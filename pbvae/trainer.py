import os
from functools import partial
import json
from copy import deepcopy
from tqdm import tqdm

import numpy as np
import torch
import wandb

from ignite.engine import Events, Engine
from ignite.metrics import Average, RunningAverage
from ignite.handlers import TerminateOnNan, Checkpoint, DiskSaver
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.handlers.wandb_logger import *

import pbvae.utils as utils
from pbvae.dataset import preproc_data


def output_fn(output, name):
    return output[name]


class Trainer:
    def __init__(
            self,
            model,
            prior_model,
            train_loader,
            train_prior_loader,
            eval_bound_loader,
            valid_loader,
            test_loader,
            train_loss_fn,
            prior_loss_fn,
            optimizer,
            prior_optimizer,
            eval_bound_fn,
            test_metric_fn,
            config,
            run_name,
            log_dir
    ):
        self.model = model
        self.prior_model = prior_model

        self.train_loader = train_loader
        self.train_prior_loader = train_prior_loader
        self.eval_bound_loader = eval_bound_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.train_loss_fn = train_loss_fn
        self.prior_loss_fn = prior_loss_fn
        self.optimizer = optimizer
        self.prior_optimizer = prior_optimizer
        self.eval_bound_fn = eval_bound_fn
        self.test_metric_fn = test_metric_fn

        self.config = config
        self.run_name = run_name
        self.log_dir = log_dir

        self.training = True

        self.trainer = Engine(self.train_batch)
        self.prior_trainer = Engine(self.train_prior_batch)

        to_save = {"trainer": self.trainer, "prior_trainer": self.prior_trainer, "optimizer": self.optimizer,
                   "prior_optimizer": self.prior_optimizer, "model": self.model, "prior_model": self.prior_model}

        # Setup trainer
        self.trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
        if not self.config["nosave"]:
            self.trainer.add_event_handler(Events.EPOCH_COMPLETED, Checkpoint(to_save, DiskSaver(
                os.path.join(self.log_dir, 'checkpoints'), create_dir=True, require_empty=False),
                                                                              filename_pattern="latest.pt"))
            self.trainer.add_event_handler(Events.COMPLETED, Checkpoint(to_save, DiskSaver(
                os.path.join(self.log_dir, 'checkpoints'), create_dir=True, require_empty=False),
                                                                        filename_pattern="final.pt"))

        self.trainer.add_event_handler(Events.EPOCH_COMPLETED(every=self.config["test_freq"]), self.test)
        RunningAverage(output_transform=lambda x: x[0]).attach(self.trainer, "train_loss")
        ProgressBar(persist=False).attach(self.trainer, ["train_loss"])

        # Setup prior_trainer
        self.prior_trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
        if not self.config["nosave"]:
            self.trainer.add_event_handler(Events.EPOCH_COMPLETED, Checkpoint(to_save, DiskSaver(
                os.path.join(self.log_dir, 'checkpoints'), create_dir=True, require_empty=False),
                                                                              filename_pattern="latest.pt"))

        RunningAverage(output_transform=lambda x: x[0]).attach(self.prior_trainer, "prior_loss")
        ProgressBar(persist=False).attach(self.prior_trainer, ["prior_loss"])

        if not self.config["nosave"]:
            try:
                with open(os.path.join(self.log_dir, 'wandb_id.txt')) as f:
                    id = f.read()

            except FileNotFoundError:
                id = wandb.util.generate_id()

                with open('wandb_id.txt', 'w') as f:
                    f.write(id)

            wandb.init(
                project="pbvae",
                name=self.run_name,
                config=self.config,
                tags=[self.config["dataset"], self.config["train_config"]["type"], self.config["net_type"]],
                resume="allow",
                id=id
            )
            wandb.watch(self.model)

            self.trainer.add_event_handler(Events.ITERATION_COMPLETED(every=self.config["train_log_freq"]),
                                           self.log_training)
            self.prior_trainer.add_event_handler(Events.ITERATION_COMPLETED(every=self.config["train_log_freq"]),
                                                 self.log_training_prior)

            train_recon_x_target_grid = utils.make_grid(self.train_loader.dataset[:64], self.config)
            wandb.log({"train_recon_x_target": wandb.Image(utils.PIL_Image_from_tensor(train_recon_x_target_grid,
                                                                                       self.config))})

            test_recon_x_target_grid = utils.make_grid(self.test_loader.dataset[:64], self.config)
            wandb.log({"test_recon_x_target": wandb.Image(utils.PIL_Image_from_tensor(test_recon_x_target_grid,
                                                                                      self.config))})

        try:
            Checkpoint.load_objects(to_save, torch.load(os.path.join(self.log_dir, 'checkpoints', "final.pt")))
            print("Loaded final.pt")
        except FileNotFoundError:
            print("Did not find final checkpoint.")
            try:
                Checkpoint.load_objects(to_save, torch.load(os.path.join(self.log_dir, 'checkpoints', "latest.pt")))
                print("Loaded latest.pt")
            except FileNotFoundError:
                print("Did not find latest checkpoint.")

        # Setup bound evaluator
        self.bound_evaluator = Engine(self.eval_batch)
        for comp_name in self.eval_bound_fn.comp_names:
            Average(output_transform=partial(output_fn, name=comp_name)).attach(self.bound_evaluator, comp_name)
        ProgressBar(persist=False, desc="Evaluating").attach(self.bound_evaluator)
        self.bound_evaluator.add_event_handler(Events.EPOCH_COMPLETED, self.log_eval_bound)
        self.bound_evaluator.add_event_handler(Events.EPOCH_STARTED, self.model.resample)

        # Setup tester
        self.tester = Engine(self.test_batch)
        for comp_name in self.test_metric_fn.comp_names:
            Average(output_transform=partial(output_fn, name=comp_name)).attach(self.tester, comp_name)
        ProgressBar(persist=False, desc="Testing").attach(self.tester)
        self.tester.add_event_handler(Events.ITERATION_STARTED, self.model.resample)

    def train(self):
        if self.config["train_prior"] is True and self.prior_trainer.state.epoch < \
                self.config["train_prior_config"]["max_epochs"]:
            self.prior_trainer.run(self.train_prior_loader, max_epochs=self.config["train_prior_config"]["max_epochs"])
            self.model.init_prior_model(self.prior_model, self.config["init_weights_with_prior"])
        if self.config["train_config"]["pacbayes"]:
            self.test_prior()
        if self.trainer.state.epoch < self.config["train_config"]["max_epochs"]:
            self.trainer.run(self.train_loader, max_epochs=self.config["train_config"]["max_epochs"])

    def train_batch(self, engine, x):
        self.model.train()
        self.optimizer.zero_grad()

        x = preproc_data(x, self.config)
        vae_result = self.model.forward(x)

        loss, loss_comp = self.train_loss_fn(vae_result)
        loss.mean().backward()

        self.optimizer.step()
        return loss, loss_comp

    def train_prior_batch(self, engine, x):
        self.prior_model.train()
        self.prior_optimizer.zero_grad()

        x = preproc_data(x, self.config)
        vae_result = self.prior_model.forward(x)

        loss, loss_comp = self.prior_loss_fn(vae_result)
        loss.mean().backward()

        self.prior_optimizer.step()
        return loss, loss_comp

    @torch.no_grad()
    def eval_bound(self):
        self.model.eval()
        self.model.set_resample_network(False)  # Random weight but bound_evaluator only resamples every epoch

        self.eval_bound_results = []
        self.bound_evaluator.run(self.eval_bound_loader, max_epochs=self.config["eval_config"]["num_mc_samples"])
        assert len(self.eval_bound_results) == self.config["eval_config"]["num_mc_samples"]
        avg_vae_result = {
            comp_name: sum([eval_bound_sample[comp_name] for eval_bound_sample in self.eval_bound_results]) / len(
                self.eval_bound_results) for comp_name in self.eval_bound_fn.comp_names}
        eval_bound_result = self.eval_bound_fn.eval_bound(avg_vae_result["rescaled_recon_loss"],
                                                          avg_vae_result["prior_kl_loss"])
        recon_loss_std = np.std([eval_bound_sample["recon_loss"] for eval_bound_sample in self.eval_bound_results])
        print("recon_loss_std: ", recon_loss_std)
        avg_vae_result["recon_loss_std"] = recon_loss_std

        unnormalized_eval_bound_result = {}
        for k, v in eval_bound_result.items():
            eval_bound_result[k] = v.item()
            unnormalized_eval_bound_result["unnormalized_" + k] = utils.unnormalize(eval_bound_result[k],
                                                                                    self.config["loss_range"])

        eval_bound_result = {**avg_vae_result, **eval_bound_result, **unnormalized_eval_bound_result}

        print(json.dumps(eval_bound_result))

        if not self.config["nosave"]:
            wandb.log({r'eval_bound/' + k: v for k, v in eval_bound_result.items()},
                      step=self.current_log_step())

    @torch.no_grad()
    def eval_batch(self, engine, x):
        x = preproc_data(x, self.config)
        vae_result = self.model(x)
        bound_comp = self.eval_bound_fn(vae_result)
        return bound_comp

    @torch.no_grad()
    def log_eval_bound(self):
        self.eval_bound_results.append(deepcopy(self.bound_evaluator.state.metrics))

    @torch.no_grad()
    def test(self):
        self.model.eval()
        self.model.set_resample_network(False)  # Random weight but tester only resamples every iteration

        self.tester.run(self.train_loader)
        train_test_results = deepcopy(self.tester.state.metrics)
        print(json.dumps(train_test_results))

        if self.config["validate"]:
            self.tester.run(self.valid_loader)
            valid_test_results = deepcopy(self.tester.state.metrics)
            print(json.dumps(valid_test_results))

        self.tester.run(self.test_loader)
        test_test_results = deepcopy(self.tester.state.metrics)
        print(json.dumps(test_test_results))

        if self.training:
            del train_test_results["iwae_loss" + str(self.config["test_config"]["iwae_num_samples"])]
            if self.config["validate"]:
                del valid_test_results["iwae_loss" + str(self.config["test_config"]["iwae_num_samples"])]
            del test_test_results["iwae_loss" + str(self.config["test_config"]["iwae_num_samples"])]

        additional_test_results = {"gen_gap": test_test_results["recon_loss"] - train_test_results["recon_loss"]}
        print(json.dumps(additional_test_results))

        if not self.config["nosave"]:
            wandb.log({r'test/train_' + k: v for k, v in train_test_results.items()},
                      step=self.current_log_step())
            if self.config["validate"]:
                wandb.log({r'test/valid_' + k: v for k, v in valid_test_results.items()},
                          step=self.current_log_step())
            wandb.log({r'test/test_' + k: v for k, v in test_test_results.items()},
                      step=self.current_log_step())
            wandb.log({r'test/' + k: v for k, v in additional_test_results.items()},
                      step=self.current_log_step())

            sampled_x = self.model.sample(64)["px_z_dist"].mean
            sampled_x_grid = utils.make_grid(sampled_x, self.config)
            wandb.log({"sampled_x": wandb.Image(utils.PIL_Image_from_tensor(sampled_x_grid, self.config))},
                      step=self.current_log_step())

            recon_x = self.model(preproc_data(self.train_loader.dataset[:64].clone(), self.config))["px_z_dist"].mean[0]
            recon_x_grid = utils.make_grid(recon_x, self.config)
            wandb.log({"train_recon_x": wandb.Image(utils.PIL_Image_from_tensor(recon_x_grid, self.config))},
                      step=self.current_log_step())

            recon_x = self.model(preproc_data(self.test_loader.dataset[:64].clone(), self.config))["px_z_dist"].mean[0]
            recon_x_grid = utils.make_grid(recon_x, self.config)
            wandb.log({"test_recon_x": wandb.Image(utils.PIL_Image_from_tensor(recon_x_grid, self.config))},
                      step=self.current_log_step())

    def test_prior(self):
        self.model.eval()
        self.model.set_sample_network(False)

        self.tester.run(self.train_loader)
        train_test_results = deepcopy(self.tester.state.metrics)
        print(json.dumps(train_test_results))

        self.tester.run(self.test_loader)
        test_test_results = deepcopy(self.tester.state.metrics)
        print(json.dumps(test_test_results))

        del train_test_results["iwae_loss" + str(self.config["test_config"]["iwae_num_samples"])]
        del test_test_results["iwae_loss" + str(self.config["test_config"]["iwae_num_samples"])]

        additional_test_results = {"gen_gap": test_test_results["recon_loss"] - train_test_results["recon_loss"]}
        print(json.dumps(additional_test_results))

        if not self.config["nosave"]:
            wandb.log({r'test_prior/train_' + k: v for k, v in train_test_results.items()},
                      step=self.current_log_step())
            wandb.log({r'test_prior/test_' + k: v for k, v in test_test_results.items()},
                      step=self.current_log_step())
            wandb.log({r'test_prior/' + k: v for k, v in additional_test_results.items()},
                      step=self.current_log_step())

        self.model.set_sample_network(True)

    @torch.no_grad()
    def test_pz(self):
        self.model.eval()
        self.model.set_resample_network(False)  # Random weight but tester only resamples every iteration

        # self.tester.run(self.train_loader)
        # train_test_results = deepcopy(self.tester.state.metrics)
        # print(json.dumps(train_test_results))

        self.tester.run(self.test_loader)
        test_test_results = deepcopy(self.tester.state.metrics)
        print(json.dumps(test_test_results))

        if not self.config["nosave"]:
            wandb.log({r'test_pz/train_' + k: v for k, v in train_test_results.items()},
                      step=self.current_log_step())
            wandb.log({r'test_pz/test_' + k: v for k, v in test_test_results.items()},
                      step=self.current_log_step())

            sampled_x = self.model.sample(64)["px_z_dist"].mean
            sampled_x_grid = utils.make_grid(sampled_x, self.config)
            wandb.log({"pz_sampled_x": wandb.Image(utils.PIL_Image_from_tensor(sampled_x_grid, self.config))},
                      step=self.current_log_step())

    @torch.no_grad()
    def test_batch(self, engine, x):
        x = preproc_data(x, self.config)
        if self.training:
            vae_result = self.model(x)
        else:
            vae_result = self.model(x, num_samples=self.config["test_config"]["iwae_num_samples"])
        metrics = self.test_metric_fn(vae_result)
        return metrics

    def log_training(self):
        wandb.log({r'train/' + k: v for k, v in self.trainer.state.output[1].items()},
                  step=self.current_log_step())

    def log_training_prior(self):
        wandb.log({r'train_prior/' + k: v for k, v in self.prior_trainer.state.output[1].items()},
                  step=self.current_log_step())
    
    def current_log_step(self):
        return self.prior_trainer.state.iteration + self.trainer.state.iteration