import datetime
import os
import json
from functools import partial
from argparse import ArgumentParser

import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import OmegaConf
import wandb

import numpy as np
import torch

from pbvae.models.vae import VAE, ProbVAE
from pbvae.models.networks import MLP, MLP_Normal, Prob_MLP, Prob_MLP_Normal, Constant_Normal
from pbvae.models.distributions import independent_normal, independent_bernoulli
from pbvae.bounds import PBVAE_Bound
from pbvae.losses import VAE_Loss
from pbvae.metrics import VAE_Metric
from pbvae.trainer import Trainer
from pbvae.dataset import setup_loaders, preproc_data


@hydra.main(config_path='config', config_name="mnist_train")
def main(config):
    if config["resume"] is not None:
        # Resume earlier run and read config
        log_dir = to_absolute_path(config["resume"])
        config_load = OmegaConf.load(os.path.join(log_dir, ".hydra", "config.yaml"))
        config_load["test"] = config["test"]
        config = config_load

        overrides = OmegaConf.from_cli(list(OmegaConf.load(os.path.join(os.getcwd(), ".hydra", "overrides.yaml"))))
        config = OmegaConf.merge(config, overrides)
        config["resume"] = log_dir

    else:
        # Initialize new run
        log_dir = os.getcwd()

    run_name = os.path.relpath(log_dir, to_absolute_path('outputs'))  # Only used for wandb

    if config["dataset"] in ["mnist", "b-mnist", "f-mnist", "omniglot", "omniglot_original"]:
        config["loss_range"] = [float(-np.log(1 - config["p_min"]) * np.prod(config["obs_dim"])),
                                float(-np.log(config["p_min"]) * np.prod(config["obs_dim"]))]

    config["train_config"]["pacbayes"] = config["train_config"]["type"] in ["classic", "quad"]
    if config["train_prior"] is True:
        config["train_prior"] = config["train_config"]["pacbayes"]
    if config["train_prior_config"]["lr"] is None:
        config["train_prior_config"]["lr"] = config["train_config"]["lr"]
    config["validate"] = not config["train_config"]["pacbayes"]
    if config["train_config"]["pacbayes"]:
        assert config["pz_config"]["type"] == "standard"

    train_loader, train_prior_loader, eval_bound_loader, valid_loader, test_loader = setup_loaders(config)
    train_data_mean = preproc_data(train_loader.dataset.clone(), config).mean(0)
    model = setup_model({**config, **config["train_config"], "pseudoinputs_mean": train_data_mean})
    prior_model = setup_prior_model({**config, **config["train_prior_config"]})

    if config["train_prior"] is True:
        prior_loss_fn = VAE_Loss(prior_model, {**config, **config["train_prior_config"]},
                                 n=train_prior_loader.dataset.shape[0])
    else:
        prior_loss_fn = None
    train_loss_fn = VAE_Loss(model, {**config, **config["train_config"]}, n=train_loader.dataset.shape[0])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train_config"]["lr"], )
    prior_optimizer = torch.optim.Adam(prior_model.parameters(), lr=config["train_prior_config"]["lr"], )
    eval_bound_fn = PBVAE_Bound(model, {**config, **config["eval_config"]}, n=eval_bound_loader.dataset.shape[0],
                                m=config["eval_config"]["num_mc_samples"])
    test_metric_fn = VAE_Metric(model, {**config, **config["test_config"]})

    trainer = Trainer(
        model=model,
        prior_model=prior_model,
        train_loader=train_loader,
        train_prior_loader=train_prior_loader,
        eval_bound_loader=eval_bound_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        train_loss_fn=train_loss_fn,
        prior_loss_fn=prior_loss_fn,
        optimizer=optimizer,
        prior_optimizer=prior_optimizer,
        eval_bound_fn=eval_bound_fn,
        test_metric_fn=test_metric_fn,
        config=config,
        run_name=run_name,
        log_dir=log_dir
    )

    if not config["test"]:
        trainer.train()
    
    if config["test_weight_std"]:
        model.set_weight_std(config["test_weight_std"])

    if not config["test"]:
        if config["train_config"]["pacbayes"]:
            trainer.eval_bound()

    trainer.training = False
    train_loader, _, _, valid_loader, test_loader = setup_loaders(config, batch_size=config["final_test_batch_size"])
    trainer.train_loader, trainer.valid_loader, trainer.test_loader = train_loader, valid_loader, test_loader
    trainer.test()

    if not config["nosave"]:
        wandb.finish()


def setup_prior_model(config):
    if config["net_type"] == "MLP":
        decoder = MLP([config["latent_dim"]] + list(config["net_dims"]) + [np.prod(config["obs_dim"])],
                      torch.nn.Tanh(), dropout_p=config["dropout_p"])
        encoder = MLP_Normal([np.prod(config["obs_dim"])] + list(config["net_dims"]) + [config["latent_dim"]],
                             torch.nn.Tanh(), dropout_p=config["dropout_p"])
    prior = Constant_Normal([config["latent_dim"]], 0, 0, True, True)

    model = VAE(decoder, prior, encoder, partial(independent_bernoulli, p_min=config["p_min"]),
                independent_normal, independent_normal).to(config["device"])
    if config["train_prior"] is True:
        print(model)
        print(f"\nNumber of parameters: {num_params(model):,}\n")
    return model


def setup_model(config):
    if config["net_type"] == "MLP":
        decoder = Prob_MLP([config["latent_dim"]] + list(config["net_dims"]) + [np.prod(config["obs_dim"])],
                           torch.nn.Tanh(), prior_std=config["prior_std"], prior_init=config["prior_init"],
                           sample_network=config["pacbayes"])
        encoder = Prob_MLP_Normal([np.prod(config["obs_dim"])] + list(config["net_dims"]) + [config["latent_dim"]],
                                  torch.nn.Tanh(), prior_std=config["prior_std"], prior_init=config["prior_init"],
                                  sample_network=config["pacbayes"])
    if config["pz_config"]["type"] == "standard":
        prior = Constant_Normal([config["latent_dim"]], 0, 0, True, True)
        model = ProbVAE(decoder, prior, encoder, partial(independent_bernoulli, p_min=config["p_min"]),
                        independent_normal, independent_normal).to(config["device"])
    else:
        raise NotImplementedError
    print(model)
    print(f"\nNumber of parameters: {num_params(model):,}\n")
    return model


def num_params(module):
    return sum(p.view(-1).shape[0] for p in module.parameters())


if __name__ == "__main__":
    main()
