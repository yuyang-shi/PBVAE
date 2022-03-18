import torch
from pbvae.models.vae_functions import *


class VAE_Metric:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.comp_names = ["recon_loss", "rescaled_recon_loss", "kl_loss", "elbo_loss",
                           "iwae_loss" + str(self.config["iwae_num_samples"])]

    def __call__(self, vae_result):
        recon_loss = recon_loss_fn(vae_result)
        rescaled_recon_loss = rescaled_recon_loss_fn(vae_result, self.config["loss_range"])
        kl_loss = kl_loss_fn(vae_result)
        elbo_loss = recon_loss + kl_loss
        if vae_result["log_px_z"].shape[0] == self.config["iwae_num_samples"]:
            iwae_loss = iwae_loss_fn(vae_result)
        else:
            iwae_loss = 0.
       
        metric_comp = {
            "recon_loss": recon_loss,
            "rescaled_recon_loss": rescaled_recon_loss,
            "kl_loss": kl_loss,
            "elbo_loss": elbo_loss,
            "iwae_loss" + str(self.config["iwae_num_samples"]): iwae_loss
        }
        return metric_comp
