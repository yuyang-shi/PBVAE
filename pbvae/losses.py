import torch
from pbvae.models.vae_functions import *


class VAE_Loss:
    def __init__(self, model, config, n):
        self.model = model
        self.config = config
        self.n = n

    def __call__(self, vae_result):
        if self.config["type"] == "beta-vae":
            return self.beta_vae(vae_result)
        elif self.config["type"] == "alpha-vae":
            return self.alpha_vae(vae_result)
        elif self.config["type"] == "classic":
            return self.classic(vae_result)
        elif self.config["type"] == "quad":
            return self.quad(vae_result)
        # elif self.config["type"] == "classic-alpha-vae":
        #     return self.classic_alpha_vae(vae_result)
        # elif self.config["type"] == "quad-alpha-vae":
        #     return self.quad_alpha_vae(vae_result)
        # elif self.config["type"] == "classic-beta-vae":
        #     return self.classic_beta_vae(vae_result)
        # elif self.config["type"] == "quad-beta-vae":
        #     return self.quad_beta_vae(vae_result)
        else:
            raise NotImplementedError

    def beta_vae(self, vae_result):
        recon_loss = recon_loss_fn(vae_result)
        rescaled_recon_loss = rescaled_recon_loss_fn(vae_result, self.config["loss_range"])
        kl_loss = kl_loss_fn(vae_result)
        if self.config["beta"] == 0:
            beta_vae_loss = recon_loss
        else:
            beta_vae_loss = recon_loss + kl_loss * self.config["beta"]
        loss_comp = {
            "recon_loss": recon_loss,
            "rescaled_recon_loss": rescaled_recon_loss,
            "kl_loss": kl_loss,
            "beta_vae_loss": beta_vae_loss
        }
        return beta_vae_loss, loss_comp

    def alpha_vae(self, vae_result):
        recon_loss = recon_loss_fn(vae_result)
        rescaled_recon_loss = rescaled_recon_loss_fn(vae_result, self.config["loss_range"])
        agg_kl_loss = agg_kl_loss_fn(vae_result, n=self.n)
        if self.config["alpha"] == 0:
            alpha_vae_loss = recon_loss
        else:
            alpha_vae_loss = recon_loss + agg_kl_loss * self.config["alpha"]
        loss_comp = {
            "recon_loss": recon_loss,
            "rescaled_recon_loss": rescaled_recon_loss,
            "agg_kl_loss": agg_kl_loss,
            "alpha_vae_loss": alpha_vae_loss
        }
        return alpha_vae_loss, loss_comp

    def classic(self, vae_result):
        recon_loss = recon_loss_fn(vae_result)
        rescaled_recon_loss = rescaled_recon_loss_fn(vae_result, self.config["loss_range"])
        prior_kl_loss = prior_kl_loss_fn(vae_result) * self.config["kl_penalty"]
        frac_term = (prior_kl_loss + delta_term_fn(n=self.n, delta=self.config["delta"],
                                                   device=self.config["device"])) / (2 * self.n)
        classic_bound = rescaled_recon_loss + torch.sqrt(frac_term)

        loss_comp = {
            "recon_loss": recon_loss,
            "rescaled_recon_loss": rescaled_recon_loss,
            "prior_kl_loss": prior_kl_loss,
            "classic_bound": classic_bound
        }
        return classic_bound, loss_comp

    def quad(self, vae_result):
        recon_loss = recon_loss_fn(vae_result)
        rescaled_recon_loss = rescaled_recon_loss_fn(vae_result, self.config["loss_range"])
        prior_kl_loss = prior_kl_loss_fn(vae_result) * self.config["kl_penalty"]
        frac_term = (prior_kl_loss + delta_term_fn(n=self.n, delta=self.config["delta"],
                                                   device=self.config["device"])) / (2 * self.n)
        quad_bound = torch.square(torch.sqrt(rescaled_recon_loss + frac_term) + torch.sqrt(frac_term))
        loss_comp = {
            "recon_loss": recon_loss,
            "rescaled_recon_loss": rescaled_recon_loss,
            "prior_kl_loss": prior_kl_loss,
            "quad_bound": quad_bound
        }
        return quad_bound, loss_comp

