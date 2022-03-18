import torch
from pbvae.models.vae_functions import *


class PBVAE_Bound:
    def __init__(self, model, config, n, m):
        self.model = model
        self.config = config  # Use entries: loss_range, (eval) type, delta, device, (eval) delta_eval
        self.n = n
        self.m = m  # num_mc_samples
        self.comp_names = ["recon_loss", "rescaled_recon_loss", "prior_kl_loss"]

    def __call__(self, vae_result):
        recon_loss = recon_loss_fn(vae_result)
        rescaled_recon_loss = rescaled_recon_loss_fn(vae_result, self.config["loss_range"])
        prior_kl_loss = prior_kl_loss_fn(vae_result)
        bound_comp = {
            "recon_loss": recon_loss,
            "rescaled_recon_loss": rescaled_recon_loss,
            "prior_kl_loss": prior_kl_loss
        }
        return bound_comp

    def eval_bound(self, rescaled_recon_loss, prior_kl_loss):
        kl_inversion_bound = self.kl_inversion(rescaled_recon_loss, prior_kl_loss)
        return {"kl_inversion_bound"+str(self.m): kl_inversion_bound}

    def kl_inversion(self, rescaled_recon_loss, prior_kl_loss):
        empirical_risk = inv_kl(rescaled_recon_loss, np.log(2 / self.config["delta_eval"]) / self.m)
        risk = inv_kl(empirical_risk, (prior_kl_loss + delta_term_fn(n=self.n, delta=self.config["delta"],
                                                                     device=self.config["device"])) / self.n)
        return torch.tensor([risk]).float().to(self.config["device"])


def inv_kl(qs, ks):
    """Inversion of the binary kl

    Parameters
    ----------
    qs : float
        Empirical risk

    ks : float
        second term for the binary kl inversion

    """
    # computation of the inversion of the binary KL
    qd = 0
    ikl = 0
    izq = qs
    dch = 1 - 1e-10
    while ((dch - izq) / dch >= 1e-5):
        p = (izq + dch) * .5
        if qs == 0:
            ikl = ks - (0 + (1 - qs) * np.log((1 - qs) / (1 - p)))
        elif qs == 1:
            ikl = ks - (qs * np.log(qs / p) + 0)
        else:
            ikl = ks - (qs * np.log(qs / p) + (1 - qs) * np.log((1 - qs) / (1 - p)))
        if ikl < 0:
            dch = p
        else:
            izq = p
        qd = p
    return qd
