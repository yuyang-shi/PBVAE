import numpy as np
import torch
import pbvae.utils as utils


# Compute VAE reconstruction loss (minibatch averaged)
def recon_loss_fn(vae_result):
    return -vae_result["log_px_z"].mean()


# Compute rescaled VAE reconstruction loss (minibatch averaged)
def rescaled_recon_loss_fn(vae_result, loss_range):
    return utils.rescale(recon_loss_fn(vae_result), loss_range)


# Compute VAE KL loss (minibatch averaged)
def kl_loss_fn(vae_result):
    return (vae_result["log_qz_x"] - vae_result["log_pz"]).mean()


# Compute VAE aggregate/marginal KL loss (minibatch averaged)
def agg_kl_loss_fn(vae_result, n):
    b = vae_result["x"].shape[0]
    log_qz_x_pairs = vae_result["qz_x_dist"].log_prob(vae_result["z"].unsqueeze(-2)).view(-1, b, b)  # log_qzi_xj
    coeffs = torch.ones([b, b], device=log_qz_x_pairs.device) * (n - 1) / (n * (b - 1))
    coeffs += torch.eye(b, device=log_qz_x_pairs.device) * (1 / n - (n - 1) / (n * (b - 1)))
    H_qz = - torch.logsumexp(log_qz_x_pairs + coeffs.log(), dim=-1).mean()
    return - vae_result["log_pz"].mean() - H_qz


# Compute VAE IWAE loss (minibatch averaged)
def iwae_loss_fn(vae_result):
    return - utils.logmeanexp(vae_result["log_px_z"] + vae_result["log_pz"] - vae_result["log_qz_x"], dim=0).mean()


# Compute KL of network weights
def prior_kl_loss_fn(vae_result):
    return vae_result["vae"].encoder.compute_prior_kl() + vae_result["vae"].decoder.compute_prior_kl()


# log(2*sqrt(n)/delta)
# n is the total number of data points
def delta_term_fn(n, delta, device):
    return torch.tensor([np.log(2 * np.sqrt(n) / delta)]).float().to(device)
