import numpy as np
import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, decoder, prior, encoder, px_z_dist_fn, pz_dist_fn, qz_x_dist_fn):
        super(VAE, self).__init__()
        self.decoder = decoder
        self.encoder = encoder
        self.prior = prior

        self.px_z_dist_fn = px_z_dist_fn
        self.pz_dist_fn = pz_dist_fn
        self.qz_x_dist_fn = qz_x_dist_fn

        self.p_params = list(self.decoder.parameters()) + list(self.prior.parameters())
        self.q_params = list(self.encoder.parameters())

    def forward(self, x, num_samples=1):
        # Pass through the VAE q(z|x)p(x|z)
        qz_x_dist = self.qz_x_dist(x)
        z = qz_x_dist.rsample((num_samples, ))  # (num_samples, n, d)
        px_z_dist = self.px_z_dist(z)
        pz_dist = self.pz_dist()
        return {"x": x,
                # "x_recon": px_z_dist.sample(),
                "z": z,
                "log_px_z": px_z_dist.log_prob(x),
                "log_pz": pz_dist.log_prob(z),
                "log_qz_x": qz_x_dist.log_prob(z),
                "px_z_dist": px_z_dist,
                "qz_x_dist": qz_x_dist
                }

    def px_z_dist(self, z):
        return self.px_z_dist_fn(*self.decoder(z))

    def pz_dist(self):
        return self.pz_dist_fn(*self.prior())

    def qz_x_dist(self, x):
        return self.qz_x_dist_fn(*self.encoder(x))

    def sample(self, num_samples):
        z = self.pz_dist().sample((num_samples, ))
        px_z_dist = self.px_z_dist(z)
        return {"px_z_dist": px_z_dist
                }


class ProbVAE(nn.Module):
    def __init__(self, decoder, prior, encoder, px_z_dist_fn, pz_dist_fn, qz_x_dist_fn):
        super(ProbVAE, self).__init__()
        self.decoder = decoder
        self.encoder = encoder
        self.prior = prior

        self.px_z_dist_fn = px_z_dist_fn
        self.pz_dist_fn = pz_dist_fn
        self.qz_x_dist_fn = qz_x_dist_fn

        self.p_params = list(self.decoder.parameters()) + list(self.prior.parameters())
        self.q_params = list(self.encoder.parameters())

    def forward(self, x, num_samples=1):
        # Pass through the ProbVAE q(z|x)p(x|z)
        qz_x_dist = self.qz_x_dist(x)
        z = qz_x_dist.rsample((num_samples, ))  # (num_samples, n, d)
        px_z_dist = self.px_z_dist(z)
        pz_dist = self.pz_dist()
        return {"x": x,
                # "x_recon": px_z_dist.sample(),
                "z": z,
                "log_px_z": px_z_dist.log_prob(x),
                "log_pz": pz_dist.log_prob(z),
                "log_qz_x": qz_x_dist.log_prob(z),
                "vae": self,
                "px_z_dist": px_z_dist,
                "qz_x_dist": qz_x_dist
                }

    def px_z_dist(self, z):
        return self.px_z_dist_fn(*self.decoder(z))

    def pz_dist(self):
        return self.pz_dist_fn(*self.prior())

    def qz_x_dist(self, x):
        return self.qz_x_dist_fn(*self.encoder(x))

    def sample(self, num_samples):
        z = self.pz_dist().sample((num_samples, ))
        px_z_dist = self.px_z_dist(z)
        return {"px_z_dist": px_z_dist
                }

    def init_prior_model(self, prior_model, init_weights_with_prior=True):
        self.decoder.init_prior_module(prior_model.decoder, init_weights_with_prior)
        self.encoder.init_prior_module(prior_model.encoder, init_weights_with_prior)

    def resample(self):
        self.decoder.resample()
        self.encoder.resample()

    def set_sample_network(self, mode):
        self.decoder.set_sample_network(mode)
        self.encoder.set_sample_network(mode)

    def set_resample_network(self, mode):
        self.decoder.set_resample_network(mode)
        self.encoder.set_resample_network(mode)
    
    def set_weight_std(self, weight_std):
        self.decoder.set_weight_std(weight_std)
        self.encoder.set_weight_std(weight_std)

