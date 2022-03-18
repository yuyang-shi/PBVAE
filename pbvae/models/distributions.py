import numpy as np
import torch
from torch.distributions import Normal, Bernoulli, Independent, Categorical, MixtureSameFamily


# Returns the independent bernoulli distribution with bounded probability
# Equivalently the entrywise bounded BCE loss
def independent_bernoulli(logits, p_min=0.):
    assert 0 <= p_min <= 0.5
    probs = torch.sigmoid(logits)
    probs = torch.clamp(probs, p_min, 1-p_min)
    return Independent(Bernoulli(probs, validate_args=False), reinterpreted_batch_ndims=1)


def independent_normal(mu, sig, reinterpreted_batch_ndims=1):
    return Independent(Normal(mu, sig), reinterpreted_batch_ndims=reinterpreted_batch_ndims)

