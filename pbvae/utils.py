import numpy as np
import torch
import torchvision

from PIL import Image


def logmeanexp(values, dim, keepdim=False):
    return torch.logsumexp(values, dim, keepdim) - np.log(values.shape[dim])


def rescale(loss, loss_range):
    assert len(loss_range) == 2
    assert loss_range[0] < loss_range[1]
    rescaled_loss = (loss - loss_range[0]) / (loss_range[1] - loss_range[0])
    assert 0 <= rescaled_loss <= 1, rescaled_loss
    return rescaled_loss


def unnormalize(rescaled_loss, loss_range):
    # assert 0 <= rescaled_loss <= 1, rescaled_loss
    assert len(loss_range) == 2
    assert loss_range[0] < loss_range[1]
    loss = rescaled_loss * (loss_range[1] - loss_range[0]) + loss_range[0]
    return loss


def make_grid(tensor, config):
    if config["dataset"] in ["mnist", "b-mnist", "f-mnist", "omniglot", "omniglot_original"]:
        return torchvision.utils.make_grid(tensor.view(-1, 1, 28, 28))[0]


def PIL_Image_from_tensor(tensor, config):
    array = tensor.detach().cpu().numpy()
    if config["dataset"] in ["mnist", "b-mnist", "f-mnist", "omniglot", "omniglot_original"]:
        if np.max(array) <= 1:
            array *= 255
        return Image.fromarray(array.astype(np.uint8), mode="L")
