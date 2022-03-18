import os
import contextlib
import urllib
from pathlib import Path

import numpy as np
import scipy.io

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets

from hydra.utils import to_absolute_path


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def setup_loaders(config, batch_size=None):
    if batch_size is None:
        train_batch_size = config["train_batch_size"]
        eval_batch_size = config["eval_batch_size"]
        test_batch_size = config["test_batch_size"]
    else:
        train_batch_size = batch_size
        eval_batch_size = batch_size
        test_batch_size = batch_size

    device = torch.device(config["device"])
    data_root = to_absolute_path("data")
    if config["dataset"] in ["mnist", "f-mnist"]:
        dataset_class = {
            "mnist": datasets.MNIST,
            "f-mnist": datasets.FashionMNIST
        }[config["dataset"]]

        train_data = dataset_class(
            root=data_root,
            train=True,
            download=True
        ).data.to(device).float()

        if config["validate"]:
            valid_data = train_data[50000:]
            train_data = train_data[:50000]
            valid_loader = DataLoader(valid_data, batch_size=test_batch_size, shuffle=True)
        else:
            valid_data = None
            valid_loader = DataLoader(valid_data)

        test_data = dataset_class(
            root=data_root,
            train=False,
            download=True
        ).data.to(device).float()

    elif config["dataset"] == "b-mnist":
        def download_binarized_mnist(data_root):
            Path(os.path.join(data_root, "binarized_mnist")).mkdir(parents=True, exist_ok=True)
            for dataset in ['train', 'valid', 'test']:
                filename = 'binarized_mnist_{}.amat'.format(dataset)
                url = 'http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_{}.amat'.format(
                    dataset)
                print('Downloading from {}...'.format(url))
                local_filename = os.path.join(data_root, "binarized_mnist", filename)
                urllib.request.urlretrieve(url, local_filename)
                print('Saved to {}'.format(local_filename))

        try:
            train_data = torch.from_numpy(
                np.genfromtxt(os.path.join(data_root, "binarized_mnist", "binarized_mnist_train.amat"),
                              delimiter=" ")).to(device).float()
            valid_data = torch.from_numpy(
                np.genfromtxt(os.path.join(data_root, "binarized_mnist", "binarized_mnist_valid.amat"),
                              delimiter=" ")).to(device).float()
            test_data = torch.from_numpy(
                np.genfromtxt(os.path.join(data_root, "binarized_mnist", "binarized_mnist_test.amat"),
                              delimiter=" ")).to(device).float()
        except:
            download_binarized_mnist(data_root)
            train_data = torch.from_numpy(
                np.genfromtxt(os.path.join(data_root, "binarized_mnist", "binarized_mnist_train.amat"),
                              delimiter=" ")).to(device).float()
            valid_data = torch.from_numpy(
                np.genfromtxt(os.path.join(data_root, "binarized_mnist", "binarized_mnist_valid.amat"),
                              delimiter=" ")).to(device).float()
            test_data = torch.from_numpy(
                np.genfromtxt(os.path.join(data_root, "binarized_mnist", "binarized_mnist_test.amat"),
                              delimiter=" ")).to(device).float()

        if config["validate"]:
            valid_loader = DataLoader(valid_data, batch_size=test_batch_size, shuffle=True)
        else:
            train_data = torch.cat([train_data, valid_data], dim=0)
            valid_data = None
            valid_loader = DataLoader(valid_data)

    elif config["dataset"] == "omniglot":
        local_filename = os.path.join(data_root, "omniglot", "chardata.mat")
        try:
            raw_data = scipy.io.loadmat(local_filename)
        except FileNotFoundError:
            Path(os.path.join(data_root, "omniglot")).mkdir(parents=True, exist_ok=True)
            url = "https://github.com/yburda/iwae/raw/master/datasets/OMNIGLOT/chardata.mat"
            print('Downloading from {}...'.format(url))
            local_filename = local_filename
            urllib.request.urlretrieve(url, local_filename)
            print('Saved to {}'.format(local_filename))
            raw_data = scipy.io.loadmat(local_filename)

        train_data = torch.from_numpy(np.transpose(raw_data["data"])).to(device).float()

        with temp_seed(0):
            perm = np.random.permutation(24345)
        assert np.array_equal(perm[:3], np.array([9825, 22946, 16348]))
        train_data = train_data[perm]

        if config["validate"]:
            valid_data = train_data[23000:]
            train_data = train_data[:23000]
            valid_loader = DataLoader(valid_data, batch_size=test_batch_size, shuffle=True)
        else:
            valid_data = None
            valid_loader = DataLoader(valid_data)

        test_data = torch.from_numpy(np.transpose(raw_data["testdata"])).to(device).float()

    elif config["dataset"] == "omniglot_original":
        try:
            train_data = torch.load(os.path.join(data_root, "omniglot_original", "processed", "training.pt"))
            test_data = torch.load(os.path.join(data_root, "omniglot_original", "processed", "test.pt"))

        except FileNotFoundError:
            import shutil

            _BASE_URL = "https://github.com/brendenlake/omniglot/"
            _DL_URL = _BASE_URL + "raw/master/python/"
            _DL_URLS = {
                "train": _DL_URL + "images_background.zip",
                "eval": _DL_URL + "images_evaluation.zip"
            }

            _DL_DIR = os.path.join(data_root, "omniglot_original", "raw")
            _DL_FILENAMES = {
                "train": os.path.join(_DL_DIR, "images_background.zip"),
                "eval": os.path.join(_DL_DIR, "images_evaluation.zip")
            }

            Path(_DL_DIR).mkdir(parents=True, exist_ok=True)

            for split in ["train", "eval"]:
                print('Downloading from {}...'.format(_DL_URLS[split]))
                urllib.request.urlretrieve(_DL_URLS[split], _DL_FILENAMES[split])
                shutil.unpack_archive(_DL_FILENAMES[split], _DL_DIR)

            transform = torchvision.transforms.Compose([
                torchvision.transforms.Grayscale(num_output_channels=1),
                torchvision.transforms.Resize((28, 28)),
                torchvision.transforms.ToTensor()
            ])
            train_dataset = torchvision.datasets.ImageFolder(os.path.join(_DL_DIR, "images_background"),
                                                             transform=transform)
            train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
            train_data = torchvision.transforms.functional.invert(next(iter(train_loader))[0]).squeeze(1)

            test_dataset = torchvision.datasets.ImageFolder(os.path.join(_DL_DIR, "images_evaluation"),
                                                            transform=transform)
            test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
            test_data = torchvision.transforms.functional.invert(next(iter(test_loader))[0]).squeeze(1)

            save_dir = os.path.join(data_root, "omniglot_original", "processed")
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            torch.save(train_data, os.path.join(save_dir, "training.pt"))
            torch.save(test_data, os.path.join(save_dir, "test.pt"))

        train_data = train_data.to(device)
        test_data = test_data.to(device)

        with temp_seed(0):
            perm = np.random.permutation(19280)
        assert np.array_equal(perm[:3], np.array([9206, 3053, 1972]))
        train_data = train_data[perm]

        if config["validate"]:
            valid_data = train_data[18000:]
            train_data = train_data[:18000]
            valid_loader = DataLoader(valid_data, batch_size=test_batch_size, shuffle=True)
        else:
            valid_data = None
            valid_loader = DataLoader(valid_data)

    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=True)

    if config["train_prior"] is True:
        if config["resume"] is not None:
            loader_idx = torch.load(os.path.join(config["resume"], "loader_idx.pt"))
            train_prior_idx = loader_idx["train_prior_loader_idx"]
            prior_free_idx = loader_idx["eval_bound_loader_idx"]
        else:
            perm = torch.randperm(train_data.shape[0])
            train_prior_idx = perm[:int(np.rint(train_data.shape[0] * config["prior_split"]))]
            prior_free_idx = perm[int(np.rint(train_data.shape[0] * config["prior_split"])):]
        train_prior_loader = DataLoader(train_data[train_prior_idx], batch_size=train_batch_size,
                                        shuffle=True)
        eval_bound_loader = DataLoader(train_data[prior_free_idx], batch_size=eval_batch_size, shuffle=True)
    else:
        train_prior_idx = None
        prior_free_idx = torch.arange(train_data.shape[0])
        train_prior_loader = DataLoader(None)
        eval_bound_loader = DataLoader(train_data, batch_size=eval_batch_size, shuffle=True)
    if config["resume"] is None:
        torch.save({"train_prior_loader_idx": train_prior_idx, "eval_bound_loader_idx": prior_free_idx},
                   "loader_idx.pt")

    return train_loader, train_prior_loader, eval_bound_loader, valid_loader, test_loader


def preproc_data(x, config):
    if config["dataset"] in ["mnist", "f-mnist"]:
        x /= 255.
        if config["binarize"]:
            x = torch.bernoulli(x)
    elif config["dataset"] in ["omniglot", "omniglot_original"]:
        if config["binarize"]:
            x = torch.bernoulli(x)
    x = x.view((-1, *config["obs_dim"]))
    return x
