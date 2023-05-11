# Description: Contains Utility functions for preprocessing the image dataset
# author: Kolade Gideon @Allaye
# github: www.github.com/allaye
# created: 2023-03-06
# last modified: 2023-03-29


import os
import json
import re
import glob
import random
import numpy as np
import torch
from typing import Callable
from PIL import Image



def add_noise_to_image(read_path: str, save_path: str) -> None:
    """
    add random noise to an image
    :return: None
    """
    # get all the files in the directory in order alphabetically
    paths = returns_files_in_order(read_path)
    for idx, path in enumerate(paths):
        # pick a random noise type to apply to the image
        noiser = random_noise_type()
        # open the image with a context manager and ensure the image is in RGB format
        with Image.open(path) as img:
            img = img.convert('RGB')
            # convert the image to a numpy array
            img = np.array(img)
            # apply the noise to the image
            img = noiser(img)
            # return the image to a PIL image/format
            img = Image.fromarray(img.astype('uint8'))
            print(f'saving image "{idx} to file')
            img.save(save_path + 'noisey' + path.split('\\')[-1])
    return None


def random_noise_type() -> Callable:
    """
    select a random noise type to add to an image
    :return: func
    """
    # select and return a random noise type
    noise_types = [gaussian_noise, poisson_noise, color_noise, speckle_noise]
    return random.choice(noise_types)


def gaussian_noise(image, mean=0, var=10) -> np.ndarray:
    """
    Gaussian-distributed additive noise.

    :return: np.ndarray
    """

    if len(image.shape) == 2:
        image = image.convert('RGB')
    row, col, ch = image.shape
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, (row, col))
    noisy_image = np.zeros(image.shape, np.float32)
    noisy_image[:, :, 0] = image[:, :, 0] + gaussian
    noisy_image[:, :, 1] = image[:, :, 1] + gaussian
    noisy_image[:, :, 2] = image[:, :, 2] + gaussian
    return noisy_image


def poisson_noise(image):
    """
    Poisson-distributed noise generated from the data.

    :return: func
    """
    if len(image.shape) == 2:
        image = image.convert('RGB')
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(image * vals) / float(vals)
    return noisy + image


def color_noise(image) -> np.ndarray:
    """
    fused a random color of the size of the image,
    with a random noise mask of the same siz

    :return: func
    """
    if len(image.shape) == 2:
        image = image.convert('RGB')
    h, w, c = image.shape
    noise = (800 * np.random.random((h, w, c))).clip(0, 255).astype(np.uint8)
    rand_color = random.choice([(255, 0, 0), (0, 255, 0), (0, 0, 255)])
    color = np.full_like(image, rand_color, dtype=np.uint8)
    masked_image = np.bitwise_and(image, noise)
    masked_color = np.bitwise_and(color, noise)
    result = masked_image + masked_color
    return result


def speckle_noise(image):
    """
    Multiplicative noise using out = image + n*image,where
    n is uniform noise with specified mean & variance.

    :return: func
    """
    if len(image.shape) == 2:
        image = image.convert('RGB')
    row, col, ch = image.shape
    gauss = np.random.randn(row, col, ch)
    gauss = gauss.reshape(row, col, ch)
    noisy = image + image * gauss
    return noisy


def returns_files_in_order(path: str) -> list:
    files = glob.glob(path)
    return sorted(files, key=lambda x: float(re.findall("(\d+)", x)[0]) if re.findall("(\d+)", x) else -1)


def inferance(image: np.ndarray) -> np.ndarray:
    """
    :param image: np.ndarray
    :return: np.ndarray
    """
    pass


def hyperparameter_tuning():
    num_epochs = 20
    learning_rate = 0.01  # karpathy's constant
    batch_size = 2
    num_workers = 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return num_epochs, learning_rate, batch_size, num_workers, device


def experiment(**kwargs):
    expe = kwargs
    # write to a file
    with open(f'./experiments/action_{expe.get("experiment", None)}.json', 'w') as f:
        json.dump(expe, f, indent=True)

#
# def args(*args, **kwargs):
#     ex = kwargs
#     return ex
#
#
# aa = {
#         'loss': 0.122,
#         'hyper': {'num_epochs': 20, 'learning_rate': 0.0001, 'batch_size': 2, 'num_workers': 2, 'device': 'cuda:0'},
#         'archi': {'model': 'unet', 'in_channels': 3, 'out_channels': 1, 'init_features': 32, 'dropout': 0.1},
#         'exp_num': 0o_0_1
#     }
# # print('arguments', args(**aa), aa['a'])
# experiment(**aa)
# img1 = returns_files_in_order("../data/train/*.jpg")
# print('images', len(img1))
# img2 = returns_files_in_order()

#
# im = Image.open("../data/test/img_0.jpg")
# print(np.array(im))
# print(im.size)
# print(im.mode)
# print(im.format)
# print(len(sorted(glob.glob("../data/train/*.tiff"), key=lambda x: float(re.findall('(\d+)', x)[0]))))
# for path in paths:
#     with Image.open(path) as img:
#         i = np.array(img)
#         print(i.shape)
#         image = color_noise(np.array(img))
#         image = Image.fromarray(image.astype('uint8'))
#         image.show()

# add_noise_to_image("../data/train/*.*", "../data/noisey_train/")

# print('files', returns_files_in_order("../data/noisey_train/*.*"))
