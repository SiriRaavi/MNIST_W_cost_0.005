import theano
import numpy as np
from scipy.ndimage import rotate, shift
import scipy.misc
import matplotlib.pyplot as mplimg
from scipy.misc import imread, imresize

import os
import random


def flatten(lis):
    c = []
    for a in lis:
        for b in a:
            c.append(b)
    return c


def get_shuffled_images(paths, labels, nb_samples=None):
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images = [(i, os.path.join(path, image))
              for i, path in zip(labels, paths)
              for image in sampler(os.listdir(path))]
    random.shuffle(images)
    return images


def modified_get_shuffled_images(paths, labels, nb_samples=None):
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images = [(i, os.path.join(path, image))
              for i, path in zip(labels, paths)
              for image in sampler(os.listdir(path))]
    images1 = images[:len(images) / 2]
    images2 = images[len(images) / 2:]
    return flatten(zip(images1, images2))


def time_offset_input(labels_and_images):
    labels, images = zip(*labels_and_images)
    time_offset_labels = (None,) + labels[:-1]
    return zip(images, time_offset_labels)


def load_transform(image_path, angle=0., s=(0, 0), var=0, amount=0.05):
    # Load the image
    original = imread(image_path, flatten=True)
    # Rotate and normalize the image
    rotated = np.maximum(np.minimum(rotate(original, angle=angle, mode='nearest', reshape=False), 255.), 0.) / 255.
    # Shift the image
    shifted = shift(rotated, shift=s)
    # Add distortion
    distorted = add_noise(shifted, mean=0, var=var, amount=amount, mode='pepper')
    return distorted


def add_noise(image, mean=0, var=0.1, amount=0.01, mode='pepper'):
    """Adding noise to image.
    :param image: Square numpy array of the input image
    :param mean: mean of the Gaussian noise
    :param var: variance of the Gaussian noise
    :param amount: amount of the noise added for 'pepper' and 's&p'
    :param mode: Noise type.
    """
    if mode == 'gaussian':
        gauss = np.random.normal(mean, var, image.shape)
        image = image + gauss
    elif mode == 'pepper':
        num_pepper = np.ceil(amount * image.size)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        image[coords] = 0
    elif mode == "s&p":
        s_vs_p = 0.5
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        image[coords] = 1
        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        image[coords] = 0
    return image