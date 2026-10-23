"""Perturbations used to remove intensity and texture information from the patches."""

import numpy as np


def normalize(img):
    """Intensity normalization (dataset D_{-i}): zero mean and unit standard deviation."""
    img = img.astype(float)
    return (img - np.mean(img)) / np.std(img)


def shuffle_pixels(img, seed):
    """Pixel shuffling (dataset D_{-t}).

    Note: the random generator is reseeded with `seed` on every call, as in the code
    that produced the results of the paper. Since all patches have the same size, the
    same permutation is applied to every patch of a training run.
    """
    np.random.seed(seed)
    shape = img.shape
    img = img.flatten()
    np.random.shuffle(img)
    return img.reshape(shape)


# Perturbation name -> (shuffle pixels, normalize intensities)
PERTURBATIONS = {
    "none": (False, False),  # D
    "i": (False, True),      # D_{-i}
    "t": (True, False),      # D_{-t}
    "it": (True, True),      # D_{-i,-t}
}

# Names used in the original checkpoint files
LEGACY_NAMES = {
    "none": "T_e_I_mantida",
    "i": "T_mantida",
    "t": "I_mantida",
    "it": "T_e_I_perdidas",
}
