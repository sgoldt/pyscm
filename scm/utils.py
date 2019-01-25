#!/usr/bin/env python3
#
# Diverse utilities
#
# Date: October 2018
#
# Author: Sebastian Goldt <goldt.sebastian@gmail.com>

import numpy as np
import numpy.random as rnd

import os

# Various constants that are useful when playing around...
MNIST = 1
FMNIST = 2
ISING = 3
TRAIN = 1
TEST = 2

# activation functions
LINEAR = 0
ERF = 1
RELU = 2

# weight initialisations
INIT_LARGE = 1
INIT_SMALL = 2
INIT_INFORMED = 3
INIT_DENOISE = 4


def activation_name(g, short=True):
    """
    Returns a string representation of the given activation function.

    Parameters:
    -----------
    g : int, function
        either an integer code for the activation function used, or the
        activation itself.
    """
    name = None
    if isinstance(g, int):
        if g == ERF:
            name = "erf"
        elif g == RELU:
            name = "relu"
        elif g == LINEAR:
            name = "lin"
    else:
        from scm import scm
        name = ("erf" if g == scm.g_erf else "relu")

    if name is None:
        raise ValueError("parameter not a valid activation function as defined "
                         "in scm.utils!")

    if not short:
        name = name.capitalize().replace("lu", "LU")

    return name


def dataset_name(dataset):
    name = ""
    if dataset == MNIST:
        name = "mnist"
    elif dataset == FMNIST:
        name = "fmnist"
    elif dataset == ISING:
        name = "ising"
    else:
        msg = "parameter not a valid dataset as defined in scm.utils!"
        raise ValueError(msg)
    return name


def get_overlaps(raw, M, K, row=-1):
    """
    Returns the student's self-overlap matrix Q and the teacher-student overlap
    matrix R from the given raw data array from a logfile

    Returns:
    --------
    Q : (K, K)
        student self-overlap
    R : (K, M)
        teacher-student overlap
    """
    Q00_idx = 4  # index of the first Q entry in the logfile
    QKK_idx = (Q00_idx - 1) + int(0.5 * K * (K + 1))  # idx of the last Q entry
    R00_idx = QKK_idx + 1  # index of the first R entry in the logfile
    RKM_idx = R00_idx + K * M - 1  # index of the last R entry in the logfile

    Q = np.zeros((K, K))
    Q[np.triu_indices(K)] = raw[row, Q00_idx:(QKK_idx + 1)]
    Q = 0.5 * (Q + Q.T)

    R = raw[row, R00_idx:(RKM_idx + 1)].reshape(K, M)

    return Q, R


def frac_error(a, b):
    """
    Returns the fraction of entries where these two arrays do not have the same
    entry.
    """
    if not a.shape == b.shape:
        raise ValueError("a and b do not have the same dimensions: ",
                         a.shape, b.shape)

    polar = (-1 in a) or (-1 in b)

    if not polar:  # ie binary encoded
        a = 2 * a - 1
        b = 2 * b - 1
    return 1 - np.mean(0.5 * (a * b + 1))


def load_ising(mode, temperature, debug=False):
    """
    Loads the ising data set which consists only of inputs!

    Parameters:
    -----------
    mode : TRAIN or TEST
    debug :
        if True, load shortened MNIST dataset for debugging purposes.

    Returns:
    --------
        numpy arrays
    """
    if mode not in [TEST, TRAIN]:
        raise ValueError("train has to be either TEST or TRAIN")

    prefix = ("train" if mode == TRAIN else "test")
    postfix = ("_debug.npz" if debug else ".npz")
    fname = prefix + ("_ising_2D_N784_T%g" % temperature) + postfix
    fname_full = os.path.join(os.path.expanduser("~"), "datasets", "ising",
                              fname)
    print("Loading data from %s" % fname_full)

    xis = np.load(fname_full)['samples']

    # center and re-scale
    xis -= np.mean(xis)
    xis /= np.sqrt(np.var(xis))

    return xis


def load_mnist(mode, dataset=MNIST, debug=False, randomise=False):
    """
    Loads the (F)MNIST data set and returns it with the images centered and
    with unit variance.

    Parameters:
    -----------
    mode : TRAIN or TEST
    dataset : int
        MNIST or FMNIST (constants defined in this module)
    debug :
        if True, load shortened MNIST dataset for debugging purposes.
    randomise :
        if True, randomise the labels of the images.

    Returns:
    --------
        numpy arrays
    """
    if mode not in [TEST, TRAIN]:
        raise ValueError("train has to be either TEST or TRAIN")

    fashion = (dataset == FMNIST)
    prefix = ("fashion-mnist_" if fashion else "mnist_")
    postfix = ("_debug.csv" if debug else ".csv")
    fname = prefix + ("train" if mode == TRAIN else "test") + postfix
    fname_full = os.path.join(os.path.expanduser("~"), "datasets",
                              ("fashion-mnist" if fashion else "mnist"),
                              fname)
    print("Loading data from %s" % fname_full)

    data = np.loadtxt(fname_full, delimiter=',')
    rnd.shuffle(data)  # in-place method

    # rescaled inputs with mean 0, variance 1
    N = data.shape[1]
    xis = data[:, 1:N+1]
    xis -= np.mean(xis)
    xis /= np.sqrt(np.var(xis))

    # labels
    ys = data[:, 0]

    if randomise:
        rnd.shuffle(ys)  # in-place method

    return (xis, ys)


def load_mnist_oe(mode, dataset=MNIST, polar=False, debug=False,
                  randomise=False):
    """
    Loads the (F)MNIST data set and returns it with the images centered and
    with unit variance.

    Labels: polar: -1 -> even, 1 -> odd
           binary: 0 -> even, 1 -> odd

    Parameters:
    -----------
    mode : TRAIN or TEST
    polar :
        if True, will encode classes as -1, 1. Default False.
    dataset : int
        MNIST or FMNIST (constants defined in this module)
    debug :
        if True, load shortened MNIST dataset for debugging purposes.
    randomise :
        if True, randomise the labels of the images.

    Returns:
    --------
        numpy arrays
    """
    # returns numpy tensors
    xis, ys = load_mnist(mode, dataset, debug, randomise)

    ys = np.mod(ys, 2)  # encode labels with 0, 1
    if polar:  # encode labels with \pm 1 for g=erf
        ys = 2 * ys - 1

    return xis, ys


def optimal_lr_erf(K):
    """
    Returns the optimal learning rate for training a SCM with Erf activation
    function when you keep the learning rate constant throughout and the student
    has the same number of hidden units as the teacher[SS].

    Parameters:
    -----------
    K : int
        number of hidden units in the student.

    References
    ----------
    .. [SS] Saad, D., & Solla, S. A., 1995, "On-line learning in soft committee
            machines". Physical Review E, 52(4), 4225–4243.
    """
    return 2 * np.pi * np.sqrt(3) / (3 * (K - 1 + 3/np.sqrt(5)))
