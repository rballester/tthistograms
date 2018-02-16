"""
Miscellaneous utility functions
"""

import numpy as np
import os
import re
import pickle
import ttrecipes as tr
import tthistogram


def read_tensor(filename):
    """
    Read an ndarray from a file of the form name_size1_[...]_sizeN_type

    Example: bonsai_256_256_256_uint8

    :param filename:
    :return: an ndarray, and its "short" name (e.g. "bonsai")

    """

    filename = os.path.expanduser(filename)
    rs = re.search(r'(\D+)((_\d+)+)_(\w+)\.[^\.]+$', os.path.basename(filename))
    basename = rs.group(1)
    shape = [int(part) for part in rs.group(2)[1:].split("_")]
    input_type = getattr(np, rs.group(4))
    X = np.reshape(np.fromfile(filename, dtype=input_type), shape)
    return X.astype(float), basename


def prepare_dataset(data_folder, filename, B, eps):
    """
    Given a data set, quantize it and generate its TT-histogram with given number of bins and tolerance error

    :param data_folder:
    :param filename: the data set name with sizes and type, e.g. hurricane_500_500_91_uint8.raw
    :param B: number of bins
    :param eps: relative error
    :return: (data set "base" name, a NumPy array containing it, and its TT-histogram)
    """

    X, basename = read_tensor(filename)
    X = X.astype(np.float32)
    dump_name = os.path.join(data_folder, '{}_{}_{:.7f}.tth'.format(basename, B, eps))

    Xmin = X.min()
    Xmax = X.max()
    X = ((X - Xmin) / (Xmax - Xmin) * (B - 1)).astype(int)

    print(dump_name)
    if not os.path.exists(dump_name):
        tth = tthistogram.TTHistogram(X, B, eps, verbose=True)
        pickle.dump(tth, open(dump_name, 'wb'))
    tth = pickle.load(open(dump_name, 'rb'))
    return basename, X, tth


def sphere(data_folder, N, S, eps):
    """
    Create (or read if available) a sphere, and compress it

    :param data_folder:
    :param S:
    :return: the S^N sphere and an eps-compression with TT

    """

    filename = os.path.join(data_folder, 'sphere_{}_{}_{}.pickle'.format(N, S, eps))
    if not os.path.exists(filename):
        mg = np.meshgrid(*list([np.linspace(-1, 1, S)]*N))
        sqsum = np.zeros([S]*N)
        for m in mg:
            sqsum += m**2
        sphere = (np.sqrt(sqsum) <= 1)
        sphere_t = tr.core.tt_svd(sphere, eps=eps)
        pickle.dump({'raw': sphere, 'tt': sphere_t}, open(filename, 'wb'))
    sphere = pickle.load(open(filename, 'rb'))
    return sphere['raw'].astype(float), sphere['tt']
