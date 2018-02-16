"""
Brute-force histogram computations using CuPy (https://cupy.chainer.org/)
"""

import numpy as np
import cupy as cp
import time


def bincount(X, B, weights=None):
    if weights is None:
        b = cp.zeros((B,), dtype=cp.int32)
        startin = time.time()
        cp.ElementwiseKernel(
            'S x', 'raw U bin',
            'atomicAdd(&bin[x], 1)',
            'bincount_kernel'
        )(X, b)
        b = b.astype(np.intp)
    else:
        b = cp.zeros((B,), dtype=cp.float32)
        cp.ElementwiseKernel(
            'S x, T w', 'raw U bin',
            'atomicAdd(&bin[x], w)',
            'bincount_with_weight_kernel'
        )(X, weights, b)
        b = b.astype(cp.float64)

    return b


def box_cp(X, B, corners):
    lim = 512
    if np.prod(X.shape) > lim ** 3:
        Xg = cp.asarray(X[:lim, :lim, :lim].astype(np.uint8))
        corners -= corners[:, 0:1]
    else:
        Xg = cp.asarray(X.astype(np.uint8))
    start = time.time()
    slicing = [slice(corner[0], corner[1]) for corner in corners]
    Xg = Xg[slicing]
    result = bruteforce_cupy.bincount(Xg.flatten(), B=B)
    elapsed = time.time() - start
    return cp.asnumpy(result).astype(np.int), elapsed


def pattern_cp(X, B, corners, pat):
    # Xg = cp.asarray(X.astype(np.uint8))
    lim = 512
    if np.prod(X.shape) > lim**3:
        Xg = cp.asarray(X[:lim, :lim, :lim].astype(np.uint8))
        corners -= corners[:, 0:1]
    else:
        Xg = cp.asarray(X.astype(np.uint8))
    patg = cp.asarray(pat.astype(np.float32).flatten())
    start = time.time()
    slicing = [slice(corner[0], corner[1]) for corner in corners]
    Xg = Xg[slicing]
    result = bincount(Xg.flatten(), B=B, weights=patg)
    elapsed = time.time() - start
    return cp.asnumpy(result).astype(np.int), elapsed


if __name__ == '__main__':  # Some tests

    B = 128
    x = np.random.randint(0, B, [512, ]*3).astype(np.uint8)
    slicing = [slice(10, 500)]*3
    # weights = np.ones(x.shape)
    xg = cp.asarray(x)
    print(xg.dtype)
    print(xg.shape)
    # weights = cp.asarray(weights)
    # xg = cp.take(xg, np.arange(10, 500), axis=0)

    xg = xg[slicing]

    start = time.time()
    # xg = xg.flatten()
    hist = bincount(xg.flatten(), B=B, weights=None)
    elapsed = time.time() - start
    print('Elapsed:', elapsed)
    print(hist)

    start = time.time()
    gt = np.histogram(x[slicing], bins=B, range=[-0.5, B - 0.5], weights=None)[0]
    print(gt)
    elapsed = time.time() - start
    print('Elapsed:', elapsed)
