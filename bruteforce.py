"""
Compute histograms via brute-force traversal in the CPU, for comparison against TT-histograms.

Important: all these functions assume that the data set X has already been quantized to B bins,
i.e. takes values in 0, ..., B-1
"""

import numpy as np
import time
import scipy.signal


def box(X, B, corners, weights=None):
    """
    Query a box histogram via brute-force traversal

    :param X: an ND data set
    :param B: number of bins
    :param corners: a list of pairs [[i0, i1], [j0, j1], ...] encoding the query box
    :return: a vector with B elements
    """

    start = time.time()
    chunk = X[[slice(corner[0], corner[1]) for corner in corners]]
    gt = np.histogram(chunk, bins=B, range=[-0.5, B - 0.5], weights=weights)[0]
    elapsed = time.time() - start
    return gt, elapsed


def pattern(X, B, corners, pat):
    """
    Query a non-rectangular histogram via brute-force traversal

    :param X: an ND data set
    :param B: number of bins
    :param corners: a list of pairs [[i0, i1], [j0, j1], ...] encoding the pattern's bounding box
    :param pat: a multiarray (its size must fit `corners`) containing the region weights
    :return: a vector with B elements
    """

    return box(X, B, corners, weights=pat)


def box_field(X, B, corners, shape, verbose=False):
    """
    Compute a box histogram field

    :param X: an ND data set
    :param B: number of bins
    :param corners: a list of pairs [[i0, i1], [j0, j1], ...] containing all window positions to compute
    :param shape: a list of N integers (all must be odd) encoding the shape of each window
    :param verbose:
    :return: an array of dimension N+1 and size (i1-i0+1) x (j1-j0+1) x ... x B
    """

    if verbose:
        print('Computing box field')
    start = time.time()
    shape = shape // 2
    N = X.ndim
    chunk = X[[slice(corner[0]-sh, corner[1]+sh) for corner, sh in zip(corners, shape)]]
    tmp = chunk
    elapsed = time.time() - start
    chunk = np.zeros(np.array(chunk.shape) + 1)
    chunk[[slice(1, None)] * N] = tmp
    start = time.time()
    result = np.zeros(list(corners[:, 1] - corners[:, 0]) + [B])
    for b in range(B):
        if verbose:
            print('b = {}'.format(b))
        sl = (chunk == b)
        for n in range(N):
            sl = np.cumsum(sl, axis=n)
        blocks = []
        for corner, sh in zip(corners, shape):
            blocks.append([slice(0, corner[1]-corner[0], 1), slice(2*sh, corner[1]-corner[0]+2*sh, 1)])
        codes = np.array(np.unravel_index(np.arange(2**N), [2]*N)).T
        for code in codes:
            sign = (-1) ** (codes.shape[1] - np.sum(code))
            result[..., b] += sign * sl[[block[c] for block, c in zip(blocks, code)]]
    elapsed += time.time() - start
    return result, elapsed


def separable_field(X, B, corners, pat, verbose=False):
    """
    As `box_field`, but for non-rectangular separable regions.
    :param pat: a list of vectors encoding the rank-1 expression of the weights. This substitutes the parameter `shape`
    """

    if verbose:
        print('Computing separable field')
    start = time.time()
    shape = np.array([len(v) for v in pat])
    shape = shape // 2
    N = X.ndim
    chunk = X[[slice(corner[0] - sh, corner[1] - 1 + sh) for corner, sh in zip(corners, shape)]]
    result = np.zeros(list(corners[:, 1] - corners[:, 0]) + [B])
    for b in range(B):
        if verbose:
            print('b = {}'.format(b))
        sl = (chunk == b)
        for i, v in enumerate(pat):
            slicing = [np.newaxis]*N
            slicing[i] = slice(None)
            sl = scipy.signal.convolve(sl, v[slicing], mode='valid')
        result[..., b] = sl
    elapsed = time.time() - start
    return result, elapsed


def nonseparable_field(X, B, corners, pat, verbose=False):
    """
    As `box_field`, but for non-rectangular general regions.
    :param pat: an ND NumPy array containing the weights
    """

    if verbose:
        print('Computing non-separable field')
    N = X.ndim
    start = time.time()
    shape = np.array(pat.shape)
    shape = shape // 2
    chunk = X[[slice(corner[0] - sh, corner[1] - 1 + sh) for corner, sh in zip(corners, shape)]]
    result = np.zeros(list(corners[:, 1] - corners[:, 0]) + [B])
    pat = pat[[slice(None, None, -1)]*N]
    for b in range(B):
        if verbose:
            print('b = {}'.format(b))
        sl = (chunk == b)
        sl = scipy.signal.convolve(sl, pat, mode='valid')
        result[..., b] = sl
    elapsed = time.time() - start
    return result, elapsed


def create_levelset(X, B):
    L = np.zeros(list(X.shape) + [B])
    for b in range(B):
        sl = (X == b)
        L[..., b] = sl
    return L


def create_ih(X, B):
    L_C = np.zeros(list(X.shape) + [B])
    for b in range(B):
        sl = (X == b)
        sl = np.cumsum(sl, axis=0)
        sl = np.cumsum(sl, axis=1)
        sl = np.cumsum(sl, axis=2)
        L_C[..., b] = sl
    return L_C
