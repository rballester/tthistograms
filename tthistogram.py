"""
The most important class: the TTHistogram. Includes code compression and look-up over a variety of ROIs
"""

import numpy as np
import tt
import scipy.signal
import time
import sys
import ttrecipes as tr


class TTHistogram(object):

    def __init__(self, X, B, eps, verbose=False):
        """
        Build (incrementally) a compressed integral histogram that can be later queried

        :param X: an ndarray
        :param B: number of bins
        :param eps: relative error
        """

        N = X.ndim
        assert X.min() >= 0
        assert X.max() < B
        start = time.time()

        def create_generator():
            for b in range(B):
                if verbose:
                    print(b)
                sys.stdout.flush()
                sl = (X == b)
                for n in range(N):
                    sl = np.cumsum(sl, axis=n)
                sys.stdout.flush()
                sl_t = tr.core.tt_svd(sl, eps, verbose=False)
                sys.stdout.flush()
                cores = tt.vector.to_list(sl_t)
                onehot = np.zeros([1, B, 1])
                onehot[0, b, 0] = 1
                cores.append(onehot)
                sl_t = tt.vector.from_list(cores)
                if verbose:
                    print('Done')
                yield sl_t

        generator = create_generator()
        self.tensor = tr.core.sum_and_compress(generator, rounding=eps/np.log2(B), verbose=verbose)
        cores = tt.vector.to_list(self.tensor)
        cores[:-1] = [np.concatenate([np.zeros([c.shape[0], 1, c.shape[2]]), c], axis=1) for c in cores[:-1]]
        self.tensor = tt.vector.from_list(cores)
        self.total_time = time.time() - start

    def box(self, corners):
        """
        :param corners: a list of pairs [[i0, i1], [j0, j1], ...] encoding the query box
        :return: a vector with B elements
        """

        cores = tt.vector.to_list(self.tensor)
        start = time.time()
        reco = np.array([[1]])
        for corner, core in zip(corners, cores[:-1]):
            reco = reco.dot(core[:, corner[1], :] - core[:, corner[0], :])
        reco = np.squeeze(reco.dot(cores[-1][:, :, 0]))
        elapsed = time.time() - start
        return reco, elapsed

    def separable(self, corners, pat):
        """
        :param corners: a list of pairs [[i0, i1], [j0, j1], ...] containing all window positions to compute
        :param pat: a rank-1 TT encoding the separable region (must fit in `corners`)
        :return: an array of dimension N+1 and size (i1-i0+1) x (j1-j0+1) x ... x B
        """

        assert np.all(corners[:, 1] - corners[:, 0] == pat.n)
        cores = tt.vector.to_list(self.tensor)
        coresp = [-np.diff(np.concatenate([np.zeros([c.shape[0], 1, c.shape[2]]), c, np.zeros([c.shape[0], 1, c.shape[2]])], axis=1), axis=1) for c in tt.vector.to_list(pat)]
        start = time.time()
        reco = np.array([[1]])
        for corner, core, corep, in zip(corners, cores[:-1], coresp):
            comb = np.einsum('ijk,ljm->ilkm', core[:, corner[0]:corner[1]+1, :], corep)
            comb = np.reshape(comb, [comb.shape[0]*comb.shape[1], comb.shape[-2]*comb.shape[-1]])
            reco = reco.dot(comb)
        reco = np.squeeze(reco.dot(cores[-1][:, :, 0]))
        elapsed = time.time() - start
        return reco, elapsed

    def nonseparable(self, corners, pat):
        """
        As `separable`, but `pat` does not need have rank 1
        """

        assert np.all(corners[:, 1] - corners[:, 0] == pat.n)
        cores = tt.vector.to_list(self.tensor)
        coresp = [-np.diff(np.concatenate([np.zeros([c.shape[0], 1, c.shape[2]]), c, np.zeros([c.shape[0], 1, c.shape[2]])], axis=1), axis=1) for c in tt.vector.to_list(pat)]
        start = time.time()
        cores[:-1] = [core[:, corner[0]:corner[1] + 1, :] for corner, core in zip(corners, cores[:-1])]
        Rprod = np.array([[1]])

        def partial_dot_right(cores1, cores2, mu, Rprod):
            Ucore = np.einsum('ij,ikl->jkl', Rprod, cores1[mu])
            Vcore = cores2[mu]
            return np.dot(tr.core.left_unfolding(Ucore).T, tr.core.left_unfolding(Vcore))

        d = len(coresp)
        for mu in range(d):
            Rprod = partial_dot_right(coresp, cores, mu, Rprod)
        Rprod = Rprod.dot(cores[-1][:, :, 0])
        elapsed = time.time() - start
        return np.squeeze(Rprod), elapsed

    def box_field(self, corners, shape):
        """
        :param corners: a list of pairs [[i0, i1], [j0, j1], ...] containing all window positions to compute
        :param shape: a list of N integers (all must be odd) encoding the shape of each window
        :return: an array of dimension N+1 and size (i1-i0+1) x (j1-j0+1) x ... x B
        """

        assert all(np.mod(shape, 2) == 0)
        shape = shape // 2
        assert all(corners[:, 0] - shape >= 0)
        assert all(corners[:, 1] + shape <= self.tensor.n[:-1])
        cores = tt.vector.to_list(self.tensor)
        start = time.time()
        reco = np.ones([1, 1, 1])
        for corner, sh, core in zip(corners, shape, cores[:-1]):
            chunk = core[:, corner[0]+sh:corner[1]+sh, :] - core[:, corner[0]-sh:corner[1]-sh, :]  # r1 x corner[1]-corner[0] x r2
            reco = np.tensordot(reco, chunk, axes=[2, 0])
            reco = np.reshape(reco, [reco.shape[0], -1, reco.shape[-1]])
        reco = np.squeeze(np.tensordot(reco, cores[-1], axes=[2, 0]))
        reco = np.reshape(reco, list(corners[:, 1] - corners[:, 0]) + [self.tensor.n[-1]])
        elapsed = time.time() - start
        return reco, elapsed

    def separable_field(self, corners, pat):
        """
        As `box_field`, but for non-rectangular separable regions.
        :param pat: a rank-1 TT
        """

        shape = pat.n
        assert all(np.mod(shape, 2) == 0)
        shape = shape // 2
        cores = tt.vector.to_list(self.tensor)
        coresp = [np.diff(np.concatenate([np.zeros([c.shape[0], 1, c.shape[2]]), c, np.zeros([c.shape[0], 1, c.shape[2]])], axis=1), axis=1) for c in tt.vector.to_list(pat)]
        start = time.time()
        reco = np.ones([1, 1])
        for corner, sh, core, corep in zip(corners, shape, cores[:-1], coresp):
            chunk = core[:, corner[0]-sh:corner[1]+sh, :]
            convolution = scipy.signal.convolve(chunk, corep, mode='valid')
            reco = np.einsum('jk,klm', reco, convolution)
            reco = np.reshape(reco, [-1, convolution.shape[-1]])
        reco = np.squeeze(np.tensordot(reco, cores[-1], axes=[1, 0]))
        reco = np.reshape(reco, list(corners[:, 1] - corners[:, 0]) + [self.tensor.n[-1]])
        elapsed = time.time() - start
        return reco, elapsed
