"""
Show example IH slices, before and after TT compression
"""

import numpy as np
import pickle
import config
import bruteforce
import tthistogram
import os
import pandas as pd
import util
import ttrecipes as tr
import scipy.misc
import matplotlib.pyplot as plt

###
input_dataset = os.path.join(config.data_folder, 'waterfall_4096_4096_uint8.raw')
B = 64
eps = 1e-3
basename, X, tth = util.prepare_dataset(config.data_folder, input_dataset, B, eps)
###

factor = 16

N = X.ndim
bs = [16, 32, 48]
origs = []
recos = []
diffs = []
for b in bs:
    # print(X.shape)
    orig = (X == b)
    tmp = orig
    orig = np.zeros(np.array(orig.shape)+1)
    orig[[slice(1, None)]*N] = tmp
    print(orig.shape)
    for n in range(N):
        orig = np.cumsum(orig, axis=n)
    orig = orig.astype(float)
    orig = orig[[slice(None, None, factor)]*N]
    origs.append(orig)

    reco = tth.tensor[[slice(None)]*N + [b]].full()
    reco = reco[[slice(None, None, factor)] * N]
    recos.append(reco)

    diffs.append(np.abs(orig-reco))

print('Original range:', origs[0].min(), origs[0].max())
print('Absolute error range:', diffs[0].min(), diffs[0].max())

vmin = float('inf')
vmax = float('-inf')
for o in orig:
    vmin = min(vmin, np.min(o))
    vmax = max(vmax, np.max(o))

fig = plt.figure()
for i, orig in enumerate(origs):
    fig.add_subplot(len(bs), 1, i+1)
    plt.imshow(orig, cmap='gray', vmin=vmin, vmax=vmax)
    plt.axis('off')
plt.savefig('orig_slices', bbox_inches='tight')
plt.clf()

fig = plt.figure()
for i, reco in enumerate(recos):
    fig.add_subplot(len(bs), 1, i+1)
    plt.imshow(reco, cmap='gray', vmin=vmin, vmax=vmax)
    plt.axis('off')
plt.savefig('reco_slices', bbox_inches='tight')
plt.clf()

fig = plt.figure()
for i, diff in enumerate(diffs):
    fig.add_subplot(len(bs), 1, i+1)
    plt.imshow(vmax - 100*diff, cmap='gray', vmin=vmin, vmax=vmax)
    plt.axis('off')
plt.savefig('diff_slices', bbox_inches='tight')
plt.clf()
