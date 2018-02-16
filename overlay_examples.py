"""
Example box- and Gaussian-shaped histograms over a waterfall image
"""

import numpy as np
import config
import bruteforce
import util
import os
import ttrecipes as tr
import matplotlib.pyplot as plt

###
input_dataset = os.path.join(config.data_folder, 'waterfall_4096_4096_uint8.raw')
B = 64
eps = 1e-3
basename, X, tth = util.prepare_dataset(config.data_folder, input_dataset, B, eps)
###


def plot_overlay(X, corners_list, pattern_list, color=(0.25, 0.25, 1), alpha=0.5):
    fig = plt.figure(figsize=(8, 8), frameon=False)
    mask = np.zeros(list(X.shape)+[4])
    for corners, pattern in zip(corners_list, pattern_list):
        # pattern = pattern[:, :, np.newaxis]*np.array(color)[np.newaxis, np.newaxis, :]
        Xrgb = np.repeat((X/B*255)[:, :, np.newaxis].astype(np.uint8), 3, axis=2)
        plt.imshow(Xrgb, vmin=0, vmax=255)
        plt.axis('off')
        mask[[slice(c[0], c[1]) for c in corners]+[slice(0, 3)]] = color
        mask[[slice(c[0], c[1]) for c in corners]+[3]] = pattern*alpha
    plt.imshow(mask, vmin=0, vmax=255)
    # plt.savefig(os.path.join(config.data_folder, 'overlaid.png'))

print('IH compression ratio:', np.prod(X.shape)*B / len(tth.tensor.core))
print(X.shape)
N = X.ndim

corners_list = []
pattern_list = []

### Box
print('*** Box ***')
offset = [1024*3, int(1024*0.7)]
S = 512
shape = [S]*N
corners = np.array([[offset[0], offset[0]+S], [offset[1], offset[1]+S]])
gt, elapsed = bruteforce.box(X, B, corners)
print('Elapsed GT:', elapsed)
reco, elapsed = tth.box(corners)
print('Elapsed TT:', elapsed)
corners_list.append(corners)
pattern_list.append(np.ones(corners[:, 1] - corners[:, 0]))
fig = plt.figure(figsize=(4, 3))
plt.plot(gt, label='Groundtruth')
plt.plot(reco, label='TT')
plt.legend()
plt.xlabel('Bin')
plt.ylabel('Count')
plt.title('Box')
plt.tight_layout()
plt.savefig(os.path.join(config.data_folder, 'overlaid_plot_box.pdf'))
###

print()

### Gaussian
print('*** Gaussian ***')
offset = [1024*1, 1024*2]
S = 768
shape = [S]*N
corners = np.array([[offset[0], offset[0]+S], [offset[1], offset[1]+S]])
print(corners)
shape = corners[:, 1] - corners[:, 0]
pattern = tr.core.gaussian_tt(shape, shape/4)
pattern *= (1./np.max(pattern.full()))
gt, elapsed = bruteforce.pattern(X, B, corners, pat=pattern.full())
print('Elapsed GT:', elapsed)
reco, elapsed = tth.separable(corners, pat=pattern)
print('Elapsed TT:', elapsed)
corners_list.append(corners)
pattern_list.append(pattern.full())
fig = plt.figure(figsize=(4, 3))
plt.plot(gt, label='Groundtruth')
plt.plot(reco, label='TT')
plt.legend()
plt.xlabel('Bin')
plt.ylabel('Count')
plt.title('Gaussian')
plt.tight_layout()
plt.savefig(os.path.join(config.data_folder, 'overlaid_plot_gaussian.pdf'))
###

plot_overlay(X, corners_list, pattern_list)
