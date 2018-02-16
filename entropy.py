"""
Compute box- and Gaussian-entropy fields from a hurricane directional histogram
"""

import numpy as np
import config
import util
import tt
import ttrecipes as tr
import bruteforce
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mpl_toolkits.mplot3d
import os


def save_image(image, filename):
    fig = plt.figure(figsize=(10,10), frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image, aspect='auto', cmap=cm.GnBu)
    fig.savefig(filename)

    
def entropy(X, axis=-1):
    P = X / np.sum(X, axis=axis, keepdims=True)
    return -np.sum(P*np.log2(P), axis=axis)


input_dataset = os.path.join(config.data_folder, 'hurricane_500_500_91_uint8.raw')
B = 128
eps = 0.0001

basename, X, tth = util.prepare_dataset(config.data_folder, input_dataset, B=B, eps=eps)
print(tth.tensor)

N = 3
shape = np.array([20, 20, 90])
Is = [500, 500, 91]
corners = np.array([[shape[0]//2, Is[0]-shape[0]//2], [shape[1]//2, Is[1]-shape[1]//2], [45, 46]])


# Box field
gt, elapsed = bruteforce.box_field(X, B, corners, shape=shape, verbose=True)
gt = np.squeeze(gt)
print('Elapsed (box GT):', elapsed)

print('[{}, {}]'.format(gt.min(), gt.max()))
field = entropy(np.abs(gt+1))
save_image(field, os.path.join(config.data_folder, 'hurricane_box_field_gt.jpg'))

print()

reco, elapsed = tth.box_field(corners, shape)
reco = np.squeeze(reco)
print('Elapsed (box TT):', elapsed)

print('[{}, {}]'.format(reco.min(), reco.max()))
field = entropy(np.abs(reco+1))
save_image(field, os.path.join(config.data_folder, 'hurricane_box_field_tt.jpg'))


# Gaussian field
shape = np.array([10, 10, 90])
pattern = tr.core.gaussian_tt(shape, shape/4)
pattern *= (1./tr.core.sum(pattern))*np.prod(shape)

gt, elapsed = bruteforce.separable_field(X, B, corners, pat=pattern, verbose=True)
gt = np.squeeze(gt)
field = entropy(np.abs(gt+1))
save_image(field, os.path.join(config.data_folder, 'hurricane_gaussian_field_gt.jpg'))
print('Elapsed (Gaussian GT):', elapsed)

reco, elapsed = tth.separable_field(corners, pat=pattern)
reco = np.squeeze(reco)
field = entropy(np.abs(reco+1))
save_image(field, os.path.join(data_folder, 'hurricane_gaussian_field_tt.jpg'))
print('Elapsed (Gaussian TT):', elapsed)
