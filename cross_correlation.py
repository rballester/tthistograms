"""
Normalized cross-correlation interactive display for the hurricane data set
"""

import numpy as np
import copy
import config
import util
import tt
import ttrecipes as tr
import matplotlib.pyplot as plt
import os
import time

input_dataset = os.path.join(config.data_folder, 'hurricane_500_500_91_uint8.raw')
B = 128
eps = 0.0001

basename, X, tth = util.prepare_dataset(config.data_folder, input_dataset, B=B, eps=eps)
print('Total computing time: {}'.format(tth.total_time))
print('NNZ: {}'.format(len(tth.tensor.core)))


def interactive_loop(tth, basis_size=8):
    """
    Display an interactive image showing the normalized cross-correlation between
    a histogram field and any window selected by the user

    :param tth:
    :param basis_size: used to approximate the norm of each individual histogram in the field
    """

    cores = tt.vector.to_list(tth.tensor)
    tr.core.orthogonalize(cores, len(cores)-1)
    basis = np.linalg.svd(cores[-1][:, :, 0], full_matrices=0)[2][:basis_size, :]

    cores[-1] = np.einsum('ijk,aj->iak', cores[-1], basis)
    pca = tt.vector.from_list(cores)
    pcatth = copy.copy(tth)
    pcatth.tensor = pca
    shape = np.array([8, 8, 90])
    Is = [500, 500, 91]
    corners = np.array([[shape[0] // 2, Is[0] - shape[0] // 2], [shape[1] // 2, Is[1] - shape[1] // 2], [45, 46]])
    pcafield, elapsed = pcatth.box_field(corners, shape)
    print('Box field computation time: {}'.format(elapsed))
    pcafield = np.squeeze(pcafield)
    norms = np.sqrt(np.sum(pcafield**2, axis=-1))

    global im
    im = None
    global sc
    sc = None
    global counter
    counter = 1

    def update(x, y):
        start = time.time()
        v, elapsed = tth.box(np.array([[x-4, x+3], [y-4, y+3], [0, 91]]))
        v = v / np.linalg.norm(v)
        cores = tt.vector.to_list(tth.tensor)
        cores[-1] = np.einsum('ijk,j->ik', cores[-1], v)[:, np.newaxis, :]
        proj = tt.vector.from_list(cores)
        projtth = copy.copy(tth)
        projtth.tensor = proj
        field, elapsed = projtth.box_field(corners, shape)
        field = np.squeeze(field.T) / norms
        global im
        global sc
        global counter
        if im is None:
            plt.axis('off')
            im = plt.imshow(field, cmap='pink', vmin=0, vmax=1)
            sc, = plt.plot(x, y, marker='+', ms=25, mew=5, color='red')
            plt.show()
        else:
            im.set_data(field)
            sc.set_data(x, y)
            fig.canvas.draw()
            extent = plt.gca().get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            plt.savefig(os.path.join(config.data_folder, 'similarity_{:03d}.pdf'.format(counter)), bbox_inches=extent)
        counter += 1

    def onclick(event):
        x = int(event.xdata)
        y = int(event.ydata)
        print(x, y)
        update(x, y)

    fig = plt.figure()
    fig.canvas.mpl_connect('button_press_event', onclick)
    update(250, 250)

interactive_loop(tth)
