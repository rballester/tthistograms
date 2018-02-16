"""
Visual results: a 3D sphere after compression in the TT format
"""

import numpy as np
import config
import util
import matplotlib.pyplot as plt
import ttrecipes as tr

N = 3
S = 64

s = util.sphere(config.data_folder, N, S, eps=0.0)
s, _ = s[0], s[1]
eps = 0

for rmax in 2**np.arange(0, 5):
    t = tr.core.tt_svd(s, eps=eps, rmax=rmax, verbose=False)
    t = t.full()
    print(rmax, np.linalg.norm(s - t) / np.linalg.norm(s))
    fig = plt.figure(frameon=False)
    fig.set_size_inches(5, 5)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(t[:, :, S//2], cmap='gray', vmin=0, vmax=1, aspect='normal')
    plt.axis('off')
    plt.savefig('tt_ball_{:03d}.jpg'.format(rmax))
    plt.clf()
