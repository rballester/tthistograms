"""
Compression quality for histogram reconstruction: a) use a integral histogram vs b) use its derivative, i.e. the raw per-bin slices. a) is much better
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
import pickle
import tt
import config
import time
import bruteforce
import ttrecipes as tr

np.random.seed(1)


def histogram_tt1(L_t, corners):
    start = time.time()
    prod = np.array([[1]])
    cores = tt.vector.to_list(L_t)
    for corner, core in zip(corners, cores[:-1]):
        prod = prod.dot(np.sum(core[:, corner[0]:corner[1], :], axis=1))
    reco = np.squeeze(prod.dot(cores[-1][:, :, 0]))
    time_tt1 = time.time() - start
    return reco, time_tt1


def histogram_tt2(L_C_t, corners):
    cores2 = tt.vector.to_list(L_C_t)
    start = time.time()
    prod = np.array([[1]])
    for corner, core in zip(corners, cores2[:-1]):
        prod = prod.dot(core[:, corner[1], :] - core[:, corner[0], :])
    reco = np.squeeze(prod.dot(cores2[-1][:, :, 0]))
    time_tt2 = time.time() - start
    return reco, time_tt2


### Parameters
input_dataset = os.path.join(config.data_folder, 'channel_512_512_512_float32.raw')
B = 32
P = 100
I = 64
S = 32
# targets1 = 10**np.linspace(-2, -4, 15)
# targets2 = 10**np.linspace(-7, -9, 15)
targets1 = 10**np.linspace(-0.75, -4, 25)
targets2 = 10**np.linspace(-3, -7, 25)
###

X, basename = bruteforce.read_tensor(input_dataset)
X = X.astype(float)
shape = X.shape
N = X.ndim
X = X[:I, :I, :I]
X = ((X - X.min()) / (X.max() - X.min()) * (B - 1)).astype(int)

# Prepare corners
clist = []
for p in range(P):
    corners = []
    for i in range(N):
        left = np.random.randint(0, I-S)
        right = left + S
        corners.append([left, right])
    clist.append(corners)

for target1 in targets1:
    print('TT1 - {}'.format(target1))
    name = basename + '_tt1_{}_{}_{:.8f}.pickle'.format(I, B, target1)
    name = os.path.join(config.data_folder, name)
    if not os.path.exists(name):
        L = bruteforce.create_levelset(X, B)
        L_t = tr.core.tt_svd(L, eps=target1)
        pickle.dump(L_t, open(name, 'wb'))

for target2 in targets2:
    print('TT2 - {}'.format(target2))
    name = basename + '_tt2_{}_{}_{:.8f}.pickle'.format(I, B, target2)
    name = os.path.join(config.data_folder, name)
    if not os.path.exists(name):
        L_C = bruteforce.create_ih(X, B)
        L_C_t = tr.core.tt_svd(L_C, eps=target2)
        cores = tt.vector.to_list(L_C_t)
        cores[:-1] = [np.concatenate([np.zeros([core.shape[0], 1, core.shape[2]]), core], axis=1) for core in cores[:-1]]
        L_C_t = tt.vector.from_list(cores)
        pickle.dump(L_C_t, open(name, 'wb'))

means_times_gt = []
nnz_tt1 = []
means_errors_tt1 = []
means_times_tt1 = []
for target1 in targets1:
    name = basename + '_tt1_{}_{}_{:.8f}.pickle'.format(I, B, target1)
    name = os.path.join(config.data_folder, name)
    L_t = pickle.load(open(name, 'rb'))
    nnz_tt1.append(len(L_t.core))
    times_gt = []
    errors_tt1 = []
    times_tt1 = []
    for corners in clist:
        gt, time_gt = bruteforce.box(X, B, corners)
        times_gt.append(time_gt)
        reco, time_tt1 = bruteforce.histogram_tt1(L_t, corners)
        errors_tt1.append(np.linalg.norm(reco - gt) / np.linalg.norm(gt))
        times_tt1.append(time_tt1)

    means_times_gt.append(np.mean(times_gt))
    means_errors_tt1.append(np.mean(errors_tt1))
    means_times_tt1.append(np.mean(times_tt1))

print(means_times_gt)
print(nnz_tt1)
print(means_errors_tt1)
print(means_times_tt1)
print()

means_times_gt = []
nnz_tt2 = []
means_errors_tt2 = []
means_times_tt2 = []
for target2 in targets2:
    name = basename + '_tt2_{}_{}_{:.8f}.pickle'.format(I, B, target2)
    name = os.path.join(config.data_folder, name)
    L_C_t = pickle.load(open(name, 'rb'))
    nnz_tt2.append(len(L_C_t.core))
    times_gt = []
    errors_tt2 = []
    times_tt2 = []
    for corners in clist:
        gt, time_gt = bruteforce.box(X, B, corners)
        times_gt.append(time_gt)
        reco, time_tt2 = bruteforce.histogram_tt2(L_C_t, corners)
        errors_tt2.append(np.linalg.norm(reco - gt) / np.linalg.norm(gt))
        times_tt2.append(time_tt2)

    means_times_gt.append(np.mean(times_gt))
    means_errors_tt2.append(np.mean(errors_tt2))
    means_times_tt2.append(np.mean(times_tt2))

print(means_times_gt)
print(nnz_tt2)
print(means_errors_tt2)
print(means_times_tt2)
print()

fig = plt.figure()
plt.plot(nnz_tt1, np.array(means_errors_tt1)*100, marker='o', label='Non-cumulative')
plt.plot(nnz_tt2, np.array(means_errors_tt2)*100, marker='o', label='Integral histogram')
plt.legend()
plt.xlabel('NNZ')
plt.ylabel(r'Relative error (\%)')
plt.show()

