"""
Main benchmarking experiments: times and accuracy for queries over various regions
"""

import numpy as np
import config
import bruteforce
import bruteforce_cupy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pandas as pd
import util
import ttrecipes as tr


def run(X, tth, eps, S, P):
    """
    Generate many random ROIs and measure their histograms (both groundtruth and from a TTHistogram)

    :param X: a tensor
    :param tth: its TTHistogram
    :param eps:
    :param S: region size
    :param P: number of samples to draw
    :return: a DataFrame

    """

    times_gt_box = []
    times_gt_gaussian = []
    times_gt_sphere = []

    times_cp_box = []
    times_cp_gaussian = []
    times_cp_sphere = []

    errors_tt_box = []
    times_tt_box = []
    errors_tt_gaussian = []
    times_tt_gaussian = []
    errors_tt_sphere = []
    times_tt_sphere = []

    N = X.ndim
    B = tth.tensor.n[-1]

    sphere, sphere_t = util.sphere(config.data_folder, N, S, eps=0.1)
    # print(sphere_t)
    # print(np.linalg.norm(sphere - sphere_t.full()) / np.linalg.norm(sphere))
    denominator = np.sum(sphere)
    sphere /= denominator
    sphere_t *= (1./denominator)

    corners = []
    for i in range(N):
        # left = np.random.randint(0, X.shape[i] + 1 - S)
        left = X.shape[i] // 2 - S // 2
        right = left + S
        corners.append([left, right])
    corners = np.array(corners)

    for p in range(P):

        # Box
        gt, time_gt = bruteforce.box(X, B, corners)
        # _, time_tf_gt = box_tf(X, B, corners)
        times_gt_box.append(time_gt)
        gt_cp, time_cp = bruteforce_cupy.box_cp(X, B, corners)
        times_cp_box.append(time_cp)
        reco, time_tt = tth.box(corners)
        errors_tt_box.append(np.linalg.norm(reco - gt) / np.linalg.norm(gt))
        times_tt_box.append(time_tt)

        # Gaussian
        pattern = tr.core.gaussian_tt(corners[:, 1] - corners[:, 0], [S/5] * 3)
        pattern *= (1. / tr.core.sum(pattern))
        patternfull = pattern.full()
        gt, time_gt = bruteforce.pattern(X, B, corners, pat=patternfull)
        times_gt_gaussian.append(time_gt)
        gt_cp, time_cp = bruteforce_cupy.pattern_cp(X, B, corners, pat=patternfull)
        times_cp_gaussian.append(time_cp)
        reco, time_tt = tth.separable(corners, pat=pattern)
        errors_tt_gaussian.append(np.linalg.norm(reco - gt) / np.linalg.norm(gt))
        times_tt_gaussian.append(time_tt)

        # Sphere
        gt, time_gt = bruteforce.pattern(X, B, corners, pat=sphere)
        times_gt_sphere.append(time_gt)
        gt_cp, time_cp = bruteforce_cupy.pattern_cp(X, B, corners, pat=sphere)
        times_cp_sphere.append(time_cp)
        reco, time_tt = tth.nonseparable(corners, pat=sphere_t)
        errors_tt_sphere.append(np.linalg.norm(reco - gt) / np.linalg.norm(gt))
        times_tt_sphere.append(time_tt)

    df = pd.DataFrame(columns=['B', 'method', 'heps', 'roitype', 'roisize', 'fieldsize', 'meantime', 'mediantime', 'maxtime', 'meanerror', 'medianerror', 'maxerror'])

    df.loc[0, :] = [B, 'gt', None, 'box', S, 1, np.mean(times_gt_box), np.median(times_gt_box), np.max(times_gt_box), 0, 0, 0]
    df.loc[1, :] = [B, 'gt', None, 'gaussian', S, 1, np.mean(times_gt_gaussian), np.median(times_gt_gaussian), np.max(times_gt_gaussian), 0, 0, 0]
    df.loc[2, :] = [B, 'gt', None, 'sphere', S, 1, np.mean(times_gt_sphere), np.median(times_gt_sphere), np.max(times_gt_sphere), 0, 0, 0]

    df.loc[3, :] = [B, 'cp', None, 'box', S, 1, np.mean(times_cp_box), np.median(times_cp_box), np.max(times_cp_box), 0, 0, 0]
    df.loc[4, :] = [B, 'cp', None, 'gaussian', S, 1, np.mean(times_cp_gaussian), np.median(times_cp_gaussian), np.max(times_cp_gaussian), 0, 0, 0]
    df.loc[5, :] = [B, 'cp', None, 'sphere', S, 1, np.mean(times_cp_sphere), np.median(times_cp_sphere), np.max(times_cp_sphere), 0, 0, 0]

    df.loc[6, :] = [B, 'tt', eps, 'box', S, 1, np.mean(times_tt_box), np.median(times_tt_box), np.max(times_tt_box), np.mean(errors_tt_box), np.median(errors_tt_box), np.max(errors_tt_box)]
    df.loc[7, :] = [B, 'tt', eps, 'gaussian', S, 1, np.mean(times_tt_gaussian), np.median(times_tt_gaussian), np.max(times_tt_gaussian), np.mean(errors_tt_gaussian), np.median(errors_tt_gaussian), np.max(errors_tt_gaussian)]
    df.loc[8, :] = [B, 'tt', eps, 'sphere', S, 1, np.mean(times_tt_sphere), np.median(times_tt_sphere), np.max(times_tt_sphere), np.mean(errors_tt_sphere), np.median(errors_tt_sphere), np.max(errors_tt_sphere)]

    return df


def run_field(X, tth, eps, S, K, P):
    """
    Field histogram reconstruction
    """

    times_gt_box_field = []
    times_gt_gaussian_field = []

    errors_tt_box_field = []
    times_tt_box_field = []
    errors_tt_gaussian_field = []
    times_tt_gaussian_field = []

    N = X.ndim
    B = tth.tensor.n[-1]
    shape = np.array([S]*N)

    for p in range(P):
        print(p)
        corners = []
        for i in range(N):
            left = np.random.randint(S // 2, X.shape[i] + 1 - S // 2 - K)
            right = left + K
            corners.append([left, right])
        corners = np.array(corners)

        # Box field
        gt, elapsed = bruteforce.box_field(X, B, corners, shape)
        times_gt_box_field.append(elapsed)
        reco, elapsed = tth.box_field(corners, shape)
        errors_tt_box_field.append(np.linalg.norm(reco - gt) / np.linalg.norm(gt))
        times_tt_box_field.append(elapsed)

        # Gaussian field
        pattern = tr.core.gaussian_tt(shape, shape/5)
        pattern *= (1. / tr.core.sum(pattern))
        gt, time_gt = bruteforce.separable_field(X, B, corners, pat=[np.squeeze(c) for c in tt.vector.to_list(pattern)])
        times_gt_gaussian_field.append(time_gt)
        reco, time_tt = tth.separable_field(corners, pat=pattern)
        errors_tt_gaussian_field.append(np.linalg.norm(reco - gt) / np.linalg.norm(gt))
        times_tt_gaussian_field.append(time_tt)

    df = pd.DataFrame(columns=['B', 'method', 'heps', 'roitype', 'roisize', 'fieldsize', 'meantime', 'mediantime', 'maxtime', 'meanerror', 'medianerror', 'maxerror'])

    df.loc[0, :] = [B, 'gt', None, 'box_field', S, K, np.mean(times_gt_box_field), np.median(times_gt_box_field), np.max(times_gt_box_field), 0, 0, 0]
    df.loc[1, :] = [B, 'gt', None, 'gaussian_field', S, K, np.mean(times_gt_gaussian_field), np.median(times_gt_gaussian_field), np.max(times_gt_gaussian_field), 0, 0, 0]

    df.loc[2, :] = [B, 'tt', eps, 'box_field', S, K, np.mean(times_tt_box_field), np.median(times_tt_box_field), np.max(times_tt_box_field), np.mean(errors_tt_box_field), np.median(errors_tt_box_field), np.max(errors_tt_box_field)]
    df.loc[3, :] = [B, 'tt', eps, 'gaussian_field', S, K, np.mean(times_tt_gaussian_field), np.median(times_tt_gaussian_field), np.max(times_tt_gaussian_field), np.mean(errors_tt_gaussian_field), np.median(errors_tt_gaussian_field), np.max(errors_tt_gaussian_field)]

    return df


def batch():
    P = 3
    shape = np.array(X.shape)
    if np.all(shape == [1024, 1024, 1024]):
        Ss = (np.linspace(shape[0]/48, 512, 11).astype(int))//2 * 2
        # print(Ss)
    else:
        Ss = (np.linspace(shape[0]/17, shape[0]/1.2, 11).astype(int))//2 * 2
    df = None
    for S in Ss:
        print('S = {}'.format(S))
        partial = run(X, tth, eps, S=S, P=P)
        if df is None:
            df = partial
        else:
            df = pd.concat([df, partial], ignore_index=True)
    return df


def batch_field():

    P = 2
    K = 16
    shape = np.array(X.shape)
    Ss = (np.linspace(shape[0]/8, shape[0]//2, 2).astype(int))//2 * 2
    # if S < shape[0] - K:
    df = None
    for S in Ss:
        partial = run_field(X, tth, eps, S=S, K=K, P=P)
        if df is None:
            df = partial
        else:
            df = pd.concat([df, partial], ignore_index=True)
    return df


def plot(basename):

    title = basename.capitalize()
    if title == 'Bonsai_volvis':
        title = 'Bonsai'

    df = pd.read_excel(os.path.join(config.data_folder, '{}.xlsx'.format(basename)), sheetname=basename)

    figsize = (6, 3)

    # Times GT vs TT (single)
    # colors = ['#1f77b4', '#ff7f0e']#, '#2ca02c']
    roitypes = ['box', 'gaussian']#, 'sphere']
    markers = ['s', 'o']
    fig = plt.figure(figsize=figsize)
    for roitype, marker in zip(roitypes, markers):
        select = df.loc[(df['method'] == 'gt') & (df['roitype'] == roitype)]
        plt.plot(select['roisize'], np.log10(select['meantime']), label='{}: brute-force'.format(roitype.capitalize()), linestyle='-', marker=marker
                 , color='#1f77b4')
        select = df.loc[(df['method'] == 'cp') & (df['roitype'] == roitype)]
        plt.plot(select['roisize'], np.log10(select['meantime']), label='{}: brute-force (CuPy)'.format(roitype.capitalize()),
                 linestyle='-', marker=marker, color='#ff7f0e')
        select = df.loc[(df['method'] == 'tt') & (df['roitype'] == roitype)]
        plt.plot(select['roisize'], np.log10(select['meantime']), label='{}: TT'.format(roitype.capitalize()), marker=marker, color='#2ca02c')
    # plt.legend(loc='upper left')
    # plt.legend()
    plt.title(title)
    plt.xlabel('ROI size')
    plt.ylabel('log10(time) (s)')
    plt.ylim([-4.15, 1])
    plt.tight_layout()
    # plt.show()
    pdf = os.path.join(config.data_folder, '{}_single_times.pdf').format(basename)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.savefig(pdf)
    os.system('pdfcrop {} {}'.format(pdf, pdf))
    plt.savefig(os.path.join(config.data_folder, '{}_single_times.png').format(basename))

    fig_legend = plt.figure(figsize=(8,8), frameon=False)
    axi = fig_legend.add_subplot(111)
    fig_legend.legend(handles, labels, loc='center', scatterpoints=1, ncol=2)
    axi.xaxis.set_visible(False)
    axi.yaxis.set_visible(False)
    plt.axis('off')
    fig_legend.canvas.draw()
    plt.tight_layout()
    plt.savefig(os.path.join(config.data_folder, '{}_single_times_legend.pdf').format(basename))
    # assert 0
    # fig_legend.show()

    # Times GT vs TT (field)
    colors = ['#1f77b4', '#ff7f0e']
    roitypes = ['box_field', 'gaussian_field']
    fig = plt.figure(figsize=figsize)
    for color, roitype in zip(colors, roitypes):
        select = df.loc[(df['method'] == 'gt') & (df['roitype'] == roitype)]
        plt.plot(select['roisize'], np.log10(select['meantime']), label='{}: brute-force'.format(roitype.capitalize()), linestyle='-', marker='o', color=color)
        select = df.loc[(df['method'] == 'tt') & (df['roitype'] == roitype)]
        plt.plot(select['roisize'], np.log10(select['meantime']), label='{}: TT'.format(roitype.capitalize()), marker='o', color=color)
    plt.legend(loc='upper left')
    plt.title(title)
    plt.xlabel('ROI size')
    plt.ylabel('log10(time) (s)')
    plt.tight_layout()

    pdf = os.path.join(config.data_folder, '{}_field_times.pdf').format(basename)
    plt.savefig(pdf)
    os.system('pdfcrop {} {}'.format(pdf, pdf))
    plt.savefig(os.path.join(config.data_folder, '{}_field_times.png').format(basename))
    # plt.show()

    # Errors for TT
    colors = ['#2ca02c', '#2ca02c']#, '#2ca02c']
    roitypes = ['box', 'gaussian']#, 'sphere']
    fig = plt.figure(figsize=figsize)
    for color, roitype, marker in zip(colors, roitypes, markers):
        select = df.loc[(df['method'] == 'tt') & (df['roitype'] == roitype)]
        plt.plot(select['roisize'], np.log10(select['medianerror']), label='{}: TT'.format(roitype.capitalize()), marker=marker, color=color)
    # plt.legend()
    plt.title(title)
    plt.xlabel('ROI size')
    plt.ylabel('log10(relative error)')
    plt.ylim([-5.2, -1])
    plt.tight_layout()
    # plt.show()
    pdf = os.path.join(config.data_folder, '{}_errors.pdf').format(basename)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.savefig(pdf)
    os.system('pdfcrop {} {}'.format(pdf, pdf))
    plt.savefig(os.path.join(config.data_folder, '{}_errors.png').format(basename))

    fig_legend = plt.figure(figsize=(2,2), frameon=False)
    axi = fig_legend.add_subplot(111)
    fig_legend.legend(handles, labels, loc='center', scatterpoints=1)
    axi.xaxis.set_visible(False)
    axi.yaxis.set_visible(False)
    plt.axis('off')
    fig_legend.canvas.draw()
    plt.tight_layout()
    plt.savefig(os.path.join(config.data_folder, '{}_errors_legend.pdf').format(basename))


###
input_datasets = [os.path.join(config.data_folder, 'waterfall_4096_4096_uint8.raw'), os.path.join(config.data_folder, 'bonsai_volvis_256_256_256_uint8.raw'), os.path.join(config.data_folder, 'lung_512_512_512_uint8.raw'), os.path.join(config.data_folder, 'flower_1024_1024_1024_uint8.raw')]
Bs = [64, 128, 128, 128]
epss = [0.0005000, 0.0001000, 0.0002000, 0.0000150]
###

for input_dataset, B, eps in zip(input_datasets, Bs, epss):
    basename, X, tth = util.prepare_dataset(config.data_folder, input_dataset, B=B, eps=eps)
    excel_name = os.path.join(config.data_folder, '{}.xlsx'.format(basename))
    if not os.path.exists(excel_name):
        df1 = batch()
        # df2 = batch_field()
        # df = pd.concat([df1, df2], ignore_index=True)
        df = df1
        writer = pd.ExcelWriter(excel_name)
        df.to_excel(writer, basename)
        writer.save()
    else:
        print('Excel exists; skipping...')
    plot(basename)

# run(X, tth, eps, 128, 10)
# run_field(X, tth, eps, 32, 32, 10)


# corners = [[0, sh//2] for sh in X.shape]
# gt, _ = bruteforce.box(X, B, corners)
# reco, _ = tth.box(corners)
# print(np.sum(gt))
# print(np.sum(reco))
# print(np.linalg.norm(gt - reco) / np.linalg.norm(gt))
# fig = plt.figure()
# plt.plot(gt)
# plt.plot(reco)
# plt.show()
