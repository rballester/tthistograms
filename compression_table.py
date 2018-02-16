"""
Gather TT-histogram offline compression statistics and display them as a LaTeX table
"""

import pickle
import numpy as np
import re
import pandas as pd
import os
import config


input_datasets = [
    os.path.join(config.data_folder, 'waterfall_64_0.0005000.tth'),
    os.path.join(config.data_folder, 'bonsai_volvis_128_0.0002000.tth'),
    os.path.join(config.data_folder, 'hnut_128_0.0000250.tth'),
    os.path.join(config.data_folder, 'flower_128_0.0000150.tth')
    ]

df = pd.DataFrame(index=['Size (MB)', '$T_D$ (s)', 'B', 'Full IH (GB)', r'\specialcell{Compression \\ target $\eps$}', r'\specialcell{Compressed \\ IH (MB)}', 'TT ranks'], columns=[r'\textbf{Waterfall}', r'\textbf{Bonsai}', r'\textbf{Hazelnut}', r'\textbf{Flower}'])

for i, filename in enumerate(input_datasets):
    tth = pickle.load(open(filename, 'rb'))
    size = np.prod(tth.tensor.n[:-1]-1) / (1024**2)
    rs = re.search(r'\D+_(\d+)_(0.\d+)', filename)
    B = int(rs.group(1))
    full_size = np.prod(tth.tensor.n[:-1]-1)*B
    if full_size // B > 2**32:  # 64-bit integers are needed
        full_size *= 8
    else:  # 32-bit integers are needed
        full_size *= 4
    full_size /= (1024**3)  # Size in GB's
    eps = float(rs.group(2))
    compressed_size = len(tth.tensor.core)*8 / (1024**2)
    tt_ranks = ", ".join([str(r) for r in tth.tensor.r[1:-1]])

    print(size)
    print(tth.total_time)
    print(B)
    print(full_size)
    print(eps)
    print(compressed_size)
    print(tt_ranks)

    df.iloc[:, i] = [size, tth.total_time, B, full_size, eps, compressed_size, tt_ranks]

    print()

print(df.to_latex(escape=False, column_format='ccccc'))
