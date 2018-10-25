## TT-histograms: Integral Histogram Compression and Look-up in the Tensor Train Format

This is a Python implementation of the method and experiments from the paper ["Tensor Decompositions for Integral Histogram Compression and Look-Up"](http://ieeexplore.ieee.org/document/8281540/) (Rafael Ballester-Ripoll and Renato Pajarola), IEEE Transactions on Visualization and Computer Graphics, 2018.

### Motivation

Computing histograms of large multidimensional data is time-consuming as it may require millions of random accesses. The [integral histogram](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1467353) trades space for speed: it needs only a few hundred accesses per query, but takes up orders of magnitude more storage.

TT-histograms compress integral histograms to make them manageable and offer fast and approximate look-up over medium to large regions. As a bonus, we can also query non-rectangular regions (separable regions are especially friendly) and compute histogram fields as well. For small regions, brute-force histogram computation is still the better option, since it is error-free and sufficiently fast.

### Usage

The main class is `TTHistogram` as contained in `tthistogram.py`. It features:

- A constructor that builds a TT-compressed integral histogram representation out of an ND array with an arbitrary number of bins;
- The function `box()`, which looks up a histogram over any given box region;
- The function `separable()`, which looks up a histogram over a separable region (e.g. Gaussian);
- The function `nonseparable()`, which looks up a histogram over a non-separable region;
- The function `box_field()`, which reconstructs a histogram field made up of box regions;
- The function `separable_field()`, which reconstructs a histogram field made up of separable regions.

Note: for benchmarking, all look-ups are also implemented using brute-force traversal in both NumPy (file `bruteforce.py`) and [CuPy](https://cupy.chainer.org/) (file `bruteforce_cupy.py`). The remaining files simply reproduce the experiments in the [paper](http://ieeexplore.ieee.org/document/8281540/).

### Dependencies

- The [ttpy](https://github.com/oseledets/ttpy/) toolbox for several elementary operations with compressed TT tensors.

- The [ttrecipes](https://github.com/rballester/ttrecipes) toolbox, which contains some extra homemade utility functions for working with the TT format, as well as the ```sum_and_compress()``` function for incremental TT decomposition that we use to compress huge integral histograms.

- From the Python ecosystem: matplotlib, SciPy

- Optionally (for benchmarking only): pandas, [CuPy](https://cupy.chainer.org/) (a NumPy-compatible matrix library accelerated by CUDA)

## Acknowledgment

This work was partially supported by the [UZH Forschungskredit "Candoc"](http://www.researchers.uzh.ch/en/funding/phd/fkcandoc.html), grant number FK-16-012.
