# spectrum_optimizer

This is the latest version of our spectrum optimizer code that was first published in [Chemistry--Methods](https://doi.org/10.1002/cmtd.202200069). It processes Gaussian or ORCA outputs located in subfolders named after the molecules in the benchmark set. In our latest work, we have benchmarked a set of 139 transition metal complexes and created the database from the experimental and optimized spectra called TMPHOTCAT-139. The database has been separated into parts based on the metal centres which can be found in the repository. Each of the `.zip` files contain a `.csv` file that can be processed directly using the pandas Python library. We have also included a Jupyter Notebook that we have used to process the database and analyze the performance of the DFT functionals. 

The spectrum optimization algorithm includes the following steps:

1. excitation energies oscillator strengths are extracted from the output files of excited state calculations 
2. the extracted stick spectrum is shifted and broadened using a sum of Gaussians with bandwidth and linear scaling factor parameters
3. the two parameters are optimized until the calculated lineshape offers the best possible fit (can be evaluated using any error metric) to the experimental reference spectrum located in a given subdirectory
4. the optimized parameters and lineshapes for every molecule/functional/error_metric combination are merged into a database file that can be analyzed using our Jupyter Notebook located in the for_analysis/ subdirectory


## Requirements

Python3 (tested on 3.8+) with standard scientific libraries (numpy, pandas, seaborn, matplotlib, scipy) and wordcloud (used for the TOC figure only, not necessary for database processing). An `environment.yml` file for conda is included.

## Usage

Run `run_optimizer.py` located in the `/optimizer` folder to obtain sub-databases for each test molecule then run `merge_all.py` to merge them into a full database.

- By default, the optimizer script uses `max_workers = 8` (8 is the number of test molecules) and runs fully parallelized if 8 cores are available.
- Default is to use only the MSE error metric. Multiple metrics can be given as a list (`error_functions`) in `run_optimizer.py`.
- The fitting algorithm has been updated from a brute-force algorithm to use the global optimization algorithms available in scipy. By default, it uses the DIRECT algorithm but the Dual Annealing and brute force options have also been tested and work fine. One molecule should take around 5-10 minutes to optimize. For faster runtimes both the options of the global optimizer function and the grid resolution can be adjusted in the `get_errors_hmap()` function of `bench_opt.py`.

> **_NOTE:_**  Currently, to change the number of cores (`max_workers`), grid resolution (`SHIFT_min_max`,`SIGMA_min_max`), the list of error functions (`error_functions`) and the optimization algorithm (`get_errors_hmap()`) the `run_optimizer.py` and `bench_opt.py` files need to be modified manually.

## Analysis

The analysis of the database can be performed with the Jupyter Notebook located in the root folder.
