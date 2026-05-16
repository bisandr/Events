# Events

## Phase Space Analysis

This repository includes `phase_space_analysis.py` for nonlinear phase space analysis of one-column CSV time series files (for example `data/diff_sums.csv`).

### What it computes

- **Time delay (τ)** via Average Mutual Information (AMI), lags 1..50
- **Embedding dimension (m)** via False Nearest Neighbors (FNN), dimensions 1..10
- **Attractor reconstruction** with 2D and 3D phase portraits
- **Poincaré section** using positive-slope crossings of the mean-value section
- **Largest Lyapunov Exponent (LLE)** using Rosenstein's algorithm
- **Recurrence plot** and RQA metrics: RR, DET, Lmax, Lmean, ENT, LAM, TT

### Run

```bash
python phase_space_analysis.py
```

Analyze a different time series file:

```bash
python phase_space_analysis.py /absolute/or/relative/path/to/series.csv
```

Optional output directory:

```bash
python phase_space_analysis.py data/diff_sums.csv --output-dir output
```

Optional recurrence threshold fraction (default `0.1`, i.e. 10% of max pairwise distance):

```bash
python phase_space_analysis.py data/diff_sums.csv --recurrence-fraction 0.1
```

### Outputs

The script creates the output directory automatically (default: `output/`) and saves:

- `output/ami.png`
- `output/fnn.png`
- `output/attractor_2d.png`
- `output/attractor_3d.png`
- `output/poincare_section.png`
- `output/lyapunov.png`
- `output/recurrence_plot.png`

It also prints a summary with optimal τ, optimal m, LLE, and RQA metrics.

## Phase Space Analysis 5D

This repository also includes `phase_space_analysis_5d.py` for multivariate phase space analysis of CSV files containing one 5D state vector per row (for example `data/vectors_5d.csv`).

### What it computes

- **Data loading and preprocessing** with per-dimension z-score normalization
- **PCA projection** in 2D and 3D using NumPy SVD for attractor visualization
- **Poincaré section** from positive-slope crossings of the hyperplane `PC1 = median(PC1)`
- **Largest Lyapunov Exponent (LLE)** using Rosenstein's algorithm directly in the 5D state space
- **Recurrence plot** and RQA metrics: RR, DET, Lmax, Lmean, ENT, LAM, TT
- **Chaos summary report** combining LLE and RQA evidence

### Input

Use a CSV with 5 columns and no header, where each row is one time step:

```text
0.12,1.03,-0.44,2.10,0.55
0.14,1.01,-0.40,2.05,0.60
...
```

### Run

Default input:

```bash
python phase_space_analysis_5d.py
```

Custom input path:

```bash
python phase_space_analysis_5d.py --input path/to/file.csv
```

Custom output directory:

```bash
python phase_space_analysis_5d.py --input data/vectors_5d.csv --output output
```

### Outputs

The script creates the output directory automatically (default: `output/`) and saves:

- `output/attractor_pca_2d.png`
- `output/attractor_pca_3d.png`
- `output/poincare_section_5d.png`
- `output/lyapunov_5d.png`
- `output/recurrence_plot_5d.png`

It also prints basic preprocessing statistics, PCA explained variance, the LLE interpretation, RQA metrics, and a structured chaos report.
