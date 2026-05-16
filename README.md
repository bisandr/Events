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
