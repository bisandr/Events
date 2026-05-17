# predict_5d.py

Forecasts the next N steps of a 5-dimensional time series using three
complementary methods and estimates how far ahead the predictions can be
trusted based on the system's largest Lyapunov exponent (LLE).

---

## Methods

| Method | How it works |
|---|---|
| **Method of Analogues** | Finds the K past states most similar to the current one and averages their future trajectories |
| **Local Linear Maps** | Fits a linear map in the neighbourhood of the current state and iterates it forward |
| **Vector Autoregression (VAR)** | Fits a global linear model across all 5 dimensions using the lag order selected by AIC |
| **Forecast Horizon (LLE)** | Uses the LLE to compute the theoretical maximum reliable prediction window: `T_max = (1/LLE) * ln(tolerance / d0)` |

---

## Input

A CSV file where each row is one time step and each column is one dimension.
No header row. Example (`data/vectors_5d.csv`):

```
5,12,21,30,38
6,13,22,31,39
7,14,23,32,40
```

---

## Output

All files are written to the `output/` directory (created automatically).

| File | Description |
|---|---|
| `future_predictions.csv` | Numerical predictions for all 3 methods in the original data scale |
| `future_predictions.png` | Per-dimension forecast plots for all 3 methods |
| `forecast_horizon.png` | Divergence curve showing the theoretical reliability limit |
| `predictions_comparison.png` | Cross-validation: predicted vs ground truth (last window) |
| `error_per_step.png` | RMSE per prediction step for each method |

A summary table is also printed to stdout at the end of the run.

---

## Usage Examples

### Basic — predict the next 5 steps

```bash
python predict_5d.py --input data/vectors_5d.csv
```

### Predict 10 steps ahead

```bash
python predict_5d.py --input data/vectors_5d.csv --horizon 10
```

### Use your own LLE value (from phase_space_analysis_5d.py)

```bash
python predict_5d.py --input data/vectors_5d.csv --lle 0.003132
```

### Save outputs to a custom directory

```bash
python predict_5d.py --input data/vectors_5d.csv --output results/my_run
```

### Combine options

```bash
python predict_5d.py \
  --input   data/vectors_5d.csv \
  --horizon 8 \
  --lle     0.003132 \
  --test-size 0.15 \
  --output  output/run_01
```

---

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--input` | `data/vectors_5d.csv` | Path to the input CSV (N × 5, no header) |
| `--horizon` | `5` | Number of future steps to forecast |
| `--lle` | `0.003132` | Largest Lyapunov Exponent (from `phase_space_analysis_5d.py`) |
| `--test-size` | `0.2` | Fraction of data held out for cross-validation |
| `--output` | `output` | Directory for all output files |

---

## Notes

- The LLE for the default dataset (`events_indices.csv`) is **0.003132**, placing the
  theoretical forecast horizon at roughly **40–50 steps**. In practice, prediction
  quality degrades noticeably past **5 steps** (Lmax = 5 from the recurrence analysis).
- All three methods operate on z-score normalized data internally; all printed
  values and saved CSVs are in the **original data scale**.
- No external chaos or statistics libraries are required — only `numpy`,
  `scipy`, and `matplotlib`.

---

## Dependencies

```bash
pip install numpy scipy matplotlib
```
