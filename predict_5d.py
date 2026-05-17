"""
predict_5d.py -- Prediction module for 5D vector time series
============================================================
Implements 4 prediction approaches:
  1. Method of Analogues (Lorenz)
  2. Local Linear Maps
  3. Vector Autoregression (VAR) -- from scratch, no statsmodels
  4. Forecast Horizon Estimation from LLE

Usage:
    python predict_5d.py --input data/vectors_5d.csv --horizon 5 --test-size 0.2
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_and_normalize(filepath):
    """Load a CSV of 5D vectors and z-score normalize.
    Returns (data_norm, mean, std, raw_data).
    """
    data = np.loadtxt(filepath, delimiter=",")
    if data.ndim != 2 or data.shape[1] != 5:
        raise ValueError(f"Expected shape (N, 5), got {data.shape}")
    mean = data.mean(axis=0)
    std  = data.std(axis=0)
    std[std == 0] = 1.0
    return (data - mean) / std, mean, std, data


def denormalize(data_norm, mean, std):
    """Convert z-scored data back to original scale."""
    return data_norm * std + mean


# ─────────────────────────────────────────────────────────────
# METHOD 1 -- METHOD OF ANALOGUES
# ─────────────────────────────────────────────────────────────

def analogues_predict(train, query_states, horizon, K=5, temporal_window=10):
    """
    Method of Analogues (Lorenz 1969).

    For each query state find K nearest neighbours in `train` (excluding
    temporal neighbours within `temporal_window`), then average their
    future H-step trajectories as the prediction.

    Parameters
    ----------
    train           : ndarray (T, 5)
    query_states    : ndarray (Q, 5)
    horizon         : int
    K               : int
    temporal_window : int

    Returns
    -------
    predictions : ndarray (Q, H, 5)
    """
    T = len(train)
    Q = len(query_states)
    predictions = np.full((Q, horizon, 5), np.nan)

    D = cdist(query_states, train, metric="euclidean")

    for q in range(Q):
        dists = D[q].copy()
        dists[max(0, T - temporal_window):] = np.inf
        dists[:temporal_window] = np.inf

        nn_idx = np.argsort(dists)[:K]
        valid  = [i for i in nn_idx if i + horizon < T]
        if len(valid) == 0:
            continue
        futures = np.stack([train[i + 1 : i + 1 + horizon] for i in valid])
        predictions[q] = futures.mean(axis=0)

    return predictions


# ─────────────────────────────────────────────────────────────
# METHOD 2 -- LOCAL LINEAR MAPS
# ─────────────────────────────────────────────────────────────

def local_linear_predict(train, query_states, horizon, K=20, temporal_window=10):
    """
    Local Linear Maps.

    Fit a linear map  x(t+1) = A @ x(t) + b  in the neighbourhood of each
    query state, then iterate for `horizon` steps.
    """
    T = len(train)
    Q = len(query_states)
    predictions = np.full((Q, horizon, 5), np.nan)

    D = cdist(query_states, train[:-1], metric="euclidean")

    for q in range(Q):
        dists = D[q].copy()
        dists[max(0, T - 1 - temporal_window):] = np.inf
        dists[:temporal_window] = np.inf

        nn_idx = np.argsort(dists)[:K]
        nn_idx = [i for i in nn_idx if i + 1 < T]
        if len(nn_idx) < 6:
            continue

        X_nn  = train[nn_idx]
        Y_nn  = train[[i + 1 for i in nn_idx]]
        X_aug = np.hstack([X_nn, np.ones((len(X_nn), 1))])
        B, _, _, _ = np.linalg.lstsq(X_aug, Y_nn, rcond=None)

        A = B[:5]
        b = B[5]

        state = query_states[q].copy()
        for h in range(horizon):
            state = A.T @ state + b
            predictions[q, h] = state

    return predictions


# ─────────────────────────────────────────────────────────────
# METHOD 3 -- VECTOR AUTOREGRESSION (VAR)
# ─────────────────────────────────────────────────────────────

def _build_var_matrices(data, p):
    """Build design matrix X and response Y for VAR(p)."""
    T, d = data.shape
    X_rows, Y_rows = [], []
    for t in range(p, T):
        row = np.concatenate([data[t - i] for i in range(1, p + 1)] + [np.ones(1)])
        X_rows.append(row)
        Y_rows.append(data[t])
    return np.array(X_rows), np.array(Y_rows)


def _var_aic(data, p):
    """AIC for VAR(p)."""
    T, d = data.shape
    X, Y = _build_var_matrices(data, p)
    B, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    residuals = Y - X @ B
    sigma = residuals.T @ residuals / T
    sign, logdet = np.linalg.slogdet(sigma)
    if sign <= 0:
        return np.inf
    k = d * (d * p + 1)
    return T * logdet + 2 * k


def fit_var(train, max_lag=10):
    """Select VAR lag order by AIC and fit model. Returns (B, p)."""
    T, d = train.shape
    best_aic, best_p = np.inf, 1
    for p in range(1, min(max_lag + 1, T // 3)):
        aic = _var_aic(train, p)
        if aic < best_aic:
            best_aic, best_p = aic, p
    X, Y = _build_var_matrices(train, best_p)
    B, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    return B, best_p


def var_predict(B, p, last_states, horizon):
    """
    Predict `horizon` steps ahead with a fitted VAR(p) model.

    Parameters
    ----------
    B           : ndarray (p*d+1, d)
    p           : int
    last_states : ndarray (>=p, d)   recent history, oldest first
    horizon     : int

    Returns
    -------
    predictions : ndarray (H, 5)
    """
    history = list(last_states[-p:])
    preds = []
    for _ in range(horizon):
        row = np.concatenate([history[-(i + 1)] for i in range(p)] + [np.ones(1)])
        next_state = row @ B
        preds.append(next_state)
        history.append(next_state)
    return np.array(preds)


# ─────────────────────────────────────────────────────────────
# METHOD 4 -- FORECAST HORIZON FROM LLE
# ─────────────────────────────────────────────────────────────

def forecast_horizon(data, lle, output_dir):
    """Estimate and plot the maximum reliable forecast window from LLE."""
    D = cdist(data, data, metric="euclidean")
    np.fill_diagonal(D, np.inf)
    d0 = float(np.median(D.min(axis=1)))
    tolerance = float(data.std())

    if lle <= 0 or d0 <= 0 or tolerance <= d0:
        t_max = np.inf
        print("\nLLE <= 0 -> system is stable, T_max = inf")
    else:
        t_max = (1.0 / lle) * np.log(tolerance / d0)
        print(f"\n--- Forecast Horizon ---")
        print(f"  LLE              = {lle:.6f}")
        print(f"  Initial sep d0   = {d0:.4f}")
        print(f"  Tolerance        = {tolerance:.4f}")
        print(f"  T_max            = {t_max:.1f} steps")

    t_plot = np.linspace(0, max(50, int(t_max * 2) if np.isfinite(t_max) else 100), 300)
    d_t    = d0 * np.exp(lle * t_plot)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_plot, d_t, "steelblue", label="d(t) = d0 * exp(LLE * t)")
    ax.axhline(tolerance, color="red",   linestyle="--", label=f"Tolerance = {tolerance:.3f}")
    if np.isfinite(t_max):
        ax.axvline(t_max, color="green", linestyle=":", label=f"T_max = {t_max:.1f} steps")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Divergence")
    ax.set_title("Theoretical Forecast Horizon (from LLE)")
    ax.legend()
    ax.set_ylim(0, tolerance * 2.5)
    path = os.path.join(output_dir, "forecast_horizon.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    return t_max, d0, tolerance


# ─────────────────────────────────────────────────────────────
# FUTURE FORECAST -- numerical predictions beyond the last step
# ─────────────────────────────────────────────────────────────

def predict_future(data, mean, std, horizon, output_dir):
    """
    Forecast the next `horizon` steps BEYOND the last observed point.

    Uses the full dataset as training. Predictions are printed and saved
    in the ORIGINAL (denormalized) scale to output/future_predictions.csv.

    Parameters
    ----------
    data       : ndarray (N, 5)  normalized time series
    mean       : ndarray (5,)    per-dimension means (original scale)
    std        : ndarray (5,)    per-dimension stds  (original scale)
    horizon    : int
    output_dir : str

    Returns
    -------
    preds_raw : dict  method -> ndarray (H, 5) in original scale
    """
    print("\n" + "=" * 62)
    print(f"  FUTURE FORECAST -- next {horizon} steps beyond the dataset")
    print("=" * 62)

    query = data[-1:].copy()   # seed: last observed normalized state

    # Run all three methods
    pred_ana = analogues_predict(data, query, horizon)[0]
    pred_llm = local_linear_predict(data, query, horizon)[0]
    B_var, p_var = fit_var(data)
    pred_var = var_predict(B_var, p_var, data, horizon)

    print(f"  (VAR lag order selected: p = {p_var})\n")

    methods    = ["Analogues", "LocalLinear", "VAR"]
    preds_norm = [pred_ana, pred_llm, pred_var]
    preds_raw  = {}

    dim_labels = [f"Dim{i+1}" for i in range(5)]
    header = f"{'Step':<6}" + "".join(f"{lbl:>14}" for lbl in dim_labels)

    for method, pred_n in zip(methods, preds_norm):
        pred_r = denormalize(pred_n, mean, std)
        preds_raw[method] = pred_r

        print(f"--- {method} ---")
        print(header)
        for step in range(horizon):
            row = f"{step+1:<6}" + "".join(f"{pred_r[step, d]:>14.4f}" for d in range(5))
            print(row)
        print()

    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "future_predictions.csv")
    with open(csv_path, "w") as f:
        f.write("method,step," + ",".join(dim_labels) + "\n")
        for method, pred_r in preds_raw.items():
            for step in range(horizon):
                vals = ",".join(f"{pred_r[step, d]:.4f}" for d in range(5))
                f.write(f"{method},{step+1},{vals}\n")
    print(f"Predictions saved to: {csv_path}")

    # Plot: one subplot per dimension
    fig, axes = plt.subplots(5, 1, figsize=(9, 12), sharex=True)
    steps  = np.arange(1, horizon + 1)
    colors = {"Analogues": "tomato", "LocalLinear": "seagreen", "VAR": "steelblue"}

    for dim, ax in enumerate(axes):
        for method, pred_r in preds_raw.items():
            ax.plot(steps, pred_r[:, dim], "o--", color=colors[method],
                    label=method, ms=5)
        ax.set_ylabel(f"Dim {dim+1}")
        ax.legend(fontsize=7, loc="best")
        ax.set_title(f"Dim {dim+1} forecast", fontsize=9)
    axes[-1].set_xlabel("Steps ahead")
    fig.suptitle(f"Future Forecast -- next {horizon} steps (original scale)", fontsize=12)
    fig.tight_layout()
    plot_path = os.path.join(output_dir, "future_predictions.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to:        {plot_path}")

    return preds_raw


# ─────────────────────────────────────────────────────────────
# CROSS-VALIDATION (ROLLING ORIGIN)
# ─────────────────────────────────────────────────────────────

def rolling_cv(data, horizon, test_size=0.2):
    """Walk-forward cross-validation over the test portion of the data."""
    N       = len(data)
    split   = int(N * (1 - test_size))
    origins = range(split, N - horizon)

    methods = ["Analogues", "LocalLinear", "VAR"]
    errors  = {m: [] for m in methods}

    last_pred      = {}
    last_true      = None
    var_p_selected = None
    B_var          = None

    print(f"\nRunning rolling-origin CV: {len(origins)} windows, horizon={horizon} ...")

    for idx, t in enumerate(origins):
        train = data[:t]
        true  = data[t:t + horizon]
        q     = data[t - 1:t]

        pred_ana = analogues_predict(train, q, horizon)[0]
        pred_llm = local_linear_predict(train, q, horizon)[0]

        if idx % 10 == 0 or B_var is None:
            B_var, var_p_selected = fit_var(train)
        pred_var = var_predict(B_var, var_p_selected, train, horizon)

        for m, pred in zip(methods, [pred_ana, pred_llm, pred_var]):
            valid = ~np.isnan(pred).any(axis=1)
            err   = np.full((horizon, 5), np.nan)
            if valid.sum() > 0:
                err[valid] = np.abs(pred[valid] - true[valid])
            errors[m].append(err)

        if t == list(origins)[-1]:
            last_true                = true
            last_pred["Analogues"]   = pred_ana
            last_pred["LocalLinear"] = pred_llm
            last_pred["VAR"]         = pred_var

    print(f"  VAR lag order selected: p = {var_p_selected}")

    results = {}
    for m in methods:
        e = np.array(errors[m])
        results[m] = {
            "mae":           float(np.nanmean(e)),
            "rmse":          float(np.sqrt(np.nanmean(e ** 2))),
            "rmse_per_step": np.sqrt(np.nanmean(e ** 2, axis=(0, 2))),
        }
    return results, last_pred, last_true


# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────

def plot_predictions_comparison(last_true, last_pred, mean, std, results, output_dir):
    """5 subplots: true vs predicted for the last CV window (original scale)."""
    H      = len(last_true)
    steps  = np.arange(1, H + 1)
    colors = {"Analogues": "tomato", "LocalLinear": "seagreen", "VAR": "steelblue"}

    true_raw = denormalize(last_true, mean, std)

    fig, axes = plt.subplots(5, 1, figsize=(9, 12), sharex=True)
    for dim, ax in enumerate(axes):
        ax.plot(steps, true_raw[:, dim], "k-o", ms=4, label="Ground truth")
        for m, pred in last_pred.items():
            pred_raw = denormalize(pred, mean, std)
            rmse     = results[m]["rmse"]
            ax.plot(steps, pred_raw[:, dim], color=colors[m], linestyle="--",
                    marker="s", ms=3, label=f"{m} (RMSE={rmse:.3f})")
        ax.set_ylabel(f"Dim {dim + 1}")
        ax.legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("Prediction step")
    fig.suptitle("Predictions vs Ground Truth -- last CV window (original scale)", fontsize=12)
    fig.tight_layout()
    path = os.path.join(output_dir, "predictions_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_error_per_step(results, output_dir):
    """RMSE vs prediction step for each method."""
    fig, ax = plt.subplots(figsize=(7, 5))
    colors  = {"Analogues": "tomato", "LocalLinear": "seagreen", "VAR": "steelblue"}
    for m, res in results.items():
        steps = np.arange(1, len(res["rmse_per_step"]) + 1)
        ax.plot(steps, res["rmse_per_step"], "o-", color=colors[m], label=m)
    ax.set_xlabel("Prediction step")
    ax.set_ylabel("RMSE")
    ax.set_title("Prediction Error Growth vs Horizon")
    ax.legend()
    path = os.path.join(output_dir, "error_per_step.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────
# SUMMARY REPORT
# ─────────────────────────────────────────────────────────────

def print_summary(filepath, N, horizon, lle, t_max, d0, tolerance, results):
    """Print a structured prediction summary to stdout."""
    best = min(results, key=lambda m: results[m]["rmse"])
    sep  = "=" * 62
    print(f"\n{sep}")
    print("              PREDICTION SUMMARY")
    print(sep)
    print(f"Dataset              : {os.path.basename(filepath)}")
    print(f"N time steps         : {N}")
    print(f"Prediction horizon H : {horizon}")
    print(f"\n--- Forecast Horizon (from LLE) ---")
    print(f"  LLE          = {lle:.6f}")
    print(f"  Initial sep  = {d0:.4f}")
    print(f"  Tolerance    = {tolerance:.4f}")
    if np.isfinite(t_max):
        print(f"  T_max        = {t_max:.1f} steps")
    else:
        print(f"  T_max        = inf")
    print(f"\n--- Cross-Validation Results (normalized scale RMSE) ---")
    print(f"  {'Method':<20} {'MAE':>8}   {'RMSE':>8}")
    print(f"  {'-'*42}")
    for m, res in results.items():
        print(f"  {m:<20} {res['mae']:>8.4f}   {res['rmse']:>8.4f}")
    print(f"\n  Best method: {best}  (lowest RMSE = {results[best]['rmse']:.4f})")
    print(sep)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def run_pipeline(filepath, horizon, test_size, lle, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    data, mean, std, _ = load_and_normalize(filepath)
    N = len(data)

    # 1. Forecast horizon from LLE
    t_max, d0, tolerance = forecast_horizon(data, lle, output_dir)

    # 2. Future predictions -- prints numerical table in original scale
    predict_future(data, mean, std, horizon, output_dir)

    # 3. Cross-validation
    results, last_pred, last_true = rolling_cv(data, horizon, test_size)

    # 4. Plots
    plot_predictions_comparison(last_true, last_pred, mean, std, results, output_dir)
    plot_error_per_step(results, output_dir)

    # 5. Summary
    print_summary(filepath, N, horizon, lle, t_max, d0, tolerance, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="5D time series prediction: analogues, local linear maps, VAR, LLE horizon."
    )
    parser.add_argument("--input",     default="data/vectors_5d.csv",
                        help="CSV file with 5D vectors (N x 5, no header).")
    parser.add_argument("--horizon",   type=int,   default=5,
                        help="Number of future steps to predict.")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of data used as CV test set.")
    parser.add_argument("--lle",       type=float, default=0.003132,
                        help="LLE value from phase_space_analysis_5d.py output.")
    parser.add_argument("--output",    default="output",
                        help="Directory for output plots and CSVs.")
    args = parser.parse_args()

    run_pipeline(args.input, args.horizon, args.test_size, args.lle, args.output)
