"""
predict_2d.py -- Prediction module for 2D vector time series (values in 1-12)
==============================================================================
Adapted from predict_5d.py for 2-dimensional integer time series.

Implements:
  1. Method of Analogues (Lorenz)
  2. Local Linear Maps
  3. Vector Autoregression (VAR) -- from scratch, no statsmodels
  4. Forecast Horizon Estimation from LLE

Predictions are rounded to integers and clipped to [1, 12] to match
the domain of the input data.

Usage:
    python predict_2d.py --input data/vectors_2d.csv --horizon 5
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

NDIM   = 2
VMIN   = 1
VMAX   = 12


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_and_normalize(filepath):
    """
    Load a CSV of 2D vectors and z-score normalize each dimension.

    Returns
    -------
    data_norm : ndarray (N, 2)
    mean      : ndarray (2,)
    std       : ndarray (2,)
    raw_data  : ndarray (N, 2)  original integer values
    """
    raw = np.loadtxt(filepath, delimiter=",")
    if raw.ndim == 1:
        raise ValueError("Expected 2D data with 2 columns, got a single row.")
    if raw.ndim != 2 or raw.shape[1] != NDIM:
        raise ValueError(f"Expected shape (N, {NDIM}), got {raw.shape}")

    mean = raw.mean(axis=0)
    std  = raw.std(axis=0)
    std[std == 0] = 1.0

    print(f"\n--- Data loaded ---")
    print(f"  N time steps : {raw.shape[0]}")
    print(f"  Value range  : [{int(raw.min())}, {int(raw.max())}]")
    for d in range(NDIM):
        print(f"  Dim {d+1}: mean={mean[d]:.3f}, std={std[d]:.3f}, "
              f"min={int(raw[:,d].min())}, max={int(raw[:,d].max())}")

    return (raw - mean) / std, mean, std, raw


def denormalize(data_norm, mean, std):
    """Convert z-scored predictions back to original scale."""
    return data_norm * std + mean


def to_domain(values):
    """Round to nearest integer and clip to [VMIN, VMAX]."""
    return np.clip(np.round(values).astype(int), VMIN, VMAX)


# ─────────────────────────────────────────────────────────────
# METHOD 1 -- METHOD OF ANALOGUES
# ─────────────────────────────────────────────────────────────

def analogues_predict(train, query_states, horizon, K=5, temporal_window=5):
    """
    Method of Analogues (Lorenz 1969).

    Finds K nearest past states and averages their future trajectories.

    Parameters
    ----------
    train           : ndarray (T, 2)
    query_states    : ndarray (Q, 2)
    horizon         : int
    K               : int    number of analogues
    temporal_window : int    minimum temporal gap to nearest neighbour

    Returns
    -------
    predictions : ndarray (Q, H, 2)
    """
    T = len(train)
    Q = len(query_states)
    predictions = np.full((Q, horizon, NDIM), np.nan)

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

def local_linear_predict(train, query_states, horizon, K=15, temporal_window=5):
    """
    Local Linear Maps.

    Fits x(t+1) = A @ x(t) + b in the neighbourhood of the query state
    and iterates forward for `horizon` steps.
    """
    T = len(train)
    Q = len(query_states)
    predictions = np.full((Q, horizon, NDIM), np.nan)

    D = cdist(query_states, train[:-1], metric="euclidean")

    for q in range(Q):
        dists = D[q].copy()
        dists[max(0, T - 1 - temporal_window):] = np.inf
        dists[:temporal_window] = np.inf

        nn_idx = np.argsort(dists)[:K]
        nn_idx = [i for i in nn_idx if i + 1 < T]
        if len(nn_idx) < NDIM + 1:   # need at least 3 points to fit 2x2 + bias
            continue

        X_nn  = train[nn_idx]
        Y_nn  = train[[i + 1 for i in nn_idx]]
        X_aug = np.hstack([X_nn, np.ones((len(X_nn), 1))])
        B, _, _, _ = np.linalg.lstsq(X_aug, Y_nn, rcond=None)

        A = B[:NDIM]   # (2, 2)
        b = B[NDIM]    # (2,)

        state = query_states[q].copy()
        for h in range(horizon):
            state = A.T @ state + b
            predictions[q, h] = state

    return predictions


# ─────────────────────────────────────────────────────────────
# METHOD 3 -- VECTOR AUTOREGRESSION (VAR)
# ─────────────────────────────────────────────────────────────

def _build_var_matrices(data, p):
    T, d = data.shape
    X_rows, Y_rows = [], []
    for t in range(p, T):
        row = np.concatenate([data[t - i] for i in range(1, p + 1)] + [np.ones(1)])
        X_rows.append(row)
        Y_rows.append(data[t])
    return np.array(X_rows), np.array(Y_rows)


def _var_aic(data, p):
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
    """Select VAR lag order by AIC and fit. Returns (B, p)."""
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
    Forecast `horizon` steps with a fitted VAR(p) model.

    Parameters
    ----------
    B           : ndarray (p*d+1, d)
    p           : int
    last_states : ndarray (>=p, d)   recent history, oldest first
    horizon     : int

    Returns
    -------
    predictions : ndarray (H, 2)
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
        print(f"  LLE             = {lle:.6f}")
        print(f"  Initial sep d0  = {d0:.4f}")
        print(f"  Tolerance       = {tolerance:.4f}")
        print(f"  T_max           = {t_max:.1f} steps")

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
    path = os.path.join(output_dir, "forecast_horizon_2d.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    return t_max, d0, tolerance


# ─────────────────────────────────────────────────────────────
# FUTURE FORECAST
# ─────────────────────────────────────────────────────────────

def predict_future(data, mean, std, horizon, output_dir):
    """
    Forecast the next `horizon` steps BEYOND the last observed point.

    Predictions are printed in original integer scale (rounded, clipped to
    [VMIN, VMAX]) and saved to output/future_predictions_2d.csv.

    Parameters
    ----------
    data       : ndarray (N, 2)  normalized
    mean       : ndarray (2,)
    std        : ndarray (2,)
    horizon    : int
    output_dir : str

    Returns
    -------
    preds_raw  : dict  method -> ndarray (H, 2) in original scale
    preds_int  : dict  method -> ndarray (H, 2) rounded integers
    """
    print("\n" + "=" * 52)
    print(f"  FUTURE FORECAST -- next {horizon} steps beyond the dataset")
    print("=" * 52)

    query = data[-1:].copy()

    pred_ana = analogues_predict(data, query, horizon)[0]
    pred_llm = local_linear_predict(data, query, horizon)[0]
    B_var, p_var = fit_var(data)
    pred_var = var_predict(B_var, p_var, data, horizon)

    print(f"  (VAR lag order selected: p = {p_var})\n")

    methods    = ["Analogues", "LocalLinear", "VAR"]
    preds_norm = [pred_ana, pred_llm, pred_var]
    preds_raw  = {}
    preds_int  = {}

    header = f"{'Step':<6}  {'Dim1 (raw)':>12}  {'Dim2 (raw)':>12}  {'Dim1 (int)':>12}  {'Dim2 (int)':>12}"

    for method, pred_n in zip(methods, preds_norm):
        pred_r = denormalize(pred_n, mean, std)
        pred_i = to_domain(pred_r)
        preds_raw[method] = pred_r
        preds_int[method] = pred_i

        print(f"--- {method} ---")
        print(header)
        for step in range(horizon):
            print(f"{step+1:<6}  "
                  f"{pred_r[step,0]:>12.3f}  {pred_r[step,1]:>12.3f}  "
                  f"{pred_i[step,0]:>12}  {pred_i[step,1]:>12}")
        print()

    # Save CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "future_predictions_2d.csv")
    with open(csv_path, "w") as f:
        f.write("method,step,dim1_raw,dim2_raw,dim1_int,dim2_int\n")
        for method in methods:
            pr = preds_raw[method]
            pi = preds_int[method]
            for step in range(horizon):
                f.write(f"{method},{step+1},"
                        f"{pr[step,0]:.4f},{pr[step,1]:.4f},"
                        f"{pi[step,0]},{pi[step,1]}\n")
    print(f"Predictions saved to: {csv_path}")

    # Plot
    steps  = np.arange(1, horizon + 1)
    colors = {"Analogues": "tomato", "LocalLinear": "seagreen", "VAR": "steelblue"}
    dim_labels = ["Dim 1", "Dim 2"]

    fig, axes = plt.subplots(NDIM, 1, figsize=(8, 7), sharex=True)
    for dim, ax in enumerate(axes):
        for method in methods:
            pr = preds_raw[method]
            pi = preds_int[method]
            ax.plot(steps, pr[:, dim], "o--", color=colors[method],
                    alpha=0.5, ms=4, label=f"{method} (raw)")
            ax.step(steps, pi[:, dim], where="mid", color=colors[method],
                    linewidth=2, label=f"{method} (int)")
        ax.set_ylim(VMIN - 0.5, VMAX + 0.5)
        ax.set_yticks(range(VMIN, VMAX + 1))
        ax.set_ylabel(dim_labels[dim])
        ax.legend(fontsize=7, loc="best")
        ax.set_title(f"{dim_labels[dim]} forecast", fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
    axes[-1].set_xlabel("Steps ahead")
    fig.suptitle(f"Future Forecast -- next {horizon} steps  (integer range {VMIN}-{VMAX})", fontsize=11)
    fig.tight_layout()
    plot_path = os.path.join(output_dir, "future_predictions_2d.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to:        {plot_path}")

    return preds_raw, preds_int


# ─────────────────────────────────────────────────────────────
# CROSS-VALIDATION (ROLLING ORIGIN)
# ─────────────────────────────────────────────────────────────

def rolling_cv(data, horizon, test_size=0.2):
    """Walk-forward cross-validation over the test portion."""
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
            err   = np.full((horizon, NDIM), np.nan)
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
    """2 subplots: true vs predicted for the last CV window (original scale)."""
    H      = len(last_true)
    steps  = np.arange(1, H + 1)
    colors = {"Analogues": "tomato", "LocalLinear": "seagreen", "VAR": "steelblue"}

    true_raw = denormalize(last_true, mean, std)

    fig, axes = plt.subplots(NDIM, 1, figsize=(8, 7), sharex=True)
    for dim, ax in enumerate(axes):
        ax.plot(steps, true_raw[:, dim], "k-o", ms=5, label="Ground truth")
        for m, pred in last_pred.items():
            pred_raw = denormalize(pred, mean, std)
            pred_int = to_domain(pred_raw)
            rmse     = results[m]["rmse"]
            ax.step(steps, pred_int[:, dim], where="mid", color=colors[m],
                    linewidth=2, label=f"{m} (RMSE={rmse:.3f})")
        ax.set_ylim(VMIN - 0.5, VMAX + 0.5)
        ax.set_yticks(range(VMIN, VMAX + 1))
        ax.set_ylabel(f"Dim {dim + 1}")
        ax.legend(fontsize=7)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
    axes[-1].set_xlabel("Prediction step")
    fig.suptitle("Predictions vs Ground Truth -- last CV window", fontsize=11)
    fig.tight_layout()
    path = os.path.join(output_dir, "predictions_comparison_2d.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_error_per_step(results, output_dir):
    """RMSE vs prediction step for each method."""
    fig, ax = plt.subplots(figsize=(7, 4))
    colors  = {"Analogues": "tomato", "LocalLinear": "seagreen", "VAR": "steelblue"}
    for m, res in results.items():
        steps = np.arange(1, len(res["rmse_per_step"]) + 1)
        ax.plot(steps, res["rmse_per_step"], "o-", color=colors[m], label=m)
    ax.set_xlabel("Prediction step")
    ax.set_ylabel("RMSE (normalized scale)")
    ax.set_title("Prediction Error Growth vs Horizon")
    ax.legend()
    path = os.path.join(output_dir, "error_per_step_2d.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────
# SUMMARY REPORT
# ─────────────────────────────────────────────────────────────

def print_summary(filepath, N, horizon, lle, t_max, d0, tolerance, results):
    best = min(results, key=lambda m: results[m]["rmse"])
    sep  = "=" * 52
    print(f"\n{sep}")
    print("           PREDICTION SUMMARY")
    print(sep)
    print(f"Dataset              : {os.path.basename(filepath)}")
    print(f"N time steps         : {N}")
    print(f"Dimensions           : {NDIM}  (integer range {VMIN}-{VMAX})")
    print(f"Prediction horizon H : {horizon}")
    print(f"\n--- Forecast Horizon (from LLE) ---")
    print(f"  LLE          = {lle:.6f}")
    print(f"  Initial sep  = {d0:.4f}")
    print(f"  Tolerance    = {tolerance:.4f}")
    if np.isfinite(t_max):
        print(f"  T_max        = {t_max:.1f} steps")
    else:
        print(f"  T_max        = inf")
    print(f"\n--- Cross-Validation Results (normalized RMSE) ---")
    print(f"  {'Method':<20} {'MAE':>8}   {'RMSE':>8}")
    print(f"  {'-'*38}")
    for m, res in results.items():
        print(f"  {m:<20} {res['mae']:>8.4f}   {res['rmse']:>8.4f}")
    print(f"\n  Best method: {best}  (RMSE = {results[best]['rmse']:.4f})")
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

    # 2. Future predictions -- numerical output in original integer scale
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
        description=(
            "2D integer time series prediction (values 1-12): "
            "analogues, local linear maps, VAR, LLE horizon."
        )
    )
    parser.add_argument("--input",     default="data/vectors_2d.csv",
                        help="CSV file with 2D vectors (N x 2, no header, values 1-12).")
    parser.add_argument("--horizon",   type=int,   default=5,
                        help="Number of future steps to predict.")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of data held out for cross-validation.")
    parser.add_argument("--lle",       type=float, default=0.003132,
                        help="Largest Lyapunov Exponent (run phase_space_analysis_5d.py to obtain).")
    parser.add_argument("--output",    default="output",
                        help="Directory for output plots and CSVs.")
    args = parser.parse_args()

    run_pipeline(args.input, args.horizon, args.test_size, args.lle, args.output)
