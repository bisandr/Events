"""
predict_5d.py — Prediction module for 5D vector time series
============================================================
Implements 4 prediction approaches:
  1. Method of Analogues (Lorenz)
  2. Local Linear Maps
  3. Vector Autoregression (VAR) — from scratch, no statsmodels
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
    """Load a CSV of 5D vectors and z-score normalize. Returns (data_norm, mean, std)."""
    data = np.loadtxt(filepath, delimiter=",")
    if data.ndim != 2 or data.shape[1] != 5:
        raise ValueError(f"Expected shape (N, 5), got {data.shape}")
    mean = data.mean(axis=0)
    std  = data.std(axis=0)
    std[std == 0] = 1.0
    return (data - mean) / std, mean, std


# ─────────────────────────────────────────────────────────────
# METHOD 1 — METHOD OF ANALOGUES
# ─────────────────────────────────────────────────────────────

def analogues_predict(train, query_states, horizon, K=5, temporal_window=10):
    """
    Method of Analogues (Lorenz 1969).

    For each query state find K nearest neighbours in `train` (excluding
    temporal neighbours within `temporal_window`), then average their
    future H-step trajectories as the prediction.

    Parameters
    ----------
    train         : ndarray (T, 5)   training trajectory
    query_states  : ndarray (Q, 5)   states to predict from
    horizon       : int              number of steps to predict
    K             : int              number of analogues
    temporal_window : int            minimum temporal gap to nearest-neighbour

    Returns
    -------
    predictions : ndarray (Q, H, 5)
    """
    T = len(train)
    Q = len(query_states)
    predictions = np.full((Q, horizon, 5), np.nan)

    # Distance matrix: query → train
    D = cdist(query_states, train, metric="euclidean")   # (Q, T)

    for q in range(Q):
        dists = D[q].copy()
        # Mask out temporal neighbours (assume query comes just after train)
        dists[max(0, T - temporal_window):] = np.inf
        dists[:temporal_window] = np.inf

        nn_idx = np.argsort(dists)[:K]
        valid  = [i for i in nn_idx if i + horizon < T]
        if len(valid) == 0:
            continue
        futures = np.stack([train[i + 1 : i + 1 + horizon] for i in valid])  # (k, H, 5)
        predictions[q] = futures.mean(axis=0)

    return predictions


# ─────────────────────────────────────────────────────────────
# METHOD 2 — LOCAL LINEAR MAPS
# ─────────────────────────────────────────────────────────────

def local_linear_predict(train, query_states, horizon, K=20, temporal_window=10):
    """
    Local Linear Maps.

    Fit a linear map  x(t+1) = A @ x(t) + b  in the neighbourhood of each
    query state, then iterate for `horizon` steps.

    Parameters
    ----------
    train         : ndarray (T, 5)
    query_states  : ndarray (Q, 5)
    horizon       : int
    K             : int   neighbourhood size
    temporal_window : int

    Returns
    -------
    predictions : ndarray (Q, H, 5)
    """
    T = len(train)
    Q = len(query_states)
    predictions = np.full((Q, horizon, 5), np.nan)

    D = cdist(query_states, train[:-1], metric="euclidean")   # avoid out-of-bounds

    for q in range(Q):
        dists = D[q].copy()
        dists[max(0, T - 1 - temporal_window):] = np.inf
        dists[:temporal_window] = np.inf

        nn_idx = np.argsort(dists)[:K]
        nn_idx = [i for i in nn_idx if i + 1 < T]
        if len(nn_idx) < 6:       # need enough points to fit 5×5 + bias
            continue

        X_nn = train[nn_idx]               # (K, 5)
        Y_nn = train[[i + 1 for i in nn_idx]]  # (K, 5)

        # Augment with bias column
        X_aug = np.hstack([X_nn, np.ones((len(X_nn), 1))])   # (K, 6)
        # Least squares: Y = X_aug @ B  →  B shape (6, 5)
        B, _, _, _ = np.linalg.lstsq(X_aug, Y_nn, rcond=None)

        A = B[:5]    # (5, 5)
        b = B[5]     # (5,)

        state = query_states[q].copy()
        for h in range(horizon):
            state = A.T @ state + b
            predictions[q, h] = state

    return predictions


# ─────────────────────────────────────────────────────────────
# METHOD 3 — VECTOR AUTOREGRESSION (VAR)
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
    """AIC for VAR(p). Returns float."""
    T, d = data.shape
    X, Y = _build_var_matrices(data, p)
    B, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    residuals = Y - X @ B
    sigma = residuals.T @ residuals / T
    sign, logdet = np.linalg.slogdet(sigma)
    if sign <= 0:
        return np.inf
    k = d * (d * p + 1)     # number of free parameters
    return T * logdet + 2 * k


def fit_var(train, max_lag=10):
    """
    Select VAR lag order by AIC and fit model.

    Returns
    -------
    B     : ndarray (p*d + 1, d)  coefficient matrix
    p_opt : int  selected lag order
    """
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
    Predict `horizon` steps with a fitted VAR(p) model.

    Parameters
    ----------
    B           : ndarray (p*d+1, d)  fitted coefficients
    p           : int                 lag order
    last_states : ndarray (p, d)      last p observed states (oldest first)
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
# METHOD 4 — FORECAST HORIZON FROM LLE
# ─────────────────────────────────────────────────────────────

def forecast_horizon(data, lle, output_dir):
    """
    Estimate maximum reliable forecast window from LLE.

    T_max = (1 / LLE) * ln(tolerance / d0)

    where tolerance = mean std of normalized data ≈ 1.0
    and d0 = median nearest-neighbour distance.
    """
    # Median nearest-neighbour distance
    D = cdist(data, data, metric="euclidean")
    np.fill_diagonal(D, np.inf)
    d0 = float(np.median(D.min(axis=1)))
    tolerance = float(data.std())

    if lle <= 0 or d0 <= 0 or tolerance <= d0:
        t_max = np.inf
        print(f"\nLLE ≤ 0 → system is stable, T_max = ∞")
    else:
        t_max = (1.0 / lle) * np.log(tolerance / d0)
        print(f"\n--- Forecast Horizon ---")
        print(f"  LLE              = {lle:.6f}")
        print(f"  Initial sep d0   = {d0:.4f}")
        print(f"  Tolerance        = {tolerance:.4f}")
        print(f"  T_max            = {t_max:.1f} steps")

    # Plot divergence envelope
    t_plot = np.linspace(0, max(50, int(t_max * 2) if np.isfinite(t_max) else 100), 300)
    d_t = d0 * np.exp(lle * t_plot)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_plot, d_t, "steelblue", label=f"d(t) = d₀·e^(LLE·t)")
    ax.axhline(tolerance, color="red", linestyle="--", label=f"Tolerance = {tolerance:.3f}")
    if np.isfinite(t_max):
        ax.axvline(t_max, color="green", linestyle=":", label=f"T_max = {t_max:.1f} steps")
    ax.set_xlabel("Time step"); ax.set_ylabel("Divergence")
    ax.set_title("Theoretical Forecast Horizon (from LLE)")
    ax.legend()
    ax.set_ylim(0, tolerance * 2.5)
    path = os.path.join(output_dir, "forecast_horizon.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    return t_max, d0, tolerance


# ─────────────────────────────────────────────────────────────
# CROSS-VALIDATION (ROLLING ORIGIN)
# ─────────────────────────────────────────────────────────────

def rolling_cv(data, horizon, test_size=0.2, lle=0.003132):
    """
    Walk-forward cross-validation over the test portion.

    For each origin in the test set, train on all prior data and predict
    the next `horizon` steps.

    Returns
    -------
    results : dict  method → {"mae": float, "rmse": float, "rmse_per_step": ndarray}
    last_pred : dict  method → ndarray (H, 5)  last prediction (for plot)
    last_true : ndarray (H, 5)  ground truth for last window
    """
    N = len(data)
    split = int(N * (1 - test_size))
    origins = range(split, N - horizon)

    methods = ["Analogues", "LocalLinear", "VAR"]
    errors  = {m: [] for m in methods}   # list of (H, 5) absolute errors

    last_pred = {}
    last_true = None
    var_p_selected = None

    print(f"\nRunning rolling-origin CV: {len(origins)} windows, horizon={horizon} ...")

    for idx, t in enumerate(origins):
        train = data[:t]
        true  = data[t:t + horizon]    # (H, 5)

        q = data[t - 1:t]              # (1, 5) — last observed state

        # 1. Analogues
        pred_ana = analogues_predict(train, q, horizon)[0]      # (H, 5)

        # 2. Local Linear
        pred_llm = local_linear_predict(train, q, horizon)[0]   # (H, 5)

        # 3. VAR — refit every 10 windows to save time
        if idx % 10 == 0 or var_p_selected is None:
            B_var, var_p_selected = fit_var(train)
        pred_var = var_predict(B_var, var_p_selected, train, horizon)  # (H, 5)

        for m, pred in zip(methods, [pred_ana, pred_llm, pred_var]):
            valid = ~np.isnan(pred).any(axis=1)
            if valid.sum() == 0:
                errors[m].append(np.full((horizon, 5), np.nan))
            else:
                err = np.full((horizon, 5), np.nan)
                err[valid] = np.abs(pred[valid] - true[valid])
                errors[m].append(err)

        # Keep last window
        if t == list(origins)[-1]:
            last_true = true
            last_pred["Analogues"]  = pred_ana
            last_pred["LocalLinear"] = pred_llm
            last_pred["VAR"]        = pred_var

    print(f"  VAR lag order selected: p = {var_p_selected}")

    results = {}
    for m in methods:
        e = np.array(errors[m])          # (n_windows, H, 5)
        mae_per  = np.nanmean(e, axis=(0, 2))    # (H,)
        rmse_per = np.sqrt(np.nanmean(e ** 2, axis=(0, 2)))
        results[m] = {
            "mae":          float(np.nanmean(e)),
            "rmse":         float(np.sqrt(np.nanmean(e ** 2))),
            "rmse_per_step": rmse_per,
        }

    return results, last_pred, last_true


# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────

def plot_predictions_comparison(last_true, last_pred, results, output_dir):
    """5 subplots — one per dimension — true vs each method."""
    H = len(last_true)
    steps = np.arange(1, H + 1)
    colors = {"Analogues": "tomato", "LocalLinear": "seagreen", "VAR": "steelblue"}

    fig, axes = plt.subplots(5, 1, figsize=(9, 12), sharex=True)
    for dim, ax in enumerate(axes):
        ax.plot(steps, last_true[:, dim], "k-o", ms=4, label="Ground truth")
        for m, pred in last_pred.items():
            rmse = results[m]["rmse"]
            ax.plot(steps, pred[:, dim], color=colors[m], linestyle="--",
                    marker="s", ms=3, label=f"{m} (RMSE={rmse:.3f})")
        ax.set_ylabel(f"Dim {dim + 1}")
        ax.legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("Prediction step")
    fig.suptitle("Predictions vs Ground Truth (last CV window)", fontsize=12)
    fig.tight_layout()
    path = os.path.join(output_dir, "predictions_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_error_per_step(results, output_dir):
    """RMSE vs prediction step for each method."""
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"Analogues": "tomato", "LocalLinear": "seagreen", "VAR": "steelblue"}
    for m, res in results.items():
        steps = np.arange(1, len(res["rmse_per_step"]) + 1)
        ax.plot(steps, res["rmse_per_step"], "o-", color=colors[m], label=m)
    ax.set_xlabel("Prediction step"); ax.set_ylabel("RMSE")
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
    best = min(results, key=lambda m: results[m]["rmse"])
    sep = "=" * 62
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
    print(f"  T_max        = {t_max:.1f} steps" if np.isfinite(t_max) else "  T_max        = ∞")
    print(f"\n--- Cross-Validation Results ---")
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

    data, _, _ = load_and_normalize(filepath)
    N = len(data)

    # Forecast horizon
    t_max, d0, tolerance = forecast_horizon(data, lle, output_dir)

    # Cross-validation
    results, last_pred, last_true = rolling_cv(data, horizon, test_size, lle)

    # Plots
    plot_predictions_comparison(last_true, last_pred, results, output_dir)
    plot_error_per_step(results, output_dir)

    # Summary
    print_summary(filepath, N, horizon, lle, t_max, d0, tolerance, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="5D time series prediction: analogues, local linear maps, VAR, LLE horizon."
    )
    parser.add_argument("--input",     default="data/vectors_5d.csv")
    parser.add_argument("--horizon",   type=int,   default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--lle",       type=float, default=0.003132,
                        help="LLE from phase space analysis (default from events_indices.csv run).")
    parser.add_argument("--output",    default="output")
    args = parser.parse_args()

    run_pipeline(args.input, args.horizon, args.test_size, args.lle, args.output)