"""Phase space analysis for univariate time series CSV data."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform


@dataclass
class AnalysisResults:
    """Container for summary metrics from phase space analysis."""

    tau: int
    embedding_dimension: int
    lle: float
    rqa_metrics: Dict[str, float]


def load_series(csv_path: str) -> np.ndarray:
    """Load a single-column time series CSV as a 1D float array."""
    series = pd.read_csv(csv_path, header=None).iloc[:, 0].astype(float).to_numpy()
    if series.size < 10:
        raise ValueError("Time series is too short for phase space analysis.")
    return series


def average_mutual_information(series: np.ndarray, max_lag: int = 50, bins: int | None = None) -> np.ndarray:
    """Compute AMI values for lags 1..max_lag."""
    if bins is None:
        bins = max(8, int(np.sqrt(series.size)))

    ami_values = []
    for lag in range(1, max_lag + 1):
        x = series[:-lag]
        y = series[lag:]
        joint_hist, _, _ = np.histogram2d(x, y, bins=bins)
        pxy = joint_hist / np.sum(joint_hist)
        px = np.sum(pxy, axis=1, keepdims=True)
        py = np.sum(pxy, axis=0, keepdims=True)

        valid = pxy > 0
        expected = px * py
        ami = np.sum(pxy[valid] * np.log(pxy[valid] / expected[valid]))
        ami_values.append(float(ami))

    return np.asarray(ami_values)


def first_local_minimum(values: np.ndarray) -> int:
    """Return 1-based index of first local minimum; fallback to global minimum."""
    for i in range(1, len(values) - 1):
        if values[i] < values[i - 1] and values[i] < values[i + 1]:
            return i + 1
    return int(np.argmin(values)) + 1


def delay_embedding(series: np.ndarray, m: int, tau: int) -> np.ndarray:
    """Construct delay embedding with dimension m and delay tau."""
    n_vectors = len(series) - (m - 1) * tau
    if n_vectors <= 1:
        raise ValueError("Insufficient data length for requested embedding parameters.")
    return np.column_stack([series[i : i + n_vectors] for i in range(0, m * tau, tau)])


def false_nearest_neighbors(
    series: np.ndarray,
    tau: int,
    max_dim: int = 10,
    rtol: float = 15.0,
    atol: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute FNN percentages for embedding dimensions 1..max_dim."""
    std_series = np.std(series)
    dims: List[int] = []
    fnn_percentages: List[float] = []

    for m in range(1, max_dim + 1):
        if len(series) - m * tau <= 1:
            break

        emb_m = delay_embedding(series, m, tau)
        emb_m1 = delay_embedding(series, m + 1, tau)
        usable = emb_m1.shape[0]
        emb_m = emb_m[:usable]

        tree = cKDTree(emb_m)
        dists, indices = tree.query(emb_m, k=2)

        nearest_dist = dists[:, 1]
        nearest_idx = indices[:, 1]
        extra_dist = np.abs(series[np.arange(usable) + m * tau] - series[nearest_idx + m * tau])

        with np.errstate(divide="ignore", invalid="ignore"):
            criterion_1 = extra_dist / np.maximum(nearest_dist, 1e-12) > rtol
            criterion_2 = np.sqrt(nearest_dist**2 + extra_dist**2) / np.maximum(std_series, 1e-12) > atol
        fnn = np.mean(np.logical_or(criterion_1, criterion_2)) * 100.0

        dims.append(m)
        fnn_percentages.append(float(fnn))

    return np.asarray(dims), np.asarray(fnn_percentages)


def choose_embedding_dimension(dimensions: np.ndarray, fnn_percentages: np.ndarray, threshold: float = 5.0) -> int:
    """Choose first embedding dimension below threshold, else global minimum."""
    below = dimensions[fnn_percentages < threshold]
    if below.size:
        return int(below[0])
    return int(dimensions[np.argmin(fnn_percentages)])


def estimate_mean_period(series: np.ndarray) -> int:
    """Estimate mean period from dominant Fourier frequency."""
    centered = series - np.mean(series)
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    freqs = np.fft.rfftfreq(len(series), d=1.0)
    if spectrum.size <= 1:
        return 1

    dominant_idx = np.argmax(spectrum[1:]) + 1
    dominant_freq = freqs[dominant_idx]
    if dominant_freq <= 0:
        return 1
    period = int(round(1.0 / dominant_freq))
    return max(period, 1)


def rosenstein_lle(embedding: np.ndarray, mean_period: int, max_t: int = 50) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate largest Lyapunov exponent with Rosenstein's method."""
    n_points = embedding.shape[0]
    max_t = min(max_t, n_points // 2)
    if max_t < 5:
        raise ValueError("Not enough embedded points for Lyapunov exponent estimation.")

    dist_matrix = squareform(pdist(embedding))
    idx = np.arange(n_points)
    temporal_mask = np.abs(idx[:, None] - idx[None, :]) <= mean_period
    dist_matrix[temporal_mask] = np.inf

    neighbors = np.argmin(dist_matrix, axis=1)
    valid = np.isfinite(dist_matrix[np.arange(n_points), neighbors])
    base_indices = np.where(valid)[0]
    neighbors = neighbors[valid]

    divergence = []
    t_values = np.arange(max_t)
    for k in t_values:
        mask = (base_indices + k < n_points) & (neighbors + k < n_points)
        if not np.any(mask):
            divergence.append(np.nan)
            continue
        a = embedding[base_indices[mask] + k]
        b = embedding[neighbors[mask] + k]
        d = np.linalg.norm(a - b, axis=1)
        divergence.append(float(np.mean(np.log(np.maximum(d, 1e-12)))))

    divergence = np.asarray(divergence)
    valid_fit = np.isfinite(divergence)
    fit_t = t_values[valid_fit]
    fit_y = divergence[valid_fit]

    fit_start = 1 if fit_t.size > 6 else 0
    fit_end = max(fit_start + 5, min(fit_start + 10, fit_t.size))
    fit_t_window = fit_t[fit_start:fit_end]
    fit_y_window = fit_y[fit_start:fit_end]

    if fit_t_window.size < 2:
        raise ValueError("Unable to fit linear region for Lyapunov exponent.")

    slope, intercept = np.polyfit(fit_t_window, fit_y_window, 1)
    fit_line = slope * fit_t + intercept
    return float(slope), fit_t, fit_y, fit_line


def run_lengths(sequence: np.ndarray, min_length: int = 2) -> List[int]:
    """Return consecutive-ones run lengths in sequence at least min_length."""
    lengths: List[int] = []
    current = 0
    for value in sequence:
        if value:
            current += 1
        elif current:
            if current >= min_length:
                lengths.append(current)
            current = 0
    if current >= min_length:
        lengths.append(current)
    return lengths


def rqa_metrics(recurrence: np.ndarray, min_line: int = 2) -> Dict[str, float]:
    """Compute RR, DET, Lmax, Lmean, ENT, LAM, and TT from recurrence matrix."""
    n = recurrence.shape[0]
    recurrence_points = float(np.sum(recurrence))
    rr = 100.0 * recurrence_points / (n * n)

    diag_lengths: List[int] = []
    for offset in range(-(n - 1), n):
        if offset == 0:
            continue
        diag = np.diag(recurrence, k=offset)
        diag_lengths.extend(run_lengths(diag, min_length=min_line))

    recurrence_without_identity = recurrence_points - n
    diag_points = float(np.sum(diag_lengths))
    det = 100.0 * diag_points / recurrence_without_identity if recurrence_without_identity > 0 else 0.0
    lmax = float(np.max(diag_lengths)) if diag_lengths else 0.0
    lmean = float(np.mean(diag_lengths)) if diag_lengths else 0.0

    if diag_lengths:
        values, counts = np.unique(diag_lengths, return_counts=True)
        probs = counts / counts.sum()
        ent = float(-np.sum(probs * np.log(probs)))
    else:
        ent = 0.0

    vertical_lengths: List[int] = []
    for col in range(n):
        vertical_lengths.extend(run_lengths(recurrence[:, col], min_length=min_line))

    vertical_points = float(np.sum(vertical_lengths))
    lam = 100.0 * vertical_points / recurrence_without_identity if recurrence_without_identity > 0 else 0.0
    tt = float(np.mean(vertical_lengths)) if vertical_lengths else 0.0

    return {
        "RR": rr,
        "DET": det,
        "Lmax": lmax,
        "Lmean": lmean,
        "ENT": ent,
        "LAM": lam,
        "TT": tt,
    }


def find_poincare_crossings(embedding3d: np.ndarray, threshold: float) -> np.ndarray:
    """Find positive-slope crossings of first coordinate through threshold."""
    x = embedding3d[:, 0]
    y = embedding3d[:, 1]
    z = embedding3d[:, 2]

    crossings: List[Tuple[float, float]] = []
    for i in range(len(x) - 1):
        x0, x1 = x[i], x[i + 1]
        if x0 < threshold <= x1 and (x1 - x0) > 0:
            alpha = (threshold - x0) / (x1 - x0)
            y_cross = y[i] + alpha * (y[i + 1] - y[i])
            z_cross = z[i] + alpha * (z[i + 1] - z[i])
            crossings.append((float(y_cross), float(z_cross)))

    if not crossings:
        return np.empty((0, 2))
    return np.asarray(crossings)


def save_ami_plot(ami_values: np.ndarray, tau: int, output_dir: str) -> None:
    """Save AMI curve plot."""
    lags = np.arange(1, len(ami_values) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(lags, ami_values, marker="o", ms=3)
    plt.axvline(tau, color="red", linestyle="--", label=f"Chosen τ={tau}")
    plt.xlabel("Lag (τ)")
    plt.ylabel("AMI")
    plt.title("Average Mutual Information")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ami.png"), dpi=200)
    plt.close()


def save_fnn_plot(dimensions: np.ndarray, fnn_percentages: np.ndarray, m: int, output_dir: str) -> None:
    """Save FNN percentage plot."""
    plt.figure(figsize=(8, 4))
    plt.plot(dimensions, fnn_percentages, marker="o")
    plt.axvline(m, color="red", linestyle="--", label=f"Chosen m={m}")
    plt.xlabel("Embedding Dimension (m)")
    plt.ylabel("FNN (%)")
    plt.title("False Nearest Neighbors")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fnn.png"), dpi=200)
    plt.close()


def save_phase_portraits(embedding3d: np.ndarray, output_dir: str) -> None:
    """Save 2D and 3D attractor plots."""
    plt.figure(figsize=(6, 6))
    plt.plot(embedding3d[:, 0], embedding3d[:, 1], linewidth=0.8)
    plt.xlabel("x(t)")
    plt.ylabel("x(t+τ)")
    plt.title("2D Attractor")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "attractor_2d.png"), dpi=200)
    plt.close()

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(embedding3d[:, 0], embedding3d[:, 1], embedding3d[:, 2], linewidth=0.8)
    ax.set_xlabel("x(t)")
    ax.set_ylabel("x(t+τ)")
    ax.set_zlabel("x(t+2τ)")
    ax.set_title("3D Attractor")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "attractor_3d.png"), dpi=200)
    plt.close(fig)


def save_poincare_plot(points: np.ndarray, output_dir: str) -> None:
    """Save Poincaré section scatter plot."""
    plt.figure(figsize=(6, 6))
    if points.size:
        plt.scatter(points[:, 0], points[:, 1], s=15)
    plt.xlabel("x(t+τ)")
    plt.ylabel("x(t+2τ)")
    plt.title("Poincaré Section")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "poincare_section.png"), dpi=200)
    plt.close()


def save_lyapunov_plot(t: np.ndarray, divergence: np.ndarray, fit_line: np.ndarray, output_dir: str) -> None:
    """Save Lyapunov divergence and fitted linear trend."""
    plt.figure(figsize=(8, 4))
    plt.plot(t, divergence, "o-", label="Average log divergence")
    plt.plot(t, fit_line, "--", label="Linear fit")
    plt.xlabel("Time steps")
    plt.ylabel("log divergence")
    plt.title("Largest Lyapunov Exponent (Rosenstein)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lyapunov.png"), dpi=200)
    plt.close()


def save_recurrence_plot(recurrence: np.ndarray, output_dir: str) -> None:
    """Save recurrence matrix as image."""
    plt.figure(figsize=(6, 6))
    plt.imshow(recurrence, cmap="binary", origin="lower", interpolation="nearest")
    plt.xlabel("j")
    plt.ylabel("i")
    plt.title("Recurrence Plot")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "recurrence_plot.png"), dpi=200)
    plt.close()


def run_analysis(csv_path: str, output_dir: str = "output") -> AnalysisResults:
    """Run full phase space analysis pipeline and save all required outputs."""
    os.makedirs(output_dir, exist_ok=True)

    series = load_series(csv_path)

    ami_values = average_mutual_information(series, max_lag=50)
    tau = first_local_minimum(ami_values)
    save_ami_plot(ami_values, tau, output_dir)

    dimensions, fnn_percentages = false_nearest_neighbors(series, tau=tau, max_dim=10, rtol=15.0, atol=2.0)
    m = choose_embedding_dimension(dimensions, fnn_percentages, threshold=5.0)
    save_fnn_plot(dimensions, fnn_percentages, m, output_dir)

    if m < 3:
        print("Note: using m=3 for 3D/Poincaré plotting while keeping estimated m for metrics.")
    embedding3d = delay_embedding(series, m=max(3, m), tau=tau)[:, :3]
    save_phase_portraits(embedding3d, output_dir)

    mean_value = float(np.mean(series))
    poincare_points = find_poincare_crossings(embedding3d, threshold=mean_value)
    save_poincare_plot(poincare_points, output_dir)

    embedding_m = delay_embedding(series, m=m, tau=tau)
    mean_period = estimate_mean_period(series)
    lle, t, divergence, fit_line = rosenstein_lle(embedding_m, mean_period=mean_period, max_t=50)
    save_lyapunov_plot(t, divergence, fit_line, output_dir)

    distances = squareform(pdist(embedding_m))
    epsilon = 0.1 * np.max(distances)  # Required threshold: 10% of maximum pairwise distance.
    recurrence = (distances <= epsilon).astype(int)
    save_recurrence_plot(recurrence, output_dir)

    metrics = rqa_metrics(recurrence, min_line=2)

    return AnalysisResults(tau=tau, embedding_dimension=m, lle=lle, rqa_metrics=metrics)


def print_summary(results: AnalysisResults) -> None:
    """Print final summary of selected parameters and nonlinear metrics."""
    print("\nPhase Space Analysis Summary")
    print("-" * 32)
    print(f"Optimal time delay (τ): {results.tau}")
    print(f"Optimal embedding dimension (m): {results.embedding_dimension}")
    print(f"Largest Lyapunov Exponent (LLE): {results.lle:.6f}")
    print("RQA Metrics:")
    for key in ["RR", "DET", "Lmax", "Lmean", "ENT", "LAM", "TT"]:
        print(f"  {key}: {results.rqa_metrics[key]:.6f}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run phase space analysis on a univariate CSV time series.")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="data/diff_sums.csv",
        help="Path to one-column CSV file (default: data/diff_sums.csv).",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for saving generated plots (default: output).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for full pipeline execution."""
    args = parse_args()
    results = run_analysis(args.csv_path, output_dir=args.output_dir)
    print_summary(results)


if __name__ == "__main__":
    main()
