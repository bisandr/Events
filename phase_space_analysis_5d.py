"""Phase space analysis for multivariate 5D vector time series CSV data."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

DEFAULT_TEMPORAL_WINDOW = 10
DEFAULT_LLE_STEPS = 20
DEFAULT_LLE_FIT_RANGE = (1, 10)
DEFAULT_RECURRENCE_PERCENTILE = 10
DEFAULT_RQA_MIN_LINE_LENGTH = 2
LLE_CHAOTIC_THRESHOLD = 0.01
DET_PERIODIC_THRESHOLD = 0.9


@dataclass
class AnalysisResults5D:
    """Container for summary metrics from 5D phase space analysis."""

    dataset: str
    n_time_steps: int
    explained_variance_ratio: np.ndarray
    lle: float
    lle_interpretation: str
    rqa_metrics: Dict[str, float]
    verdict: str


def load_and_normalize(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a 5-column CSV and z-score normalize each dimension."""
    data = pd.read_csv(csv_path, header=None).astype(float).to_numpy()
    if data.ndim != 2 or data.shape[1] != 5:
        raise ValueError(f"Expected CSV with shape (N, 5), got {data.shape}.")
    if data.shape[0] < 25:
        raise ValueError("Need at least 25 time steps for the requested analyses.")

    raw_mean = data.mean(axis=0)
    raw_std = data.std(axis=0, ddof=1)
    safe_std = np.where(raw_std == 0.0, 1.0, raw_std)
    normalized = (data - raw_mean) / safe_std

    print("\n--- Data Loading and Preprocessing ---")
    print(f"N = {data.shape[0]}")
    print("Raw data statistics:")
    for dim, (mean_value, std_value) in enumerate(zip(raw_mean, raw_std), start=1):
        print(f"  Dim {dim}: mean={mean_value:.6f}, std={std_value:.6f}")

    norm_mean = normalized.mean(axis=0)
    norm_std = normalized.std(axis=0, ddof=1)
    print("Normalized statistics:")
    for dim, (mean_value, std_value) in enumerate(zip(norm_mean, norm_std), start=1):
        print(f"  Dim {dim}: mean={mean_value:.6f}, std={std_value:.6f}")

    return normalized, raw_mean, raw_std


def compute_pca(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute PCA in 5D using NumPy SVD."""
    centered = data - data.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    explained_variance_ratio = singular_values**2 / np.sum(singular_values**2)
    projected = centered @ vt.T

    print("\n--- PCA Explained Variance Ratio ---")
    for index, ratio in enumerate(explained_variance_ratio, start=1):
        print(f"  PC{index}: {ratio:.4%}")

    return projected, vt, explained_variance_ratio


def save_pca_attractor_2d(projected: np.ndarray, output_dir: str) -> None:
    """Save the 2D PCA attractor plot."""
    time_index = np.arange(projected.shape[0])
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(projected[:, 0], projected[:, 1], c=time_index, cmap="viridis", s=16, linewidths=0)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("5D Attractor in PCA Space (2D)")
    fig.colorbar(scatter, ax=ax, label="Time index")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "attractor_pca_2d.png"), dpi=200)
    plt.close(fig)


def save_pca_attractor_3d(projected: np.ndarray, output_dir: str) -> None:
    """Save the 3D PCA attractor plot."""
    time_index = np.arange(projected.shape[0])
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(projected[:, 0], projected[:, 1], projected[:, 2], color="lightgray", linewidth=0.8, alpha=0.8)
    scatter = ax.scatter(
        projected[:, 0],
        projected[:, 1],
        projected[:, 2],
        c=time_index,
        cmap="viridis",
        s=10,
        linewidths=0,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("5D Attractor in PCA Space (3D)")
    fig.colorbar(scatter, ax=ax, pad=0.1, label="Time index")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "attractor_pca_3d.png"), dpi=200)
    plt.close(fig)


def find_poincare_crossings(projected: np.ndarray) -> Tuple[np.ndarray, float]:
    """Find positive-slope crossings of the PC1 median hyperplane."""
    pc1 = projected[:, 0]
    threshold = float(np.percentile(pc1, 50))

    crossings: List[Tuple[float, float]] = []
    for index in range(len(pc1) - 1):
        start_value = pc1[index]
        end_value = pc1[index + 1]
        if start_value < threshold <= end_value:
            fraction = (threshold - start_value) / (end_value - start_value)
            point = projected[index] + fraction * (projected[index + 1] - projected[index])
            crossings.append((float(point[1]), float(point[2])))

    if not crossings:
        return np.empty((0, 2)), threshold
    return np.asarray(crossings), threshold


def save_poincare_section(points: np.ndarray, threshold: float, output_dir: str) -> None:
    """Save the Poincaré section plot in PCA space."""
    fig, ax = plt.subplots(figsize=(6, 6))
    if points.size:
        ax.scatter(points[:, 0], points[:, 1], color="steelblue", s=20)
    ax.set_xlabel("PC2")
    ax.set_ylabel("PC3")
    ax.set_title(f"Poincaré Section (PC1 = median = {threshold:.4f})")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "poincare_section_5d.png"), dpi=200)
    plt.close(fig)


def rosenstein_lle(
    data: np.ndarray,
    temporal_window: int = DEFAULT_TEMPORAL_WINDOW,
    n_steps: int = DEFAULT_LLE_STEPS,
    fit_range: Tuple[int, int] = DEFAULT_LLE_FIT_RANGE,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate the largest Lyapunov exponent directly in 5D state space."""
    n_points = data.shape[0]
    max_start = n_points - n_steps
    if max_start <= temporal_window + 1:
        raise ValueError("Not enough samples for LLE estimation with the configured temporal window and steps.")

    truncated = data[:max_start]
    distance_matrix = squareform(pdist(truncated, metric="euclidean"))
    indices = np.arange(max_start)
    temporal_mask = np.abs(indices[:, None] - indices[None, :]) <= temporal_window
    # Mark temporally adjacent candidates as unusable before nearest-neighbor selection.
    distance_matrix[temporal_mask] = np.inf

    neighbors = np.argmin(distance_matrix, axis=1)
    valid_rows = np.isfinite(distance_matrix[np.arange(max_start), neighbors])
    base_indices = np.where(valid_rows)[0]
    neighbor_indices = neighbors[valid_rows]
    if base_indices.size < 2:
        raise ValueError("Unable to find enough valid nearest neighbors for LLE estimation.")

    steps = np.arange(1, n_steps + 1)
    mean_log_divergence = np.full(n_steps, np.nan)
    for offset in steps:
        separations = np.linalg.norm(data[base_indices + offset] - data[neighbor_indices + offset], axis=1)
        valid_separations = separations[separations > 0]
        if valid_separations.size:
            mean_log_divergence[offset - 1] = float(np.mean(np.log(valid_separations)))

    fit_start, fit_end = fit_range
    fit_mask = np.isfinite(mean_log_divergence[fit_start - 1 : fit_end])
    fit_steps = steps[fit_start - 1 : fit_end][fit_mask]
    fit_values = mean_log_divergence[fit_start - 1 : fit_end][fit_mask]
    if fit_steps.size < 2:
        raise ValueError("Not enough valid divergence points in the requested linear fit region.")

    slope, intercept = np.polyfit(fit_steps, fit_values, 1)
    fit_line = slope * steps + intercept
    return float(slope), steps, mean_log_divergence, fit_line


def save_lyapunov_plot(
    steps: np.ndarray,
    mean_log_divergence: np.ndarray,
    fit_line: np.ndarray,
    lle: float,
    output_dir: str,
) -> None:
    """Save the mean log divergence curve and linear fit."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, mean_log_divergence, "o-", label="Average log divergence", color="steelblue")
    ax.plot(steps, fit_line, "--", label=f"Linear fit (LLE={lle:.4f})", color="crimson")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Mean log divergence")
    ax.set_title("Largest Lyapunov Exponent (Rosenstein, 5D)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "lyapunov_5d.png"), dpi=200)
    plt.close(fig)


def interpret_lle(lle: float) -> str:
    """Interpret the largest Lyapunov exponent using the requested thresholds."""
    if lle > LLE_CHAOTIC_THRESHOLD:
        return "System is likely CHAOTIC"
    if lle < -LLE_CHAOTIC_THRESHOLD:
        return "System is NOT chaotic (stable/periodic)"
    return "System is at the edge of chaos (periodic or quasi-periodic)"


def build_recurrence_matrix(
    data: np.ndarray,
    percentile: float = DEFAULT_RECURRENCE_PERCENTILE,
) -> Tuple[np.ndarray, float]:
    """Build a recurrence matrix in the original 5D space.

    The default threshold is the 10th percentile of all pairwise Euclidean
    distances, matching the requested analysis recipe for this repository.
    """
    pairwise_distances = pdist(data, metric="euclidean")
    epsilon = float(np.percentile(pairwise_distances, percentile))
    recurrence = (squareform(pairwise_distances) <= epsilon).astype(int)
    np.fill_diagonal(recurrence, 1)
    return recurrence, epsilon


def save_recurrence_plot(recurrence: np.ndarray, output_dir: str) -> None:
    """Save the recurrence plot with black recurrent points."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(recurrence, cmap="gray_r", origin="lower", interpolation="nearest", aspect="auto")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Time step")
    ax.set_title("Recurrence Plot (5D)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "recurrence_plot_5d.png"), dpi=200)
    plt.close(fig)


def run_lengths(sequence: Iterable[int], min_length: int = 2) -> List[int]:
    """Return consecutive-one run lengths with minimum accepted length."""
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


def compute_rqa_metrics(
    recurrence: np.ndarray,
    min_line_length: int = DEFAULT_RQA_MIN_LINE_LENGTH,
) -> Dict[str, float]:
    """Compute RR, DET, Lmax, Lmean, ENT, LAM, and TT from the recurrence matrix."""
    n_points = recurrence.shape[0]
    off_diagonal_mask = ~np.eye(n_points, dtype=bool)
    off_diagonal_points = recurrence[off_diagonal_mask]
    rr = float(np.mean(off_diagonal_points)) if off_diagonal_points.size else 0.0

    diagonal_lengths: List[int] = []
    for offset in range(-(n_points - 1), n_points):
        if offset == 0:
            continue
        diagonal_lengths.extend(run_lengths(np.diag(recurrence, k=offset), min_length=min_line_length))

    recurrent_points_off_diagonal = float(np.sum(off_diagonal_points))
    diagonal_points = float(np.sum(diagonal_lengths))
    det = diagonal_points / recurrent_points_off_diagonal if recurrent_points_off_diagonal else 0.0
    lmax = float(np.max(diagonal_lengths)) if diagonal_lengths else 0.0
    lmean = float(np.mean(diagonal_lengths)) if diagonal_lengths else 0.0

    if diagonal_lengths:
        _, counts = np.unique(diagonal_lengths, return_counts=True)
        probabilities = counts / counts.sum()
        ent = max(float(-np.sum(probabilities * np.log(probabilities))), 0.0)
    else:
        ent = 0.0

    vertical_lengths: List[int] = []
    for column in range(n_points):
        vertical_lengths.extend(run_lengths(recurrence[:, column], min_length=min_line_length))
    vertical_points = float(np.sum(vertical_lengths))
    lam = vertical_points / recurrent_points_off_diagonal if recurrent_points_off_diagonal else 0.0
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


def synthesize_verdict(lle: float, rqa_metrics: Dict[str, float]) -> str:
    """Synthesize LLE and RQA into a final chaos verdict."""
    det = rqa_metrics["DET"]
    if lle > LLE_CHAOTIC_THRESHOLD and det < DET_PERIODIC_THRESHOLD:
        return "Chaotic: positive LLE and lower DET both support sensitive, irregular dynamics."
    if lle > LLE_CHAOTIC_THRESHOLD:
        return "Likely chaotic: the positive LLE indicates exponential divergence, while high DET shows the dynamics remain strongly deterministic."
    if lle < -LLE_CHAOTIC_THRESHOLD and det > DET_PERIODIC_THRESHOLD:
        return "Not chaotic: negative LLE and high DET are consistent with stable or periodic dynamics."
    if lle < -LLE_CHAOTIC_THRESHOLD:
        return "Likely not chaotic: the negative LLE does not support chaos, though recurrence structure is less clearly periodic."
    return "Edge of chaos or mixed evidence: the LLE is near zero, so longer or cleaner data may be needed for a stronger conclusion."


def print_rqa_metrics(rqa_metrics: Dict[str, float]) -> None:
    """Print RQA metrics and requested DET interpretation."""
    print("\n--- Recurrence Quantification Analysis ---")
    for key in ["RR", "DET", "Lmax", "Lmean", "ENT", "LAM", "TT"]:
        print(f"  {key} = {rqa_metrics[key]:.6f}")
    if rqa_metrics["DET"] > DET_PERIODIC_THRESHOLD:
        print("  DET interpretation: DET > 0.9 suggests deterministic/periodic dynamics.")
    else:
        print("  DET interpretation: lower DET suggests chaos or noise.")


def print_summary(results: AnalysisResults5D) -> None:
    """Print the requested structured chaos report."""
    explained = results.explained_variance_ratio
    rqa_metrics = results.rqa_metrics

    print("\n============================================================")
    print("           PHASE SPACE ANALYSIS - CHAOS REPORT")
    print("============================================================")
    print(f"Dataset         : {results.dataset}")
    print(f"N time steps    : {results.n_time_steps}")
    print("Dimensions      : 5")
    print("")
    print("--- Embedding ---")
    print(
        "PCA explained variance: "
        f"PC1={explained[0] * 100:.2f}%, PC2={explained[1] * 100:.2f}%, PC3={explained[2] * 100:.2f}%"
    )
    print("")
    print("--- Largest Lyapunov Exponent ---")
    print(f"LLE = {results.lle:.6f}")
    print(f"Interpretation  : {results.lle_interpretation}")
    print("")
    print("--- Recurrence Quantification Analysis ---")
    print(f"RR    = {rqa_metrics['RR']:.6f}")
    print(f"DET   = {rqa_metrics['DET']:.6f}")
    print(f"Lmax  = {rqa_metrics['Lmax']:.6f}")
    print(f"Lmean = {rqa_metrics['Lmean']:.6f}")
    print(f"ENT   = {rqa_metrics['ENT']:.6f}")
    print(f"LAM   = {rqa_metrics['LAM']:.6f}")
    print(f"TT    = {rqa_metrics['TT']:.6f}")
    print("")
    print("--- Overall Verdict ---")
    print(results.verdict)
    print("============================================================")


def run_analysis(csv_path: str, output_dir: str = "output") -> AnalysisResults5D:
    """Run the full 5D phase space analysis pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    data, _, _ = load_and_normalize(csv_path)
    projected, _, explained_variance_ratio = compute_pca(data)
    save_pca_attractor_2d(projected, output_dir)
    save_pca_attractor_3d(projected, output_dir)

    poincare_points, threshold = find_poincare_crossings(projected)
    print(f"\nPoincaré crossings found: {len(poincare_points)}")
    save_poincare_section(poincare_points, threshold, output_dir)

    lle, steps, mean_log_divergence, fit_line = rosenstein_lle(
        data,
        temporal_window=DEFAULT_TEMPORAL_WINDOW,
        n_steps=DEFAULT_LLE_STEPS,
        fit_range=DEFAULT_LLE_FIT_RANGE,
    )
    lle_interpretation = interpret_lle(lle)
    print(f"\nLargest Lyapunov Exponent (LLE) = {lle:.6f}")
    print(f"Interpretation: {lle_interpretation}")
    save_lyapunov_plot(steps, mean_log_divergence, fit_line, lle, output_dir)

    recurrence, epsilon = build_recurrence_matrix(data, percentile=DEFAULT_RECURRENCE_PERCENTILE)
    print(f"\nRecurrence threshold epsilon (10th percentile) = {epsilon:.6f}")
    save_recurrence_plot(recurrence, output_dir)
    rqa_metrics = compute_rqa_metrics(recurrence, min_line_length=DEFAULT_RQA_MIN_LINE_LENGTH)
    print_rqa_metrics(rqa_metrics)

    verdict = synthesize_verdict(lle, rqa_metrics)
    results = AnalysisResults5D(
        dataset=os.path.basename(csv_path),
        n_time_steps=data.shape[0],
        explained_variance_ratio=explained_variance_ratio,
        lle=lle,
        lle_interpretation=lle_interpretation,
        rqa_metrics=rqa_metrics,
        verdict=verdict,
    )
    print_summary(results)
    return results


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run multivariate phase space analysis on 5D vector CSV data.")
    parser.add_argument(
        "--input",
        default="data/vectors_5d.csv",
        help="Path to a CSV with shape (N, 5) and no header (default: data/vectors_5d.csv).",
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Directory for saved plots (default: output).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for full 5D analysis."""
    args = parse_args()
    run_analysis(args.input, output_dir=args.output)


if __name__ == "__main__":
    main()
