"""Minibatch dictionary learning with nonnegative codes and l1 regularization.

This is a minibatch version of ``solve_dictionary_learning_l1.py``.

The model is

    min_{W, R >= 0} ||X - W R||_F^2 / n_samples
                  + lambda1 * ||R||_1 / n_samples
                  + lambda2 * ||W||_F^2.

Conventions
-----------
Internally, X has shape (input_dim, n_samples), W has shape
(input_dim, n_atoms), and R has shape (n_atoms, n_samples), matching the
original script.

For an arbitrary data matrix stored on disk, the default assumption is that
rows are datapoints. The script transposes it internally. Use
``--data-orientation columns`` if your file already uses columns as samples.

Examples
--------
Original toy data, minibatch training:

    python solve_dictionary_learning_l1_minibatch_fixed.py --a 10 \
        --n-atoms 20 --epochs 100 --batch-size 16 --steps-per-epoch 20

Arbitrary data matrix with one datapoint per row:

    python solve_dictionary_learning_l1_minibatch_fixed.py --data-path X.npy \
        --data-orientation rows --n-atoms 64 --epochs 100 --batch-size 256

Optional new input inference:

    python solve_dictionary_learning_l1_minibatch_fixed.py --new-input 1 0 0 0 0 0
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# Data loading / toy data
# -----------------------------------------------------------------------------

def build_samples(a: float) -> np.ndarray:
    """Return the 6 x 4 toy training data matrix X = [x1, x2, x3, x4]."""
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [a, a, 0.0, 0.0],
            [0.0, 0.0, a, a],
        ],
        dtype=float,
    )


def build_reference_input(a: float) -> np.ndarray:
    """Reference vector used in the cosine-similarity comparison for the toy data."""
    return np.array([1.0, 0.0, 0.0, 0.0, a, 0.0], dtype=float)


def load_data_matrix(path: str, orientation: str = "rows") -> np.ndarray:
    """Load an arbitrary data matrix and return it as X with columns as samples.

    Parameters
    ----------
    path:
        .npy, .npz, .csv, or .txt file.
    orientation:
        ``rows`` means the loaded array has shape (n_samples, input_dim).
        ``columns`` means the loaded array already has shape (input_dim, n_samples).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Cannot find data file: {path}")

    if p.suffix.lower() == ".npy":
        data = np.load(p)
    elif p.suffix.lower() == ".npz":
        payload = np.load(p)
        if "X" in payload:
            data = payload["X"]
        else:
            first_key = list(payload.keys())[0]
            data = payload[first_key]
    elif p.suffix.lower() in {".csv", ".txt"}:
        data = np.loadtxt(p, delimiter="," if p.suffix.lower() == ".csv" else None)
    else:
        raise ValueError("data_path must be a .npy, .npz, .csv, or .txt file")

    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"Data matrix must be 2D, got shape {data.shape}")

    if orientation == "rows":
        return data.T.copy()
    if orientation == "columns":
        return data.copy()
    raise ValueError("orientation must be either 'rows' or 'columns'")


# -----------------------------------------------------------------------------
# Core objective and updates
# -----------------------------------------------------------------------------

def objective_normalized(X: np.ndarray, W: np.ndarray, R: np.ndarray, lambda1: float, lambda2: float) -> float:
    """Normalized objective used by the minibatch solver."""
    n_samples = X.shape[1]
    residual = X - W @ R
    return float(np.sum(residual ** 2) / n_samples + lambda1 * np.sum(R) / n_samples + lambda2 * np.sum(W ** 2))


def objective_unnormalized(X: np.ndarray, W: np.ndarray, R: np.ndarray, lambda1: float, lambda2: float) -> float:
    """Original unnormalized objective, kept for comparison with the old script."""
    residual = X - W @ R
    return float(np.sum(residual ** 2) + lambda1 * np.sum(R) + lambda2 * np.sum(W ** 2))


def update_dictionary_full_batch(X: np.ndarray, R: np.ndarray, lambda2: float, normalized: bool = False) -> np.ndarray:
    """Closed-form full-batch update for W once all codes R are fixed."""
    n_atoms = R.shape[0]
    if normalized:
        A = R @ R.T / X.shape[1]
        C = X @ R.T / X.shape[1]
        gram = A + lambda2 * np.eye(n_atoms)
        return np.linalg.solve(gram, C.T).T
    gram = R @ R.T + lambda2 * np.eye(n_atoms)
    rhs = X @ R.T
    return np.linalg.solve(gram, rhs.T).T


def update_dictionary_from_stats(A: np.ndarray, C: np.ndarray, lambda2: float) -> np.ndarray:
    """Update W from online statistics.

    A approximates E[r r^T], shape (H, H).
    C approximates E[x r^T], shape (d, H).

    The ridge dictionary update is

        W = C (A + lambda2 I)^{-1}.
    """
    n_atoms = A.shape[0]
    gram = A + lambda2 * np.eye(n_atoms)
    return np.linalg.solve(gram, C.T).T


# -----------------------------------------------------------------------------
# Nonnegative sparse coding by FISTA / proximal gradient
# -----------------------------------------------------------------------------

def solve_code_l1(W: np.ndarray,
    x: np.ndarray,
    lambda1: float,
    warm_start: np.ndarray | None = None,
    max_iter: int = 500,
    tol: float = 1e-9) -> np.ndarray:
    """Infer one nonnegative sparse code r for a single input x.

    Solves

        min_{r >= 0} ||x - W r||_2^2 + lambda1 ||r||_1.
    """
    r = np.zeros(W.shape[1]) if warm_start is None else warm_start.copy()
    y = r.copy()
    t = 1.0
    lipschitz = max(2.0 * np.linalg.norm(W, ord=2) ** 2, 1e-12)
    step = 1.0 / lipschitz
    previous_value: float | None = None

    for _ in range(max_iter):
        gradient = 2.0 * W.T @ (W @ y - x)
        r_next = np.maximum(y - step * gradient - step * lambda1, 0.0)
        value = np.sum((x - W @ r_next) ** 2) + lambda1 * np.sum(r_next)

        if previous_value is not None:
            delta = abs(value - previous_value)
            if delta <= tol * max(1.0, abs(previous_value)):
                return r_next

        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = r_next + ((t - 1.0) / t_next) * (r_next - r)
        r = r_next
        t = t_next
        previous_value = value

    return r


def solve_codes_l1(W: np.ndarray,
    X: np.ndarray,
    lambda1: float,
    warm_start: np.ndarray | None = None,
    inner_max_iter: int = 500,
    inner_tol: float = 1e-9) -> np.ndarray:
    """Infer one sparse code per column of X using the single-vector solver."""
    codes = []
    for k in range(X.shape[1]):
        start = None if warm_start is None else warm_start[:, k]
        code = solve_code_l1(
            W=W,
            x=X[:, k],
            lambda1=lambda1,
            warm_start=start,
            max_iter=inner_max_iter,
            tol=inner_tol,
        )
        codes.append(code)
    return np.column_stack(codes)


def solve_codes_l1_matrix(W: np.ndarray,
    X: np.ndarray,
    lambda1: float,
    warm_start: np.ndarray | None = None,
    inner_max_iter: int = 200,
    inner_tol: float = 1e-8,
    monotone: bool = True) -> np.ndarray:
    """Infer nonnegative sparse codes for a whole minibatch at once.

    Solves

        min_{R >= 0} ||X - W R||_F^2 + lambda1 ||R||_1.

    This is vectorized FISTA. If ``monotone`` is true, it restarts the momentum
    whenever the objective increases, which improves robustness for minibatches.
    """
    n_atoms = W.shape[1]
    n_batch = X.shape[1]
    if warm_start is None:
        R = np.zeros((n_atoms, n_batch), dtype=float)
    else:
        R = np.asarray(warm_start, dtype=float).copy()
        if R.shape != (n_atoms, n_batch):
            raise ValueError(f"warm_start shape {R.shape} != {(n_atoms, n_batch)}")

    Y = R.copy()
    t = 1.0
    lipschitz = max(2.0 * np.linalg.norm(W, ord=2) ** 2, 1e-12)
    step = 1.0 / lipschitz

    def value(Z: np.ndarray) -> float:
        return float(np.sum((X - W @ Z) ** 2) + lambda1 * np.sum(Z))

    previous_value = value(R)
    for _ in range(inner_max_iter):
        grad = 2.0 * W.T @ (W @ Y - X)
        R_next = np.maximum(Y - step * grad - step * lambda1, 0.0)
        next_value = value(R_next)

        if monotone and next_value > previous_value:
            # Restart without momentum.
            Y = R.copy()
            t = 1.0
            grad = 2.0 * W.T @ (W @ Y - X)
            R_next = np.maximum(Y - step * grad - step * lambda1, 0.0)
            next_value = value(R_next)

        if abs(previous_value - next_value) <= inner_tol * max(1.0, abs(previous_value)):
            return R_next

        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        Y = R_next + ((t - 1.0) / t_next) * (R_next - R)
        R = R_next
        t = t_next
        previous_value = next_value

    return R


def infer_pattern_l1(
    W: np.ndarray,
    x_new: np.ndarray,
    lambda1: float,
    inner_max_iter: int = 500,
    inner_tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Infer the neural pattern r_new and reconstructed input W @ r_new."""
    x_new = np.asarray(x_new, dtype=float).reshape(-1)
    if x_new.size != W.shape[0]:
        raise ValueError(f"x_new must have length {W.shape[0]}, got {x_new.size}.")
    r_new = solve_code_l1(W, x_new, lambda1, max_iter=inner_max_iter, tol=inner_tol)
    return r_new, W @ r_new


# -----------------------------------------------------------------------------
# Minibatch dictionary learning
# -----------------------------------------------------------------------------

def initialize_dictionary(X: np.ndarray, n_atoms: int, rng: np.random.Generator, mode: str = "data") -> np.ndarray:
    """Initialize W with random data columns or Gaussian noise."""
    d, n_samples = X.shape
    if mode == "data" and n_samples > 0:
        idx = rng.integers(0, n_samples, size=n_atoms)
        W = X[:, idx].copy()
        # Add small noise to break duplicated atoms.
        scale = max(np.std(X), 1e-3)
        W += 0.01 * scale * rng.standard_normal(size=W.shape)
        return W
    if mode == "random":
        scale = max(np.std(X), 1.0) / np.sqrt(max(n_atoms, 1))
        return scale * rng.standard_normal((d, n_atoms))
    raise ValueError("init mode must be 'data' or 'random'")


def solve_dictionary_learning_l1_minibatch(X: np.ndarray,
    n_atoms: int,
    lambda1: float,
    lambda2: float,
    epochs: int = 50,
    steps_per_epoch: int | None = None,
    batch_size: int = 128,
    stats_lr: float = 0.05,
    seed: int = 0,
    restarts: int = 3,
    inner_max_iter: int = 100,
    inner_tol: float = 1e-8,
    init: str = "data",
    batch_noise_std: float = 0.0,
    eval_every: int = 10,
    eval_code_max_iter: int = 300,
    final_code_max_iter: int = 500) -> tuple[np.ndarray, np.ndarray, list[float], int]:
    """Minibatch / online dictionary learning.

    The algorithm samples minibatches of columns of X. For each minibatch it
    re-infers local nonnegative sparse codes, updates online sufficient
    statistics A ~= E[rr^T], C ~= E[xr^T], and performs the ridge dictionary
    update W = C(A + lambda2 I)^{-1}.

    Returns
    -------
    best_W : ndarray, shape (d, n_atoms)
    best_R_final : ndarray, shape (n_atoms, n_samples)
        Codes inferred for all data points using the final dictionary.
    best_history : list[float]
        Approximate normalized objective evaluated every eval_every epochs.
    best_restart : int
    """
    d, n_samples = X.shape
    if n_samples == 0:
        raise ValueError("X has zero samples.")
    if steps_per_epoch is None:
        steps_per_epoch = max(1, int(np.ceil(n_samples / batch_size)))

    best_W = None
    best_R_final = None
    best_history: list[float] | None = None
    best_restart = -1
    best_value = np.inf

    for restart in range(restarts):
        rng = np.random.default_rng(seed + 1009 * restart)
        W = initialize_dictionary(X, n_atoms, rng, mode=init)

        # Small positive diagonal avoids singular early dictionary updates.
        A = 1e-8 * np.eye(n_atoms)
        C = np.zeros((d, n_atoms), dtype=float)
        history: list[float] = []

        total_steps = epochs * steps_per_epoch
        for step_idx in range(total_steps):
            batch_idx = rng.integers(0, n_samples, size=batch_size)
            X_batch_clean = X[:, batch_idx]
            if batch_noise_std > 0:
                X_batch = X_batch_clean + batch_noise_std * rng.standard_normal(size=X_batch_clean.shape)
            else:
                X_batch = X_batch_clean

            R_batch = solve_codes_l1_matrix(
                W=W,
                X=X_batch,
                lambda1=lambda1,
                warm_start=None,
                inner_max_iter=inner_max_iter,
                inner_tol=inner_tol,
            )

            # If batch_noise_std is used as data augmentation, update the
            # dictionary to reconstruct the noisy samples. To use denoising,
            # replace X_batch by X_batch_clean here.
            A_batch = (R_batch @ R_batch.T) / X_batch.shape[1]
            C_batch = (X_batch @ R_batch.T) / X_batch.shape[1]

            if step_idx == 0:
                A = A_batch + 1e-8 * np.eye(n_atoms)
                C = C_batch
            else:
                rho = stats_lr
                A = (1.0 - rho) * A + rho * A_batch
                C = (1.0 - rho) * C + rho * C_batch

            W = update_dictionary_from_stats(A, C, lambda2)

            # Optional approximate full-data objective. This is expensive for
            # huge data, so it is done only every eval_every epochs.
            epoch = (step_idx + 1) // steps_per_epoch
            at_epoch_end = (step_idx + 1) % steps_per_epoch == 0
            if at_epoch_end and (eval_every > 0) and (epoch % eval_every == 0 or epoch == epochs):
                R_eval = infer_all_codes_in_batches(
                    W,
                    X,
                    lambda1=lambda1,
                    batch_size=max(batch_size, 256),
                    inner_max_iter=eval_code_max_iter,
                    inner_tol=inner_tol,
                )
                value = objective_normalized(X, W, R_eval, lambda1, lambda2)
                history.append(value)

        R_final = infer_all_codes_in_batches(
            W,
            X,
            lambda1=lambda1,
            batch_size=max(batch_size, 256),
            inner_max_iter=final_code_max_iter,
            inner_tol=inner_tol,
        )
        final_value = objective_normalized(X, W, R_final, lambda1, lambda2)
        if not history or history[-1] != final_value:
            history.append(final_value)

        if final_value < best_value:
            best_W = W.copy()
            best_R_final = R_final.copy()
            best_history = history
            best_restart = restart
            best_value = final_value

    if best_W is None or best_R_final is None or best_history is None:
        raise RuntimeError("Minibatch solver did not produce a solution.")

    return best_W, best_R_final, best_history, best_restart


def infer_all_codes_in_batches(
    W: np.ndarray,
    X: np.ndarray,
    lambda1: float,
    batch_size: int = 512,
    inner_max_iter: int = 500,
    inner_tol: float = 1e-8,
) -> np.ndarray:
    """Infer codes for all columns of X in manageable batches."""
    n_atoms = W.shape[1]
    n_samples = X.shape[1]
    R = np.zeros((n_atoms, n_samples), dtype=float)
    for start in range(0, n_samples, batch_size):
        stop = min(start + batch_size, n_samples)
        R[:, start:stop] = solve_codes_l1_matrix(
            W=W,
            X=X[:, start:stop],
            lambda1=lambda1,
            warm_start=None,
            inner_max_iter=inner_max_iter,
            inner_tol=inner_tol,
        )
    return R


# -----------------------------------------------------------------------------
# Full-batch solver from original script, kept as an option
# -----------------------------------------------------------------------------

def solve_dictionary_learning_l1_fullbatch(
    X: np.ndarray,
    n_atoms: int,
    lambda1: float,
    lambda2: float,
    max_iter: int = 200,
    tol: float = 1e-8,
    seed: int = 0,
    restarts: int = 8,
    inner_max_iter: int = 500,
    inner_tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, list[float], int]:
    """Original full-batch alternating solver."""
    n_samples = X.shape[1]
    best_W = best_R = None
    best_history: list[float] | None = None
    best_restart = -1
    best_value = np.inf

    for restart in range(restarts):
        rng = np.random.default_rng(seed + restart)
        R = np.maximum(rng.random((n_atoms, n_samples)), 1e-8)
        history: list[float] = []

        for _ in range(max_iter):
            W = update_dictionary_full_batch(X, R, lambda2, normalized=False)
            R = solve_codes_l1(
                W=W,
                X=X,
                lambda1=lambda1,
                warm_start=R,
                inner_max_iter=inner_max_iter,
                inner_tol=inner_tol,
            )
            W = update_dictionary_full_batch(X, R, lambda2, normalized=False)
            value = objective_unnormalized(X, W, R, lambda1, lambda2)
            history.append(value)

            if len(history) > 1:
                delta = abs(history[-1] - history[-2])
                if delta <= tol * max(1.0, abs(history[-2])):
                    break

        if history[-1] < best_value:
            best_W = W.copy()
            best_R = R.copy()
            best_history = history
            best_restart = restart
            best_value = history[-1]

    if best_W is None or best_R is None or best_history is None:
        raise RuntimeError("Full-batch solver did not produce a solution.")

    return best_W, best_R, best_history, best_restart


# -----------------------------------------------------------------------------
# Plotting and summaries
# -----------------------------------------------------------------------------

def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)
    if x_norm == 0.0 or y_norm == 0.0:
        raise ValueError("Cosine similarity is undefined for a zero vector.")
    return float(np.dot(x, y) / (x_norm * y_norm))


def plot_cosine_similarity(value: float, path: str, title: str) -> None:
    """Save a simple bar plot of the cosine similarity."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(["cosine"], [value], color="darkorange")
    ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel("Cosine similarity")
    ax.set_title(title)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.text(0, value, f"{value:.4f}", ha="center", va="bottom" if value >= 0 else "top")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def representation_similarity_matrix(R: np.ndarray) -> np.ndarray:
    """Cosine-similarity matrix between optimized code vectors."""
    n_samples = R.shape[1]
    similarity = np.eye(n_samples, dtype=float)

    norms = np.linalg.norm(R, axis=0)
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            if norms[i] == 0.0 or norms[j] == 0.0:
                value = 0.0
            else:
                value = float(np.dot(R[:, i], R[:, j]) / (norms[i] * norms[j]))
            similarity[i, j] = value
            similarity[j, i] = value

    return similarity


def plot_representation_similarity_matrix(matrix: np.ndarray, path: str, max_labels: int = 40) -> None:
    """Save representation similarity matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5.5))
    image = ax.imshow(matrix, cmap="viridis", vmin=-1.0, vmax=1.0)
    n = matrix.shape[0]
    if n <= max_labels:
        ax.set_xticks(range(n), labels=[f"x{k + 1}" for k in range(n)], rotation=90)
        ax.set_yticks(range(n), labels=[f"x{k + 1}" for k in range(n)])
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title("Representation similarity matrix")
    fig.colorbar(image, ax=ax, label="Cosine similarity")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(history: list[float], path: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(np.arange(1, len(history) + 1), history)
    ax.set_xlabel("evaluation index")
    ax.set_ylabel("objective")
    ax.set_title("Training objective")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# CLI helpers
# -----------------------------------------------------------------------------

def parse_new_input(values: list[float] | None, expected_dim: int) -> np.ndarray | None:
    """Turn --new-input values into a NumPy vector and check its size."""
    if values is None:
        return None
    x_new = np.asarray(values, dtype=float)
    if x_new.size != expected_dim:
        raise ValueError(f"--new-input expects exactly {expected_dim} numbers, got {x_new.size}.")
    return x_new


def save_results(
    path: str,
    X: np.ndarray,
    W: np.ndarray,
    R: np.ndarray,
    history: list[float],
    args: argparse.Namespace,
    similarity_matrix: np.ndarray | None = None,
    x_new: np.ndarray | None = None,
    r_new: np.ndarray | None = None,
    cosine_value: float | None = None,
    pattern_cosine_value: float | None = None,
) -> None:
    payload = {
        "X": X,
        "W": W,
        "R": R,
        "reconstruction": W @ R,
        "objective_history": np.array(history),
        "lambda1": args.lambda1,
        "lambda2": args.lambda2,
        "n_atoms": args.n_atoms,
        "method": args.method,
    }
    if hasattr(args, "a"):
        payload["a"] = args.a
    if x_new is not None and r_new is not None:
        payload["x_new"] = x_new
        payload["r_new"] = r_new
        payload["reconstruction_new"] = W @ r_new
    if similarity_matrix is not None:
        payload["representation_similarity_matrix"] = similarity_matrix
    if cosine_value is not None:
        payload["cosine_similarity"] = cosine_value
    if pattern_cosine_value is not None:
        payload["pattern_cosine_similarity"] = pattern_cosine_value
    np.savez(path, **payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # Data.
    parser.add_argument("--data-path", type=str, default="", help="Optional arbitrary data matrix file: .npy, .npz, .csv, .txt.")
    parser.add_argument("--data-orientation", choices=["rows", "columns"], default="rows", help="For --data-path: are samples rows or columns?")
    parser.add_argument("--a", type=float, default=5.0, help="Toy data parameter, used only if --data-path is empty.")

    # Optimization.
    parser.add_argument("--method", choices=["minibatch", "fullbatch"], default="minibatch")
    parser.add_argument("--lambda1", type=float, default=0.01)
    parser.add_argument("--lambda2", type=float, default=0.01)
    parser.add_argument("--n-atoms", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--restarts", type=int, default=3)

    # Full-batch options.
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1e-8)

    # Minibatch options.
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=0, help="0 means ceil(n_samples / batch_size).")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--stats-lr", type=float, default=0.05, help="EMA learning rate for sufficient statistics.")
    parser.add_argument("--batch-noise-std", type=float, default=0.0, help="Optional Gaussian noise added to each sampled minibatch.")
    parser.add_argument("--init", choices=["data", "random"], default="data")
    parser.add_argument("--eval-every", type=int, default=10, help="Evaluate full objective every this many epochs. 0 disables intermediate evaluation.")

    # Sparse-code solver.
    parser.add_argument("--inner-max-iter", type=int, default=200)
    parser.add_argument("--inner-tol", type=float, default=1e-8)
    parser.add_argument("--final-code-max-iter", type=int, default=500)

    # Test input / plots / save.
    parser.add_argument("--new-input",
        type=float,
        nargs="+",
        default=None,
        help="Give a new input x_new. Its length must match input dimension.")
    parser.add_argument("--cosine-plot", type=str, default="cosine_similarity_l1.png")
    parser.add_argument("--representation-similarity-plot", type=str, default="representation_similarity_l1.png")
    parser.add_argument("--history-plot", type=str, default="training_history_l1.png")
    parser.add_argument("--max-similarity-plot-samples", type=int, default=80)
    parser.add_argument("--save", type=str, default="")
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.data_path:
        X = load_data_matrix(args.data_path, orientation=args.data_orientation)
    else:
        X = build_samples(args.a)

    x_new = parse_new_input(args.new_input, X.shape[0])

    if args.method == "fullbatch":
        W, R, history, best_restart = solve_dictionary_learning_l1_fullbatch(
            X=X,
            n_atoms=args.n_atoms,
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            max_iter=args.max_iter,
            tol=args.tol,
            seed=args.seed,
            restarts=args.restarts,
            inner_max_iter=args.inner_max_iter,
            inner_tol=args.inner_tol,
        )
        final_objective = objective_unnormalized(X, W, R, args.lambda1, args.lambda2)
    else:
        steps_per_epoch = None if args.steps_per_epoch == 0 else args.steps_per_epoch
        W, R, history, best_restart = solve_dictionary_learning_l1_minibatch(
            X=X,
            n_atoms=args.n_atoms,
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            batch_size=args.batch_size,
            stats_lr=args.stats_lr,
            seed=args.seed,
            restarts=args.restarts,
            inner_max_iter=args.inner_max_iter,
            inner_tol=args.inner_tol,
            init=args.init,
            batch_noise_std=args.batch_noise_std,
            eval_every=args.eval_every,
            final_code_max_iter=args.final_code_max_iter,
        )
        final_objective = objective_normalized(X, W, R, args.lambda1, args.lambda2)

    np.set_printoptions(precision=6, suppress=True)
    print("X shape:", X.shape, "(input_dim, n_samples)")
    print("Best restart:", best_restart)
    print("Final objective:", final_objective)
    print("W shape:", W.shape)
    print("R shape:", R.shape)

    if X.shape[1] <= 20 and X.shape[0] <= 20:
        print("\nX =")
        print(X)
        print("\nW =")
        print(W)
        print("\nR =")
        print(R)
        print("\nW @ R =")
        print(W @ R)

    plot_history(history, args.history_plot)
    print("\nTraining-history plot saved to:", args.history_plot)

    similarity_matrix: Optional[np.ndarray]
    if X.shape[1] <= args.max_similarity_plot_samples:
        similarity_matrix = representation_similarity_matrix(R)
        plot_representation_similarity_matrix(similarity_matrix, args.representation_similarity_plot)
        print("Representation similarity plot saved to:", args.representation_similarity_plot)
        if X.shape[1] <= 20:
            print("\nRepresentation similarity matrix =")
            print(similarity_matrix)
    else:
        similarity_matrix = None
        print("Skipping representation similarity plot because n_samples is large.")

    r_new = None
    cosine_value = None
    pattern_cosine_value = None
    if x_new is not None:
        r_new, x_hat = infer_pattern_l1(
            W=W,
            x_new=x_new,
            lambda1=args.lambda1,
            inner_max_iter=args.final_code_max_iter,
            inner_tol=args.inner_tol,
        )
        print("\nNew input x_new =")
        print(x_new)
        print("\nNeural pattern r_new =")
        print(r_new)
        print("\nReconstructed input W @ r_new =")
        print(x_hat)

        if X.shape[0] == 6 and not args.data_path:
            reference = build_reference_input(args.a)
            cosine_value = cosine_similarity(x_hat, reference)
            old_pattern = R[:, 0]
            if np.linalg.norm(r_new) == 0.0 or np.linalg.norm(old_pattern) == 0.0:
                pattern_cosine_value = np.nan
            else:
                pattern_cosine_value = cosine_similarity(r_new, old_pattern)
            plot_cosine_similarity(cosine_value, args.cosine_plot, "L1 cosine similarity")
            print("\nReference input [1, 0, 0, 0, a, 0] =")
            print(reference)
            print("\nCosine similarity =", cosine_value)
            print("Cosine similarity plot saved to:", args.cosine_plot)

    if args.save:
        save_results(
            args.save,
            X,
            W,
            R,
            history,
            args,
            similarity_matrix=similarity_matrix,
            x_new=x_new,
            r_new=r_new,
            cosine_value=cosine_value,
            pattern_cosine_value=pattern_cosine_value,
        )
        print("Saved NPZ results to:", args.save)


if __name__ == "__main__":
    main()
