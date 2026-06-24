"""Toy dictionary learning with nonnegative codes and l1 regularization.

Where to give a new input:
    Pass it on the command line with --new-input.
    Example:
        py -3 solve_dictionary_learning_l1.py --new-input 1 0 0 0 2 0

The six numbers above are the entries of your new vector x_new.
"""

from __future__ import annotations

import argparse
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt


def build_samples(a: float) -> np.ndarray:
    """Return the 6 x 4 training data matrix X = [x1, x2, x3, x4]."""
    return np.array(
        [[1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [a, a, 0.0, 0.0],
            [0.0, 0.0, a, a]],dtype=float)


def build_reference_input(a: float) -> np.ndarray:
    """Reference vector used in the cosine-similarity comparison."""
    return np.array([1.0, 0.0, 0.0, 0.0, a, 0.0], dtype=float)


def objective(X: np.ndarray, W: np.ndarray, R: np.ndarray, lambda1: float, lambda2: float) -> float:
    residual = X - W @ R
    return float(np.sum(residual**2) + lambda1 * np.sum(R) + lambda2 * np.sum(W**2))


def update_dictionary(X: np.ndarray, R: np.ndarray, lambda2: float) -> np.ndarray:
    """Closed-form update for W once the codes R are fixed."""
    n_atoms = R.shape[0]
    gram = R @ R.T + lambda2 * np.eye(n_atoms)
    rhs = X @ R.T
    return np.linalg.solve(gram, rhs.T).T


def solve_code_l1(W: np.ndarray,
    x: np.ndarray,
    lambda1: float,
    warm_start: np.ndarray | None = None,
    max_iter: int = 500,
    tol: float = 1e-9) -> np.ndarray:
    """Infer one nonnegative sparse code r for a single input x."""
    r = np.zeros(W.shape[1]) if warm_start is None else warm_start.copy()
    y = r.copy()
    t = 1.0
    lipschitz = max(2.0 * np.linalg.norm(W, ord=2) ** 2, 1e-12)
    step = 1.0 / lipschitz
    previous_value: float | None = None

    for _ in range(max_iter):
        # Gradient step for the quadratic term, then soft-threshold and clip at 0.
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


def solve_codes_l1(
    W: np.ndarray,
    X: np.ndarray,
    lambda1: float,
    warm_start: np.ndarray | None = None,
    inner_max_iter: int = 500,
    inner_tol: float = 1e-9,
) -> np.ndarray:
    """Infer one sparse code per column of X."""
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
    """Cosine-similarity matrix between the optimized code vectors for x1,...,x4."""
    n_samples = R.shape[1]
    similarity = np.eye(n_samples, dtype=float)

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            norm_i = np.linalg.norm(R[:, i])
            norm_j = np.linalg.norm(R[:, j])
            if norm_i == 0.0 or norm_j == 0.0:
                value = 0.0
            else:
                value = cosine_similarity(R[:, i], R[:, j])
            similarity[i, j] = value
            similarity[j, i] = value

    return similarity


def plot_representation_similarity_matrix(matrix: np.ndarray, path: str) -> None:
    """Save the 4x4 representation similarity matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    image = ax.imshow(matrix, cmap="viridis", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(matrix.shape[1]), labels=[f"x{k + 1}" for k in range(matrix.shape[1])])
    ax.set_yticks(range(matrix.shape[0]), labels=[f"x{k + 1}" for k in range(matrix.shape[0])])
    ax.set_title("Representation similarity matrix")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white")

    fig.colorbar(image, ax=ax, label="Cosine similarity")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def solve_dictionary_learning_l1(
    X: np.ndarray,
    n_atoms: int,
    lambda1: float,
    lambda2: float,
    max_iter: int = 200,
    tol: float = 1e-8,
    seed: int = 0,
    restarts: int = 8,
    inner_max_iter: int = 500,
    inner_tol: float = 1e-9) -> tuple[np.ndarray, np.ndarray, list[float], int]:
    """Alternate between updating R and W, then keep the best restart."""
    n_samples = X.shape[1]
    best_W = best_R = None
    best_history: list[float] | None = None
    best_restart = -1
    best_value = np.inf

    for restart in range(restarts):
        rng = np.random.default_rng(seed + restart)
        # Start from random nonnegative codes, then alternate updates.
        R = np.maximum(rng.random((n_atoms, n_samples)), 1e-8)
        history: list[float] = []

        for _ in range(max_iter):
            # Step 1: update the dictionary for the current codes.
            W = update_dictionary(X, R, lambda2)
            # Step 2: update all sparse codes for the current dictionary.
            R = solve_codes_l1(
                W=W,
                X=X,
                lambda1=lambda1,
                warm_start=R,
                inner_max_iter=inner_max_iter,
                inner_tol=inner_tol,
            )
            # One more W update keeps the returned pair (W, R) consistent.
            W = update_dictionary(X, R, lambda2)
            value = objective(X, W, R, lambda1, lambda2)
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
        raise RuntimeError("Solver did not produce a solution.")

    return best_W, best_R, best_history, best_restart


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
        "a": args.a,
        "n_atoms": args.n_atoms,
    }
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
    parser.add_argument("--a", type=float, default=5.0)
    parser.add_argument("--lambda1", type=float, default=0.01)
    parser.add_argument("--lambda2", type=float, default=0.01)
    parser.add_argument("--n-atoms", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--inner-max-iter", type=int, default=500)
    parser.add_argument("--inner-tol", type=float, default=1e-9)
    parser.add_argument("--seed", type=int, default=np.random.randint(0, 1000000))
    parser.add_argument("--restarts", type=int, default=np.random.randint(1, 10))
    parser.add_argument("--new-input",
        type=float,
        nargs="+",
        default=[1, 0, 0, 0, 0, 0],
        help="Give your new input x_new here, for example: --new-input 1 0 0 0 2 0")
    parser.add_argument("--cosine-plot",
        type=str,
        default="cosine_similarity_l1.png",
        help="PNG file where the cosine-similarity plot will be saved.")
    parser.add_argument("--representation-similarity-plot",
        type=str,
        default="representation_similarity_l1.png",
        help="PNG file where the 4x4 representation similarity matrix will be saved.")
    parser.add_argument("--save", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    X = build_samples(args.a)
    # This is where the script reads your new input from the command line.
    x_new = parse_new_input(args.new_input, X.shape[0])

    W, R, history, best_restart = solve_dictionary_learning_l1(
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

    np.set_printoptions(precision=6, suppress=True)
    print("X =")
    print(X)
    print("\nBest restart:", best_restart)
    print("Final objective:", history[-1])
    print("\nW =")
    print(W)
    print("\nR =")
    print(R)
    print("\nW @ R =")
    print(W @ R)

    similarity_matrix = representation_similarity_matrix(R)
    plot_representation_similarity_matrix(similarity_matrix, args.representation_similarity_plot)
    print("\nRepresentation similarity matrix =")
    print(similarity_matrix)
    print("\nRepresentation similarity plot saved to:")
    print(args.representation_similarity_plot)

    r_new = None
    if x_new is not None:
        # After training W, infer the neural pattern for the new input.
        r_new, x_hat = infer_pattern_l1(
            W=W,
            x_new=x_new,
            lambda1=args.lambda1,
            inner_max_iter=args.inner_max_iter,
            inner_tol=args.inner_tol,
        )
        reference = build_reference_input(args.a)
        cosine_value = cosine_similarity(x_hat, reference)
        old_pattern = R[:, 0]
        if np.linalg.norm(r_new) == 0.0 or np.linalg.norm(old_pattern) == 0.0:
            pattern_cosine_value = np.nan
        else:
            pattern_cosine_value = cosine_similarity(r_new, old_pattern)
        plot_cosine_similarity(cosine_value, args.cosine_plot, "L1 cosine similarity")
        print("\nNew input x_new =")
        print(x_new)
        print("\nNeural pattern r_new =")
        print(r_new)
        print("\nNeural pattern for the 1st old input r_1 =")
        print(old_pattern)
        print("\nReconstructed input W @ r_new =")
        print(x_hat)
        print("\nReference input [1, 0, 0, 0, a, 0] =")
        print(reference)
        print("\nCosine similarity =")
        print(cosine_value)
        print("\nCosine similarity between r_new and r_1 =")
        print(pattern_cosine_value)
        print("\nCosine plot saved to:")
        print(args.cosine_plot)
    else:
        cosine_value = None
        pattern_cosine_value = None

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


if __name__ == "__main__":
    main()
