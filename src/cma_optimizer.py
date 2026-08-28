from __future__ import annotations

import numpy as np
import cma


def _validate_inputs(
	z_t: np.ndarray, mu_s: np.ndarray, V_k: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	latent = np.asarray(z_t, dtype=np.float64)
	mean = np.asarray(mu_s, dtype=np.float64)
	subspace = np.asarray(V_k, dtype=np.float64)
	if latent.ndim != 1 or mean.ndim != 1:
		raise ValueError("z_t and mu_s must be one-dimensional latent vectors")
	if latent.shape != mean.shape:
		raise ValueError("z_t and mu_s must have the same latent dimension")
	if subspace.ndim != 2 or subspace.shape[0] != latent.size:
		raise ValueError("V_k must have shape (latent_dimension, components)")
	if k < 1 or k > subspace.shape[1]:
		raise ValueError("k must be between 1 and the number of V_k components")
	if not np.isfinite(latent).all() or not np.isfinite(mean).all() or not np.isfinite(subspace).all():
		raise ValueError("z_t, mu_s, and V_k must contain finite values")
	return latent, mean, subspace[:, :k]


def forward_subspace_cmaes(
	z_t: np.ndarray,
	mu_s: np.ndarray,
	V_k: np.ndarray,
	k: int = 8,
	max_iter: int = 3,
	lambda_reg: float = 0.05,
	initial_sigma: float = 0.1,
	seed: int = 42,
) -> np.ndarray:
	"""Return ``z_t + p_star @ V_k.T`` using forward-only CMA-ES.

	The objective uses NumPy vector operations only. No model parameters,
	gradients, or backward passes are involved.
	"""
	if max_iter < 1:
		raise ValueError("max_iter must be positive")
	if lambda_reg < 0 or not np.isfinite(lambda_reg):
		raise ValueError("lambda_reg must be finite and non-negative")
	if initial_sigma <= 0 or not np.isfinite(initial_sigma):
		raise ValueError("initial_sigma must be finite and positive")
	latent, mean, basis = _validate_inputs(z_t, mu_s, V_k, k)

	def loss(coordinates: list[float]) -> float:
		p = np.asarray(coordinates, dtype=np.float64)
		adapted = latent + p @ basis.T
		return float(np.sum((adapted - mean) ** 2) + lambda_reg * np.sum(p**2))

	optimizer = cma.CMAEvolutionStrategy(
		[0.0] * k,
		initial_sigma,
		{"maxiter": max_iter, "seed": seed, "verbose": -9},
	)
	while not optimizer.stop():
		candidates = optimizer.ask()
		optimizer.tell(candidates, [loss(candidate) for candidate in candidates])
	best_coordinates = np.asarray(optimizer.result.xbest, dtype=np.float64)
	return (latent + best_coordinates @ basis.T).astype(np.float32)


def adapt_latent(
	z_t: np.ndarray,
	mu_s: np.ndarray,
	V_k: np.ndarray,
	**kwargs: object,
) -> np.ndarray:
	"""Alias for :func:`forward_subspace_cmaes` for online pipeline callers."""
	return forward_subspace_cmaes(z_t, mu_s, V_k, **kwargs)
