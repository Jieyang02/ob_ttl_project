from __future__ import annotations

import numpy as np


def _validate_inputs(
    z_t: np.ndarray, R_fp: np.ndarray, inv_cov: np.ndarray, V_k: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    latent = np.asarray(z_t, dtype=np.float64)
    references = np.asarray(R_fp, dtype=np.float64)
    inverse = np.asarray(inv_cov, dtype=np.float64)
    
    if latent.ndim != 1:
        raise ValueError("z_t must be a one-dimensional latent vector")
    if references.ndim != 2 or references.shape[0] == 0:
        raise ValueError("R_fp must have shape (reference_count, latent_dimension)")
    return latent, references, inverse


def compute_mahalanobis_min(
    z_t: np.ndarray, 
    R_fp: np.ndarray, 
    inv_cov: np.ndarray, 
    V_k: np.ndarray | None = None
) -> float:
    """Return the minimum squared Mahalanobis distance to R_fp in the principal subspace."""
    latent, references, inverse = _validate_inputs(z_t, R_fp, inv_cov)
    
    # If subspace V_k is provided, project onto well-conditioned k-dimensional subspace
    if V_k is not None:
        basis = np.asarray(V_k, dtype=np.float64)
        latent_proj = latent @ basis          # (k,)
        ref_proj = references @ basis         # (N_fp, k)
        diff = ref_proj - latent_proj         # (N_fp, k)
        distances = np.einsum("ni,ij,nj->n", diff, inverse, diff)
        return float(np.min(distances))

    diff = references - latent
    distances = np.einsum("ni,ij,nj->n", diff, inverse, diff)
    return float(np.min(distances))


def is_benign_drift(
    z_t: np.ndarray,
    R_fp: np.ndarray,
    inv_cov: np.ndarray,
    delta: float,
    V_k: np.ndarray | None = None,
) -> bool:
    """Return True for benign drift (distance < delta) and False for true faults."""
    if not np.isfinite(delta) or delta < 0:
        raise ValueError("delta must be a finite non-negative value")
    return compute_mahalanobis_min(z_t, R_fp, inv_cov, V_k=V_k) < delta


def route_sample(
    z_t: np.ndarray,
    R_fp: np.ndarray,
    inv_cov: np.ndarray,
    delta: float,
    V_k: np.ndarray | None = None,
) -> bool:
    """Route one latent sample; True means trigger adaptation."""
    return is_benign_drift(z_t, R_fp, inv_cov, delta, V_k=V_k)