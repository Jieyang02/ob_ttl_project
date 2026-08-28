from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import onnxruntime as ort

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


@dataclass(frozen=True)
class CalibrationResult:
    """Statistics and latent reference data used by the online OB-TTL stage."""

    mu_s: np.ndarray
    V_k: np.ndarray
    tau: float
    R_fp: np.ndarray
    inv_cov: np.ndarray
    reconstruction_errors: np.ndarray

    def save(self, path: str | Path) -> None:
        """Save calibration arrays in a portable NumPy archive."""
        np.savez(
            path,
            mu_s=self.mu_s,
            V_k=self.V_k,
            tau=np.asarray(self.tau, dtype=np.float32),
            R_fp=self.R_fp,
            inv_cov=self.inv_cov,
            reconstruction_errors=self.reconstruction_errors,
        )


def _run(session: ort.InferenceSession, values: np.ndarray, batch_size: int = 1) -> np.ndarray:
    """Run an ONNX graph in bounded batches to limit calibration memory."""
    input_name = session.get_inputs()[0].name
    if batch_size < 1:
        raise ValueError("inference_batch_size must be positive")
    outputs = []
    for start in range(0, len(values), batch_size):
        batch = values[start : start + batch_size].astype(np.float32)
        outputs.append(np.asarray(session.run(None, {input_name: batch})[0]))
    return np.concatenate(outputs, axis=0)


def _validate_windows(windows: np.ndarray) -> np.ndarray:
    values = np.asarray(windows, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] == 0:
        raise ValueError("windows must have shape (samples, channels, length)")
    if not np.isfinite(values).all():
        raise ValueError("calibration windows must contain only finite values")
    return values


def _latent_matrix(latent: np.ndarray) -> np.ndarray:
    if latent.ndim < 2:
        raise ValueError("encoder output must include a batch and latent dimension")
    if latent.ndim == 2:
        return latent.astype(np.float32)
    # Pool structural axes if encoder outputs raw multi-dimensional tensors
    return latent.mean(axis=tuple(range(1, latent.ndim - 1))).astype(np.float32)


def _reconstruction_errors(windows: np.ndarray, reconstruction: np.ndarray) -> np.ndarray:
    if windows.shape != reconstruction.shape:
        raise ValueError(
            "decoder output shape does not match input windows: "
            f"{reconstruction.shape} != {windows.shape}"
        )
    return np.mean((windows - reconstruction) ** 2, axis=tuple(range(1, windows.ndim)))


def calibrate(
    encoder_session: ort.InferenceSession,
    decoder_session: ort.InferenceSession,
    calibration_windows: np.ndarray,
    validation_windows: Optional[np.ndarray] = None,
    k: int = 8,
    threshold_percentile: float = 95.0,
    false_positive_band: float = 0.10,
    covariance_regularization: float = 1e-4,
    inference_batch_size: int = 1,
) -> CalibrationResult:
    """Fit the baseline latent subspace, threshold, FP memory, and covariance.

    Calibration windows must be the normalized 0 HP baseline slice. Optional
    normal validation windows are used to build R_fp and the covariance; if
    omitted, the calibration slice is used as the reference population.
    """
    calibration = _validate_windows(calibration_windows)
    if k < 1:
        raise ValueError("k must be positive")
    if not 0 < threshold_percentile <= 100:
        raise ValueError("threshold_percentile must be in (0, 100]")
    if false_positive_band < 0:
        raise ValueError("false_positive_band must be non-negative")
    if covariance_regularization < 0:
        raise ValueError("covariance_regularization must be non-negative")
    if inference_batch_size < 1:
        raise ValueError("inference_batch_size must be positive")

    # 1. Feature extraction and reconstruction
    calibration_latent = _run(encoder_session, calibration, inference_batch_size)
    calibration_reconstruction = _run(decoder_session, calibration_latent, inference_batch_size)
    calibration_errors = _reconstruction_errors(calibration, calibration_reconstruction)
    calibration_matrix = _latent_matrix(calibration_latent)

    # 2. Extract baseline center and SVD subspace basis (V_k)
    mu_s = calibration_matrix.mean(axis=0).astype(np.float32)
    centered = calibration_matrix - mu_s
    _, singular_values, right_singular_vectors = np.linalg.svd(centered, full_matrices=False)

    component_count = min(k, right_singular_vectors.shape[0])
    if component_count == 0:
        raise ValueError("calibration latent features contain no usable SVD components")

    # Transpose so V_k has shape (d, k) matching: z_adapted = z_t + p @ V_k.T
    V_k = right_singular_vectors[:component_count, :].T.astype(np.float32)
    tau = float(np.percentile(calibration_errors, threshold_percentile))

    # 3. Process validation slice for False Positive Reference Memory (R_fp)
    validation = calibration if validation_windows is None else _validate_windows(validation_windows)
    validation_latent = _run(encoder_session, validation, inference_batch_size)
    validation_reconstruction = _run(decoder_session, validation_latent, inference_batch_size)
    validation_errors = _reconstruction_errors(validation, validation_reconstruction)
    validation_matrix = _latent_matrix(validation_latent)

    # Select borderline samples near threshold tau
    band = max(false_positive_band * max(tau, np.finfo(np.float32).eps), np.finfo(np.float32).eps)
    borderline = np.flatnonzero(np.abs(validation_errors - tau) <= band)
    if len(borderline) == 0:
        borderline = np.array([int(np.argmin(np.abs(validation_errors - tau)))])
    R_fp = validation_matrix[borderline].astype(np.float32)

    # 4. Compute regularized inverse covariance matrix in the k-dimensional subspace
    # Project validation features onto V_k (shape: (N, k))
    validation_proj = validation_matrix @ V_k
    k_dim = V_k.shape[1]
    subspace_cov = np.atleast_2d(np.cov(validation_proj, rowvar=False))
    if subspace_cov.shape != (k_dim, k_dim):
        subspace_cov = np.eye(k_dim, dtype=np.float32)

    # Add small regularization to guarantee invertibility in subspace
    subspace_cov = subspace_cov + covariance_regularization * np.eye(k_dim, dtype=np.float32)
    inv_cov = np.linalg.pinv(subspace_cov).astype(np.float32)

    return CalibrationResult(
        mu_s=mu_s,
        V_k=V_k,
        tau=tau,
        R_fp=R_fp,
        inv_cov=inv_cov,
        reconstruction_errors=calibration_errors.astype(np.float32),
    )


def calibrate_from_paths(
    encoder_path: str | Path,
    decoder_path: str | Path,
    calibration_windows: np.ndarray,
    validation_windows: Optional[np.ndarray] = None,
    **kwargs: object,
) -> CalibrationResult:
    """Load ONNX graphs from disk and run :func:`calibrate`."""
    encoder_session = ort.InferenceSession(str(encoder_path))
    decoder_session = ort.InferenceSession(str(decoder_path))
    return calibrate(
        encoder_session,
        decoder_session,
        calibration_windows,
        validation_windows,
        **kwargs,
    )


if __name__ == "__main__":
    encoder_path = "models/encoder.onnx"
    decoder_path = "models/decoder.onnx"
    artifact_output_path = "models/calibration_artifacts.npz"

    print("Loading calibration data...")
    try:
        try:
            from src.data_loader import get_cwru_calibration_and_test_data
        except ModuleNotFoundError:
            from data_loader import get_cwru_calibration_and_test_data

        X_cal, _, _ = get_cwru_calibration_and_test_data()
        print(f"Loaded {len(X_cal)} real calibration windows from data_loader.")
    except Exception as e:
        print(f"Warning: Could not load data_loader ({e}). Using synthetic baseline slice for testing...")
        N_samples, n_channels, seq_len = 50, 2, 512
        X_cal = np.random.randn(N_samples, n_channels, seq_len).astype(np.float32)

    print("Running offline calibration on baseline slice...")
    result = calibrate_from_paths(
        encoder_path=encoder_path,
        decoder_path=decoder_path,
        calibration_windows=X_cal,
        k=8,
        threshold_percentile=95.0,
    )

    result.save(artifact_output_path)

    print("\n" + "=" * 65)
    print("           OFFLINE BASELINE CALIBRATION COMPLETE")
    print("=" * 65)
    print(f"Latent Mean Vector (mu_s)       : shape {result.mu_s.shape}")
    print(f"Subspace Basis (V_k)            : shape {result.V_k.shape}")
    print(f"Baseline Anomaly Threshold (τ)  : {result.tau:.5f}")
    print(f"Reference FP Memory (R_fp)      : {len(result.R_fp)} samples (shape {result.R_fp.shape})")
    print(f"Subspace Inverse Covariance (Σ^-1): shape {result.inv_cov.shape}")
    print("=" * 65)
    print(f"Successfully saved artifacts to: {artifact_output_path}")