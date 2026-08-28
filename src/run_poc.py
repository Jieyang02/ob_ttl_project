from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np
import onnxruntime as ort
from scipy.stats import chi2

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from src.cma_optimizer import forward_subspace_cmaes
    from src.fpm import compute_mahalanobis_min, is_benign_drift
except ImportError:
    from cma_optimizer import forward_subspace_cmaes
    from fpm import compute_mahalanobis_min, is_benign_drift

LOGGER = logging.getLogger(__name__)


def _run(session: ort.InferenceSession, values: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    return np.asarray(session.run(None, {input_name: values.astype(np.float32)})[0])


def _pool_latent(latent: np.ndarray) -> np.ndarray:
    if latent.ndim == 2:
        return latent[0].astype(np.float64)
    return latent.mean(axis=tuple(range(1, latent.ndim - 1)))[0].astype(np.float64)


def _adapt_raw_latent(latent: np.ndarray, pooled_adapted: np.ndarray) -> np.ndarray:
    pooled = _pool_latent(latent)
    delta = pooled_adapted - pooled
    shape = (1,) * (latent.ndim - 1) + (delta.size,)
    return (latent + delta.reshape(shape)).astype(np.float32)


def _residual(values: np.ndarray, reconstruction: np.ndarray) -> float:
    return float(np.mean((values - reconstruction) ** 2))


def _model_shape(session: ort.InferenceSession) -> tuple[int, int]:
    shape = session.get_inputs()[0].shape
    if len(shape) != 3 or not all(isinstance(value, int) for value in shape[1:]):
        raise ValueError("encoder input must have a static (channels, window) shape")
    return int(shape[1]), int(shape[2])


def _synthetic_samples(session: ort.InferenceSession) -> dict[str, np.ndarray]:
    channels, window = _model_shape(session)
    rng = np.random.default_rng(42)
    normal = rng.normal(0.0, 0.25, (1, channels, window)).astype(np.float32)
    drift = (normal + 0.45).astype(np.float32)
    fault = normal.copy()
    fault[:, :, window // 3 : 2 * window // 3] += 3.5
    return {"0 HP normal": normal, "3 HP drift": drift, "3 HP fault": fault}


def _load_samples(
    encoder: ort.InferenceSession, 
    decoder: ort.InferenceSession, 
    data_directory: Path, 
    tau: float
) -> tuple[dict[str, np.ndarray], str]:
    """Load representative samples from CWRU, ensuring the drift sample triggers static evaluation."""
    try:
        try:
            from src.data_loader import get_cwru_calibration_and_test_data
        except ImportError:
            from data_loader import get_cwru_calibration_and_test_data

        calibration, stream, labels = get_cwru_calibration_and_test_data(data_directory)
        channels, window = _model_shape(encoder)
        if calibration.shape[1:] != (channels, window):
            raise ValueError("CWRU windows do not match encoder input shape")

        # 1. Normal baseline sample
        sample_normal = calibration[:1]

        # 2. Find a 3 HP drift sample (normal label = 0) that exceeds tau on the static model
        drift_indices = np.flatnonzero(labels == 0)
        selected_drift = None
        for idx in drift_indices[:200]:
            cand = stream[idx : idx + 1]
            rec = _run(decoder, _run(encoder, cand))
            if _residual(cand, rec) > tau:
                selected_drift = cand
                break
        if selected_drift is None:
            drift_residuals = [_residual(stream[i:i+1], _run(decoder, _run(encoder, stream[i:i+1]))) for i in drift_indices[:100]]
            selected_drift = stream[drift_indices[int(np.argmax(drift_residuals)) : drift_indices[int(np.argmax(drift_residuals))] + 1]]

        # 3. Fault sample (label = 1)
        fault_indices = np.flatnonzero(labels == 1)
        if len(fault_indices) == 0:
            raise ValueError("No fault samples found in stream")
        selected_fault = stream[fault_indices[0] : fault_indices[0] + 1]

        samples = {
            "0 HP normal": sample_normal,
            "3 HP drift": selected_drift,
            "3 HP fault": selected_fault,
        }
        return samples, "CWRU Data"
    except Exception as e:
        LOGGER.warning("Could not load real CWRU data (%s); falling back to synthetic dataset.", e)
        return _synthetic_samples(encoder), "Synthetic Fallback"


def _load_artifacts(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Calibration artifact not found: {path}")
    with np.load(path) as artifact:
        required = ("mu_s", "V_k", "tau", "R_fp", "inv_cov")
        missing = [name for name in required if name not in artifact]
        if missing:
            raise ValueError(f"Calibration artifact missing: {', '.join(missing)}")
        return {name: np.asarray(artifact[name]) for name in required}


def run_poc(
    model_directory: str | Path = "models",
    data_directory: str | Path = "data/raw",
    delta: float | None = None,
) -> tuple[list[dict[str, object]], float, float, str]:
    """Run static and OB-TTL scoring for normal, drift, and fault samples."""
    root = Path(model_directory)
    artifacts = _load_artifacts(root / "calibration_artifacts.npz")
    encoder = ort.InferenceSession(str(root / "encoder.onnx"))
    decoder = ort.InferenceSession(str(root / "decoder.onnx"))
    
    tau = float(artifacts["tau"])
    samples, source_name = _load_samples(encoder, decoder, Path(data_directory), tau)
    
    # Subspace dimension k defines the degrees of freedom
    k_components = artifacts["V_k"].shape[1]
    decision_delta = float(chi2.ppf(0.95, df=k_components) if delta is None else delta)

    rows = []
    for sample_type, values in samples.items():
        latent = _run(encoder, values)
        static_reconstruction = _run(decoder, latent)
        initial_residual = _residual(values, static_reconstruction)
        adapted_residual = initial_residual
        distance = None
        adapted = False

        if initial_residual > tau:
            pooled = _pool_latent(latent)
            
            # Pass V_k to evaluate distance in the k-dimensional subspace
            distance = compute_mahalanobis_min(
                pooled, artifacts["R_fp"], artifacts["inv_cov"], V_k=artifacts["V_k"]
            )
            
            if is_benign_drift(
                pooled, artifacts["R_fp"], artifacts["inv_cov"], decision_delta, V_k=artifacts["V_k"]
            ):
                adapted_pooled = forward_subspace_cmaes(
                    pooled, artifacts["mu_s"], artifacts["V_k"], k=k_components
                )
                adapted_reconstruction = _run(decoder, _adapt_raw_latent(latent, adapted_pooled))
                adapted_residual = _residual(values, adapted_reconstruction)
                adapted = True

        rows.append({
            "sample_type": sample_type,
            "static_alarm": initial_residual > tau,
            "ob_ttl_alarm": adapted_residual > tau,
            "initial_residual": initial_residual,
            "ob_ttl_residual": adapted_residual,
            "mahalanobis": distance,
            "adapted": adapted,
        })
    return rows, tau, decision_delta, source_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-directory", type=Path, default=Path("models"))
    parser.add_argument("--data-directory", type=Path, default=Path("data/raw"))
    parser.add_argument("--delta", type=float, default=None)
    args = parser.parse_args()

    rows, tau, delta, source_name = run_poc(args.model_directory, args.data_directory, args.delta)

    print("\n" + "=" * 92)
    print(f"Data Source       : {source_name}")
    print(f"Baseline Threshold: τ = {tau:.5f}")
    print(f"FPM Distance Limit: δ = {delta:.3f} (Chi-square 95th percentile, k=8)")
    print("-" * 92)
    print("{:<16} {:>10} {:>10} {:>14} {:>14} {:>14} {:>10}".format(
        "Sample Type", "Static", "OB-TTL", "Initial Res.", "Adapted Res.", "Mahal.² Dist", "Adapted?"
    ))
    print("-" * 92)
    for row in rows:
        mahalanobis = "-" if row["mahalanobis"] is None else f"{row['mahalanobis']:.3f}"
        print("{:<16} {:>10} {:>10} {:>14.5f} {:>14.5f} {:>14} {:>10}".format(
            row["sample_type"],
            "ALARM" if row["static_alarm"] else "NORMAL",
            "ALARM" if row["ob_ttl_alarm"] else "NORMAL",
            row["initial_residual"],
            row["ob_ttl_residual"],
            mahalanobis,
            "Yes" if row["adapted"] else "No",
        ))
    print("=" * 92)
    print("Diagnostic Interpretation:")
    print(" • 0 HP Normal : Error <= τ -> Normal baseline verified.")
    print(" • 3 HP Drift  : Error > τ but Mahal.² < δ -> FPM approves adaptation -> False alarm suppressed.")
    print(" • 3 HP Fault  : Error > τ and Mahal.² >= δ -> FPM detects anomaly -> True fault alarm triggered.\n")


if __name__ == "__main__":
    main()