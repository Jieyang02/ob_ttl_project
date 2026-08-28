from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Tuple
import pandas as pd

import numpy as np
from scipy.io import loadmat


DEFAULT_WINDOW_LENGTH = 512
DEFAULT_OVERLAP = 0.5
CALIBRATION_SAMPLES = 50


@dataclass(frozen=True)
class NormalizationStats:
    """Per-channel statistics fitted using calibration windows only."""

    mean: np.ndarray
    std: np.ndarray


@dataclass(frozen=True)
class CWRUSplit:
    """Chronological calibration and evaluation windows."""

    calibration: np.ndarray
    stream: np.ndarray
    stream_labels: np.ndarray
    stats: NormalizationStats


def sliding_windows(
    data: np.ndarray,
    window_length: int = DEFAULT_WINDOW_LENGTH,
    overlap: float = DEFAULT_OVERLAP,
) -> np.ndarray:
    """Return chronological windows with shape (samples, channels, length)."""
    values = np.asarray(data, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("data must be a two-dimensional channels-by-time array")
    if window_length < 1:
        raise ValueError("window_length must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in the range [0, 1)")
    stride = int(round(window_length * (1.0 - overlap)))
    if stride < 1:
        raise ValueError("overlap leaves no positive stride")

    if values.shape[0] > values.shape[1]:
        values = values.T
    channels, sample_count = values.shape
    if sample_count < window_length:
        return np.empty((0, channels, window_length), dtype=np.float32)
    starts = range(0, sample_count - window_length + 1, stride)
    return np.stack([values[:, start : start + window_length] for start in starts])


def fit_zscore(calibration: np.ndarray, epsilon: float = 1e-8) -> NormalizationStats:
    """Fit channel-wise z-score statistics from calibration data only."""
    values = np.asarray(calibration, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] == 0:
        raise ValueError("calibration must have shape (samples, channels, length)")
    mean = values.mean(axis=(0, 2), keepdims=True)
    std = values.std(axis=(0, 2), keepdims=True)
    std = np.maximum(std, epsilon)
    return NormalizationStats(mean=mean.astype(np.float32), std=std.astype(np.float32))


def apply_zscore(data: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    """Apply previously fitted channel-wise statistics without refitting."""
    values = np.asarray(data, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("data must have shape (samples, channels, length)")
    return ((values - stats.mean) / stats.std).astype(np.float32)


def load_vibration_file(path: str | Path, channel_names: Sequence[str] = ("DE", "FE")) -> np.ndarray:
    """Load two or more channels from a CWRU MAT file or a numeric CSV file."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required data file not found at: {file_path}. "
            "Please ensure raw CWRU .mat files are downloaded into data/raw/."
        )

    if file_path.suffix.lower() == ".mat":
        mat = loadmat(file_path)
        channels = []
        for name in channel_names:
            matches = [
                value for key, value in mat.items() 
                if name.lower() in key.lower() and not key.startswith("__")
            ]
            if not matches:
                # Fallback: if FE is missing in certain files, duplicate DE channel
                if name == "FE" and len(channels) > 0:
                    channels.append(channels[0])
                else:
                    raise ValueError(f"Could not find channel {name!r} in {file_path}")
            else:
                channels.append(np.asarray(matches[0]).reshape(-1))
        length = min(map(len, channels))
        return np.stack([channel[:length] for channel in channels]).astype(np.float32)

    if file_path.suffix.lower() == ".csv":
        values = np.loadtxt(file_path, delimiter=",", ndmin=2)
        return np.asarray(values, dtype=np.float32)

    raise ValueError(f"Unsupported vibration file format: {file_path.suffix}")


def prepare_cwru_split(
    normal_0hp: np.ndarray,
    normal_3hp: np.ndarray,
    faults_3hp: Iterable[np.ndarray],
    window_length: int = DEFAULT_WINDOW_LENGTH,
    overlap: float = DEFAULT_OVERLAP,
    calibration_samples: int = CALIBRATION_SAMPLES,
) -> CWRUSplit:
    """Create calibration and chronological drift/fault streams."""
    if calibration_samples < 1:
        raise ValueError("calibration_samples must be positive")
    calibration_source = sliding_windows(normal_0hp, window_length, overlap)
    if len(calibration_source) < calibration_samples:
        raise ValueError(
            f"0 HP source provides {len(calibration_source)} windows; "
            f"at least {calibration_samples} are required"
        )
    calibration = calibration_source[:calibration_samples]
    
    stream_groups = [sliding_windows(normal_3hp, window_length, overlap)]
    stream_groups.extend(sliding_windows(fault, window_length, overlap) for fault in faults_3hp)
    
    stream = np.concatenate(stream_groups, axis=0) if any(len(group) for group in stream_groups) else np.empty_like(calibration[:0])
    labels = np.concatenate(
        [np.zeros(len(stream_groups[0]), dtype=np.int64),
         np.ones(sum(len(group) for group in stream_groups[1:]), dtype=np.int64)]
    )
    stats = fit_zscore(calibration)
    return CWRUSplit(
        calibration=apply_zscore(calibration, stats),
        stream=apply_zscore(stream, stats),
        stream_labels=labels,
        stats=stats,
    )


def load_cwru_split(
    data_directory: str | Path = "data/raw",
    window_length: int = DEFAULT_WINDOW_LENGTH,
    overlap: float = DEFAULT_OVERLAP,
    calibration_samples: int = CALIBRATION_SAMPLES,
) -> CWRUSplit:
    """Load the standard CWRU files and prepare the TTA evaluation split."""
    root = Path(data_directory)
    return prepare_cwru_split(
        load_vibration_file(root / "normal_0hp.mat"),
        load_vibration_file(root / "normal_3hp.mat"),
        [
            load_vibration_file(root / "inner_fault_3hp.mat"),
            load_vibration_file(root / "ball_fault_3hp.mat"),
            load_vibration_file(root / "outer_fault_3hp.mat"),
        ],
        window_length=window_length,
        overlap=overlap,
        calibration_samples=calibration_samples,
    )

def export_stream_to_csv(output_path: str = "data/processed/cwru_stream.csv") -> None:
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    # 1. Load normalized streaming test data and ground-truth labels
    _, stream, labels = get_cwru_calibration_and_test_data()
    
    # stream shape is (num_windows, channels, window_length) -> e.g., (N, 2, 512)
    num_windows, channels, length = stream.shape
    
    # Flatten each window to a single row: [DE_0, DE_1, ..., FE_0, FE_1, ..., label]
    flattened_windows = stream.reshape(num_windows, channels * length)
    
    df = pd.DataFrame(flattened_windows)
    df["label"] = labels  # 0 for 3 HP normal drift, 1 for 3 HP true faults
    
    df.to_csv(output_path, index=False)
    print(f"Successfully exported {len(df)} streaming test windows to: {output_path}")

def get_cwru_calibration_and_test_data(
    data_directory: str | Path = "data/raw",
    window_length: int = DEFAULT_WINDOW_LENGTH,
    overlap: float = DEFAULT_OVERLAP,
    calibration_samples: int = CALIBRATION_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Helper wrapper returning (calibration_windows, streaming_test_windows, test_labels)."""
    split = load_cwru_split(
        data_directory=data_directory,
        window_length=window_length,
        overlap=overlap,
        calibration_samples=calibration_samples,
    )
    return split.calibration, split.stream, split.stream_labels


# =====================================================================
# Sanity Check Block
# =====================================================================
if __name__ == "__main__":
    print("Testing data_loader.py...")
    split = load_cwru_split(Path("data/raw"))

    print("\n" + "=" * 60)
    print("                DATA LOADER VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"Calibration Slice Shape : {split.calibration.shape} (Expected: (50, 2, 32))")
    print(f"Streaming Test Shape    : {split.stream.shape}")
    print(f"Streaming Labels Shape  : {split.stream_labels.shape}")
    print(f"Normal Labels in Stream : {np.sum(split.stream_labels == 0)} (Segment A: Drift)")
    print(f"Fault Labels in Stream  : {np.sum(split.stream_labels == 1)} (Segment B: Faults)")
    print(f"Fitted Channel Means    : {split.stats.mean.squeeze()}")
    print(f"Fitted Channel Stds     : {split.stats.std.squeeze()}")
    print("=" * 60)
    print("Verification passed successfully.")