from pathlib import Path
from typing import Tuple, Literal
import torch
import h5py

from src.common.paths import get_data_dir

# --- Tag-specific fallback constants (used only if H5 attrs are unavailable) ---
T250_FALLBACK_MEAN = 0.001013
T250_FALLBACK_STD = 0.021286
T500_FALLBACK_MEAN = 0.000460
T500_FALLBACK_STD = 0.020841


def infer_dataset_tag(model_path: str | Path) -> Literal["T250", "T500"]:
    """
    Infers dataset name ('T250'/'T500') from a model checkpoint path or its config.
    Tries reading checkpoint config first, then falls back to path heuristics.
    Defaults to T500 if still ambiguous.
    """
    path_str = str(model_path)
    # Priority 1: try reading checkpoint config
    try:
        ckpt = torch.load(path_str, map_location="cpu")
        if isinstance(ckpt, dict):
            cfg = ckpt.get("config") or {}
            ds_name = (cfg.get("dataset_name") or "").upper()
            ds_path = (cfg.get("dataset_path") or "").upper()
            if ds_name in {"T250", "T500"}:
                return ds_name
            if "T250" in ds_path:
                return "T250"
            if "T500" in ds_path:
                return "T500"
    except Exception:
        pass

    # Priority 2: heuristics from model_path string
    path_lower = path_str.lower()
    if "t250" in path_lower:
        return "T250"
    if "t500" in path_lower:
        return "T500"

    # Fallback: default to T500 but warn
    print(
        f"INFO: Could not determine dataset from model path '{path_str}'. Defaulting to T500 normalization."
    )
    return "T500"


def load_training_stats(tag: Literal["T250", "T500"]) -> Tuple[float, float]:
    """
    Loads wave_mean/std from the corresponding TRAINING HDF5 file attributes.
    Uses data/wave_dataset_T250.h5 or data/wave_dataset_T500.h5.
    Falls back to tag-specific historical constants if files are unavailable.
    """
    ds = tag.upper()
    h5_filename = f"wave_dataset_{ds}.h5"
    h5_path = get_data_dir() / h5_filename

    try:
        if h5_path.exists():
            with h5py.File(str(h5_path), "r") as f:
                mean = float(f.attrs["wave_mean"])
                std = float(f.attrs["wave_std"])
                print(
                    f"INFO: Using training normalization ({ds}): mean={mean:.6f}, std={std:.6f}"
                )
                return mean, std
    except Exception as e:
        print(f"WARNING: Failed to read HDF5 attrs from {h5_path}: {e}")

    # Tag-specific fallback
    if ds == "T250":
        print(
            f"WARNING: Using fallback normalization (T250): mean={T250_FALLBACK_MEAN:.6f}, std={T250_FALLBACK_STD:.6f}"
        )
        return T250_FALLBACK_MEAN, T250_FALLBACK_STD
    # Default to T500 if unknown
    print(
        f"WARNING: Using fallback normalization (T500): mean={T500_FALLBACK_MEAN:.6f}, std={T500_FALLBACK_STD:.6f}"
    )
    return T500_FALLBACK_MEAN, T500_FALLBACK_STD


def infer_dataset_tag_from_path(ds_path: str) -> str:
    """
    Infer dataset tag ('T250'/'T500') from a dataset path string; defaults to 'T500' if ambiguous.
    """
    p = ds_path.lower()
    if "t250" in p:
        return "T250"
    if "t500" in p:
        return "T500"
    print(
        f"INFO: No T250/T500 tag in dataset path '{ds_path}'. Assuming T500 (default)."
    )
    return "T500"


def ensure_dataset_model_match(dataset_path: str, model_path: str | Path):
    """
    Verifies that the inferred dataset tag from the path matches the model's inferred tag.
    Raises ValueError on mismatch.
    """
    model_tag = infer_dataset_tag(model_path)
    ds_tag = infer_dataset_tag_from_path(dataset_path) or infer_dataset_tag_from_path(
        Path(dataset_path).name
    )

    if ds_tag and ds_tag != model_tag:
        raise ValueError(
            f"Dataset/model mismatch: dataset appears to be {ds_tag} but model normalization is {model_tag}.\n"
            f"  - dataset_path={dataset_path}\n"
            f"  - model_ckpt={model_path}"
        )
    print(f"INFO: Dataset-model tag match verified ({model_tag})")
