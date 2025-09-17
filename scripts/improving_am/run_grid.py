#!/usr/bin/env python3
"""
Deterministic Activation Maximization (AM) Grid Search Runner

Runs AM for a single fixed layer and a single fixed sample with a grid of
hyperparameters. Records convergence metrics and saves concise CSV/JSON.

Outputs are written under experiments/improving_am/<timestamp>/.
"""

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch

# Lazy import yaml to allow running without it (Colab cell will install pyyaml)
try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

# Ensure project root on sys.path for 'src.*'
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.activation_maximization.simple_activation_max import SimpleActivationMaximizer
from src.activation_maximization.layer_hooks import find_best_cv_model
from src.models.wave_source_resnet import create_wave_source_model
from src.data.wave_dataset import WaveDataset
from src.common.normalization import (
    load_training_stats,
    infer_dataset_tag,
)
from src.common.paths import get_configs_dir, get_experiments_dir, get_data_dir


def set_determinism(seed: int = 42) -> None:
    """Sets random seeds for deterministic runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: Path) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    if path is None or not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    if yaml is None:
        raise RuntimeError(
            "pyyaml is not installed. Please install it (pip install pyyaml)."
        )
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_model(
    model_path: str | None, device: torch.device
) -> Tuple[torch.nn.Module, Path]:
    """
    Loads a model checkpoint, either from an explicit path or by finding the best
    model from a cross-validation experiment directory.
    """
    if model_path:
        ckpt_path = Path(model_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")
    else:
        cv_root = Path("experiments") / "cv_full"
        fold_id, error, model_path_resolved = find_best_cv_model(cv_root)
        ckpt_path = Path(model_path_resolved)
    model = create_wave_source_model(grid_size=128)
    checkpoint = torch.load(str(ckpt_path), map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    return model.to(device).eval(), ckpt_path


def infer_dataset_from_path(ds_path: str) -> str | None:
    """Infer dataset tag ('T250'/'T500') from a dataset path string; None if ambiguous."""
    p = ds_path.lower()
    if "t250" in p:
        return "T250"
    if "t500" in p:
        return "T500"
    return None


def get_conv_layers(model: torch.nn.Module) -> List[Tuple[int, str, torch.nn.Module]]:
    """Find Conv2d layers, excluding skip/projection convs.

    Returns a list of tuples: (internal_idx, qualified_name, module).

    Exclusion rules:
    - Names containing "skip_connection" (projection path inside residuals)
    - 1x1 convolutions (typical projection convs)
    """
    layers: List[Tuple[int, str, torch.nn.Module]] = []
    for idx, (name, module) in enumerate(model.named_modules()):
        if isinstance(module, torch.nn.Conv2d):
            name_has_skip = "skip_connection" in name
            is_projection_1x1 = getattr(module, "kernel_size", None) == (1, 1)
            if name_has_skip or is_projection_1x1:
                continue
            layers.append((idx, name, module))
    return layers


def compute_convergence(
    monitoring: Dict[str, List[float]],
    last_k: int,
    slope_eps: float,
    std_eps: float,
    grad_eps: float,
) -> Dict[str, Any]:
    """
    Computes convergence metrics over the last K iterations of optimization.

    Calculates the slope of the loss, standard deviation of the loss, and the
    mean gradient magnitude. Determines overall convergence based on thresholds.
    """
    iters = monitoring["iteration"]
    losses = monitoring["loss"]
    grads = monitoring["grad_magnitude"]

    if len(iters) < last_k:
        last_k = len(iters)
    x = np.array(iters[-last_k:], dtype=np.float64)
    y = np.array(losses[-last_k:], dtype=np.float64)
    g = np.array(grads[-last_k:], dtype=np.float64)

    # Linear regression slope for loss over last_k
    if len(x) >= 2:
        slope = np.polyfit(x, y, 1)[0]
    else:
        slope = float("nan")

    loss_std = float(np.std(y)) if len(y) > 0 else float("nan")
    grad_mean = float(np.mean(g)) if len(g) > 0 else float("nan")

    converged = (
        (abs(slope) < slope_eps) and (loss_std < std_eps) and (grad_mean < grad_eps)
    )

    return {
        "loss_slope_lastk": float(slope),
        "loss_std_lastk": loss_std,
        "grad_mean_lastk": grad_mean,
        "converged": bool(converged),
    }


def run_one(
    model: torch.nn.Module,
    model_path: Path,
    layer: torch.nn.Module,
    layer_idx: int,
    init_tensor: torch.Tensor,
    device: torch.device,
    params: Dict[str, Any],
    out_dir: Path,
    final_dir: Path,
) -> Dict[str, Any]:
    """
    Executes a single activation maximization run for one set of hyperparameters.

    Initializes the maximizer, runs the optimization, computes convergence,
    and saves a per-run summary JSON.

    Returns:
        A dictionary containing the summary of the run's results and metrics.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    maximizer = SimpleActivationMaximizer(
        model,
        device,
        model_path=str(model_path),
        dataset_name=str(params.get("dataset_name", "")) or None,
    )
    maximizer.register_hook(f"layer{layer_idx}", layer)

    # Build concise run-identifying base name for files (no activation mode)
    def _fmt_num(x: float) -> str:
        try:
            xi = int(x)
            if float(xi) == float(x):
                return str(xi)
        except Exception:
            pass
        return ("%g" % float(x))

    try:
        samp = int(params.get("sample_idx", -1))
    except Exception:
        samp = -1
    file_base = (
        f"s{int(samp)}_layer{int(layer_idx)}_f{int(params.get('filter_idx', 0))}"
        f"_it{int(params['iterations'])}_lr{_fmt_num(params['learning_rate'])}"
        f"_tv{_fmt_num(params.get('tv_reg', 0.0))}_l2{_fmt_num(params.get('l2_reg', 0.0))}"
        f"_sup{_fmt_num(params.get('suppression_weight', 1.0))}"
    )

    try:
        results = maximizer.optimize_filter(
            layer_name=f"layer{layer_idx}",
            filter_idx=int(params.get("filter_idx", 0)),  # default 0 if not set
            iterations=int(params["iterations"]),
            learning_rate=float(params["learning_rate"]),
            use_real_data_init=False,
            init_tensor=init_tensor,  # RAW tensor: [1, 1, H, W]
            save_dir=str(final_dir),
            tv_reg=float(params.get("tv_reg", 0.0)),
            l2_reg=float(params.get("l2_reg", 0.0)),
            activation_mode=str(params.get("activation_mode", "mean_abs")),
            suppression_weight=float(params.get("suppression_weight", 1.0)),
            output_filename_base=file_base,
            ground_truth_xy=params.get("ground_truth_xy"),
        )

        mon = results["monitoring_data"]
        conv_cfg = params.get("convergence", {})
        conv = compute_convergence(
            mon,
            last_k=int(conv_cfg.get("last_k", 100)),
            slope_eps=float(conv_cfg.get("slope_eps", 1e-2)),
            std_eps=float(conv_cfg.get("std_eps", 5e-2)),
            grad_eps=float(conv_cfg.get("grad_eps", 1e-2)),
        )

        cfg = results["config"]
        summary = {
            "layer_idx": layer_idx,
            "filter_idx": cfg["filter_idx"],
            "iterations": cfg["iterations"],
            "learning_rate": cfg["learning_rate"],
            "tv_reg": cfg.get("tv_reg", 0.0),
            "l2_reg": cfg.get("l2_reg", 0.0),
            "activation_mode": params.get("activation_mode", "mean_abs"),
            "final_activation": cfg["final_activation"],
            "final_target_loss": cfg["final_target_loss"],
            "final_suppression_loss": cfg["final_suppression_loss"],
            "final_tv_loss": cfg.get("final_tv_loss", 0.0),
            "final_l2_loss": cfg.get("final_l2_loss", 0.0),
            "loss_reduction": cfg["loss_reduction"],
            "grad_variation": cfg["grad_variation"],
            **conv,
        }

        # Plot is already saved with file_base in out_dir by the maximizer

        # Save JSON per-run with short name to avoid long paths
        summary_path = out_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        # Also place a copy in the layer folder with concise name
        try:
            (final_dir / f"{file_base}.json").write_text(summary_path.read_text())
        except Exception:
            pass

        return summary

    finally:
        maximizer.cleanup_hooks()


def get_top_active_filter(
    model: torch.nn.Module,
    layer: torch.nn.Module,
    device: torch.device,
    raw_tensor: torch.Tensor,
    wave_mean: float,
    wave_std: float,
) -> int:
    """Return index of most active filter for given layer and normalized sample.

    Uses mean(abs(.)) over spatial dims to match AM objective used before ReLU.
    """
    activations: Dict[str, torch.Tensor] = {}

    def hook_fn(module, input, output):
        activations["target"] = output.detach()

    hook = layer.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            norm_sample = (raw_tensor - wave_mean) / wave_std
            _ = model(norm_sample)
        if "target" not in activations:
            raise RuntimeError("Failed to capture layer activations for ranking")
        layer_output = activations["target"]  # [1, C, H, W]
        filter_scores = layer_output.abs().mean(dim=(0, 2, 3))  # [C]
        top_idx = int(torch.argmax(filter_scores).item())
        return top_idx
    finally:
        hook.remove()


def main():
    """Main execution function for the grid search runner."""
    parser = argparse.ArgumentParser(
        description="Activation Maximization (AM) Grid Search Runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(get_configs_dir() / "improving_am/baseline.yaml"),
        help="Path to the grid search configuration YAML file.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(get_experiments_dir() / "improving_am"),
        help="Base output directory where timestamped run folders will be created.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for deterministic runs."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Explicit path to a model checkpoint (.pth). Overrides the model_path in the config file.",
    )
    args = parser.parse_args()

    set_determinism(args.seed)

    cfg = load_config(Path(args.config))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # CLI --model_path overrides config (useful in Colab/Drive)
    model_path_cli = args.model_path if args.model_path else cfg.get("model_path")
    model, ckpt_path = resolve_model(model_path_cli, device)

    # Load dataset once
    default_dataset_path = get_data_dir() / "wave_dataset_analysis_20samples.h5"
    dataset_path = cfg.get("dataset_path", str(default_dataset_path))
    dataset = WaveDataset(dataset_path, normalize_wave_fields=False)

    # Enforce dataset-model T-tag alignment
    ds_tag_check = infer_dataset_from_path(str(dataset_path))
    model_tag_check = infer_dataset_tag(ckpt_path)
    if ds_tag_check and model_tag_check and ds_tag_check != model_tag_check:
        print(
            f"WARNING: Dataset/model tag mismatch: dataset={ds_tag_check}, model={model_tag_check}. Proceeding with dataset tag {ds_tag_check}."
        )

    # Select layers for analysis
    conv_layers = get_conv_layers(model)
    if not conv_layers:
        raise RuntimeError("No Conv2d layers found.")
    selection = cfg.get("selection", {})
    mode = selection.get("mode", "").lower()
    selected_layers: List[Tuple[int, str, torch.nn.Module]]
    if mode == "last_n":
        n = int(selection.get("last_n_layers", 1))
        selected_layers = conv_layers[-n:]
    elif mode == "explicit":
        indices = selection.get("layer_indices", [])
        raw_map_for_indices = selection.get("filters_by_layer", {}) or {}
        # Case A: explicit indices refer to positions within conv_layers array
        if indices:
            selected_layers = [
                conv_layers[i] for i in indices if 0 <= i < len(conv_layers)
            ]
        else:
            # Case B: infer by internal named_modules indices present as keys in filters_by_layer
            try:
                desired_internal_ids = set(int(k) for k in raw_map_for_indices.keys())
            except Exception:
                desired_internal_ids = set()
            selected_layers = [
                entry for entry in conv_layers if entry[0] in desired_internal_ids
            ]
    else:
        # fallback to single-layer legacy path
        layer_choice = int(cfg.get("layer_idx", 0))
        if not (0 <= layer_choice < len(conv_layers)):
            raise ValueError(
                f"layer_idx out of range (0..{len(conv_layers)-1}): {layer_choice}"
            )
        selected_layers = [conv_layers[layer_choice]]

    if not selected_layers:
        raise RuntimeError(
            "No layers selected. Check selection.mode, layer_indices, or filters_by_layer in the config."
        )
    top_k_filters = int(selection.get("top_k_filters", 1))
    # Optional explicit per-layer filters mapping: {"59": [56,123], "61": [18,167,...]}
    raw_map = selection.get("filters_by_layer", {}) or {}
    filters_by_layer = {}
    for k, v in raw_map.items():
        try:
            li = int(k)
            filters_by_layer[li] = [int(f) for f in v]
        except Exception:
            continue
    # Optional explicit filter index for legacy path; otherwise we'll rank per-layer below
    filter_idx_cfg = cfg.get("filter_idx", "auto")

    # Grid
    grid = cfg.get("grid", {})
    learning_rates = grid.get("learning_rates", [0.01])
    iterations_list = grid.get("iterations", [500])
    tv_regs = grid.get("tv_regs", [0.0])
    l2_regs = grid.get("l2_regs", [0.0])
    suppression_weights = grid.get("suppression_weights", [1.0])
    activation_modes = grid.get("activation_modes", ["mean_abs"])
    sample_indices = grid.get("sample_indices", [int(cfg.get("sample_idx", 0))])

    # Convergence thresholds
    convergence_cfg = cfg.get(
        "convergence",
        {
            "last_k": 100,
            "slope_eps": 1e-2,
            "std_eps": 5e-2,
            "grad_eps": 1e-2,
        },
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out = Path(args.out_dir) / timestamp
    base_out.mkdir(parents=True, exist_ok=True)

    # Master CSV
    csv_path = base_out / "results.csv"
    csv_fields = [
        "layer_idx",
        "layer_name",
        "filter_idx",
        "sample_idx",
        "iterations",
        "learning_rate",
        "tv_reg",
        "l2_reg",
        "activation_mode",
        "final_activation",
        "final_target_loss",
        "final_suppression_loss",
        "final_tv_loss",
        "final_l2_loss",
        "loss_reduction",
        "grad_variation",
        "loss_slope_lastk",
        "loss_std_lastk",
        "grad_mean_lastk",
        "converged",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()

    run_idx = 0
    # Iterate over samples
    for sample_idx in sample_indices:
        sample_dir = base_out / f"sample_{int(sample_idx):04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        wave_field, coords = dataset[int(sample_idx)]
        init_tensor = wave_field
        # Ensure 4D shape [B, C, H, W] for model input (dataset gives [C, H, W])
        if hasattr(init_tensor, "dim") and init_tensor.dim() == 3:
            init_tensor = init_tensor.unsqueeze(0)
        init_tensor = init_tensor.to(device)

        # For each selected layer, compute top-K filters using training normalization
        ds_tag = infer_dataset_from_path(str(dataset_path))
        if ds_tag is None:
            ds_tag = infer_dataset_tag(ckpt_path)
        wave_mean, wave_std = load_training_stats(ds_tag)

        for layer_internal_idx, layer_name, target_layer in selected_layers:
            layer_dir = sample_dir / f"layer_{int(layer_internal_idx)}"
            layer_dir.mkdir(parents=True, exist_ok=True)
            # For each activation mode, rank filters using the SAME activation used for optimization
            for act in activation_modes:
                # Determine top filters for this act
                try:
                    # Highest priority: explicit per-layer filters mapping
                    if layer_internal_idx in filters_by_layer:
                        top_filters = filters_by_layer[layer_internal_idx]
                    elif not (
                        isinstance(filter_idx_cfg, str)
                        and filter_idx_cfg.lower() == "auto"
                    ) and not (isinstance(filter_idx_cfg, int) and filter_idx_cfg < 0):
                        top_filters = [int(filter_idx_cfg)]
                    else:
                        with torch.no_grad():
                            norm_sample = (init_tensor - wave_mean) / wave_std
                            activations: Dict[str, torch.Tensor] = {}

                            def hook_fn(m, i, o):
                                activations["t"] = o.detach()

                            h = target_layer.register_forward_hook(hook_fn)
                            _ = model(norm_sample)
                            h.remove()
                            layer_out = activations.get("t")
                            if layer_out is None:
                                raise RuntimeError(
                                    "Failed to capture activations for ranking"
                                )
                            if act == "mean_abs":
                                scores = layer_out.abs().mean(dim=(0, 2, 3))
                            elif act == "mean":
                                scores = layer_out.mean(dim=(0, 2, 3))
                            elif act == "l2":
                                scores = (layer_out.pow(2).sum(dim=(2, 3)).sqrt()).mean(
                                    dim=0
                                )
                            elif act == "post_relu_mean":
                                scores = torch.relu(layer_out).mean(dim=(0, 2, 3))
                            else:
                                raise ValueError(
                                    f"Unsupported activation_mode for ranking: {act}"
                                )
                            top_filters = (
                                torch.argsort(scores, descending=True)[:top_k_filters]
                                .cpu()
                                .tolist()
                            )
                except Exception as e:
                    print(
                        f"WARNING: Ranking failed for layer {layer_internal_idx} (act={act}): {e}"
                    )
                    top_filters = [0]

                for filt in top_filters:
                    for iters in iterations_list:
                        for lr in learning_rates:
                            for tv in tv_regs:
                                for l2 in l2_regs:
                                    for sup_w in suppression_weights:
                                        run_params = {
                                            "iterations": iters,
                                            "learning_rate": lr,
                                            "tv_reg": tv,
                                            "l2_reg": l2,
                                            "activation_mode": act,
                                            "suppression_weight": sup_w,
                                            "filter_idx": int(filt),
                                            "sample_idx": int(sample_idx),
                                            "dataset_name": str(ds_tag),
                                            "ground_truth_xy": (
                                                float(coords[0].item()) if hasattr(coords[0], "item") else float(coords[0]),
                                                float(coords[1].item()) if hasattr(coords[1], "item") else float(coords[1]),
                                            ),
                                            "convergence": convergence_cfg,
                                        }
                                        # Working directory for this run lives under the sample/layer folder
                                        run_dir = layer_dir / (
                                            f"run_{run_idx:04d}_s{sample_idx}_layer{layer_internal_idx}_f{filt}_it{iters}_lr{lr}_tv{tv}_l2{l2}_sup{sup_w}_act{act}"
                                        )

                                        print(f"\n===== RUN {run_idx} =====")
                                        print(
                                            f"Sample {sample_idx} | Layer {layer_internal_idx} ({layer_name}) | filter {filt}"
                                        )
                                        print(
                                            f"iters={iters}, lr={lr}, tv_reg={tv}, l2_reg={l2}, sup_w={sup_w}, act={act}"
                                        )
                                        # Pass final target folder (sample/layer) for organized outputs
                                        summary = run_one(
                                            model,
                                            ckpt_path,
                                            target_layer,
                                            layer_internal_idx,
                                            init_tensor,
                                            device,
                                            run_params,
                                            run_dir,
                                            layer_dir,
                                        )

                                        # Augment summary with identifiers
                                        summary["layer_name"] = layer_name
                                        summary["sample_idx"] = int(sample_idx)

                                        with open(csv_path, "a", newline="") as f:
                                            writer = csv.DictWriter(
                                                f, fieldnames=csv_fields
                                            )
                                            writer.writerow(
                                                {
                                                    k: summary.get(k, "")
                                                    for k in csv_fields
                                                }
                                            )

                                        run_idx += 1

    # Save config snapshot
    with open(base_out / "config_used.json", "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\n✅ Grid search complete. Results saved to: {base_out}")


if __name__ == "__main__":
    main()
