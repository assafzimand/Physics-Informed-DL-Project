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
import os
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


def set_determinism(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: Path) -> Dict[str, Any]:
    if path is None or not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    if yaml is None:
        raise RuntimeError("pyyaml is not installed. Please install it (pip install pyyaml).")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_model(model_path: str | None, device: torch.device) -> Tuple[torch.nn.Module, Path]:
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
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    return model.to(device).eval(), ckpt_path


def get_conv_layers(model: torch.nn.Module) -> List[Tuple[int, str, torch.nn.Module]]:
    layers: List[Tuple[int, str, torch.nn.Module]] = []
    for idx, (name, module) in enumerate(model.named_modules()):
        if isinstance(module, torch.nn.Conv2d):
            layers.append((idx, name, module))
    return layers


def compute_convergence(monitoring: Dict[str, List[float]], last_k: int,
                        slope_eps: float, std_eps: float, grad_eps: float) -> Dict[str, Any]:
    iters = monitoring['iteration']
    losses = monitoring['loss']
    grads = monitoring['grad_magnitude']

    if len(iters) < last_k:
        last_k = len(iters)
    x = np.array(iters[-last_k:], dtype=np.float64)
    y = np.array(losses[-last_k:], dtype=np.float64)
    g = np.array(grads[-last_k:], dtype=np.float64)

    # Linear regression slope for loss over last_k
    if len(x) >= 2:
        slope = np.polyfit(x, y, 1)[0]
    else:
        slope = float('nan')

    loss_std = float(np.std(y)) if len(y) > 0 else float('nan')
    grad_mean = float(np.mean(g)) if len(g) > 0 else float('nan')

    converged = (abs(slope) < slope_eps) and (loss_std < std_eps) and (grad_mean < grad_eps)

    return {
        'loss_slope_lastk': float(slope),
        'loss_std_lastk': loss_std,
        'grad_mean_lastk': grad_mean,
        'converged': bool(converged),
    }


def run_one(model: torch.nn.Module,
            model_path: Path,
            layer: torch.nn.Module,
            layer_idx: int,
            init_tensor: torch.Tensor,
            device: torch.device,
            params: Dict[str, Any],
            out_dir: Path) -> Dict[str, Any]:

    out_dir.mkdir(parents=True, exist_ok=True)

    maximizer = SimpleActivationMaximizer(model, device, model_path=str(model_path))
    maximizer.register_hook("target_layer", layer)

    try:
        results = maximizer.optimize_filter(
            layer_name="target_layer",
            filter_idx=int(params.get('filter_idx', 0)),  # default 0 if not set
            iterations=int(params['iterations']),
            learning_rate=float(params['learning_rate']),
            use_real_data_init=False,
            init_tensor=init_tensor,  # RAW tensor: [1, 1, H, W]
            save_dir=str(out_dir),
            tv_reg=float(params.get('tv_reg', 0.0)),
        )

        mon = results['monitoring_data']
        conv_cfg = params.get('convergence', {})
        conv = compute_convergence(
            mon,
            last_k=int(conv_cfg.get('last_k', 100)),
            slope_eps=float(conv_cfg.get('slope_eps', 1e-2)),
            std_eps=float(conv_cfg.get('std_eps', 5e-2)),
            grad_eps=float(conv_cfg.get('grad_eps', 1e-2)),
        )

        cfg = results['config']
        summary = {
            'layer_idx': layer_idx,
            'filter_idx': cfg['filter_idx'],
            'iterations': cfg['iterations'],
            'learning_rate': cfg['learning_rate'],
            'tv_reg': cfg.get('tv_reg', 0.0),
            'final_activation': cfg['final_activation'],
            'final_target_loss': cfg['final_target_loss'],
            'final_suppression_loss': cfg['final_suppression_loss'],
            'final_tv_loss': cfg.get('final_tv_loss', 0.0),
            'loss_reduction': cfg['loss_reduction'],
            'grad_variation': cfg['grad_variation'],
            **conv,
        }

        # Save JSON per-run
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        return summary

    finally:
        maximizer.cleanup_hooks()


def get_top_active_filter(model: torch.nn.Module,
                          layer: torch.nn.Module,
                          device: torch.device,
                          raw_tensor: torch.Tensor,
                          wave_mean: float,
                          wave_std: float) -> int:
    """Return index of most active filter for given layer and normalized sample.

    Uses mean(abs(.)) over spatial dims to match AM objective used before ReLU.
    """
    activations: Dict[str, torch.Tensor] = {}

    def hook_fn(module, input, output):
        activations['target'] = output.detach()

    hook = layer.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            norm_sample = (raw_tensor - wave_mean) / wave_std
            _ = model(norm_sample)
        if 'target' not in activations:
            raise RuntimeError("Failed to capture layer activations for ranking")
        layer_output = activations['target']  # [1, C, H, W]
        filter_scores = layer_output.abs().mean(dim=(0, 2, 3))  # [C]
        top_idx = int(torch.argmax(filter_scores).item())
        return top_idx
    finally:
        hook.remove()

def main():
    parser = argparse.ArgumentParser(description="AM Grid Search Runner")
    parser.add_argument("--config", type=str, default="configs/improving_am/baseline.yaml",
                        help="Path to grid config YAML")
    parser.add_argument("--out_dir", type=str, default="experiments/improving_am",
                        help="Base output directory")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    parser.add_argument("--model_path", type=str, default="",
                        help="Explicit model checkpoint path (overrides config)")
    args = parser.parse_args()

    set_determinism(args.seed)

    cfg = load_config(Path(args.config))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # CLI --model_path overrides config (useful in Colab/Drive)
    model_path_cli = args.model_path if args.model_path else cfg.get('model_path')
    model, ckpt_path = resolve_model(model_path_cli, device)

    # Select layer
    conv_layers = get_conv_layers(model)
    if not conv_layers:
        raise RuntimeError("No Conv2d layers found.")
    layer_choice = int(cfg.get('layer_idx', 0))
    if not (0 <= layer_choice < len(conv_layers)):
        raise ValueError(f"layer_idx out of range (0..{len(conv_layers)-1}): {layer_choice}")
    layer_idx, layer_name, target_layer = conv_layers[layer_choice]

    # Load fixed RAW sample to use as deterministic init_tensor
    dataset_path = cfg.get('dataset_path', 'data/wave_dataset_analysis_20samples.h5')
    sample_idx = int(cfg.get('sample_idx', 0))
    dataset = WaveDataset(dataset_path, normalize_wave_fields=False)
    wave_field, _ = dataset[sample_idx]
    init_tensor = wave_field.to(device)  # Expect shape [1, 1, H, W]

    # Determine filter index: if not provided or set to 'auto'/-1, pick most active
    filter_idx_cfg = cfg.get('filter_idx', 'auto')
    if isinstance(filter_idx_cfg, str) and filter_idx_cfg.lower() == 'auto' or (
        isinstance(filter_idx_cfg, int) and filter_idx_cfg < 0
    ):
        # Use training stats to normalize for ranking (consistent with AM)
        stats_resolver = SimpleActivationMaximizer(model, device, model_path=str(ckpt_path))
        wave_mean, wave_std = stats_resolver.wave_mean, stats_resolver.wave_std
        top_filter_idx = get_top_active_filter(model, target_layer, device, init_tensor, wave_mean, wave_std)
        print(f"🔎 Auto-selected top active filter: {top_filter_idx}")
    else:
        top_filter_idx = int(filter_idx_cfg)
        print(f"🎯 Using configured filter index: {top_filter_idx}")

    # Grid
    grid = cfg.get('grid', {})
    learning_rates = grid.get('learning_rates', [0.01])
    iterations_list = grid.get('iterations', [500])
    tv_regs = grid.get('tv_regs', [0.0])

    # Convergence thresholds
    convergence_cfg = cfg.get('convergence', {
        'last_k': 100,
        'slope_eps': 1e-2,
        'std_eps': 5e-2,
        'grad_eps': 1e-2,
    })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out = Path(args.out_dir) / timestamp
    base_out.mkdir(parents=True, exist_ok=True)

    # Master CSV
    csv_path = base_out / "results.csv"
    csv_fields = [
        'layer_idx', 'filter_idx', 'iterations', 'learning_rate', 'tv_reg',
        'final_activation', 'final_target_loss', 'final_suppression_loss', 'final_tv_loss',
        'loss_reduction', 'grad_variation',
        'loss_slope_lastk', 'loss_std_lastk', 'grad_mean_lastk', 'converged',
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()

    run_idx = 0
    for iters in iterations_list:
        for lr in learning_rates:
            for tv in tv_regs:
                run_params = {
                    'iterations': iters,
                    'learning_rate': lr,
                    'tv_reg': tv,
                    'filter_idx': top_filter_idx,
                    'convergence': convergence_cfg,
                }
                run_dir = base_out / f"run_{run_idx:03d}_it{iters}_lr{lr}_tv{tv}"

                print(f"\n===== RUN {run_idx} =====")
                print(f"Layer {layer_idx} ({layer_name}) | filter {top_filter_idx}")
                print(f"iters={iters}, lr={lr}, tv_reg={tv}")
                summary = run_one(model, ckpt_path, target_layer, layer_idx, init_tensor, device, run_params, run_dir)

                # Append to CSV
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=csv_fields)
                    writer.writerow({k: summary.get(k, '') for k in csv_fields})

                run_idx += 1

    # Save config snapshot
    with open(base_out / "config_used.json", 'w') as f:
        json.dump(cfg, f, indent=2)

    print(f"\n✅ Grid search complete. Results saved to: {base_out}")


if __name__ == "__main__":
    main()


