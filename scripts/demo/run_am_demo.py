#!/usr/bin/env python3
"""
Run Activation Maximization demo for selected layers and top-K filters.

- Uses dataset tag inferred from dataset path; warns (does not fail) if the
  checkpoint tag disagrees, and proceeds with the dataset tag
- Ranks filters per chosen activation metric (default: post_relu_mean)
- Runs activation maximization and saves comprehensive plots
- Minimal, clear logs

Usage (Windows):
  .\venv\Scripts\python.exe scripts\demo\run_am_demo.py \
      --model_path models\best_T250.pth \
      --dataset_path data\wave_dataset_T250_validation.h5 \
      --layer_mode last_n --last_n 3 --top_k 3 \
      --activation_mode post_relu_mean --iterations 1000 --lr 0.005
"""

import argparse
from pathlib import Path
from typing import List, Tuple
import sys

# Add project root to sys.path to allow imports like `src.common`
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import h5py
import torch
import random

from src.activation_maximization.simple_activation_max import (
    SimpleActivationMaximizer,
)
from src.common.normalization import (
    infer_dataset_tag_from_path,
    infer_dataset_tag,
    load_training_stats,
)
from src.common.paths import get_experiments_dir


def enforce_tag(dataset_path: Path, model_path: Path) -> str:
    tag_d = infer_dataset_tag_from_path(str(dataset_path))
    tag_m = infer_dataset_tag(model_path)
    if tag_d and tag_m and tag_d != tag_m:
        # Warn only; proceed with dataset tag to ensure correct normalization
        print(
            f"[am-demo][WARN] Tag mismatch: dataset={tag_d}, model={tag_m}. Using dataset tag {tag_d}."
        )
    return tag_d or tag_m or "T500"


def capture_layer_activation(
    model: torch.nn.Module,
    device: torch.device,
    layer: torch.nn.Module,
    sample_tensor: torch.Tensor,
    activation_mode: str,
) -> torch.Tensor:
    activations = {}

    def hook_fn(module, input, output):
        activations["target"] = output.detach()

    hook = layer.register_forward_hook(hook_fn)
    try:
        model.eval()
        with torch.no_grad():
            _ = model(sample_tensor)
        layer_out = activations["target"]  # [1, C, H, W]
        if activation_mode == "mean_abs":
            filt = layer_out.abs().mean(dim=(0, 2, 3))
        elif activation_mode == "mean":
            filt = layer_out.mean(dim=(0, 2, 3))
        elif activation_mode == "l2":
            filt = (layer_out.pow(2).sum(dim=(2, 3)).sqrt()).mean(dim=0)
        elif activation_mode == "post_relu_mean":
            filt = torch.relu(layer_out).mean(dim=(0, 2, 3))
        else:
            raise ValueError(f"Unsupported activation_mode: {activation_mode}")
        return filt
    finally:
        hook.remove()


def rank_filters(
    model: torch.nn.Module,
    device: torch.device,
    layer: torch.nn.Module,
    sample_tensor: torch.Tensor,
    top_k: int,
    activation_mode: str,
) -> List[int]:
    scores = capture_layer_activation(
        model, device, layer, sample_tensor, activation_mode
    )
    order = torch.argsort(scores, descending=True)[:top_k]
    return [int(i) for i in order.cpu().numpy().tolist()]


def load_reference_sample(
    dataset_path: Path,
    device: torch.device,
) -> torch.Tensor:
    with h5py.File(dataset_path, "r") as f:
        wave_key = "wave_fields" if "wave_fields" in f else "inputs"
        x = f[wave_key][0][...]
    x = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0).to(device)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument(
        "--layer_mode",
        required=True,
        choices=["last_n", "explicit"],
    )
    ap.add_argument("--last_n", type=int, default=3)
    ap.add_argument("--layer_indices", type=str)
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--activation_mode", default="post_relu_mean")
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--suppression_weight", type=float, default=1.0)
    ap.add_argument("--tv_reg", type=float, default=0.0)
    ap.add_argument("--l2_reg", type=float, default=0.0)
    ap.add_argument("--num_samples", type=int, default=1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(args.model_path)
    dataset_path = Path(args.dataset_path)

    tag = enforce_tag(dataset_path, model_path)
    mean, std = load_training_stats(tag)

    # Load model via your usual training factory
    # Caller provides a model instance; here we load through checkpoint
    ckpt = torch.load(model_path, map_location=device)
    from src.models.wave_source_resnet import create_wave_source_model

    model = create_wave_source_model(grid_size=128).to(device).eval()
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    # Force correct normalization tag in the AM engine
    am = SimpleActivationMaximizer(
        model=model, device=str(device), model_path=str(model_path), dataset_name=tag
    )

    # Ensure AM has real samples from the provided dataset
    try:
        am.load_real_wave_samples(dataset_path)
    except Exception:
        pass

    # Choose layers
    conv_layers: List[Tuple[str, torch.nn.Module]] = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            conv_layers.append((name, module))

    if args.layer_mode == "last_n":
        selected = conv_layers[-args.last_n:]
    else:
        assert args.layer_indices, "--layer_indices required for explicit mode"
        indices = [int(i) for i in args.layer_indices.split(",")]
        selected = [conv_layers[i] for i in indices]

    # Prepare save dir
    out_dir = project_root / "outputs" / "am_demo" / model_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[am-demo] tag={tag} layers={len(selected)} top_k={args.top_k} out={out_dir}")

    # Utility to get dataset length
    with h5py.File(dataset_path, "r") as f_len:
        wave_key = "wave_fields" if "wave_fields" in f_len else "inputs"
        total_samples = len(f_len[wave_key])

    for run_idx in range(args.num_samples):
        sample_idx = random.randrange(total_samples)
        with h5py.File(dataset_path, "r") as f:
            wave_key = "wave_fields" if "wave_fields" in f else "inputs"
            x_np = f[wave_key][sample_idx][...]
        x_t = torch.from_numpy(x_np).float().unsqueeze(0).unsqueeze(0).to(device)
        x_norm = (x_t - mean) / (std + 1e-8)

        sample_dir = out_dir / f"sample_{sample_idx:05d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        print(f"  - Sample {sample_idx}: ranking and optimizing")

        for layer_name, layer in selected:
            top_filters = rank_filters(
                model,
                device,
                layer,
                x_norm,
                top_k=args.top_k,
                activation_mode=args.activation_mode,
            )
            for filt in top_filters:
                am.register_hook(layer_name, layer)
                am.optimize_filter(
                    layer_name=layer_name,
                    filter_idx=int(filt),
                    iterations=args.iterations,
                    learning_rate=args.lr,
                    image_size=128,
                    use_real_data_init=False,
                    save_intermediate=True,
                    save_every=max(1, args.iterations // 10),
                    save_dir=str(sample_dir),
                    init_tensor=x_norm.clone().detach(),
                    tv_reg=args.tv_reg,
                    l2_reg=args.l2_reg,
                    suppression_weight=args.suppression_weight,
                    activation_mode=args.activation_mode,
                )
                am.cleanup_hooks()

    print(f"[am-demo] Completed. Outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
