#!/usr/bin/env python3
"""Stream Qwen3.8 fused experts into a compact Colibri expert overlay.

The official dense/QSA/PLE tensors remain in the source checkpoint. In
particular, the ~95 GiB PLE embedding is never copied. The runtime opens the
official shards plus this directory and streams quantized experts from the
overlay.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def expert_rows(config: dict) -> list[tuple[int, str]]:
    text = config["text_config"]
    rows = [
        (layer, f"model.language_model.layers.{layer}.mlp.experts")
        for layer in range(text["num_hidden_layers"])
    ]
    rows.extend(
        (text["num_hidden_layers"] + layer, f"mtp.layers.{layer}.mlp.experts")
        for layer in range(text.get("mtp_num_hidden_layers", 0))
    )
    return rows


def source_names(prefix: str) -> tuple[str, str]:
    return prefix + ".gate_up_proj", prefix + ".down_proj"


def output_names(row: int, expert: int) -> tuple[str, str]:
    base = f"model.layers.{row}.mlp.experts.{expert}"
    return base + ".merged_weight", base + ".qs"


def validate_source_shapes(gate_up_shape, down_shape, config: dict) -> None:
    text = config["text_config"]
    expected_gate_up = (
        text["num_experts"], 2 * text["moe_intermediate_size"],
        text["hidden_size"])
    expected_down = (
        text["num_experts"], text["hidden_size"],
        text["moe_intermediate_size"])
    if tuple(gate_up_shape) != expected_gate_up:
        raise ValueError(f"gate_up shape {tuple(gate_up_shape)} != {expected_gate_up}")
    if tuple(down_shape) != expected_down:
        raise ValueError(f"down shape {tuple(down_shape)} != {expected_down}")


def load_weight_map(model: Path, safe_open) -> dict[str, str]:
    index_path = model / "model.safetensors.index.json"
    if index_path.is_file():
        return json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    weight_map: dict[str, str] = {}
    for shard in sorted(model.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as source:
            for name in source.keys():
                weight_map[name] = shard.name
    return weight_map


def complete_output(path: Path, experts: int, safe_open) -> bool:
    if not path.is_file():
        return False
    try:
        with safe_open(str(path), framework="pt", device="cpu") as source:
            keys = set(source.keys())
        expected = {
            name for expert in range(experts)
            for name in output_names(int(path.stem.rsplit("-", 1)[-1]), expert)
        }
        return keys == expected
    except (OSError, ValueError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True,
                        help="official Qwen3.8 checkpoint directory")
    parser.add_argument("--out", type=Path, required=True,
                        help="expert overlay output directory")
    parser.add_argument("--ebits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=0,
                        help="0 for per-row scales; otherwise input group size")
    args = parser.parse_args()
    if not 2 <= args.ebits <= 8:
        parser.error("--ebits must be in 2..8")
    if args.group_size < 0:
        parser.error("--group-size must be non-negative")
    try:
        from safetensors.torch import safe_open, save_file
        from convert_qwen36 import make_merged
    except ImportError as error:
        raise SystemExit(
            f"missing conversion dependency: {error}; install torch and safetensors")

    config_path = args.model / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"missing {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text = config["text_config"]
    experts = text["num_experts"]
    weight_map = load_weight_map(args.model, safe_open)
    args.out.mkdir(parents=True, exist_ok=True)

    for row, prefix in expert_rows(config):
        output = args.out / f"qwen38-experts-{row:03d}.safetensors"
        if complete_output(output, experts, safe_open):
            print(f"[row {row}] complete, skip")
            continue
        gate_name, down_name = source_names(prefix)
        missing = [name for name in (gate_name, down_name) if name not in weight_map]
        if missing:
            raise SystemExit("missing source tensors: " + ", ".join(missing))
        gate_path = args.model / weight_map[gate_name]
        down_path = args.model / weight_map[down_name]
        tensors = {}
        with safe_open(str(gate_path), framework="pt", device="cpu") as gate_file, \
             safe_open(str(down_path), framework="pt", device="cpu") as down_file:
            gate_slice = gate_file.get_slice(gate_name)
            down_slice = down_file.get_slice(down_name)
            validate_source_shapes(gate_slice.get_shape(), down_slice.get_shape(), config)
            intermediate = text["moe_intermediate_size"]
            for expert in range(experts):
                gate_up = gate_slice[expert].float()
                down = down_slice[expert].float()
                merged, scales = make_merged(
                    gate_up[:intermediate], gate_up[intermediate:], down,
                    args.ebits, gs=args.group_size)
                merged_name, scales_name = output_names(row, expert)
                tensors[merged_name] = merged
                tensors[scales_name] = scales
                if (expert + 1) % 32 == 0 or expert + 1 == experts:
                    print(f"[row {row}] {expert + 1}/{experts}", flush=True)
        temporary = output.with_suffix(output.suffix + ".partial")
        save_file(tensors, str(temporary), metadata={
            "colibri.family": "qwen38_flash_next",
            "colibri.expert_bits": str(args.ebits),
            "colibri.group_size": str(args.group_size),
            "colibri.source_row": str(row),
        })
        temporary.replace(output)

    metadata = {
        "schema_version": 1,
        "family": "qwen38_flash_next",
        "source_model_type": config.get("model_type"),
        "text_model_type": text.get("model_type"),
        "rows": len(expert_rows(config)),
        "experts_per_row": experts,
        "expert_bits": args.ebits,
        "group_size": args.group_size,
        "source_checkpoint": str(args.model.resolve()),
    }
    (args.out / "qwen38_expert_meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"PASS Qwen3.8 expert overlay: {metadata['rows']} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
