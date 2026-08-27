#!/usr/bin/env python3
"""Validate the text tensors consumed by the Qwen3.8-Flash-Next runtime."""
from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path


FLOAT_DTYPES = ("BF16", "F16", "F32")


def read_header(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        raw_size = stream.read(8)
        if len(raw_size) != 8:
            raise ValueError(f"truncated safetensors header: {path}")
        size = struct.unpack("<Q", raw_size)[0]
        return json.loads(stream.read(size))


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    return all(value % divisor for divisor in range(3, math.isqrt(value) + 1, 2))


def _next_prime(value: int) -> int:
    value += 1
    while not _is_prime(value):
        value += 1
    return value


def ple_padded_rows(config: dict) -> int:
    heads = (config["ngram_size"] - 1) * config["heads_per_ngram"]
    prime = config["ngram_vocab_size_base"] - 1
    total = 0
    for _ in range(heads):
        prime = _next_prime(prime)
        total += prime
    divisor = config["make_ngram_vocab_size_divisible_by"]
    return (total + divisor - 1) // divisor * divisor


def expected_tensors(text: dict) -> dict[str, tuple[list[int], tuple[str, ...]]]:
    hidden = text["hidden_size"]
    layers = text["num_hidden_layers"]
    vocab = text["vocab_size"]
    experts = text["num_experts"]
    intermediate = text["moe_intermediate_size"]
    shared = text["shared_expert_intermediate_size"]
    hc = text["hc_count"]
    hc_width = hc * hidden
    hc_lowrank = text["hc_lowrank"]
    heads = text["num_attention_heads"]
    kv_heads = text["num_key_value_heads"]
    head_dim = text["head_dim"]
    index_heads = text["indexer_n_heads"]
    index_kv_heads = text["indexer_kv_heads"]
    index_dim = text["indexer_head_dim"]
    key_heads = text["linear_num_key_heads"]
    key_dim = text["linear_key_head_dim"]
    value_heads = text["linear_num_value_heads"]
    value_dim = text["linear_value_head_dim"]
    conv_width = 2 * key_heads * key_dim + value_heads * value_dim
    value_width = value_heads * value_dim
    tensors: dict[str, tuple[list[int], tuple[str, ...]]] = {}

    def add(name: str, shape: list[int], dtypes=FLOAT_DTYPES) -> None:
        tensors[name] = (shape, tuple(dtypes))

    add("model.language_model.embed_tokens.weight", [vocab, hidden])
    add("lm_head.weight", [vocab, hidden])

    def hyper(prefix: str, combine: bool = True) -> None:
        add(prefix + "hc_norm.weight", [hc_width])
        add(prefix + "input_mix_weight_down.weight", [hc_lowrank, hc_width])
        add(prefix + "input_mix_weight_up.weight", [hc_width, hc_lowrank])
        if combine:
            add(prefix + "block_inject_weight.weight", [hc, hc_width])

    hyper("model.language_model.hyper_connection_mixer.", False)

    def moe(prefix: str) -> None:
        add(prefix + "experts.gate_up_proj", [experts, 2 * intermediate, hidden])
        add(prefix + "experts.down_proj", [experts, hidden, intermediate])
        add(prefix + "gate.weight", [experts, hidden])
        add(prefix + "shared_expert.gate_proj.weight", [shared, hidden])
        add(prefix + "shared_expert.up_proj.weight", [shared, hidden])
        add(prefix + "shared_expert.down_proj.weight", [hidden, shared])
        add(prefix + "shared_expert_gate.weight", [1, hidden])

    def full_attention(prefix: str) -> None:
        # Qwen3.8 projects query and an output sigmoid gate together.
        add(prefix + "q_proj.weight", [2 * heads * head_dim, hidden])
        add(prefix + "k_proj.weight", [kv_heads * head_dim, hidden])
        add(prefix + "v_proj.weight", [kv_heads * head_dim, hidden])
        add(prefix + "o_proj.weight", [hidden, heads * head_dim])
        add(prefix + "q_norm.weight", [head_dim])
        add(prefix + "k_norm.weight", [head_dim])
        add(prefix + "indexer.index_qk_proj.weight",
            [(index_heads + index_kv_heads) * index_dim, hidden])
        add(prefix + "indexer.q_layernorm.weight", [index_dim])
        add(prefix + "indexer.k_layernorm.weight", [index_dim])

    def linear_attention(prefix: str) -> None:
        add(prefix + "in_proj_qkv.weight", [conv_width, hidden])
        add(prefix + "in_proj_z.weight", [value_width, hidden])
        add(prefix + "in_proj_b.weight", [value_heads, hidden])
        add(prefix + "in_proj_a.weight", [value_heads, hidden])
        add(prefix + "conv1d.weight", [conv_width, 1, text["linear_conv_kernel_dim"]])
        add(prefix + "dt_bias", [value_heads])
        add(prefix + "A_log", [value_heads])
        add(prefix + "norm.weight", [value_dim])
        add(prefix + "out_proj.weight", [hidden, value_width])

    for layer, kind in enumerate(text["layer_types"]):
        base = f"model.language_model.layers.{layer}."
        hyper(base + "attn_hyper_connection.")
        hyper(base + "mlp_hyper_connection.")
        moe(base + "mlp.")
        if kind == "linear_attention":
            linear_attention(base + "linear_attn.")
        elif kind == "full_attention":
            full_attention(base + "self_attn.")
        else:
            raise ValueError(f"unsupported layer type {kind!r}")

    for one_based_layer in text.get("ple_layer_ids", []):
        base = f"model.language_model.layers.{one_based_layer - 1}.ple."
        ple_dim = text["ple_embed_dim"]
        ngram_heads = (text["ngram_size"] - 1) * text["heads_per_ngram"]
        add(base + "key_proj.weight", [hc_width, ple_dim])
        add(base + "value_proj.weight", [hidden, ple_dim])
        for norm in ("norm_key", "norm_query", "norm_conv"):
            add(base + norm + ".weight", [hc_width])
        add(base + "conv1d.weight", [hc_width, 1, text["ple_conv_kernel_size"]])
        meta = base + "ple_embedding."
        add(meta + "layer_multipliers", [text["ngram_size"]], ("I64", "U64"))
        add(meta + "ngram_heads_vocab_sizes", [ngram_heads], ("I64", "U64"))
        add(meta + "ngram_heads_offsets", [ngram_heads], ("I64", "U64"))

    mtp_layers = text.get("mtp_num_hidden_layers", 0)
    if mtp_layers:
        add("mtp.fc_embedding.weight", [hidden, hidden])
        add("mtp.fc_hidden.weight", [hidden, hidden])
        add("mtp.pre_fc_norm_embedding.weight", [hidden])
        add("mtp.pre_fc_norm_hidden.weight", [hc_width])
        hyper("mtp.hyper_connection_mixer.", False)
        for layer in range(mtp_layers):
            base = f"mtp.layers.{layer}."
            hyper(base + "attn_hyper_connection.")
            hyper(base + "mlp_hyper_connection.")
            full_attention(base + "self_attn.")
            moe(base + "mlp.")
    return tensors


def validate(config: dict, tensors: dict[str, object], *, shapes: bool) -> list[str]:
    text = config["text_config"]
    failures: list[str] = []
    for name, (shape, dtypes) in expected_tensors(text).items():
        actual = tensors.get(name)
        if actual is None:
            failures.append(f"missing {name}")
        elif shapes and (actual.get("shape") != shape or actual.get("dtype") not in dtypes):
            failures.append(
                f"invalid {name}: {actual.get('dtype')} {actual.get('shape')}, "
                f"expected {dtypes} {shape}")

    ple_shards = text.get("split_ngram_parts", 0)
    for one_based_layer in text.get("ple_layer_ids", []):
        prefix = (f"model.language_model.layers.{one_based_layer - 1}.ple."
                  "ple_embedding.ngram_embedding")
        total_rows = 0
        expected_width = text["ple_embed_dim"] // (
            (text["ngram_size"] - 1) * text["heads_per_ngram"])
        for shard in range(ple_shards):
            name = f"{prefix}.shard_{shard}.weight"
            actual = tensors.get(name)
            if actual is None:
                failures.append(f"missing {name}")
            elif shapes:
                shape = actual.get("shape", [])
                if (actual.get("dtype") not in FLOAT_DTYPES or len(shape) != 2 or
                        shape[0] < 1 or shape[1] != expected_width):
                    failures.append(f"invalid {name}: {actual.get('dtype')} {shape}")
                else:
                    total_rows += shape[0]
        if shapes and total_rows != ple_padded_rows(text):
            failures.append(
                f"invalid PLE row total: {total_rows}, expected {ple_padded_rows(text)}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--headers-json", type=Path)
    parser.add_argument("--index", type=Path)
    args = parser.parse_args()
    if not args.model and not args.config:
        parser.error("provide --model or --config")
    config_path = args.config or args.model / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.headers_json:
        tensors = json.loads(args.headers_json.read_text(encoding="utf-8"))
        shapes = True
    elif args.index:
        index = json.loads(args.index.read_text(encoding="utf-8"))
        tensors = {name: {} for name in index["weight_map"]}
        shapes = False
    elif args.model:
        tensors = {}
        for shard in sorted(args.model.glob("*.safetensors")):
            tensors.update({name: spec for name, spec in read_header(shard).items()
                            if name != "__metadata__"})
        shapes = True
    else:
        parser.error("--config requires --headers-json or --index")
    failures = validate(config, tensors, shapes=shapes)
    if failures:
        raise SystemExit("Qwen3.8 checkpoint contract failed:\n" + "\n".join(failures[:100]))
    mode = "shape" if shapes else "index"
    print(f"PASS Qwen3.8 checkpoint {mode} contract: {len(tensors)} tensors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
