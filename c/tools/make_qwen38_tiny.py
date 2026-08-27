#!/usr/bin/env python3
"""Generate the deterministic Qwen3.8 runtime oracle checkpoint."""

import argparse
import json
import struct
from collections import OrderedDict
from pathlib import Path

import numpy as np


def bf16(values):
    f32 = np.asarray(values, dtype="<f4")
    return (f32.view("<u4") >> 16).astype("<u2")


def write_safetensors(path, tensors):
    header = OrderedDict()
    payload = bytearray()
    for name, (dtype, array) in tensors.items():
        data = np.ascontiguousarray(array).tobytes()
        start = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def zero(shape):
    return bf16(np.zeros(shape, dtype=np.float32))


def add_linear(tensors, layer):
    prefix = f"model.language_model.layers.{layer}."
    shapes = {
        "linear_attn.in_proj_qkv.weight": (32, 16),
        "linear_attn.in_proj_z.weight": (16, 16),
        "linear_attn.in_proj_b.weight": (4, 16),
        "linear_attn.in_proj_a.weight": (4, 16),
        "linear_attn.conv1d.weight": (32, 1, 4),
        "linear_attn.A_log": (4,),
        "linear_attn.dt_bias": (4,),
        "linear_attn.norm.weight": (4,),
        "linear_attn.out_proj.weight": (16, 16),
    }
    add_hyper_moe(shapes)
    for suffix, shape in shapes.items():
        tensors[prefix + suffix] = ("BF16", zero(shape))


def add_hyper_moe(shapes):
    for block in ("attn", "mlp"):
        prefix = f"{block}_hyper_connection."
        shapes.update({
            prefix + "hc_norm.weight": (32,),
            prefix + "input_mix_weight_down.weight": (4, 32),
            prefix + "input_mix_weight_up.weight": (32, 4),
            prefix + "block_inject_weight.weight": (2, 32),
        })
    shapes.update({
        "mlp.experts.gate_up_proj": (8, 12, 16),
        "mlp.experts.down_proj": (8, 16, 6),
        "mlp.gate.weight": (8, 16),
        "mlp.shared_expert.gate_proj.weight": (6, 16),
        "mlp.shared_expert.up_proj.weight": (6, 16),
        "mlp.shared_expert.down_proj.weight": (16, 6),
        "mlp.shared_expert_gate.weight": (1, 16),
    })


def source_tensors():
    tensors = OrderedDict()
    embedding = np.array([
        [(((token + 1) * (dim + 1)) % 17 - 8) / 16
         for dim in range(16)] for token in range(32)], dtype=np.float32)
    lm_head = np.array([
        [(((token + 3) * (dim + 5)) % 19 - 9) / 32
         for dim in range(16)] for token in range(32)], dtype=np.float32)
    tensors["model.language_model.embed_tokens.weight"] = ("BF16", bf16(embedding))
    tensors["lm_head.weight"] = ("BF16", bf16(lm_head))
    for shard in range(2):
        name = ("model.language_model.layers.1.ple.ple_embedding."
                f"ngram_embedding.shard_{shard}.weight")
        tensors[name] = ("BF16", zero((102, 8)))
    for layer in range(3):
        add_linear(tensors, layer)
    full = {
        "self_attn.q_proj.weight": (32, 16),
        "self_attn.k_proj.weight": (8, 16),
        "self_attn.v_proj.weight": (8, 16),
        "self_attn.o_proj.weight": (16, 16),
        "self_attn.q_norm.weight": (4,),
        "self_attn.k_norm.weight": (4,),
        "self_attn.indexer.index_qk_proj.weight": (12, 16),
        "self_attn.indexer.q_layernorm.weight": (4,),
        "self_attn.indexer.k_layernorm.weight": (4,),
    }
    add_hyper_moe(full)
    for suffix, shape in full.items():
        tensors["model.language_model.layers.3." + suffix] = ("BF16", zero(shape))
    ple = {
        "key_proj.weight": (32, 16), "value_proj.weight": (16, 16),
        "norm_key.weight": (32,), "norm_query.weight": (32,),
        "norm_conv.weight": (32,), "conv1d.weight": (32, 1, 3),
    }
    for suffix, shape in ple.items():
        tensors["model.language_model.layers.1.ple." + suffix] = ("BF16", zero(shape))
    ple_meta = "model.language_model.layers.1.ple.ple_embedding."
    tensors[ple_meta + "layer_multipliers"] = (
        "I64", np.zeros(2, dtype="<i8"))
    tensors[ple_meta + "ngram_heads_vocab_sizes"] = (
        "I64", np.zeros(2, dtype="<i8"))
    tensors[ple_meta + "ngram_heads_offsets"] = (
        "I64", np.zeros(2, dtype="<i8"))
    global_hyper = {
        "hc_norm.weight": (32,), "input_mix_weight_down.weight": (4, 32),
        "input_mix_weight_up.weight": (32, 4),
    }
    for suffix, shape in global_hyper.items():
        tensors["model.language_model.hyper_connection_mixer." + suffix] = (
            "BF16", zero(shape))
    mtp_global = {
        "fc_embedding.weight": (16, 16), "fc_hidden.weight": (16, 16),
        "pre_fc_norm_embedding.weight": (16,),
        "pre_fc_norm_hidden.weight": (32,),
        "hyper_connection_mixer.hc_norm.weight": (32,),
        "hyper_connection_mixer.input_mix_weight_down.weight": (4, 32),
        "hyper_connection_mixer.input_mix_weight_up.weight": (32, 4),
    }
    for suffix, shape in mtp_global.items():
        tensors["mtp." + suffix] = ("BF16", zero(shape))
    mtp_layer = {
        "self_attn.q_proj.weight": (32, 16),
        "self_attn.k_proj.weight": (8, 16),
        "self_attn.v_proj.weight": (8, 16),
        "self_attn.o_proj.weight": (16, 16),
        "self_attn.q_norm.weight": (4,),
        "self_attn.k_norm.weight": (4,),
        "self_attn.indexer.index_qk_proj.weight": (12, 16),
        "self_attn.indexer.q_layernorm.weight": (4,),
        "self_attn.indexer.k_layernorm.weight": (4,),
    }
    add_hyper_moe(mtp_layer)
    for suffix, shape in mtp_layer.items():
        tensors["mtp.layers.0." + suffix] = ("BF16", zero(shape))
    return tensors


def config():
    return {"model_type": "qwen4_exp", "text_config": {
        "model_type": "qwen4_exp_text", "hidden_size": 16,
        "num_hidden_layers": 4, "vocab_size": 32, "num_experts": 8,
        "num_experts_per_tok": 2, "moe_intermediate_size": 6,
        "shared_expert_intermediate_size": 6, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 4, "linear_num_key_heads": 2,
        "linear_key_head_dim": 4, "linear_num_value_heads": 4,
        "linear_value_head_dim": 4, "linear_conv_kernel_dim": 4,
        "hc_count": 2, "hc_lowrank": 4, "ngram_size": 2,
        "ngram_vocab_size_base": 100, "split_ngram_parts": 2,
        "make_ngram_vocab_size_divisible_by": 4,
        "ple_conv_kernel_size": 3, "ple_embed_dim": 16,
        "heads_per_ngram": 2, "ple_layer_ids": [2], "indexer_budget": 8,
        "indexer_compress_ratio": 2, "indexer_head_dim": 4,
        "indexer_kv_heads": 1, "indexer_n_heads": 2,
        "mtp_num_hidden_layers": 1, "eos_token_id": 31,
        "rms_norm_eps": 0.000001, "partial_rotary_factor": 0.5,
        "rope_parameters": {"rope_theta": 10000},
        "layer_types": ["linear_attention", "linear_attention",
                        "linear_attention", "full_attention"],
    }}


def tokenizer():
    added = [
        {"id": token, "content": f"<t{token:03d}>", "single_word": False,
         "lstrip": False, "rstrip": False, "normalized": False, "special": True}
        for token in range(32)
    ]
    return {
        "version": "1.0", "truncation": None, "padding": None,
        "added_tokens": added, "normalizer": None, "pre_tokenizer": None,
        "post_processor": None, "decoder": None,
        "model": {"type": "BPE", "dropout": None, "unk_token": None,
                  "continuing_subword_prefix": "", "end_of_word_suffix": "",
                  "fuse_unk": False, "byte_fallback": False,
                  "ignore_merges": True, "vocab": {"x": 31}, "merges": []},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, experts = args.output / "source", args.output / "experts"
    source.mkdir(parents=True, exist_ok=True)
    experts.mkdir(parents=True, exist_ok=True)
    (source / "config.json").write_text(json.dumps(config()), encoding="utf-8")
    (source / "tokenizer.json").write_text(
        json.dumps(tokenizer(), separators=(",", ":")), encoding="utf-8")
    write_safetensors(source / "model.safetensors", source_tensors())
    overlay = OrderedDict()
    for row in range(5):
        for expert in range(8):
            base = f"model.layers.{row}.mlp.experts.{expert}"
            overlay[base + ".merged_weight"] = ("U8", np.zeros(144, dtype=np.uint8))
            overlay[base + ".qs"] = ("F32", np.zeros(28, dtype="<f4"))
    write_safetensors(experts / "model.safetensors", overlay)
    metadata = {"family": "qwen38_flash_next", "rows": 5,
                "experts_per_row": 8, "expert_bits": 4, "group_size": 0}
    (experts / "qwen38_expert_meta.json").write_text(
        json.dumps(metadata), encoding="utf-8")
    print(f"PASS Qwen3.8 tiny checkpoint: {args.output}")


if __name__ == "__main__":
    main()
