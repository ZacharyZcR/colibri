#!/usr/bin/env python3
"""Regenerate the Qwen3.8 tiny oracle with official Transformers code.

This maintainer tool intentionally is not part of CI: it requires torch and a
Transformers release containing qwen4_exp. CI consumes the committed result.
"""

import argparse
import json
from pathlib import Path

import torch
import transformers
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = Qwen4ExpTextConfig(
        vocab_size=32, hidden_size=16, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=4,
        linear_num_key_heads=2, linear_key_head_dim=4,
        linear_num_value_heads=4, linear_value_head_dim=4,
        linear_conv_kernel_dim=4, hc_count=2, hc_lowrank=4,
        ngram_size=2, ngram_vocab_size_base=100, split_ngram_parts=2,
        ple_conv_kernel_size=3, ple_embed_dim=16, heads_per_ngram=2,
        ple_layer_ids=[2], indexer_budget=8, indexer_compress_ratio=2,
        indexer_head_dim=4, indexer_kv_heads=1, indexer_n_heads=2,
        num_experts=8, num_experts_per_tok=2, moe_intermediate_size=6,
        shared_expert_intermediate_size=6, eos_token_id=31,
        rms_norm_eps=1e-6, partial_rotary_factor=0.5,
        rope_parameters={"rope_type": "default", "rope_theta": 10000},
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        output_gate_type="sigmoid")
    config._experts_implementation = "eager"
    model = Qwen4ExpForCausalLM(config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        embedding = torch.tensor([
            [(((token + 1) * (dim + 1)) % 17 - 8) / 16
             for dim in range(16)] for token in range(32)])
        lm_head = torch.tensor([
            [(((token + 3) * (dim + 5)) % 19 - 9) / 32
             for dim in range(16)] for token in range(32)])
        model.model.embed_tokens.weight.copy_(embedding)
        model.lm_head.weight.copy_(lm_head)
        input_ids = [1, 7, 9]
        output = model(torch.tensor([input_ids]), use_cache=False).logits[0]
        teacher = output.argmax(dim=-1).tolist()
        greedy = []
        generated = list(input_ids)
        for _ in range(2):
            logits = model(torch.tensor([generated]), use_cache=False).logits[0, -1]
            token = int(logits.argmax())
            greedy.append(token)
            generated.append(token)
    reference = {
        "source": f"transformers {transformers.__version__} Qwen4ExpForCausalLM eager expert path",
        "input_ids": input_ids,
        "teacher": teacher,
        "greedy": greedy,
        "last_logits": output[-1].tolist(),
    }
    args.output.write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")
    print(f"PASS Qwen3.8 Transformers oracle: {teacher} greedy={greedy}")


if __name__ == "__main__":
    main()
