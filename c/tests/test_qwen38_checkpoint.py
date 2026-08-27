import unittest

from tools.check_qwen38_checkpoint import (
    expected_tensors,
    ple_padded_rows,
    validate,
)


def tiny_config():
    text = {
        "hidden_size": 16, "vocab_size": 32, "num_hidden_layers": 4,
        "num_experts": 4, "num_experts_per_tok": 2,
        "moe_intermediate_size": 8, "shared_expert_intermediate_size": 8,
        "hc_count": 2, "hc_lowrank": 4,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
        "indexer_n_heads": 2, "indexer_kv_heads": 1, "indexer_head_dim": 4,
        "linear_num_key_heads": 1, "linear_key_head_dim": 4,
        "linear_num_value_heads": 2, "linear_value_head_dim": 4,
        "linear_conv_kernel_dim": 3,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
        "ple_layer_ids": [2], "ple_embed_dim": 8, "ple_conv_kernel_size": 3,
        "ngram_size": 3, "heads_per_ngram": 2,
        "ngram_vocab_size_base": 11,
        "make_ngram_vocab_size_divisible_by": 8,
        "split_ngram_parts": 2,
        "mtp_num_hidden_layers": 1,
    }
    return {"model_type": "qwen4_exp", "text_config": text}


class Qwen38CheckpointContractTest(unittest.TestCase):
    def headers(self, config):
        text = config["text_config"]
        headers = {
            name: {"shape": shape, "dtype": dtypes[0]}
            for name, (shape, dtypes) in expected_tensors(text).items()
        }
        rows = ple_padded_rows(text)
        prefix = ("model.language_model.layers.1.ple.ple_embedding."
                  "ngram_embedding")
        headers[prefix + ".shard_0.weight"] = {
            "shape": [rows // 2, 2], "dtype": "BF16"}
        headers[prefix + ".shard_1.weight"] = {
            "shape": [rows - rows // 2, 2], "dtype": "BF16"}
        return headers

    def test_tiny_complete_contract(self):
        config = tiny_config()
        headers = self.headers(config)
        self.assertEqual(validate(config, headers, shapes=True), [])
        self.assertEqual(
            headers["model.language_model.layers.0.linear_attn.in_proj_qkv.weight"]["shape"],
            [16, 16])
        self.assertEqual(
            headers["model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight"]["shape"],
            [12, 16])
        self.assertEqual(
            headers["model.language_model.layers.3.self_attn.q_proj.weight"]["shape"],
            [32, 16])
        self.assertEqual(headers["mtp.pre_fc_norm_hidden.weight"]["shape"], [32])
        self.assertEqual(headers["mtp.layers.0.mlp.experts.gate_up_proj"]["shape"],
                         [4, 16, 16])

    def test_missing_and_wrong_shape_are_rejected(self):
        config = tiny_config()
        headers = self.headers(config)
        del headers["mtp.fc_hidden.weight"]
        headers["model.language_model.layers.0.mlp.gate.weight"]["shape"] = [3, 16]
        failures = validate(config, headers, shapes=True)
        self.assertTrue(any("missing mtp.fc_hidden.weight" in item for item in failures))
        self.assertTrue(any("invalid model.language_model.layers.0.mlp.gate.weight"
                            in item for item in failures))

    def test_ple_total_rows_are_checked(self):
        config = tiny_config()
        headers = self.headers(config)
        name = ("model.language_model.layers.1.ple.ple_embedding."
                "ngram_embedding.shard_1.weight")
        headers[name]["shape"][0] -= 1
        failures = validate(config, headers, shapes=True)
        self.assertTrue(any("invalid PLE row total" in item for item in failures))


if __name__ == "__main__":
    unittest.main()
