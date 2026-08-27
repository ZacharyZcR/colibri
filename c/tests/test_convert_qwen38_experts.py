import unittest

from tools.convert_qwen38_experts import (
    expert_rows,
    output_names,
    source_names,
    validate_source_shapes,
)


class Qwen38ExpertConverterTest(unittest.TestCase):
    def setUp(self):
        self.config = {"text_config": {
            "num_hidden_layers": 4, "mtp_num_hidden_layers": 1,
            "num_experts": 8, "moe_intermediate_size": 6,
            "hidden_size": 10,
        }}

    def test_text_and_mtp_rows_are_contiguous(self):
        self.assertEqual(expert_rows(self.config), [
            (0, "model.language_model.layers.0.mlp.experts"),
            (1, "model.language_model.layers.1.mlp.experts"),
            (2, "model.language_model.layers.2.mlp.experts"),
            (3, "model.language_model.layers.3.mlp.experts"),
            (4, "mtp.layers.0.mlp.experts"),
        ])

    def test_source_and_output_names_match_runtime_layout(self):
        self.assertEqual(source_names("mtp.layers.0.mlp.experts"), (
            "mtp.layers.0.mlp.experts.gate_up_proj",
            "mtp.layers.0.mlp.experts.down_proj"))
        self.assertEqual(output_names(4, 7), (
            "model.layers.4.mlp.experts.7.merged_weight",
            "model.layers.4.mlp.experts.7.qs"))

    def test_fused_shapes_are_strict(self):
        validate_source_shapes((8, 12, 10), (8, 10, 6), self.config)
        with self.assertRaisesRegex(ValueError, "gate_up shape"):
            validate_source_shapes((8, 6, 10), (8, 10, 6), self.config)
        with self.assertRaisesRegex(ValueError, "down shape"):
            validate_source_shapes((8, 12, 10), (8, 6, 10), self.config)


if __name__ == "__main__":
    unittest.main()
