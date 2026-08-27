#!/usr/bin/env python3
import argparse
import json
import subprocess
import unittest
from pathlib import Path

if __name__ != "__main__":
    raise unittest.SkipTest("command-line Qwen3.8 engine oracle")

parser = argparse.ArgumentParser()
parser.add_argument("--binary", type=Path, required=True)
parser.add_argument("--fixture", type=Path, required=True)
args = parser.parse_args()
reference = json.loads((Path(__file__).with_name("qwen38_tiny_ref.json")).read_text())
ids = ",".join(map(str, reference["input_ids"]))
result = subprocess.run([
    str(args.binary.resolve()), str((args.fixture / "source").resolve()),
    "--experts", str((args.fixture / "experts").resolve()),
    "--ids", ids, "--greedy", "2",
], text=True, capture_output=True, check=True)
lines = result.stdout.splitlines()
teacher = list(map(int, lines[0].split()[1:]))
logits = list(map(float, lines[1].split()[1:]))
greedy = [int(line.split()[1]) for line in lines[2:]]
assert teacher == reference["teacher"], (teacher, reference["teacher"])
assert len(logits) == len(reference["last_logits"])
error = max(abs(actual - expected)
            for actual, expected in zip(logits, reference["last_logits"]))
assert error < 2e-5, error
assert greedy == reference["greedy"], (greedy, reference["greedy"])
print(f"PASS Qwen3.8 engine: token-exact, max logit error {error:.3g}")
