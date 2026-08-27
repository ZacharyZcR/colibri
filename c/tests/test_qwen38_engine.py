#!/usr/bin/env python3
import argparse
import json
import re
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

summary = subprocess.run([
    str(args.binary.resolve()), str((args.fixture / "source").resolve()),
    "--experts", str((args.fixture / "experts").resolve()),
    "--ids", ids, "--greedy", "2", "--summary", "--benchmark",
], text=True, capture_output=True, check=True)
summary_lines = summary.stdout.splitlines()
assert summary_lines[0] == lines[0]
assert summary_lines[1:] == lines[2:]
match = re.fullmatch(
    r"QWEN38_BENCH load_s=([0-9.]+) teacher_tokens=(\d+) "
    r"teacher_s=([0-9.]+) teacher_tok_s=([0-9.]+) "
    r"decode_tokens=(\d+) decode_s=([0-9.]+) decode_tok_s=([0-9.]+)",
    summary.stderr.strip())
assert match, summary.stderr
assert int(match.group(2)) == len(reference["input_ids"])
assert int(match.group(5)) == len(reference["greedy"])
assert all(float(match.group(index)) >= 0.0 for index in (1, 3, 4, 6, 7))
print(f"PASS Qwen3.8 engine: token-exact, max logit error {error:.3g}")
