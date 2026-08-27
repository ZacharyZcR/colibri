#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import unittest
from pathlib import Path

if __name__ != "__main__":
    raise unittest.SkipTest("command-line Qwen3.8 gateway driver")

parser = argparse.ArgumentParser()
parser.add_argument("--binary", type=Path, required=True)
parser.add_argument("--fixture", type=Path, required=True)
args = parser.parse_args()
source = (args.fixture / "source").resolve()
experts = (args.fixture / "experts").resolve()
payload = "<t001><t007><t009>".encode()
environment = dict(os.environ, SERVE="1", SERVE_BATCH="1", SNAP=str(source),
                   Q38_EXPERTS=str(experts), Q38_MAXT="8")
process = subprocess.Popen([str(args.binary.resolve()), "1"], stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment)
try:
    assert process.stdout.readline() == b"\x01\x01READY\x01\x01\n"
    assert process.stdout.readline().startswith(b"STAT ")
    process.stdin.write(f"SUBMIT request-1 0 {len(payload)} 1 0 1\n".encode()
                        + payload + b"\n")
    process.stdin.flush()
    assert process.stdout.readline() == b"ACCEPT request-1 3\n"
    assert process.stdout.readline() == b"DATA request-1 6\n"
    assert process.stdout.readline() == b"<t001>\n"
    done = process.stdout.readline().decode().split()
    assert done[:3] == ["DONE", "request-1", "STAT"]
    assert done[3] == "1" and done[7:] == ["3", "1"]
    process.stdin.write(f"SUBMIT request-2 0 {len(payload)} 1 0 1\n".encode()
                        + payload + b"\n")
    process.stdin.flush()
    assert process.stdout.readline() == b"ACCEPT request-2 3\n"
    assert process.stdout.readline() == b"DATA request-2 6\n"
    assert process.stdout.readline() == b"<t001>\n"
    assert process.stdout.readline().startswith(b"DONE request-2 STAT 1 ")
finally:
    process.stdin.close()
    process.wait(timeout=10)
assert process.returncode == 0, process.stderr.read().decode()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openai_server
from family_registry import family_by_id

openai_server.ARCH = "qwen38_flash_next"
engine = openai_server.Engine(args.binary.resolve(), source, cap=1, max_tokens=1,
                              env=environment,
                              family=family_by_id("qwen38_flash_next"))
chunks = []
try:
    stats = engine.generate(payload.decode(), 1, 0.0, 1.0, chunks.append)
finally:
    engine.close()
assert "".join(chunks) == "<t001>"
assert stats["prompt_tokens"] == 3 and stats["completion_tokens"] == 1
print("PASS Qwen3.8 gateway adapter: READY, ACCEPT, DATA, DONE")
