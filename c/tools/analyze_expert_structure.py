#!/usr/bin/env python3
"""Measure cross-expert structure in Colibri per-row INT4 MoE weights.

`extract` needs only the Python standard library and reads sampled submatrices
directly from safetensors offsets. `analyze` uses NumPy on the much smaller
artifact, so the model host does not need a Python ML environment.
"""

import argparse
import glob
import json
import os
import random
import struct
from pathlib import Path


MAGIC = b"COLIEXS1"


def tensor_index(model):
    out = {}
    for path in sorted(glob.glob(os.path.join(model, "*.safetensors"))):
        with open(path, "rb") as f:
            header_len = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(header_len))
        data_base = 8 + header_len
        for name, meta in header.items():
            if name != "__metadata__":
                out[name] = (path, data_base, meta)
    return out


def read_at(entry):
    path, base, meta = entry
    start, end = meta["data_offsets"]
    with open(path, "rb", buffering=0) as f:
        return os.pread(f.fileno(), end - start, base + start)


def sample_tensor(index, layer, expert, proj, rows, cols):
    stem = f"model.layers.{layer}.mlp.experts.{expert}.{proj}_proj.weight"
    weight = index[stem]
    scales = index[stem + ".qs"]
    scale_shape = scales[2]["shape"]
    if scales[2]["dtype"] != "F32" or len(scale_shape) != 1:
        raise ValueError(f"{stem}: expected per-row F32 scales")
    outputs = scale_shape[0]
    packed_bytes = weight[2]["data_offsets"][1] - weight[2]["data_offsets"][0]
    inputs = packed_bytes * 2 // outputs
    row_bytes = (inputs + 1) // 2
    if max(rows) >= outputs or max(cols) >= inputs:
        raise ValueError(f"{stem}: sample outside [{outputs}, {inputs}]")

    scale_blob = read_at(scales)
    path, base, meta = weight
    data_start = base + meta["data_offsets"][0]
    values = []
    with open(path, "rb", buffering=0) as f:
        for row in rows:
            packed = os.pread(f.fileno(), row_bytes, data_start + row * row_bytes)
            scale = struct.unpack_from("<f", scale_blob, row * 4)[0]
            for col in cols:
                byte = packed[col >> 1]
                q = (byte >> 4) if col & 1 else (byte & 15)
                values.append((q - 8) * scale)
    return outputs, inputs, values


def evenly_sample(size, count, seed):
    rng = random.Random(seed)
    bins = []
    for i in range(count):
        lo = i * size // count
        hi = (i + 1) * size // count
        bins.append(rng.randrange(lo, max(lo + 1, hi)))
    return bins


def extract(args):
    index = tensor_index(args.model)
    blocks = []
    payload = bytearray()
    for layer in args.layers:
        for proj_i, proj in enumerate(("gate", "up", "down")):
            first = f"model.layers.{layer}.mlp.experts.0.{proj}_proj.weight.qs"
            if first not in index:
                raise KeyError(f"missing {first}")
            outputs = index[first][2]["shape"][0]
            packed = index[first.removesuffix(".qs")][2]["data_offsets"]
            inputs = (packed[1] - packed[0]) * 2 // outputs
            rows = evenly_sample(outputs, min(args.rows, outputs),
                                 args.seed + layer * 101 + proj_i * 17)
            cols = evenly_sample(inputs, min(args.cols, inputs),
                                 args.seed + layer * 211 + proj_i * 29)
            offset = len(payload)
            for expert in range(args.experts):
                got_o, got_i, values = sample_tensor(
                    index, layer, expert, proj, rows, cols)
                if (got_o, got_i) != (outputs, inputs):
                    raise ValueError(f"shape drift at layer {layer} expert {expert} {proj}")
                payload.extend(struct.pack(f"<{len(values)}f", *values))
            blocks.append({
                "layer": layer, "projection": proj, "experts": args.experts,
                "outputs": outputs, "inputs": inputs, "rows": rows, "cols": cols,
                "samples_per_expert": len(rows) * len(cols), "offset": offset,
                "bytes": len(payload) - offset,
            })
            print(f"extracted layer={layer} projection={proj} "
                  f"shape={outputs}x{inputs} sample={len(rows)}x{len(cols)}")

    header = json.dumps({
        "model": os.path.realpath(args.model), "seed": args.seed,
        "blocks": blocks,
    }, separators=(",", ":")).encode()
    with open(args.output, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        f.write(payload)
    print(f"wrote {args.output}: {(len(payload) + len(header) + 16) / 2**20:.1f} MiB")


def load_artifact(path):
    import numpy as np

    with open(path, "rb") as f:
        if f.read(8) != MAGIC:
            raise ValueError("not a Colibri expert-structure artifact")
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        payload_base = 16 + header_len
    mm = np.memmap(path, mode="r", dtype="<f4", offset=payload_base)
    return header, mm


def analyze(args):
    import numpy as np

    header, payload = load_artifact(args.input)
    ranks = [r for r in args.ranks if r < 256]
    rng = np.random.default_rng(args.seed)
    results = []
    for block in header["blocks"]:
        e = block["experts"]
        m = block["samples_per_expert"]
        start = block["offset"] // 4
        x = np.asarray(payload[start:start + e * m]).reshape(e, m).astype(np.float64)
        mean = x.mean(axis=0, keepdims=True)
        centered = x - mean
        total = np.square(x).sum()
        mean_energy = e * np.square(mean).sum()
        gram = centered @ centered.T
        eigvals, eigvecs = np.linalg.eigh(gram)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[order], 0.0)
        eigvecs = eigvecs[:, order]
        normalized = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-30)
        similarities = normalized @ normalized.T
        offdiag = similarities[~np.eye(e, dtype=bool)]

        # Destroy expert identity independently at every sampled coordinate.
        # This preserves every coordinate's distribution and scale while
        # removing cross-coordinate expert structure; it is the right null for
        # deciding whether finite-sample PCA is finding more than noise.
        null = x.copy()
        for col in range(m):
            rng.shuffle(null[:, col])
        null_mean = null.mean(axis=0, keepdims=True)
        null_centered = null - null_mean
        null_eigvals = np.linalg.eigvalsh(null_centered @ null_centered.T)[::-1]
        null_eigvals = np.maximum(null_eigvals, 0.0)
        null_total = np.square(null).sum()
        null_mean_energy = e * np.square(null_mean).sum()

        rows = len(block["rows"])
        cols = len(block["cols"])
        activations = rng.standard_normal((args.activations, cols))
        exact = np.einsum("erc,ac->era", x.reshape(e, rows, cols), activations)
        exact_norm = np.square(exact).sum()
        rank_results = []
        for rank in ranks:
            u = eigvecs[:, :rank]
            reconstructed = mean + u @ (u.T @ centered)
            captured = 1.0 - np.square(x - reconstructed).sum() / total
            null_captured = (
                null_mean_energy + null_eigvals[:rank].sum()) / null_total
            approx = np.einsum(
                "erc,ac->era", reconstructed.reshape(e, rows, cols), activations)
            output_nrmse = np.sqrt(np.square(exact - approx).sum() / exact_norm)
            flat_exact = exact.reshape(e, -1)
            flat_approx = approx.reshape(e, -1)
            cosine = np.sum(flat_exact * flat_approx, axis=1) / (
                np.linalg.norm(flat_exact, axis=1) *
                np.linalg.norm(flat_approx, axis=1) + 1e-30)
            rank_results.append({
                "rank": rank, "weight_energy_captured": float(captured),
                "null_energy_captured": float(null_captured),
                "excess_over_null": float(captured - null_captured),
                "output_nrmse": float(output_nrmse),
                "output_cosine_mean": float(cosine.mean()),
                "output_cosine_p05": float(np.quantile(cosine, 0.05)),
            })
        result = {
            "layer": block["layer"], "projection": block["projection"],
            "shape": [block["outputs"], block["inputs"]],
            "sample_shape": [rows, cols],
            "mean_energy_fraction": float(mean_energy / total),
            "effective_rank_90": int(np.searchsorted(
                np.cumsum(eigvals), 0.90 * eigvals.sum()) + 1),
            "effective_rank_95": int(np.searchsorted(
                np.cumsum(eigvals), 0.95 * eigvals.sum()) + 1),
            "null_effective_rank_90": int(np.searchsorted(
                np.cumsum(null_eigvals), 0.90 * null_eigvals.sum()) + 1),
            "expert_cosine_mean": float(offdiag.mean()),
            "expert_cosine_p99": float(np.quantile(offdiag, 0.99)),
            "expert_cosine_max": float(offdiag.max()),
            "ranks": rank_results,
        }
        results.append(result)
        rline = " ".join(
            f"r{r['rank']}={100*r['weight_energy_captured']:.1f}%/"
            f"{100*r['excess_over_null']:+.2f}%null/"
            f"{100*r['output_nrmse']:.1f}%err" for r in rank_results)
        print(f"L{block['layer']:02d} {block['projection']:4s} "
              f"mean={100*result['mean_energy_fraction']:.2f}% "
              f"er90={result['effective_rank_90']}/{result['null_effective_rank_90']}null "
              f"cos={result['expert_cosine_mean']:+.4f}/"
              f"{result['expert_cosine_p99']:+.4f}p99 "
              f"{rline}")

    summary = {"source": header, "analysis_seed": args.seed, "results": results}
    Path(args.output).write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {args.output}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("extract")
    p.add_argument("model")
    p.add_argument("output")
    p.add_argument("--layers", type=int, nargs="+", default=[3, 18, 33, 48, 63, 77])
    p.add_argument("--experts", type=int, default=256)
    p.add_argument("--rows", type=int, default=32)
    p.add_argument("--cols", type=int, default=128)
    p.add_argument("--seed", type=int, default=20260730)
    p.set_defaults(func=extract)
    p = sub.add_parser("analyze")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--ranks", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64, 128])
    p.add_argument("--activations", type=int, default=32)
    p.add_argument("--seed", type=int, default=20260731)
    p.set_defaults(func=analyze)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
