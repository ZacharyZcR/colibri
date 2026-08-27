# Qwen3.8-Flash-Next

Colibri supports the text-only `qwen4_exp` / `qwen4_exp_text` checkpoint with
the CPU runtime `qwen38_flash_next`. The runtime executes the official hybrid
layout: DeltaNet linear-attention layers, QSA full-attention layers, mHC,
PLE, routed MoE, the final global mixer, and the LM head. Vision and tool-call
parsing are not part of this first runtime.

The official dense, QSA, and PLE tensors remain in the source checkpoint. PLE
tables are memory-mapped. Routed experts are converted once to a compact
overlay and streamed from that overlay during decode, so the 512-expert source
tensors are not copied into resident memory.

## Prepare and run

Build the CPU engine and create the default overlay location:

```sh
make -C c qwen38_flash_next
python3 c/tools/convert_qwen38_experts.py \
  --model /models/Qwen3.8-Flash-Next \
  --out /models/Qwen3.8-Flash-Next/qwen38_experts \
  --ebits 4
```

`coli chat`, `coli serve`, and the OpenAI-compatible API then resolve the
engine from `config.json`. If the overlay is elsewhere, set
`Q38_EXPERTS=/path/to/overlay`. `Q38_MAXT` controls the allocated context
capacity and defaults to 8192.

## Routed-expert acceleration

CUDA, HIP and Metal builds can execute the routed expert matrices on the GPU.
The default per-row overlay (`--group-size 0`) is required; grouped overlays
remain correct but fall back to the CPU path. CUDA and HIP use the fused expert
pipeline and retain recently used experts in a device-resident LRU cache. A
cache hit skips both the overlay read and weight upload. Metal uses the shared
int8 matmul backend but keeps streaming because its zero-copy tensor handles
depend on the lifetime of the host allocation.

```sh
# NVIDIA on Linux
make -C c CUDA=1 qwen38_flash_next
COLI_CUDA=1 COLI_GPU=0 c/qwen38_flash_next MODEL --experts EXPERTS --ids 1,2,3

# AMD on Linux
make -C c HIP=1 HIP_ARCH=gfxNNNN qwen38_flash_next
COLI_CUDA=1 COLI_GPU=0 c/qwen38_flash_next MODEL --experts EXPERTS --ids 1,2,3

# Apple Silicon
make -C c METAL=1 qwen38_flash_next
COLI_METAL=1 c/qwen38_flash_next MODEL --experts EXPERTS --ids 1,2,3
```

`Q38_GPU_EXPERT_GB` sets the CUDA/HIP cache budget per process. It defaults to
`auto`, which reserves 1 GiB of reported free device memory for other runtime
state; set it to `0` to disable residency. Shutdown telemetry reports cache
hits, misses, uploads and evictions.

This stage accelerates routed experts only. Dense projections, DeltaNet, QSA,
mHC, PLE and the global mixer still execute on the CPU. Full-model GPU
residency and throughput claims therefore remain out of scope.

For a tokenizer-independent diagnostic, pass token IDs directly:

```sh
c/qwen38_flash_next /models/Qwen3.8-Flash-Next \
  --experts /models/Qwen3.8-Flash-Next/qwen38_experts \
  --ids 1,2,3 --greedy 8
```

For an official checkpoint acceptance run, use the maintainer gate. It checks
all source tensor names, shapes and dtypes, opens the converted overlay, runs
teacher and cached greedy decode, suppresses the full vocabulary dump, and
prints a machine-readable timing line:

```sh
make -C c qwen38-full-check \
  QWEN38_MODEL=/models/Qwen3.8-Flash-Next \
  QWEN38_EXPERTS=/models/Qwen3.8-Flash-Next/qwen38_experts \
  QWEN38_ACCEPT_IDS=1,2,3 QWEN38_ACCEPT_GREEDY=2
```

The result includes `QWEN38_BENCH load_s=... teacher_tok_s=...
decode_tok_s=...`. Record the build flags, accelerator environment and hardware
with that line; the gate deliberately does not turn one machine's output into
a repository-wide performance claim.

## Correctness boundary

`make -C c qwen38-tiny-check` generates a four-layer hybrid checkpoint and
checks teacher tokens, logits, cached greedy decode, runtime reset, the child
serve protocol, and the real Python gateway adapter. Its committed reference
is reproducible with official Transformers 5.16.1 via
`c/tools/make_qwen38_ref.py`. ASan/UBSan covers the same executable and gateway
flow in development. Mock-backed CUDA/HIP and Metal gates prove accelerator
selection, exact int8 dispatch, fused CUDA/HIP expert execution, LRU
hit/eviction behavior, and grouped overlay fallback without requiring CI
runners with GPUs.

The engine-owned Segment adapter advertises the expanded mHC stream as its
boundary state, opens only the requested layer range, keeps QSA, DeltaNet and
PLE state isolated per session, and supports transactional streamed snapshots.
The same tiny gate compares a full range with two chained ranges and verifies
snapshot continuation.

The tiny oracle proves implementation parity at the covered geometry. It is
not a performance or quality claim for the 125B-A6B checkpoint. Full-model
throughput, long-context stability, and hardware-specific acceptance still
require measurement on the target machine.
