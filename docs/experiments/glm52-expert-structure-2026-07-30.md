# GLM-5.2 cross-expert structure experiment

Date: 2026-07-30

## Question

Can a trained GLM-5.2 expert be represented as a small set of shared linear
basis matrices plus small expert-specific coefficients/residuals, reducing the
weight traffic enough to justify TR-MoE?

This experiment tests the prerequisite directly on the real Colibri INT4
checkpoint. It does not infer structure from router frequency or from another
model.

## Model and method

- Checkpoint: `/data/models/GLM-5.2-colibri-int4`
- 76 ordinary MoE layers (`3..78`, with layer 78 excluded from the main sample
  because it is the wider MTP extension)
- 256 routed experts per layer
- Expert shapes:
  - gate/up: `2048 x 6144`
  - down: `6144 x 2048`
- Sampled ordinary layers: `3, 18, 33, 48, 63, 77`
- All 256 experts and all three projections in every sampled layer
- 32 stratified output rows x 128 stratified input columns per matrix
- Two independent coordinate seeds
- Per-row INT4 values were dequantized directly from the checkpoint's U8
  payload and F32 `.qs` scales.

For every layer/projection block, the analysis measures:

1. energy represented by the common expert mean;
2. rank-R PCA reconstruction across the expert axis;
3. the same PCA after independently shuffling expert identity at every sampled
   coordinate (finite-sample null);
4. normalized output error on random activation vectors;
5. pairwise cosine similarity between experts.

The extractor needs only Python's standard library on the model host. The
72 MiB sampled artifact is analyzed with NumPy elsewhere:

```bash
python3 c/tools/analyze_expert_structure.py extract \
  /data/models/GLM-5.2-colibri-int4 expert-samples.bin
python3 c/tools/analyze_expert_structure.py analyze \
  expert-samples.bin expert-structure-results.json
```

## Results

Aggregate values across 18 layer/projection blocks:

| Metric | Seed 1 | Seed 2 |
|---|---:|---:|
| Common-mean energy | 0.423% | 0.429% |
| Effective rank for 90% centered energy | 214.5 / 256 | 214.7 / 256 |
| Rank 32 captured energy | 18.95% | 18.93% |
| Rank 32 excess over shuffled null | 0.73% | 0.71% |
| Rank 32 output NRMSE | 90.05% | 90.05% |
| Rank 64 captured energy | 34.89% | 34.85% |
| Rank 64 excess over shuffled null | 0.94% | 0.90% |
| Rank 64 output NRMSE | 80.69% | 80.71% |
| Rank 128 captured energy | 61.88% | 61.82% |
| Rank 128 excess over shuffled null | 0.91% | 0.84% |
| Rank 128 output NRMSE | 61.75% | 61.80% |

Expert pairwise cosine similarity is effectively zero in every sampled block.
The 99th percentile is only about `0.036..0.040`.

Layer 3 is the only meaningful exception to the shuffled null: depending on
rank and projection it has roughly 3.2-4.8 percentage points of excess captured
energy. That is still unusably weak: rank 32 leaves about 88% output error.
Layer 77 has less than one percentage point of excess structure. The middle
layers are statistically almost indistinguishable from the shuffled-expert
null.

The high-rank sweep makes the storage boundary explicit:

| Rank | Basis storage floor vs 256 independent experts | Typical output NRMSE |
|---:|---:|---:|
| 128 | ~50% | ~62% |
| 192 | ~75% | ~41% |
| 224 | ~87.5% | ~28% |
| 240 | ~93.8% | ~19% |
| 248 | ~96.9% | ~12-13% |

The storage column is already optimistic: it ignores coefficients, metadata
and any residual needed to recover quality.

## Verdict

**Rejected for frozen-weight linear factorization.**

The raw trained GLM-5.2 experts do not contain enough cross-expert linear
structure to support the proposed `global basis + group basis + small
residual` representation without retraining. Most apparent PCA compression is
the finite-sample spectrum expected from a random matrix, not learned shared
expert structure.

The original TR-MoE gate required at least 35% expert-data reduction with less
than 1% quality loss. The measured weight/output errors miss that gate by a
large margin. Even retaining 248 of 256 expert directions saves at most about
3% before overhead and still produces double-digit output error.

This rejects:

- a shared mean plus small per-expert delta;
- a small global linear expert basis;
- a small per-layer linear expert basis;
- the same representations applied independently to gate/up/down.

It does **not** reject representations learned with retraining, nonlinear or
activation-conditioned expert similarity, or lossy task-specific expert
substitution. Those are different hypotheses and require a quality dataset,
not another raw-weight PCA sweep.

