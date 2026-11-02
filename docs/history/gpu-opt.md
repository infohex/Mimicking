English | [中文](gpu-opt.zh-CN.md)

# Export-side GPU optimisation: direct fp16, fused operators (2026-10-26)

## Conclusion

**DiT single step 4.95 → 2.10 ms (CUDA Graph), the 3.6 ms target exceeded.** HiFT body 14.3 → 10.2 ms. Single-stream whole block down from 82 ms to **48 ms p50 / 53 ms p95** (gate 80 ms); the PyTorch streaming version is 199 / 265. One A4000 carries 4 streams at 10 steps and 5 streams at 6 steps (block 0.25 s).

## Diagnosis

`tools/export/profile_graph.py` on the autocast-exported `dit_step.onnx`: 792 nodes, only 75 of them Gemm/MatMul, the rest Cast 93, Reshape 106, Mul 169, Transpose 48, Unsqueeze 57, Gather 36, Tile 18 (no-ops from `repeat_interleave(1)`). CUDA Graph removes the CPU launches, but each kernel still has a fixed GPU-side cost of a few microseconds, and about 790 kernels is more than 2 ms of pure waste. So the goal is not "faster operators" but "fewer operators".

## DiT: rewriting the forward pass over the same weights (`tools/export/export_dit_opt.py`)

The export wrapper `DiTClean` no longer calls the original seed-vc `forward`; it rewrites the same mathematics over the original seed-vc weights and emits ORT's fused operators directly through `torch.autograd.Function.symbolic`:

| Original seed-vc form | Emitted operator | Saved per layer |
|---|---|---|
| RMSNorm: Mul, ReduceMean, Add, Sqrt, Div, Mul, Cast | `SimplifiedLayerNormalization` (ORT internal operator in the default domain, fp32 accumulation inside) | about 6 |
| RoPE: paired reshape, 4 Mul, Stack, Flatten | `com.microsoft::RotaryEmbedding(interleaved=1)`, cos/sin tables baked into constants | about 10 × 2 |
| SDPA unrolled: Transpose ×4, MatMul ×2, Softmax, Tile ×2 | `com.microsoft::MultiHeadAttention(num_heads=6)` | about 9 |
| SiLU: Sigmoid, Mul | `com.microsoft::QuickGelu(alpha=1)` | 1 |
| The 93 Casts from autocast | whole graph fp16, Cast only at the 5 inputs and 1 output | 87 |

Result: 190 nodes (MatMul 51, Add 25, SimplifiedLayerNormalization 19, RotaryEmbedding 18, MultiHeadAttention 9, QuickGelu 9). The fused version (Euler update inside the graph) has 201.

Two accuracy points that must be respected:
- The timestep sine table `1000·t·freq` reaches about 1000 radians; fp16 resolution there is 0.5 radians, so `.half()` on the whole model destroys this table; the wrapper saves an fp32 copy before `.half()`, computes the embedding in fp32 and only then converts to fp16.
- The RoPE cos/sin cache must be fp16 like the activations, otherwise ORT reports inconsistent type parameters.

At export the ONNX checker does not know `SimplifiedLayerNormalization` in the default domain; `OperatorExportTypes.ONNX_FALLTHROUGH` skips the check.

| | Nodes | eager | CUDA Graph | Relative L2 against fixtures |
|---|---|---|---|---|
| autocast export | 792 | 8.3 ms | 4.95 ms | 8.8e-4 |
| rewritten + fused + fp16 | 190 | 3.0 ms | **2.10 ms** | 9.4e-4 |
| rewritten fp32 version (only verifies the rewrite) | | | | 7.1e-4 |

Fused graph, 10 steps (Python loop including host copies): 22.5 ms, output mel spectrogram relative L2 3.8e-4.

## HiFT body: pure fp16 + overflow-safe Snake (`tools/export/export_misc_opt.py`)

The autocast version's 425 nodes contain many Casts. Exporting directly after `.half()` gives NaN: in the Snake activation `x + sin²(αx)/(α+1e-9)`, α is on a linear scale, ranges from −2.48 to 2.30 and crosses zero; in fp16 the `1e-9` vanishes and `1/α` overflows. Changed to the mathematically equivalent `x + sign(α)·(sin(αx)·|α|^-½)²`, with `|α|^-½` precomputed in fp32 (maximum about 3.2e4, within fp16 range); all intermediates are bounded.

| | Nodes | eager | CUDA Graph | Relative L2 |
|---|---|---|---|---|
| autocast | 425 | 16.2 ms | 14.26 ms | 3.2e-4 |
| fp16 + safe Snake | 372 | 11.9 ms | **10.17 ms** | 6.8e-4 |

## XLSR: abandoned

The ORT transformers optimiser (bert mode) only matched GELU and SkipLayerNorm and did not match wav2vec2's attention pattern; CUDA Graph 6.20 → 5.96 ms, and after `convert_float_to_float16` the accuracy collapses (relative L2 1.12, probably overflow in the feature encoder's normalisation or the positional convolution). Of the 6 ms, only the attention part is worth chasing; it would need a hand-written wrapper emitting MHA as for DiT, gaining about 1 to 2 ms; not done for now.

## Whole block and multi-stream (Rust, `mimic`, A4000, block 0.25 s = 240 ms period)

| | Per-block p50 | p95 | content | solve | vocode |
|---|---|---|---|---|---|
| End of M2 | 81.9 | 85.1 | 6.3 | 52.0 | 23.0 |
| + optimised DiT graph | 53.7 | 59.2 | 6.3 | 23.2 | 23.9 |
| + optimised HiFT graph | **47.8** | **53.1** | 6.3 | 23.0 | 19.1 |

| Streams | Steps | Per-round p50 / p95 | Overruns |
|---|---|---|---|
| 3 | 10 | 158 / 168 | 0 |
| 4 | 10 | 194 / 206 | 0 |
| 5 | 6 | 222 / 235 | 0 |
| 6 | 6 | 242 / 257 | 26/42 |

All seven ORT parity tests pass (DiT single step 9.4e-4, 10 steps 3.8e-4, full HiFT chain 3.0e-3); end-to-end CAM++ cosine 0.669 (PyTorch 0.647).

**An operational observation**: a single stream under real-time pacing (busy only 48 ms every 240 ms) measures p50 94 ms, almost twice the full-speed value, with both content and solve doubling. This is the GPU downclocking while idle between blocks, not a code problem; with several streams running concurrently the GPU stays busy and it does not happen. For stable latency in a single-stream deployment, lock the clocks (`nvidia-smi -lgc`) or keep a constant load on the GPU.

## Files and workflow

- Export workflow: `export_all.py` (base graphs + fixtures) → `export_dit_opt.py --install` (two DiT graphs) → `export_misc_opt.py --skip-xlsr --install` (HiFT body). On install the original autocast graphs are backed up as `*.autocast.onnx` and the optimised graphs are also saved as `*_opt.onnx`.
- Remaining breakdown (48 ms): DiT 10 steps 21 + XLSR 6.2 + HiFT 10.6 + host DSP 7.7 + copies and scheduling about 2. Options further down: hand-written MHA wrapper for XLSR (about 1 to 2 ms), SIMD for the host DSP (about 3 ms), step distillation (model level).
