English | [中文](capacity-a4000.zh-CN.md)

# Concurrent multi-stream performance evaluation template: RTX A4000 (16 GB)

Measured multi-stream concurrent capacity on one NVIDIA RTX A4000, as a baseline for estimating other cards. The A4000 is a workstation card on the Ampere GA104 core
(6144 CUDA cores, 16 GB GDDR6, 140 W), **comparable in compute to an RTX 3070** (same core; the 3070 has 5888 cores, 8 GB, 220 W),
so the stream counts here can be taken directly as an RTX 3070-class reference; cards with 8 GB also have room (about 1.9 GB per model set).

Test machine: two A4000s, Windows 11, onnxruntime-gpu 1.22, CUDA 12 / cuDNN 9. GPU 0 also drives the display.
Date: 2026-10-26/27. Tool: `MimickingLab` (`mimic-lab --headless`); the numbers are taken from its output.

## Definitions

- **Block**: 0.24 s (block length 0.25 rounded to 20 ms); every stream processes one block every 240 ms.
- **Per-stream per-block time**: wall-clock time of `Runner::push` (content encoding + length regulation + 10-step flow matching + vocoder + host DSP).
- **Round**: the total time for all streams on one card to each process one block. Streams on the same card run **sequentially**, and no stream's output comes out until the whole round has finished,
  so **the processing latency each stream experiences = one round, not its own block's time**. A round < 240 ms is what keeps blocks from being dropped; but the whole round counts towards latency,
  so a round of no more than half the period (about 120 ms) is recommended.
- Stream counts are total concurrency (1 foreground stream + N-1 background streams); all streams consume the same input and infer independently; with several cards they are assigned round-robin by stream number.
- 10-step flow matching, CFG 0.7, CUDA Graph on, fp16 fused-operator graphs.

## Two model sets

| Model set | Right context | Window | DiT sequence | Algorithmic delay | Single stream, one block (full speed) |
|---|---|---|---|---|---|
| Low-latency set (same geometry as the 2.0 `Mimicking-v2.0-lowlat`) | 0.02 s | 2.82 s | 328 | 0.32 s | 28 ms |

The low-latency set has half the window, each block costs nearly half, and timbre similarity is level (0.664 vs 0.649).

## Single-card capacity (A4000)

Low-latency set, real-time mode (microphone input, muted output), 20 s per group:

| Streams | First stream per block p50 / p95 | Round (= per-stream processing latency) | Software chain latency | GPU clock | Power | Usable |
|---|---|---|---|---|---|---|
| 1 | 44–60 / 113 ms | 44–60 ms | about 0.45 s | 630–1230 MHz (downclocked) | 35 W | Yes |
| 2 | 46 / 50 ms | 92 ms | about 0.48 s | 840 MHz | 40 W | Yes |
| 4 | 29 / 32 ms | 116 ms (p95 126) | about 0.55 s | 1935 MHz | 99 W | **Yes, recommended ceiling** |
| 6 | 29 / 38 ms | 172 ms (p95 183) | about 0.59 s | 1850 MHz | 120 W | Marginal, no margin |
| 8 | 28 / 30 ms | 231 ms (p95 242) | about 0.59 s, 6 blocks dropped in 20 s | 1800 MHz | 135 W | **No** |

Software chain latency (measured p50) = block 0.24 + one round + output margin 0.15 and its compensation, plus the algorithmic delay (0.32 s for the low-latency set) and the sound card latency.
"Dropped blocks" = the number of blocks where a round exceeded 240 ms (the test bench's “超时” (overrun) column).

Default set (2.0 s right context): 48 ms/block, 4 streams give a round of about 200 ms, likewise the ceiling; 38 ms/block with 6-step flow matching.

Conclusion: **each A4000 holds at most 4 concurrent streams**, for both the low-latency set and the default set. At 8 streams the 231 ms round hugs the period, p95 already crosses 240 ms,
6 blocks are dropped in 20 s, per-stream latency is 115 ms more than at 4 streams, and 135 W of power is up against the 140 W TDP; not usable in production.
The gain of the low-latency set at 4 streams is more margin and more stability (round 115 vs 200 ms), not more streams.

With a single stream the GPU idles and downclocks, so per-block time is actually the highest and jitter the largest; with several streams it drops to the full-speed value and stays stable; keep this in mind when evaluating single-stream latency
(or lock the clocks with `nvidia-smi -i N -lgc 1500,2100`).

## Two cards (2 × A4000)

| Configuration | First stream p50 / p95 | Round per card | Result |
|---|---|---|---|
| 8 streams, 4 per card | 30 / 32 ms | 119 / 121 ms | Zero overruns, half the margin left, **recommended** |
| 16 streams, 8 per card | 30 / 33 ms | 242 / 256 ms | Crosses the period, drops blocks, not usable |

The two cards are timed independently and run in parallel; the engine waits for the slowest one; capacity adds linearly per card: **4 streams per card, 8 streams on two cards**.
GPU 0 also drives the display, so its round is slightly longer than GPU 1's.

## Resources

- GPU memory: about 1.9 GB per model set per card (including the CUDA context), independent of the stream count (session state lives on the host).
- Power: roughly linear in the stream count, 8 streams 135 W (A4000 TDP 140 W).
- Host: engine thread + one worker thread per card, low CPU usage (the ORT thread pool does not spin).

## Estimating for other cards

Within a generation, scale by the product of CUDA core count and clock; across generations, scale by fp16 Tensor throughput; streams ≈ floor(120 ms / single-stream per-block time) (keeping a round within half the period;
A4000 low-latency set 120 / 28 ≈ 4).
GPU memory is not the bottleneck (≥ 4 GB is enough). The single-stream per-block time on the target card comes from one run of `mimic convert` (full-speed value),
or measure the stream count directly with the test bench's headless command:

```bash
# one card, 4 streams, real-time 20 s (muted output)
mimic-lab --headless --gpu 0 --load 3 --live 20 --mute
# two cards, 8 streams
mimic-lab --headless --gpus 0,1 --load 7 --live 20 --mute
```

In the output, `round p50/p95` is the round time, `misses` is the number of blocks where a round exceeded the period, and `sw latency` is the software chain latency; the “实时” (Live) and “负载” (Load) pages of the interface show the same.
