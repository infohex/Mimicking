English | [中文](runtime-pitfalls.zh-CN.md)

# Runtime pitfalls and rules

The pitfalls hit while running multiple sessions, multiple threads and multiple GPUs on ONNX Runtime + CUDA, and the rules set as a result. Where the library can backstop, it already does (marked "handled by the library");
the rest must be followed by the integrating application (the test bench, any future integration). Source: measurements made on 2026-10-26/27 while doing real-time and multi-GPU work in MimickingLab.

## 1. ONNX Runtime allows only one environment per process — handled by the library

- Symptom: the second `ort::init_from(...)` reports `only one environment is allowed per process`. The test bench's "restart to change model" was this.
- Rule: create the runtime only once per process. `OnnxRuntime::load` now creates it the first time and returns the same environment on every later call (ignoring any newly passed dll path);
  changing device, reloading, opening another pool all use `rt.clone().with_device(...)`.
- Corollary: the onnxruntime.dll path only takes effect on the first load; changing the path requires restarting the process.

## 2. CUDA Graph capture must not run concurrently with any other CUDA call in the process — handled by the library (during loading)

- Symptom: loading two cards at the same time reports `CUDA failure 901: operation failed due to a previous error during capture`,
  on any node such as Gelu/Erf. While one thread is inside `cudaStreamBeginCapture`, another thread doing cudaMalloc, synchronous cudaMemcpy
  or other "non-capture-safe" calls invalidates the capture and triggers the error; **different cards are no exception**, because the check is process-wide.
- Rule: `BoundGraph::load` and `warm_up` (the second run is the capture) hold the process-wide `CAPTURE_LOCK`, so threads loading at the same time queue up automatically.
- The integrator must still observe: **load all models first, then start real-time inference**; do not load a new model while another thread is running graphs (the host↔device copies
  made while running land inside the capture window). The test bench stops all sessions and unloads the old pool before loading a new one for exactly this reason.
- Replay (`run_binding` reusing an already captured graph) is safe to run concurrently across threads, because replay does not capture.

## 3. One run is a synchronous call: one thread per GPU

- `Session::run` blocks until the result is ready. One thread polling several cards makes the cards take turns rather than work at the same time.
- Rule: one worker thread and one `Runner` per device (`Stages` are bound to the device they were loaded on); sessions (`Session`, no model) are assigned to the cards by
  `r % G`; the scheduling thread broadcasts one block and collects the results. The test bench's `pool.rs` is the reference implementation.
- A `BoundGraph`'s device-resident buffers belong to the thread it lives on; do not share one `Runner` across threads.

## 4. ORT thread pools spin by default — handled by the library

- Symptom: seven graphs, each with its own thread pool, keep spinning while idle, burning CPU and competing with the audio threads for cores.
- Rule: `session.intra_op.allow_spinning = 0`, likewise `inter_op`, 2 intra threads (under the CUDA EP there are almost no operators on the CPU side).

## 5. Static shapes: parameters cannot be changed at runtime

- Block length, right context and prompt length determine every graph's shapes; changing any of them means re-exporting a set of graphs (about 2 minutes) into its own directory, whose
  `manifest.json` records the configuration, read back by `ModelFiles::config()`. The integrator loads by directory and no longer hard-codes the geometry.
- The only graph with dynamic shapes is CAM++ (variable reference length), and it **cannot have CUDA Graph enabled** (shape operators fall back to the CPU EP). Its convolutions also run in cuDNN fallback mode
  in the ORT CUDA EP, about 0.2 s per call (about 25× real time), which is the bulk of the "about 350 ms per reference prompt"; batch embedding computation (training cache)
  uses the PyTorch version (150× real time). To speed up the inference side, export several static graphs bucketed by reference length.

## 6. A single stream makes the GPU downclock

- One stream is busy only about 30–50 ms out of every 240 ms, and the A4000 drops from 2100 MHz to 630 MHz; per-block p50 rises 1.5–2× and p95 shows spikes.
  Several concurrent streams (the GPU is never idle) or locked clocks (`nvidia-smi -i N -lgc 1800,1800`, administrator) eliminate it. Keep this in mind when evaluating single-stream latency.

## 7. No slow work on the inference thread

- Quality probes (CAM++ embedding of 10 s of audio, 100–500 ms), file IO and export subprocesses each get their own thread. Every pause of the engine thread
  means input backlog, speaker underrun and latency climbing one notch.

## 8. Gating decisions must be aligned to the output block they belong to

- With a 2.0 s right context, the output at this moment corresponds to the input from 2 seconds ago; gating it with the VAD decision of "this moment" chops speech to pieces. `Session` already queues
  decisions by the algorithmic delay (handled by the library); a custom VAD implementation must take the same care: the decision's time axis is the input, the gate's time axis is the output.

## 9. Audio device side

- WASAPI capture's "data discontinuity" flag can come from the device itself (some USB microphones raise it every few seconds, even with no model loaded); PortAudio does not report it;
  just count it, do not log every one (console writes in turn slow the capture thread down).
- The speaker queue needs a controller: production and consumption are equal on average, and the queue only drifts one way when "something happens"; pad with silence / drop input against the target margin, do nothing inside the dead band.

## 10. The regulator statistics of a codebook model must be frozen (2.0, 2026-10-31)

- The length regulator is Conv3 + GroupNorm(1, d), with statistics taken over the whole segment it receives. 2.0's codebook output is nearly constant (first-layer convolution output variance 0.0095, raw features 0.25),
  so statistics computed over a 40-frame block swing with the content, the normalised scale swings with them, and the DiT output smears over the whole segment: streaming WER 16 % while offline whole utterances are fine. Whole-utterance statistics differ only 1–3 % across utterances,
  so freezing them to constants with `--norm-stats` at export is enough (graph version 2.0.1, streaming 7.6 %). This applies to any codebook model; the diagnostic chain is in MimickingTrain `docs/despeaker-design.md` §13.5.
- Accompanying rule: offline WER does not stand for streaming; `tools/stream_diag.py --segmented` reproduces the Rust session block by block in PyTorch, matches the Lab, and must be run before exporting.

## 11. The energy gate cannot tell typing from speech

- Continuous keyboard typing and desk tapping have the same loudness dynamics as speech; on real recordings the energy gate opens on 75–95 % of blocks, and once open the model "invents" speech at full level (same with both weight sets).
  `SileroVad` (`mimic-onnx`, CPU 1 ms/block): keyboard 0/144, desk tapping 0/179, clean speech 32/32. It only separates speech / non-speech: other people's speech and some coughs are let through.
- The energy gate's floor is initialised from the first block's level: a trimmed file starts with speech → the floor is pinned to the speech level and the first seconds are silent; digital silence at the start of a recording → the floor is pinned at −96 dB
  and afterwards rises only 0.1–1 dB per block. Do not attach a gate to offline file conversion.
- Calling protocol of the Silero v5/v6 ONNX: 64 samples of context + 512 new samples (576) per call, with the (2,1,128) state carried along; without the context the probability is always 0.

## 12. Speaker queue: absorb clock drift with rate steering, not with padding silence / dropping input

- Microphone and speaker do not share a crystal (a USB microphone commonly differs from another speaker by 0.1–0.3 %), and the queue drifts a few milliseconds per second. Threshold-based padding / dropping cuts 80 ms about every 30 s,
  which sounds like stuttering. The right way: the queue error drives one stage of ±1 % variable-rate resampling (P term with a 4 s time constant + slow integral), and padding / dropping only backstops real hiccups (implemented in MimickingLab).
- With a single stream, GPU downclocking between blocks puts the per-block p95 60 ms above p50; a margin below that underruns; with locked clocks the difference is only 2 ms and the margin can go down to 50 ms.
- The "algorithmic delay" contains only right context + crossfade + SOLA search (80 ms for the low-latency set); the block length is already in the "accumulating one block" time, do not count it twice.

## 13. Microphone side

- The system's / driver's automatic gain pushes the gain down for several seconds after a burst of loud taps, which shows up as "cannot speak for a few seconds after typing"; neither the gate nor the model is to blame. The speaking level should be about −25 dBFS,
  −44 dBFS is too low; turn off "audio enhancements / noise suppression".
