---
license: mit
library_name: onnxruntime-web
pipeline_tag: audio-classification
tags:
  - audio
  - beat-tracking
  - downbeat-tracking
  - tempo-estimation
  - music-information-retrieval
  - onnx
  - onnxruntime-web
  - webgpu
---

# Beat This! — beat & downbeat tracking (ONNX, WebGPU)

ONNX export of **Beat This!**, the JKU beat/downbeat tracker (checkpoint
`final0`), packaged for the [`musetric`][musetric] `packages/ai` runtime on
`onnxruntime-web`'s **WebGPU** execution provider.

The graph covers the network only:

```text
beat_this.onnx  spect [windows, 1500, 128] → beat, downbeat [windows, 1500]
```

Feature extraction is the host's job, and `mel-filterbank.bin` is what it needs
to do it: the host computes the log-mel spectrogram on the GPU (STFT + this
filterbank + `log1p`) and feeds windows straight into the graph as GPU buffers,
never round-tripping features through the CPU.

**Feed one 1500-frame window per call** — the author's window, 30 s of audio.
Longer tracks are covered by overlapping windows in the host (`chunkSize` 1500,
`borderSize` 6). Batching every window into a single call materializes a huge
time-attention tensor; one window per call is why the graph takes a window axis
at all. Short clips must be padded to 1500 frames.

The export substitutes no algorithm. The one construct that cannot be traced —
`PartialFTTransformer.forward`'s `b = len(x)`, which would freeze the window
count into the graph — is re-expressed with `x.shape`, and is bit-identical to
the original in torch. So the graph carries **no approximation** and its parity
does not depend on the material.

## Intended uses & limitations

**Intended:**
- Beat and downbeat times for a music track, as a stage in an audio pipeline;
  tempo and meter are derived from them host-side.
- GPU inference through `onnxruntime-web`. The wasm EP also works and gives
  identical results, at roughly 12× the wall clock.

**Out of scope:**
- Use in other training frameworks — this is an inference-only export.

**Limitations:**
- **The host owns feature extraction, and the contract is exact.** 22050 Hz mono;
  channels downmixed as the **arithmetic mean**, matching the reference
  `Audio2Frames`; STFT with `n_fft` 1024, hop 441, a **periodic Hann** window,
  `center=True` with reflect padding; magnitude divided by `sqrt(n_fft)`
  (torchaudio's `normalized='frame_length'`); projected onto
  `mel-filterbank.bin`; then `log1p(1000 · x)`. `config.json` records all of it.
- **`ffmpeg -ac 1` is the wrong downmix.** It is an energy-preserving rematrix
  and comes out √2 louder; because features are `log1p(1000 · mel)`, that gain is
  *not* a constant offset and will move beats. `config.json` says
  `"downmix": "mean"`.
- **The host also owns the chunk layout.** The graph does not know about
  `chunkSize`, `borderSize` or the shifted final window. Reproduce `split_piece`
  / `aggregate_prediction` (`keep_first`) or results drift at window seams.
- Validate on the material fed in production. For this pipeline that is the
  **instrumental stem**.
- Training-data provenance of the upstream checkpoint is not documented here.

### Known defect: onnxruntime WebGPU on Adreno, and the static rewrite (2026-09-12)

Strict on-device verification (one recorded input, wasm EP of the same build as
the bit-exact reference, comparator validated by negative controls) showed the
WebGPU EP returns **silently wrong fp32 logits** for the original dynamic graph
on mobile Adreno:

- Adreno 660: shared-tile `Transpose`/`Conv` (a quarter of the output left as
  zeros) and inference `BatchNormalization` (~50% mismatch, nondeterministic).
- Adreno 660 and 750: the time attention at 1500 frames. Every attention core
  keeps its scores in one `[b, h, 1500, 1500]` tensor — 137 MiB in the main
  transformer, 275 MiB in a frontend block — and that does not fit one storage
  binding: the Adreno 750 (Galaxy S24) reports `maxStorageBufferBindingSize`
  128 MiB and returns garbage above it, the Adreno 660 (OnePlus 9RT) reports
  512 MiB yet breaks between 137 and 275 MiB. Isolated attention chains are
  exact on both up to 120 MiB. Desktop NVIDIA is exact on the same input.

`beat_this.onnx` ships the **static rewrite at 1500 frames**: broken Transpose
families become `Reshape -> Gather(const int32 idx) -> Reshape`, every Conv
becomes Pad + im2col Gather + MatMul, the stem BatchNorm is folded to Mul+Add,
and every attention core is cut along its head axis into groups of 68.7 MiB
(`Slice`, then MatMul → Softmax → MatMul per group, then `Concat`), which
changes no arithmetic. Verified 2026-10-08 on a Galaxy S24 and a OnePlus 9RT
against the CPU provider on real log-mel windows: within **7.5e-5**,
deterministic, the two GPUs agreeing with each other; on the CPU provider it is
within 4.7e-5 of the torch model. Rebuild: `capture_shapes.py --frames 1500` +
`rewrite_static_adreno.py` + `split_attention_webgpu.py` in musetric-toolkit
`scripts/onnx/beat_this/`. The host must always feed `[1, 1500, 128]`.

From 2026-09-12 until this revision the graph shipped at **513 frames**: the
failure at 1500 had been taken for a MatMul defect, and 513-frame windows kept
the scores small. They also gave the model a third of the context it was
trained with, and without drums it lost the beat far more often (see
Validation).

## How to use

```js
import * as ort from 'onnxruntime-web/webgpu';

const session = await ort.InferenceSession.create('beat_this.onnx', {
  executionProviders: ['webgpu'],
});

// window: Float32Array(1500 * 128), one log-mel chunk (see config.json).
const { beat, downbeat } = await session.run({
  spect: new ort.Tensor('float32', window, [1, 1500, 128]),
});
```

`mel-filterbank.bin` is a raw row-major float32 matrix, `[513, 128]` — load it
with `new Float32Array(await (await fetch(url)).arrayBuffer())` and project
magnitudes onto it.

Beat times come from the logits host-side, with either postprocessing of the
reference. `minimal` keeps maxima within ±3 frames (`peakKernel` 7) whose logit
exceeds `peakThreshold` 0, merges peaks ≤ `deduplicateWidth` frames apart,
converts frames to seconds at `fps` 50 and snaps each downbeat to its nearest
beat. `dbn` runs madmom's `DBNDownBeatTrackingProcessor` (3 or 4 beats per bar,
55–215 BPM, `transition_lambda` 100) on the sigmoid probabilities: it keeps one
tempo through passages where the network is unsure, where `minimal` clicks at
random. The `musetric` host runs `dbn`, ported to TypeScript and equal to
madmom's output; see `packages/ai` (`runtime/rhythm/`,
`rhythm/downbeatTracker.ts`) for the full pipeline.

## Files

| File | Size | SHA256 |
|---|---|---|
| `beat_this.onnx` | 190,649,026 B | `1337dc6c21257ed6b803418efc320df42b5d8eb296ef16648ef60457cd4b9e55` |
| `mel-filterbank.bin` | 262,656 B | `1ee975d96f44ccf2c3bfe37825c1c1f0b089f5703c7a12a84b1f0a3bce004533` |
| `config.json` | 1,012 B | `966f271e9a44c8f1b9e438622b34165ecdf6df9adedc8a9c0a6f7fd9d3458cc2` |

`mel-filterbank.bin` is torchaudio's own `MelScale.fb` written verbatim, so a
host reuses the reference filterbank instead of reimplementing the slaney mel
scale. The analysis window is deliberately *not* shipped: it is a plain periodic
Hann, `0.5 - 0.5·cos(2πn/N)`, and the export refuses to run if the reference
window ever stops matching that.

**The graph ships `Slice` where the export emits `Split`.** onnxruntime's
WebGPU `Split` kernel fails to build a compute pipeline on Adreno: the graph
loses its very first `Split` with `CreateComputePipelines failed with
VK_ERROR_UNKNOWN`, the session falls back to the CPU provider, and the tracker
still returns beats with nothing in the log naming the cause. A shader that
will not compile, not a binding too large — tensor sizes have nothing to say
about it. So all 37 fixed-size `Split` nodes are lowered to one static `Slice`
per output, reading the split sizes off the `Constant` that fed the node and
writing them as `starts`/`ends`/`axes` initializers. Same partition of the same
axis, no weight read, 34,463 bytes larger.

**Signature** — float32 weights, opset `ai.onnx` 17:

| Tensor | Type | Shape | Meaning |
|---|---|---|---|
| `spect` (in) | float32 | `[windows, 1500, 128]` | one 1500-frame log-mel window per call |
| `beat` (out) | float32 | `[windows, 1500]` | beat logits; `> 0` is a candidate |
| `downbeat` (out) | float32 | `[windows, 1500]` | downbeat logits |

## Validation

**This revision, end to end (2026-10-08).** The musetric parity chain — the
author's inference, the torch reference, this graph on the ONNX CPU provider,
wasm, WebGPU and the product's own runtime — on three cases of a CC BY 3.0 track
(the whole track, a short excerpt and 30 s of silence), on a desktop NVIDIA
GPU, a Galaxy S24 and a OnePlus 9RT:

| Boundary | Desktop | Galaxy S24 | OnePlus 9RT |
|---|---|---|---|
| WebGPU logits vs ONNX CPU, max abs | 6.9e-5 | 6.0e-5 | 6.0e-5 |
| product logits vs reference, max abs | 1.5e-4 | 1.1e-4 | 1.3e-4 |
| product beats / downbeats vs reference, F-measure | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| tempo and meter | equal | equal | equal |

**Why 1500 frames.** Over 26 instrumentals of an open corpus (CC-licensed
tracks, 105 min) against 513-frame windows of the same network, both with the
`minimal` postprocessing: spurious beats (an interval shorter than 0.8 of the
local beat) drop from 405 to 316, and gaps (longer than 1.25 of it) from 556 s
to 492 s. 14 tracks improve, 7 get worse and 5 do not change by more than half
a percentage point of track time. The window the author trained with keeps the
beat through drumless passages that a 10 s window loses.

The tables below were measured on the earlier dynamic graph and carry over to
the network itself; the rewrite adds float32 reordering noise under 5e-5.

**The graph**, against the PyTorch `File2Beats` reference sharing its log-mel and
`minimal` postprocessing, over 20 instrumental stems (the production input,
10 s–285 s):

| Metric | Value |
|---|---|
| tracks with identical beat **and** downbeat counts | **20/20** |
| max beat / downbeat time difference | **0.0 s** |

Beat times are **bit-identical**: logits agree to ~1e-5 and peak picking
quantizes to 20 ms frames, so the residual cannot move a beat.

**The WebGPU front end**, against torchaudio's `LogMelSpect` on the same
waveforms:

| Metric | Value |
|---|---|
| frame counts | **20/20 exact** |
| max absolute difference | **3.2e-04** (relative to peak: 3.8e-05) |
| mean absolute difference | ~1e-06 |

That residual is float32 GPU arithmetic, not a different transform, and it
changes **nothing** downstream: the WebGPU path reproduces the wasm path's
results **byte for byte on 20/20 tracks**.

**End to end**, the shipped host (ffmpeg mean-downmix decode + WebGPU) against
the Python CLI it replaces (torchaudio + `soxr`), over the same 20 stems:

| Metric | Value |
|---|---|
| `bpm` — the user-visible tempo | **20/20 identical** |
| `meter` | **20/20 identical** |
| beat F-measure (±70 ms) | mean **0.9982**, median 1.0000, min 0.9898 |
| downbeat F-measure (±70 ms) | mean **0.9962**, median 1.0000, min 0.9722 |
| beats differing over the whole set | 20 missed + 9 extra of 7,853 (**0.37 %**) |

Neither the graph nor the front end causes that, and neither does gain: the mean
downmix matches the reference to a ratio of 1.0000 and the two decodes correlate
at 0.99998 with zero sample shift. What is left is the **resampler** — ffmpeg's
`swr` versus `soxr` — and peak picking turns that into a hard decision, so a
handful of marginal beats near the `logit > 0` threshold flip. Tempo and meter
are medians over hundreds of beats and absorb it. Feeding the reference's own
decode through this stack reproduces the CLI exactly, which is what pins the
residual on the decoder.

The tables above were measured before the `Split` → `Slice` lowering. The two
graphs return bit-identical `beat` and `downbeat` logits on the CPU provider on
the same input, at one window and at three, so the numbers carry over.

Re-run the parity gate on the exact published bytes before relying on it.

## Source & lineage

Code license and weight license are separate; ONNX conversion does not change the
weight license. Documented only as far as it is verifiable.

- **Architecture:** log-mel front end (`n_fft` 1024, hop 441, 128 mels,
  `log1p(1000 · x)`) → a conv stem and three frontend blocks, each a partial
  transformer attending across frequency then time → six rotary transformer
  layers (dim 512, 16 heads) → a sum head emitting beat and downbeat logits.
  20.25 M parameters.
- **Reference implementation and weights:** [`CPJKU/beat_this`][upstream],
  **MIT** (per its `LICENSE`, Copyright (c) 2024 Institute of Computational
  Perception, JKU Linz, Austria), consumed as the `beat-this` package (1.1.0).
- **Paper:** Foscarin, Schlüter, Widmer — *Beat this! Accurate beat tracking
  without DBN postprocessing*, ISMIR 2024.
- **Checkpoint:** `final0` — `beat_this-final0.ckpt`, 81,058,141 B, sha256
  `8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331` — fetched at
  export time from the upstream [checkpoint host][ckpt] via `torch.hub`, as the
  package's `load_checkpoint` does. Upstream serves it from a cloud share rather
  than a content-addressed URL, so the sha256 above is the pin.
- **Export tooling:** `scripts/onnx/beat_this` in [musetric-toolkit][toolkit];
  see its [`thirdPartyNotices.md`][notices].
- **Host runtime:** `packages/ai` in [`musetric`][musetric].

This export preserves the upstream **MIT** license; we do not claim authorship of
the original weights.

## License

MIT, inherited from the upstream weights.

[upstream]: https://github.com/CPJKU/beat_this
[ckpt]: https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt
[toolkit]: https://github.com/popelenkow/musetric-toolkit
[notices]: https://github.com/popelenkow/musetric-toolkit/blob/main/thirdPartyNotices.md
[musetric]: https://github.com/popelenkow/musetric
