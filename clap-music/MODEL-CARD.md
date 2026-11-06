# LAION-CLAP music — audio embeddings for "Sounds like" (ONNX)

The audio encoder of [LAION-CLAP](https://github.com/LAION-AI/CLAP)'s music checkpoint
[`laion/larger_clap_music`](https://huggingface.co/laion/larger_clap_music) (revision
`a0b4534a14f58e20944452dff00a22a06ce629d1`, Apache-2.0), exported with
`tools/export-clap.py` from this repository. CLAP (Wu, Chen, Zhang, Hui, Berg-Kirkpatrick,
Dubnov — *Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and
Keyword-to-Caption Augmentation*, ICASSP 2023) learns one space for audio and text; this
checkpoint was trained further on music. MixCraft uses only the audio half: a 10-second
window goes in, a unit-length 512-dimensional vector comes out, and tracks whose vectors point
the same way sound alike (cosine similarity). That is what the Sounds like suggestion ranks by
(parity row `smart.radar`, ADR-0029).

The graph is `ClapModel.get_audio_features` (HTSAT audio encoder, audio projection, L2
normalisation) and nothing else; the RoBERTa text encoder is not included. The weights are
not changed. The ONNX graph matches PyTorch to 4.9e-7 (batch of three) and 6.5e-8 (one
window) on the export machine.

## Contract `embed-clap-mel64-48k`

| Tensor | Shape | Meaning |
|---|---|---|
| `mel` (in) | `[N, 1, 1001, 64]` float32 | one log-mel spectrogram per 10 s window (below); `N` is dynamic |
| `embedding` (out) | `[N, 512]` float32 | unit-length audio embedding per window |

Front end, as transformers' `ClapFeatureExtractor` does it for this checkpoint (`truncation =
rand_trunc`, `padding = repeatpad`; MixCraft reimplements it in `mixcraft_ml::clap` and checks
it against `reference.json`):

1. Mono floats in −1..1 at **48 000 Hz**; a window is **480 000 samples** (10 s). A clip
   shorter than that is repeated whole as many times as it fits, then zero padded at the end.
   The extractor cuts a longer clip at a random place; MixCraft takes several windows at fixed
   places instead and averages their embeddings.
2. Frames of **1024 samples every 480**, each centred on its sample (reflect padding of 512 at
   both ends; 1001 frames), **periodic Hann** window, **power** spectrum (513 bins).
3. Times the **64-band mel filterbank**: Slaney mel scale, Slaney area normalisation
   (`2 / (f[m+2] − f[m])`), **50–14 000 Hz**, triangles on the linear bin frequencies
   `k · 48000 / 1024`; floored at 1e-10.
4. `10 · log10(x)` (dB, reference 1, no top-dB clamp).

## Files

| File | Bytes | SHA-256 |
|---|---|---|
| `clap-music.onnx` (joined from `.001` … `.003`) | 275 836 874 | `f4e5c9d3b0f0a463ece8241420e75db1c8441278effc723f83342f3d9b042384` |
| `config.json` | 660 | `ac9265b771e635e4f4e9686411aaacddf963a177167e7fcdc2010b61b5a6cf3d` |

The parts are the graph cut at 95 MiB (`split -b 95m -d -a 3 --numeric-suffixes=1`), each
hashed in `SHA256SUMS`; MixCraft joins them and checks the joined file's hash before using
it. `reference.json` is the export's test vector: the mel rows and the embedding of a
deterministic 3.5 s signal (MixCraft's front-end test keeps a copy).

Exported 2026-11-06 with torch 2.15.0 (CPU), transformers 5.22.1, onnx 1.23.2, onnxruntime
1.31.1; ONNX IR 8, opset 17, plain ONNX operators only (runs on ONNX Runtime 1.20.1, the version
MixCraft ships, on the processor and through DirectML, CUDA and CoreML).

To reproduce: `python tools/export-clap.py <out dir>` in an environment with the packages
above, then split as described. The script downloads the pinned revision itself. Two runs on the
export machine wrote byte-identical files (same SHA-256).

## Licence and use

Apache License 2.0 (LAION); `LICENSE` next to this file carries the full text and the
statement that this graph is a format conversion. MixCraft (MIT OR Apache-2.0) downloads it
only when the user asks for it, keeps it in its model folder and computes the embeddings on
the user's computer; they stay in the user's library and are never sent anywhere.

Limits: CLAP was trained on general audio and on music with text descriptions, not on DJ
practice. Similar vectors mean similar timbre, instrumentation and style; they do not know
tempo or key, which MixCraft filters separately.
