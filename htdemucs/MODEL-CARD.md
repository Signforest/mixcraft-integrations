# Hybrid Transformer Demucs (htdemucs) — four-stem separation (ONNX, host STFT)

The `htdemucs` checkpoint of [Demucs v4](https://github.com/facebookresearch/demucs) (Meta
AI Research, MIT; Rouard, Massa, Défossez — *Hybrid Transformers for Music Source
Separation*, ICASSP 2023), exported with `tools/export-demucs.py` from this repository.
`torch.stft` / `torch.istft` have no ONNX export, so the graph is the network between them
and the host (MixCraft: `mixcraft_ml::demucs`) does the transforms. On random audio the graph
plus the host transforms reproduce the PyTorch model end to end to 3.8e-5. Float32, opset 17,
174 264 755 bytes, stored here in two parts (`htdemucs.onnx.001`, `.002`) that MixCraft joins
(SHA-256 of the joined file below).

## Contract `stems-htdemucs-cac4096`

| Tensor | Shape | Meaning |
|---|---|---|
| `mix` (in) | `[1, 2, 343980]` float32 | 7.8 s of the stereo mix at 44 100 Hz (shorter input zero-padded) |
| `spec` (in) | `[1, 4, 2048, 336]` float32 | the mix's spectrogram as channels: L re, L im, R re, R im |
| `spec_out` (out) | `[1, 4, 4, 2048, 336]` float32 | per source (drums, bass, other, vocals) the estimated spectrogram, same layout |
| `time_out` (out) | `[1, 4, 2, 343980]` float32 | per source the time-branch waveform |

Source waveform = iSTFT(`spec_out[s]`) + `time_out[s]`.

Spectrogram, as `HTDemucs._spec`: pad the 343 980 samples by 1536 on both sides (reflect),
STFT n_fft 4096, hop 1024, periodic Hann, normalized (÷ √4096), centred with reflect
padding; drop the last frequency bin (2049 → 2048) and the first two and last two frames
(340 → 336). Inverse as `_ispec`: append a zero bin and two zero frames on each side,
normalized iSTFT, cut 1536 samples from each end. Long tracks: 7.8 s segments with 25 %
overlap and a triangular weight (demucs `apply_model` without shifts).

## Files

| File | Bytes | SHA-256 |
|---|---|---|
| `htdemucs.onnx` (joined) | 174 264 755 | `1490b84d04e8fc2f98bbdca63b84cc18f085ea8597269a3474fda9ef05f1e2c9` |

Exported 2026-10-26 with torch 2.14.1 (CPU), demucs 4.1.0, onnx 1.23.2; the attention fast
path was disabled for the export (`torch.backends.mha.set_fastpath_enabled(False)`), which
changes no arithmetic.

## Licence

MIT (code and weights, Meta Platforms, Inc.); `LICENSE` next to this file.
