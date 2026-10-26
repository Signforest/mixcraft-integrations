# All-In-One — song structure, beats and downbeats (ONNX, dynamic length)

The `harmonix-all` ensemble of [All-In-One](https://github.com/mir-aidj/all-in-one) (Taejun
Kim, MIT; Kim & Nam — *All-In-One Metrical and Functional Structure Analysis with Neighborhood
Attentions on Demixed Audio*, WASPAA 2023), exported with `tools/export-allinone.py` from this
repository. The original runs on NATTEN's dilated neighbourhood attention, which has no ONNX
export; the script rebuilds the network in plain PyTorch with NATTEN's window arithmetic
(gathers with the same clamping at the sequence ends and the same relative-position-bias
indices) and puts the eight Harmonix fold checkpoints (`taejunkim/allinone` on Hugging Face)
into one graph that averages their logits. The gathered attention matches a literal
transcription of the NATTEN CPU kernel to 2e-7; the ONNX graph matches the PyTorch rebuild to
2e-6 at two sequence lengths. Float32, opset 17, 23039335 bytes.

## Contract `structure-allinone-81x100fps`

| Tensor | Shape | Meaning |
|---|---|---|
| `spec` (in) | `[4, T, 81]` float32 | per stem — bass, drums, other, vocals — madmom's log-filtered spectrogram (below); `T` is dynamic, at least 10 240 (pad with zeros) |
| `beat` (out) | `[T]` float32 | beat logits per frame (sigmoid → probability) |
| `downbeat` (out) | `[T]` float32 | downbeat logits |
| `section` (out) | `[T]` float32 | section-boundary logits |
| `function` (out) | `[10, T]` float32 | label logits: start, end, intro, outro, break, bridge, inst, solo, verse, chorus |

Spectrogram, as `allin1.spectrogram` (madmom): the stem as mono floats in −1..1 at 44 100 Hz,
frames of 2048 samples every 441 (100 frames per second), each centred on its reference
sample (zero padding at the edges, `ceil(samples / 441)` frames), symmetric Hann window,
magnitude of bins 0–1023 (the Nyquist bin dropped), times `filterbank.bin` (1024 × 81 float32,
row major: madmom's `LogarithmicFilterbank`, 12 bands per octave, 30–17 000 Hz, triangular,
normalised), then `log10(x + 1)`. Checked against madmom's processors to 3e-7.

Post-processing, as `allin1.postprocessing`: boundaries are local maxima of the section
probability (97-frame filter) that stand above the mean of the 12 s before and after; a
segment's label is the argmax of the mean softmax over its frames; beats and downbeats are
peaks above 0.19 / 0.22, a downbeat moved to its nearest beat.

## Files

| File | Bytes | SHA-256 |
|---|---|---|
| `allinone.onnx` | 23039335 | `bc8a8c0b914d4f81c2dd95e9a5a3a4a46c9f4cfb515c8c07542208b57c34bddb` |
| `filterbank.bin` | 331 776 | `04261bfd073b7f2ba71a13ff853f6dc7d6b03d499087b5cfbf6ca0053641cdad` |

Exported 2026-10-26 with torch 2.14.1 (CPU), onnx 1.23.2, madmom 0.17.dev0.

## Licence

MIT (code and weights, Taejun Kim); `LICENSE` next to this file. The weights were trained
on the Harmonix Set annotations (MIT).
