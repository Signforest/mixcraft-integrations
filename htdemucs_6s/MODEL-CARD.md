# Hybrid Transformer Demucs, six stems (htdemucs_6s) — ONNX, host STFT

The `htdemucs_6s` checkpoint of [Demucs v4](https://github.com/facebookresearch/demucs)
(Meta AI Research, MIT): drums, bass, other, vocals, **guitar, piano**. Exported like
`htdemucs` with `tools/export-demucs.py` (see `../htdemucs/MODEL-CARD.md` for the contract
`stems-htdemucs-cac4096`); the only difference is six sources: `spec_out [1, 6, 4, 2048, 336]`,
`time_out [1, 6, 2, 343980]`. Graph plus host transforms match the PyTorch model to 3.2e-5.
Demucs' authors note the piano source is weak and the guitar source bleeds; MixCraft's deck
folds guitar and piano into the INST pad.

| File | Bytes | SHA-256 |
|---|---|---|
| `htdemucs_6s.onnx` (joined from `.001`, `.002`) | 114 560 337 | `6b6eee09786a060f3246912affaca6115c5e5f59d43e5ba8b51197a9aceed1f2` |

Exported 2026-10-26 with torch 2.14.1 (CPU), demucs 4.1.0, onnx 1.23.2. Licence MIT
(`LICENSE` next to this file).
