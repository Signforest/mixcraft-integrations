# Open-Unmix UMX-HQ — four-stem separation (ONNX)

Four graphs, one per stem (vocals, drums, bass, other), exported from the
[Open-Unmix](https://github.com/sigsep/open-unmix-pytorch) `umxhq` weights (trained on the
uncompressed MUSDB18-HQ, released under the MIT licence by Inria / Fabian-Robert Stöter and
Antoine Liutkus) with `tools/export-openunmix.py` from this repository. The weights come
from Zenodo record 3370489 through `openunmix.umxhq_spec`; the graphs reproduce the PyTorch
model to 1.3e-7 on random spectrograms and accept any number of frames.

**Not** the `umxl` weights: those are CC BY-NC-SA 4.0 and are not shipped.

## Contract `stems-umx-mag4096`

| Tensor | Shape | Meaning |
|---|---|---|
| `mag` (in) | `[1, 2, 2049, frames]` float32 | magnitude spectrogram of the stereo mix at 44 100 Hz: STFT n_fft 4096, hop 1024, periodic Hann, centre (reflect) padding, \|X\| |
| `est` (out) | `[1, 2, 2049, frames]` float32 | the stem's magnitude estimate |

The host runs the four graphs on the same input, forms ratio masks (`est_i / Σ est_j`,
floored), multiplies the complex mix spectrogram by each mask and inverse-transforms with the
same window and hop (MixCraft: `mixcraft_ml::stems`). A mono input is duplicated to two
channels; other sample rates are resampled to 44 100 Hz first. Open-Unmix's optional Wiener
filtering iterations are not part of the graphs; MixCraft's host applies one Wiener-style
refinement when asked for the slower quality setting.

## Files

| File | Bytes | SHA-256 |
|---|---|---|
| `umxhq-vocals.onnx` | 35 625 978 | `2fb10092822313c00fcd7977ecd35b07ee08fc6d9b7d69aa90a0ab46e5e191eb` |
| `umxhq-drums.onnx` | 35 625 978 | `15e5d173137279623715f7f00d8494071091e80509b9964beb3505fe1af21348` |
| `umxhq-bass.onnx` | 35 625 978 | `2143bab3feebe71f462eefeb2aeb226fb3640d353577417f1bd4abcb9ed9fb28` |
| `umxhq-other.onnx` | 35 625 978 | `ee1fbaec703a5ac8dac2927128c097d7cd06d25979d022dfe9cf7bedb15deab1` |

Opset 17, float32, exported with torch 2.14.1 (CPU) and onnx 1.23.2 on 2026-10-26.

## Licence

MIT (the Open-Unmix code and the UMX-HQ weights); `LICENSE` next to this file. Paper:
Stöter, Uhlich, Liutkus, Mitsufuji — *Open-Unmix – A Reference Implementation for Music
Source Separation*, JOSS 2019.
