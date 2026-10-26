---
tags:
  - onnx
  - audio
  - music
  - musical-key-estimation
  - onnxruntime
  - wasm
---

# S-KEY — ONNX for Key & Tempo

FP32 ONNX conversion of the pretrained [Deezer S-KEY](https://github.com/deezer/skey) model, used by [Key & Tempo](https://github.com/Hikari-Tsai/music-detection) to estimate a single global major or minor key from audio.

This is a community conversion of the upstream checkpoint, not a newly trained model or an official Deezer release. Hugging Face hosts the files; the application runs inference locally and does not send audio to Hugging Face.

## Files

| File | Purpose |
| --- | --- |
| `skey.onnx` | FP32, opset 17 model, 324,536 bytes |
| `manifest.json` | Hash, source commit, sample rate, ordered labels and parity results |
| `skey-keys.json` | Application mapping for all 24 classes |
| `LICENSE` | Original S-KEY copyright and MIT license text |
| `ConvNeXt-LICENSE`, `nnAudio-LICENSE` | Notices for architecture and preprocessing components |

## Input and output

Input `audio` is float32 mono PCM at **22,050 Hz**, shape `[1, samples]`. Output `scores` has shape `[1, 24]`. Peak normalization, VQT/dB preprocessing, frequency cropping and ChromaNet are included in the graph. Scores already include the upstream softmax; do not apply softmax again.

Use the exact label order in `manifest.json`: major classes start at A Major, minor classes start at B minor. G Major is index 10; A minor is index 22. Key & Tempo requires at least 3 seconds and rejects silent input before inference. This minimum is an application policy.

Browser inference uses ONNX Runtime Web WASM CPU. See the [export script](https://github.com/Hikari-Tsai/music-detection/blob/main/scripts/export_skey_onnx.py), [browser implementation](https://github.com/Hikari-Tsai/music-detection/blob/main/frontend/inference/key-engine.js) and [technical notes](https://github.com/Hikari-Tsai/music-detection/blob/main/docs/skey.md).

## Provenance and validation

- Source commit: `918b83d273568d5041569bb8068843d19a335726`
- Upstream checkpoint: `skey/models/skey.pt`
- Checkpoint SHA-256: `78dfd0ad4fa9434bf7cec70a25934b7c575bda9c80e994700140770ad3a5ead4`
- ONNX SHA-256: `50bac66d5b02868afad07a0b159f08495cc3f5d8492654587e828ed4686412b1`

The export rewrites affine-free LayerNorm using mean/variance operations and width-one adaptive average pooling using a time-axis mean for dynamic audio length. Checks compare the original model, rewritten copy and ONNX Runtime. The manifest records native/ONNX comparisons for approximately 3-, 25- and 75-second fixtures, with maximum score difference below 0.000004 on these examples.

These checks establish conversion consistency, not musical-key accuracy. The model estimates one global key, does not locate modulations or recognize chords, and its softmax scores are not calibrated confidence. Full-track inference can use substantial memory; the application's 20-minute limit does not guarantee every device can process that duration.

## License and attribution

Copyright (c) 2019-present, Deezer SA.

The [upstream README](https://github.com/deezer/skey/blob/918b83d273568d5041569bb8068843d19a335726/README.md#-license) explicitly describes the S-KEY **code** as MIT-licensed. That paragraph does not separately state a license for pretrained weights; this model card does not assign them a new license. The original license is preserved in `LICENSE`. Retain upstream copyright and license notices when redistributing covered material.

S-KEY includes ConvNeXt-derived code and nnAudio preprocessing. Their bundled license texts are also retained. Sources: [ConvNeXt](https://github.com/facebookresearch/ConvNeXt), [nnAudio](https://github.com/KinWaiCheuk/nnAudio).

## Citation

The upstream README asks research users to cite:

```bibtex
@INPROCEEDINGS{kongskey2025,
  author={Kong, Yuexuan and Meseguer-Brocal, Gabriel and Lostanlen, Vincent and Lagrange, Mathieu and Hennequin, Romain},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title={S-KEY: Self-supervised Learning of Major and Minor Keys from Audio},
  year={2025},
  pages={1-5},
  doi={10.1109/ICASSP49660.2025.10890222}}
```

Related references: [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545), CVPR 2022; [nnAudio](https://doi.org/10.1109/ACCESS.2020.3019084), IEEE Access 2020.
