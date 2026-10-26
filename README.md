# mixcraft-integrations

Storage for the files [MixCraft](https://github.com/Signforest/mixcraft) downloads on demand:
AI model files and the inference runtime, each as a **GitHub release** whose tag is the
model id from MixCraft's registry (`crates/ml/models/models.toml`) and whose assets are the
files that registry entry lists. MixCraft fetches
`https://github.com/Signforest/mixcraft-integrations/releases/download/<model id>/<file>`
and checks the SHA-256 the registry pins.

| Release (model id) | What | Licence | Origin |
|---|---|---|---|
| `onnxruntime` | ONNX Runtime 1.20.1 CPU libraries (Windows x64, Linux x64, macOS arm64/x64) | MIT | [microsoft/onnxruntime v1.20.1](https://github.com/microsoft/onnxruntime/releases/tag/v1.20.1), unmodified |
| `skey` | S-KEY key detection (also bundled in MixCraft) | MIT | [deezer/skey](https://github.com/deezer/skey), ONNX export [aaatmy/skey-onnx](https://huggingface.co/aaatmy/skey-onnx) |
| `beat-this` | Beat This! beat/downbeat tracker | MIT | [CPJKU/beat_this](https://github.com/CPJKU/beat_this), ONNX export [musetric/beat-this-onnx](https://huggingface.co/musetric/beat-this-onnx) |

Every asset carries its upstream LICENSE next to it. Nothing here is modified; the SHA-256 of
each file is recorded in MixCraft's registry and in each release's notes.

Only release assets matter to MixCraft; this repository's tree holds this README and
`tools/` (the scripts that produced the assets). Plugins may be added later.
