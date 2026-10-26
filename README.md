# mixcraft-integrations

Storage for the files [MixCraft](https://github.com/Signforest/mixcraft) downloads on demand:
AI model files and the inference runtime, one folder per model id from MixCraft's registry
(`crates/ml/models/models.toml`), holding the files that registry entry lists. MixCraft fetches
`https://github.com/Signforest/mixcraft-integrations/raw/main/<model id>/<file>` and checks
the SHA-256 the registry pins. Files above GitHub's 100 MB limit are stored in numbered parts
(`<file>.001`, `<file>.002`, …) that MixCraft joins.

| Folder (model id) | What | Licence | Origin |
|---|---|---|---|
| `onnxruntime` | ONNX Runtime 1.20.1 CPU libraries (Windows x64, Linux x64, macOS arm64/x64) | MIT | [microsoft/onnxruntime v1.20.1](https://github.com/microsoft/onnxruntime/releases/tag/v1.20.1), unmodified |
| `skey` | S-KEY key detection (also bundled in MixCraft) | MIT | [deezer/skey](https://github.com/deezer/skey), ONNX export [aaatmy/skey-onnx](https://huggingface.co/aaatmy/skey-onnx) |
| `beat-this` | Beat This! beat/downbeat tracker | MIT | [CPJKU/beat_this](https://github.com/CPJKU/beat_this), ONNX export [musetric/beat-this-onnx](https://huggingface.co/musetric/beat-this-onnx) |

Every asset carries its upstream LICENSE next to it. Nothing here is modified; the SHA-256 of
each file is recorded in MixCraft's registry and in each folder's `SHA256SUMS`.

`tools/` holds the scripts that produced the files. Plugins may be added later.
