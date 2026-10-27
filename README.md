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
| `umxhq` | Open-Unmix UMX-HQ four-stem separation (vocals, drums, bass, other) | MIT | [sigsep/open-unmix-pytorch](https://github.com/sigsep/open-unmix-pytorch) weights (Zenodo 3370489), exported here with `tools/export-openunmix.py` |
| `htdemucs` | Hybrid Transformer Demucs four-stem separation (drums, bass, other, vocals), STFT on the host | MIT | [facebookresearch/demucs](https://github.com/facebookresearch/demucs) `htdemucs` checkpoint, exported here with `tools/export-demucs.py` (two parts) |
| `htdemucs_6s` | Hybrid Transformer Demucs six-stem separation (+ guitar, piano) | MIT | same source and script, `htdemucs_6s` checkpoint (two parts) |
| `allinone` | All-In-One song structure (intro, verse, chorus …), beats and downbeats from the four stems | MIT | [mir-aidj/all-in-one](https://github.com/mir-aidj/all-in-one), the eight `harmonix` folds from `taejunkim/allinone`, exported here with `tools/export-allinone.py` (neighbourhood attention rebuilt from gathers) |

Every asset carries its upstream LICENSE next to it. Runtime and models are unmodified; the `umxhq`, `htdemucs`, `htdemucs_6s` and `allinone` graphs are our own ONNX exports of the unmodified weights (scripts in `tools/`); the SHA-256 of
each file is recorded in MixCraft's registry and in each folder's `SHA256SUMS`.

`tools/` holds the scripts that produced the files.

`plugins/` holds the official plug-ins (wasm modules for MixCraft's plug-in host, source
next to each): see [plugins/README.md](plugins/README.md). First: `deezer-links`, a resolver
for Deezer links in request lists.
