#!/usr/bin/env python3
"""Export the audio encoder of LAION-CLAP's music checkpoint (Apache-2.0) to ONNX.

LAION-CLAP (Wu, Chen, Zhang, Hui, Berg-Kirkpatrick, Dubnov — *Large-scale Contrastive
Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation*,
ICASSP 2023) maps audio and text into one 512-dimensional space. MixCraft only needs the audio
half: two tracks that sound alike land close together, which is what "Sounds like" ranks by.
This script takes the `laion/larger_clap_music` checkpoint (Hugging Face, Apache-2.0, pinned
to one revision below), keeps the HTSAT audio encoder and the audio projection, drops the
RoBERTa text encoder, and exports one graph. Nothing is retrained or changed: the graph is
the PyTorch model's `get_audio_features`, L2-normalised as there.

Contract `embed-clap-mel64-48k` (MixCraft's `mixcraft_ml::clap`):
  input   "mel"        float32 [N, 1, 1001, 64]  log-mel spectrogram of a 10 s window as
                                                 transformers' ClapFeatureExtractor makes it
                                                 for this checkpoint (truncation rand_trunc,
                                                 padding repeatpad): mono at 48 000 Hz, 480 000
                                                 samples (a shorter clip is repeated whole as
                                                 often as it fits, then zero padded), frames
                                                 of 1024 samples every 480 centred on their
                                                 sample (reflect padding, 1001 frames),
                                                 periodic Hann window, power spectrum, times
                                                 the 64-band Slaney mel filterbank (Slaney
                                                 scale, Slaney area norm, 50–14 000 Hz),
                                                 floored at 1e-10, 10·log10. N is dynamic.
  output  "embedding"  float32 [N, 512]          unit-length audio embedding (cosine
                                                 similarity = dot product)

Usage: python export-clap.py <out dir>   (torch, transformers, onnx, onnxruntime, numpy,
huggingface_hub). Downloads the checkpoint (about 780 MB) and writes:
  clap-music.onnx          the graph (stored in numbered parts when above 95 MB: run
                           `split -b 95m -d -a 3 --numeric-suffixes=1` as for the others)
  config.json              the contract, the front-end numbers and where the weights came from
  reference.json           a deterministic test signal's mel rows and embedding, for
                           MixCraft's own front-end tests (crates/ml/testdata/clap-reference.json)
  SHA256SUMS               hashes of the written files
and prints the parity of the ONNX graph against PyTorch.
"""
import hashlib
import json
import pathlib
import sys

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import ClapFeatureExtractor, ClapModel

REPO = "laion/larger_clap_music"
REVISION = "a0b4534a14f58e20944452dff00a22a06ce629d1"
CONTRACT = "embed-clap-mel64-48k"
OPSET = 17
SAMPLE_RATE = 48_000
WINDOW = 480_000
FRAMES = 1001
MELS = 64


class AudioEmbedding(torch.nn.Module):
    """`ClapModel.get_audio_features` without the text half: HTSAT, projection, L2 norm.
    The checkpoint is not a fusion model, so `is_longer` plays no part and is left out of the
    graph's inputs."""

    def __init__(self, model: ClapModel):
        super().__init__()
        self.audio = model.audio_model
        self.projection = model.audio_projection

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        pooled = self.audio(input_features=mel).pooler_output
        return F.normalize(self.projection(pooled), dim=-1)


def test_signal() -> np.ndarray:
    """3.5 s of two tones and a slow chirp at 48 kHz: shorter than a window, so the
    repeat-and-pad path is covered. MixCraft's test rebuilds it sample for sample."""
    n = np.arange(168_000, dtype=np.float64)
    t = n / SAMPLE_RATE
    x = 0.5 * np.sin(2 * np.pi * 440.0 * t) + 0.25 * np.sin(2 * np.pi * 3000.0 * t) + 0.1 * np.sin(2 * np.pi * (200.0 + 400.0 * t) * t)
    return x.astype(np.float32)


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    snap = snapshot_download(REPO, revision=REVISION, allow_patterns=["config.json", "preprocessor_config.json", "pytorch_model.bin"])
    model = ClapModel.from_pretrained(snap).eval()
    fe = ClapFeatureExtractor.from_pretrained(snap)
    assert fe.sampling_rate == SAMPLE_RATE and fe.nb_max_samples == WINDOW and fe.feature_size == MELS
    assert fe.truncation == "rand_trunc" and fe.padding == "repeatpad"
    assert not model.config.audio_config.enable_fusion
    net = AudioEmbedding(model).eval()

    signal = test_signal()
    feats = fe(signal, sampling_rate=SAMPLE_RATE, return_tensors="np")["input_features"].astype(np.float32)
    assert feats.shape == (1, 1, FRAMES, MELS), feats.shape
    rng = np.random.default_rng(7)
    noise = fe([rng.uniform(-0.5, 0.5, WINDOW).astype(np.float32) for _ in range(2)], sampling_rate=SAMPLE_RATE, return_tensors="np")["input_features"].astype(np.float32)
    batch = np.concatenate([feats, noise], axis=0)

    onnx_path = out / "clap-music.onnx"
    with torch.no_grad():
        torch.onnx.export(
            net,
            (torch.from_numpy(batch),),
            str(onnx_path),
            input_names=["mel"],
            output_names=["embedding"],
            dynamic_axes={"mel": {0: "n"}, "embedding": {0: "n"}},
            opset_version=OPSET,
            dynamo=False,
        )
        want = net(torch.from_numpy(batch)).numpy()
        want_one = net(torch.from_numpy(feats)).numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(["embedding"], {"mel": batch})[0]
    got_one = sess.run(["embedding"], {"mel": feats})[0]
    err = float(np.abs(got - want).max())
    err_one = float(np.abs(got_one - want_one).max())
    print(f"ONNX vs PyTorch: max |Δ| {err:.2e} (batch 3), {err_one:.2e} (batch 1); norms {np.linalg.norm(got, axis=1)}")
    assert err < 1e-4 and err_one < 1e-4

    mel = feats[0, 0]
    reference = {
        "contract": CONTRACT,
        "signal": "0.5 sin(2π·440 t) + 0.25 sin(2π·3000 t) + 0.1 sin(2π·(200 + 400 t)·t), t = n / 48000, n < 168000, float32",
        "frames": FRAMES,
        "mels": MELS,
        "rows": {str(i): [round(float(v), 4) for v in mel[i]] for i in (0, 1, 2, 350, 700, 1000)},
        "mean": round(float(mel.mean()), 5),
        "filterbank": {
            "bins": int(fe.mel_filters_slaney.shape[0]),
            "band_peaks": [[int(np.argmax(fe.mel_filters_slaney[:, m])), round(float(fe.mel_filters_slaney[:, m].max()), 7)] for m in range(MELS)],
        },
        "embedding": [round(float(v), 6) for v in got_one[0]],
    }
    (out / "reference.json").write_text(json.dumps(reference, indent=1) + "\n")

    config = {
        "contract": CONTRACT,
        "source": "https://github.com/LAION-AI/CLAP",
        "weights": f"https://huggingface.co/{REPO}/tree/{REVISION}",
        "licence": "Apache-2.0",
        "sample_rate": SAMPLE_RATE,
        "window_samples": WINDOW,
        "padding": "repeatpad",
        "n_fft": 1024,
        "hop": 480,
        "window": "hann (periodic)",
        "power": 2.0,
        "mels": MELS,
        "f_min": 50.0,
        "f_max": 14_000.0,
        "mel_scale": "slaney",
        "mel_norm": "slaney",
        "log": "10*log10(max(x, 1e-10))",
        "frames": FRAMES,
        "dim": int(got.shape[1]),
        "opset": OPSET,
        "parity_max_abs": err,
        "torch": torch.__version__,
        "onnx_runtime": ort.__version__,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    sums = [f"{sha256(out / name)}  {name}" for name in ("clap-music.onnx", "config.json")]
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print("\n".join(sums))
    print(f"clap-music.onnx: {onnx_path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
