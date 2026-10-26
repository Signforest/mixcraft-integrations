#!/usr/bin/env python3
"""Export Open-Unmix (UMX-HQ, MIT) to ONNX, one graph per stem.

Contract `stems-umx-mag4096` (what MixCraft's `mixcraft_ml::stems` feeds and reads):
  input  "mag"  float32 [1, 2, 2049, frames]  magnitude spectrogram of the stereo mix at
                44 100 Hz: STFT n_fft 4096, hop 1024, periodic Hann window, centre padding
                (reflect), |X| (power 1, no normalisation)
  output "est"  float32 [1, 2, 2049, frames]  the stem's magnitude estimate (mask · mix)
The host turns the four estimates into ratio masks (est / sum of estimates), applies them to
the complex mix spectrogram and inverse-transforms with the same window and hop.

Usage: python export-openunmix.py <out dir>   (needs torch, torchaudio, openunmix, onnx,
onnxruntime; downloads the UMX-HQ weights from Zenodo through torch.hub on first use).
Writes umxhq-<stem>.onnx, SHA256SUMS and a check against the PyTorch model.
"""
import hashlib
import json
import pathlib
import sys

import numpy as np
import onnxruntime as ort
import torch

import openunmix

OPSET = 17
STEMS = ["vocals", "drums", "bass", "other"]


class Traceable(torch.nn.Module):
    """OpenUnmix.forward with the frame count taken from the tensor's shape, so the tracer
    keeps it dynamic (the original reads `x.data.shape`, which freezes it)."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        m = self.m
        x = x.permute(3, 0, 1, 2)
        nb_frames, nb_samples, nb_channels = x.shape[0], x.shape[1], x.shape[2]
        mix = x.detach().clone()
        x = x[..., : m.nb_bins]
        x = x + m.input_mean
        x = x * m.input_scale
        x = m.fc1(x.reshape(-1, nb_channels * m.nb_bins))
        x = m.bn1(x)
        x = x.reshape(nb_frames, nb_samples, m.hidden_size)
        x = torch.tanh(x)
        lstm_out = m.lstm(x)
        x = torch.cat([x, lstm_out[0]], -1)
        x = m.fc2(x.reshape(-1, x.shape[-1]))
        x = m.bn2(x)
        x = torch.relu(x)
        x = m.fc3(x)
        x = m.bn3(x)
        x = x.reshape(nb_frames, nb_samples, nb_channels, m.nb_output_bins)
        x = x * m.output_scale
        x = x + m.output_mean
        x = torch.relu(x) * mix
        return x.permute(1, 2, 3, 0)


def main(out_dir: str) -> None:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    models = openunmix.umxhq_spec(targets=STEMS, device="cpu", pretrained=True)
    sums = []
    frames = 64
    mag = torch.rand(1, 2, 2049, frames) * 0.1
    for stem in STEMS:
        m = models[stem].eval()
        wrapped = Traceable(m).eval()
        path = out / f"umxhq-{stem}.onnx"
        with torch.no_grad():
            assert float((wrapped(mag) - m(mag)).abs().max()) < 1e-5, "wrapper differs from the original"
            torch.onnx.export(
                wrapped,
                (mag,),
                str(path),
                input_names=["mag"],
                output_names=["est"],
                dynamic_axes={"mag": {3: "frames"}, "est": {3: "frames"}},
                opset_version=OPSET,
                dynamo=False,
            )
            ref = m(mag).numpy()
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        got = sess.run(["est"], {"mag": mag.numpy()})[0]
        err = float(np.abs(got - ref).max())
        # A different length must work too (dynamic frames axis).
        mag2 = torch.rand(1, 2, 2049, 200) * 0.1
        got2 = sess.run(["est"], {"mag": mag2.numpy()})[0]
        assert got2.shape == (1, 2, 2049, 200), got2.shape
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        size = path.stat().st_size
        sums.append(f"{digest}  {path.name}")
        print(json.dumps({"stem": stem, "file": path.name, "bytes": size, "sha256": digest, "max_abs_err": err}))
        assert err < 1e-3, f"{stem}: ONNX differs from torch by {err}"
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "umxhq")
