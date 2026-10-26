#!/usr/bin/env python3
"""Export Hybrid Transformer Demucs (htdemucs, MIT) to ONNX with the STFT on the host.

`torch.stft` / `torch.istft` do not export, so the graph covers the network between them:

Contract `stems-htdemucs-cac4096` (MixCraft's `mixcraft_ml::demucs`):
  inputs   "mix"   float32 [1, 2, 343980]         7.8 s of the stereo mix at 44 100 Hz
           "spec"  float32 [1, 4, 2048, 336]      the mix's spectrogram as channels
                                                  (L re, L im, R re, R im), see below
  outputs  "spec_out" float32 [1, 4, 4, 2048, 336]  per source (drums, bass, other, vocals):
                                                  the estimated spectrogram, same layout
           "time_out" float32 [1, 4, 2, 343980]   per source: the time-branch waveform
  The source waveform = host iSTFT(spec_out[s]) + time_out[s].

Spectrogram, exactly as `HTDemucs._spec`: pad the 343980 samples by 1536 on both sides
(reflect), STFT n_fft 4096, hop 1024, periodic Hann window, normalized (÷ √4096), centred
with reflect padding; drop the last frequency bin (2049 → 2048) and the first two and last
two frames (340 → 336). Inverse as `_ispec`: append a zero bin and two zero frames on each
side, iSTFT normalized, cut 1536 from each end. The host overlap-adds 7.8 s segments with
25 % overlap and a triangular weight (demucs `apply_model` without shifts).

Usage: python export-demucs.py <out dir>   (torch, demucs, onnx, onnxruntime; the
checkpoint is fetched through demucs.pretrained). Writes htdemucs.onnx, SHA256SUMS and the
parity against the PyTorch model on random audio.
"""
import hashlib
import json
import pathlib
import sys

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from einops import rearrange

from demucs.pretrained import get_model

OPSET = 17
SEGMENT = 343980  # int(39/5 * 44100)
FRAMES = SEGMENT // 1024  # 335 → ceil → 336
FRAMES = -(-SEGMENT // 1024)


class Core(torch.nn.Module):
    """HTDemucs.forward between `_spec` and `_ispec`, for a fixed segment."""

    def __init__(self, h):
        super().__init__()
        self.h = h

    def forward(self, mix, spec):
        h = self.h
        x = spec
        B, C, Fq, T = x.shape
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)
        xt = mix
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)
        saved, saved_t, lengths, lengths_t = [], [], [], []
        for idx, encode in enumerate(h.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(h.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = h.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and h.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = h.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + h.freq_emb_scale * emb
            saved.append(x)
        if h.crosstransformer:
            if h.bottom_channels:
                b, c, f, t = x.shape
                x = rearrange(x, "b c f t-> b c (f t)")
                x = h.channel_upsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = h.channel_upsampler_t(xt)
            x, xt = h.crosstransformer(x, xt)
            if h.bottom_channels:
                x = rearrange(x, "b c f t-> b c (f t)")
                x = h.channel_downsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = h.channel_downsampler_t(xt)
        for idx, decode in enumerate(h.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))
            offset = h.depth - len(h.tdecoder)
            if idx >= offset:
                tdec = h.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip, length_t)
        S = len(h.sources)
        x = x.view(B, S, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]
        xt = xt.view(B, S, -1, SEGMENT)
        xt = xt * stdt[:, None] + meant[:, None]
        return x, xt


def host_spec(h, mix):
    """Reference spectrogram as the host must compute it (uses the model's own _spec)."""
    z = h._spec(mix)
    B, C, Fr, T = z.shape
    m = torch.view_as_real(z).permute(0, 1, 4, 2, 3)
    return m.reshape(B, C * 2, Fr, T)


def host_ispec(h, spec_out):
    B, S, C2, Fr, T = spec_out.shape
    out = spec_out.view(B, S, -1, 2, Fr, T).permute(0, 1, 2, 4, 5, 3)
    z = torch.view_as_complex(out.contiguous())
    return h._ispec(z, SEGMENT)


def main(out_dir: str) -> None:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # The fused attention fast path has no ONNX export: use the plain implementation.
    torch.backends.mha.set_fastpath_enabled(False)
    bag = get_model("htdemucs")
    h = bag.models[0].eval()
    assert h.cac and h.nfft == 4096 and h.hop_length == 1024 and int(h.segment * h.samplerate) == SEGMENT, (h.segment, h.nfft)
    core = Core(h).eval()
    torch.manual_seed(0)
    mix = (torch.rand(1, 2, SEGMENT) - 0.5) * 0.4
    with torch.no_grad():
        spec = host_spec(h, mix)
        assert tuple(spec.shape) == (1, 4, 2048, FRAMES), spec.shape
        ref = h(mix)  # the original end to end
        x, xt = core(mix, spec)
        rebuilt = host_ispec(h, x) + xt
        err_wrap = float((rebuilt - ref).abs().max())
        print(json.dumps({"check": "wrapper + host transforms vs original", "max_abs_err": err_wrap}))
        assert err_wrap < 1e-4, err_wrap
        path = out / "htdemucs.onnx"
        torch.onnx.export(
            core,
            (mix, spec),
            str(path),
            input_names=["mix", "spec"],
            output_names=["spec_out", "time_out"],
            opset_version=OPSET,
            dynamo=False,
        )
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    so, to = sess.run(["spec_out", "time_out"], {"mix": mix.numpy(), "spec": spec.numpy()})
    with torch.no_grad():
        rebuilt2 = host_ispec(h, torch.from_numpy(so)) + torch.from_numpy(to)
    err = float((rebuilt2 - ref).abs().max())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(json.dumps({"file": path.name, "bytes": path.stat().st_size, "sha256": digest, "max_abs_err_vs_torch": err, "sources": h.sources}))
    assert err < 2e-3, err
    (out / "SHA256SUMS").write_text(f"{digest}  {path.name}\n")
    (out / "config.json").write_text(
        json.dumps(
            {
                "contract": "stems-htdemucs-cac4096",
                "sampleRate": 44100,
                "segmentSamples": SEGMENT,
                "nFft": 4096,
                "hopLength": 1024,
                "frames": FRAMES,
                "bins": 2048,
                "normalized": True,
                "padReflect": 1536,
                "sources": h.sources,
                "overlap": 0.25,
                "opset": OPSET,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "htdemucs")
