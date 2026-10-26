#!/usr/bin/env python3
"""Export the All-In-One music structure analyzer (Taejun Kim, MIT) to ONNX.

All-In-One (https://github.com/mir-aidj/all-in-one, WASPAA 2023) predicts beats, downbeats,
section boundaries and functional labels (intro, verse, chorus, …) from the four Demucs stems
of a track. The original uses NATTEN's dilated neighbourhood attention, which has no ONNX
export; this script rebuilds the network in plain PyTorch with the same window arithmetic
(transcribed from NATTEN 0.14's `get_window_start` / `get_pb_start`) and gathers, then
exports the eight Harmonix folds as one graph that averages their logits (the `harmonix-all`
ensemble).

Contract `structure-allinone-81x100fps` (MixCraft's `mixcraft_ml::structure`):
  input    "spec"      float32 [4, T, 81]   per stem (bass, drums, other, vocals): madmom's
                                            log-filtered spectrogram — frames of 2048 samples
                                            at 44 100 Hz every 441 samples (100 fps), frame
                                            centred on its reference sample (zero padded),
                                            symmetric Hann window, magnitude of the samples
                                            as floats in -1..1 (madmom scales the window by
                                            the int16 range of the stem WAVs), bins 0–1023
                                            (the Nyquist bin dropped, as madmom does), times
                                            the filterbank in filterbank.bin (1024 × 81, row
                                            major f32: 12 bands per octave, 30–17000 Hz,
                                            triangular, normalised), then log10(x + 1).
                                            T ≥ 10240 (pad with zeros); T is dynamic.
  outputs  "beat"      float32 [T]          logits (sigmoid → probability per frame)
           "downbeat"  float32 [T]
           "section"   float32 [T]          boundary logits
           "function"  float32 [10, T]      label logits: start, end, intro, outro, break,
                                            bridge, inst, solo, verse, chorus

Post-processing (as `allin1.postprocessing`): boundaries = local maxima of the section
probability (window 97 frames) that stand above the mean of the 12 s before and after;
a segment's label = argmax of the mean softmax over its frames.

Usage: python export-allinone.py <out dir>   (torch, onnx, onnxruntime, madmom,
huggingface_hub). Writes allinone.onnx, filterbank.bin, SHA256SUMS, config.json and the
parity of the vectorised attention against a literal loop implementation.
"""
import hashlib
import json
import math
import pathlib
import sys

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

FOLDS = [
    "harmonix-fold0-0vra4ys2.pth",
    "harmonix-fold1-3ozjhtsj.pth",
    "harmonix-fold2-gmgo0nsy.pth",
    "harmonix-fold3-i92b7m8p.pth",
    "harmonix-fold4-1bql5qo0.pth",
    "harmonix-fold5-x4z5zeef.pth",
    "harmonix-fold6-x7t226rq.pth",
    "harmonix-fold7-qwwskhg6.pth",
]
LABELS = ["start", "end", "intro", "outro", "break", "bridge", "inst", "solo", "verse", "chorus"]


# --- NATTEN window arithmetic (natten 0.14, csrc/cpu/natten_cpu_commons.h) ----------------


def window_start(index: torch.Tensor, length: torch.Tensor, kernel: int, dilation: int) -> torch.Tensor:
    """First key position of each query's neighbourhood (`get_window_start`)."""
    nb = kernel // 2
    if dilation <= 1:
        start = torch.clamp(index - nb, min=0)
        tail = (index + nb >= length).to(index.dtype) * (length - index - nb - 1)
        return start + tail
    ni = index - nb * dilation
    imodd = index % dilation
    a = (length // dilation) * dilation
    b = length - a
    tail = torch.where(imodd < b, length - b + imodd - 2 * nb * dilation, a + imodd - kernel * dilation)
    out = torch.where(ni < 0, imodd, ni)
    out = torch.where((ni >= 0) & (index + nb * dilation >= length), tail, out)
    return out


def pb_start(index: torch.Tensor, length: torch.Tensor, kernel: int, dilation: int) -> torch.Tensor:
    """First relative-position-bias index of each query (`get_pb_start`)."""
    nb = kernel // 2
    if dilation <= 1:
        head = (index < nb).to(index.dtype) * (nb - index)
        tail = (index + nb >= length).to(index.dtype) * (length - index - 1 - nb)
        return nb + head + tail
    out = index * 0 + nb
    out = torch.where(index - nb * dilation < 0, kernel - 1 - index // dilation, out)
    out = torch.where((index - nb * dilation >= 0) & (index + nb * dilation >= length), (length - index - 1) // dilation, out)
    return out


def positions(along: torch.Tensor) -> torch.Tensor:
    """0, 1, … along a tensor's length, as a traced op (dynamic in the export)."""
    # `ones_like` / `full_like` export as constants of the example length; arithmetic on the
    # data does not.
    return (along * 0.0 + 1.0).to(torch.int64).cumsum(0) - 1


def na1d(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, rpb: torch.Tensor, kernel: int, dilation: int) -> torch.Tensor:
    """Neighbourhood attention over the last-but-one axis. q, k, v: [B, H, T, D]; rpb: [H, 2K-1]."""
    # The positions from a tensor op, not `arange(int)`: the tracer would bake the example
    # length into the graph.
    idx = positions(k[0, 0, :, 0])
    length = idx[-1] + 1
    ks = torch.arange(kernel, dtype=torch.int64)
    key_idx = window_start(idx, length, kernel, dilation)[:, None] + ks[None, :] * dilation  # [T, K]
    pb_idx = pb_start(idx, length, kernel, dilation)[:, None] + ks[None, :]  # [T, K]
    kg = k[:, :, key_idx]  # [B, H, T, K, D]
    vg = v[:, :, key_idx]
    scores = (q[:, :, :, None, :] * kg).sum(-1) + rpb[:, pb_idx][None]  # [B, H, T, K]
    probs = scores.softmax(-1)
    return (probs[..., None] * vg).sum(-2)  # [B, H, T, D]


def _na2d_general(q, k, v, rpb, kernel, dilation):
    """The general 2-D path (gathers along both axes), for the parity check of the fast one."""
    ks = torch.arange(kernel, dtype=torch.int64)
    iy = positions(k[0, 0, :, 0, 0])
    ix = positions(k[0, 0, 0, :, 0])
    ly, lx = iy[-1] + 1, ix[-1] + 1
    ky = window_start(iy, ly, kernel, dilation)[:, None] + ks[None, :] * dilation
    kx = window_start(ix, lx, kernel, dilation)[:, None] + ks[None, :] * dilation
    py = pb_start(iy, ly, kernel, dilation)[:, None] + ks[None, :]
    px = pb_start(ix, lx, kernel, dilation)[:, None] + ks[None, :]
    bias = rpb[:, py][:, :, :, px].permute(0, 1, 3, 2, 4)
    kg = k[:, :, ky][:, :, :, :, kx].permute(0, 1, 2, 4, 3, 5, 6)
    vg = v[:, :, ky][:, :, :, :, kx].permute(0, 1, 2, 4, 3, 5, 6)
    scores = (q[:, :, :, :, None, None, :] * kg).sum(-1) + bias[None]
    probs = scores.flatten(-2).softmax(-1).unflatten(-1, (kernel, kernel))
    return (probs[..., None] * vg).sum((-2, -3))


def na2d(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, rpb: torch.Tensor, kernel: int, dilation: int) -> torch.Tensor:
    """2-D neighbourhood attention. q, k, v: [B, H, Y, X, D]; rpb: [H, 2K-1, 2K-1]."""
    ks = torch.arange(kernel, dtype=torch.int64)
    iy = positions(k[0, 0, :, 0, 0])
    ix = positions(k[0, 0, 0, :, 0])
    ly = iy[-1] + 1
    lx = ix[-1] + 1
    ky = window_start(iy, ly, kernel, dilation)[:, None] + ks[None, :] * dilation  # [Y, K]
    kx = window_start(ix, lx, kernel, dilation)[:, None] + ks[None, :] * dilation  # [X, K]
    py = pb_start(iy, ly, kernel, dilation)[:, None] + ks[None, :]
    px = pb_start(ix, lx, kernel, dilation)[:, None] + ks[None, :]
    bias = rpb[:, py][:, :, :, px].permute(0, 1, 3, 2, 4)  # [H, Y, X, Ky, Kx]
    if k.size(2) == kernel and dilation == 1:
        # The whole height is every query's window (the instrument axis, 5 rows): gather
        # along X only and let the rows broadcast — a fifth of the memory of the general path.
        kx_g = k[:, :, :, kx]  # [B, H, Y(=Ky), X, Kx, D]
        vx_g = v[:, :, :, kx]
        scores = torch.einsum("bhyxd,bhzxkd->bhyxzk", q, kx_g) + bias[None]
        probs = scores.flatten(-2).softmax(-1).unflatten(-1, (kernel, kernel))
        return torch.einsum("bhyxzk,bhzxkd->bhyxd", probs, vx_g)
    # Gather keys: [B, H, Y, X, K, K, D].
    kg = k[:, :, ky][:, :, :, :, kx]  # [B, H, Y, Ky, X, Kx, D]
    kg = kg.permute(0, 1, 2, 4, 3, 5, 6)
    vg = v[:, :, ky][:, :, :, :, kx].permute(0, 1, 2, 4, 3, 5, 6)
    scores = (q[:, :, :, :, None, None, :] * kg).sum(-1) + bias[None]
    probs = scores.flatten(-2).softmax(-1).unflatten(-1, (kernel, kernel))
    return (probs[..., None] * vg).sum((-2, -3))  # [B, H, Y, X, D]


def na1d_loop(q, k, v, rpb, kernel, dilation):
    """Literal transcription of the NATTEN CPU kernel, for the parity check."""
    B, H, T, D = q.shape
    nb = kernel // 2

    def ws(i):
        if dilation <= 1:
            return max(i - nb, 0) + (i + nb >= T) * (T - i - nb - 1)
        ni = i - nb * dilation
        if ni < 0:
            return i % dilation
        if i + nb * dilation >= T:
            imodd = i % dilation
            a = (T // dilation) * dilation
            b = T - a
            if imodd < b:
                return T - b + imodd - 2 * nb * dilation
            return a + imodd - kernel * dilation
        return ni

    def ps(i):
        if dilation <= 1:
            return nb + (i < nb) * (nb - i) + (i + nb >= T) * (T - i - 1 - nb)
        if i - nb * dilation < 0:
            return kernel - 1 - i // dilation
        if i + nb * dilation >= T:
            return (T - i - 1) // dilation
        return nb

    out = torch.zeros_like(q)
    for b in range(B):
        for h in range(H):
            for i in range(T):
                ni, pi = ws(i), ps(i)
                sc = torch.stack([(q[b, h, i] * k[b, h, ni + ki * dilation]).sum() + rpb[h, pi + ki] for ki in range(kernel)])
                pr = sc.softmax(0)
                out[b, h, i] = sum(pr[ki] * v[b, h, ni + ki * dilation] for ki in range(kernel))
    return out


# --- The network ---------------------------------------------------------------------------


class Attention(nn.Module):
    def __init__(self, dim, heads, kernel, dilation, two_d):
        super().__init__()
        self.heads, self.kernel, self.dilation, self.two_d = heads, kernel, dilation, two_d
        self.hd = dim // heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.rpb = nn.Parameter(torch.zeros(heads, 2 * kernel - 1, 2 * kernel - 1) if two_d else torch.zeros(heads, 2 * kernel - 1))

    def forward(self, x):
        def split(t):
            t = t.unflatten(-1, (self.heads, self.hd))
            return t.permute(0, 3, 1, 2, 4) if self.two_d else t.permute(0, 2, 1, 3)

        q = split(self.query(x)) / math.sqrt(self.hd)
        k, v = split(self.key(x)), split(self.value(x))
        if self.two_d:
            c = na2d(q, k, v, self.rpb, self.kernel, self.dilation).permute(0, 2, 3, 1, 4)
        else:
            c = na1d(q, k, v, self.rpb, self.kernel, self.dilation).permute(0, 2, 1, 3)
        return c.flatten(-2)


class AttentionModule(nn.Module):
    def __init__(self, dim, heads, kernel, dilation, two_d):
        super().__init__()
        self.self = Attention(dim, heads, kernel, dilation, two_d)
        self.output = nn.Module()
        self.output.dense = nn.Linear(dim, dim)

    def forward(self, x):
        return self.output.dense(self.self(x))


class Layer(nn.Module):
    """`DinatLayer1d` (double attention: dilation d and 2d) or `DinatLayer2d`."""

    def __init__(self, dim, heads, kernel, dilation, two_d, double, mlp_ratio):
        super().__init__()
        self.two_d, self.double, self.kernel, self.dilation = two_d, double, kernel, dilation
        self.window = kernel * dilation * (2 if double else 1)
        self.layernorm_before = nn.LayerNorm(dim, eps=1e-5)
        self.attention = AttentionModule(dim, heads, kernel, dilation, two_d)
        if double:
            self.attention2 = AttentionModule(dim, heads, kernel, dilation * 2, two_d)
        dim_after = dim * 2 if double else dim
        self.layernorm_after = nn.LayerNorm(dim_after, eps=1e-5)
        self.intermediate = nn.Module()
        self.intermediate.dense = nn.Linear(dim_after, dim_after * mlp_ratio)
        self.output = nn.Module()
        self.output.dense = nn.Linear(dim_after * mlp_ratio, dim)

    def forward(self, x):
        shortcut = x
        h = self.layernorm_before(x)
        # Padding to the window: the exported graph requires T >= the largest window, so only
        # the instrument axis (4 < 5) is ever padded, and the time slice stays dynamic.
        if self.two_d:
            K, T = h.shape[1], h.shape[2]
            pad_k, pad_t = max(0, self.window - K), max(0, self.window - T)
            h = F.pad(h, (0, 0, 0, pad_t, 0, pad_k))
        else:
            T = h.shape[1]
            pad_t = max(0, self.window - T)
            h = F.pad(h, (0, 0, 0, pad_t))
        outs = []
        for att in [self.attention] + ([self.attention2] if self.double else []):
            a = att(h)
            if self.two_d:
                a = a[:, :K]
            if pad_t > 0:
                a = a[:, :, :T] if self.two_d else a[:, :T]
            outs.append(shortcut + a)
        if self.double:
            h = torch.cat(outs, dim=-1)
            shortcut = (outs[0] + outs[1]) / 2.0
        else:
            h = outs[0]
            shortcut = h
        y = self.layernorm_after(h)
        y = self.output.dense(F.gelu(self.intermediate.dense(y)))
        return shortcut + y


class Block(nn.Module):
    def __init__(self, cfg, dilation):
        super().__init__()
        self.timelayer = Layer(cfg["dim_embed"], cfg["num_heads"], cfg["kernel_size"], dilation, False, cfg["double_attention"], cfg["mlp_ratio"])
        assert cfg["instrument_attention"]
        self.instlayer = Layer(cfg["dim_embed"], cfg["num_heads"], 5, 1, True, False, cfg["mlp_ratio"])
        self.inst = cfg["data"]["num_instruments"]

    def forward(self, x):
        x = self.timelayer(x)
        x = x.unflatten(0, (-1, self.inst))
        x = self.instlayer(x)
        return x.flatten(0, 1)


class Embeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hs = cfg["dim_embed"]
        self.conv0 = nn.Conv2d(1, hs // 2, (3, 3), padding=(1, 0))
        self.conv1 = nn.Conv2d(hs // 2, hs, (1, 12))
        self.conv2 = nn.Conv2d(hs, hs, (3, 3), padding=(1, 0))
        self.norm = nn.LayerNorm(hs)

    def forward(self, x):
        x = F.elu(F.max_pool2d(self.conv0(x), (1, 3)))
        x = F.elu(F.max_pool2d(self.conv1(x), (1, 3)))
        x = F.elu(F.max_pool2d(self.conv2(x), (1, 3)))
        return self.norm(x.squeeze(-1).permute(0, 2, 1))


class Head(nn.Module):
    def __init__(self, cfg, classes):
        super().__init__()
        self.classifier = nn.Linear(cfg["data"]["num_instruments"] * cfg["dim_embed"], classes)

    def forward(self, x):
        logits = self.classifier(x.permute(0, 2, 1, 3).flatten(2)).permute(0, 2, 1)
        return logits.squeeze(1) if logits.shape[1] == 1 else logits


class AllInOne(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embeddings = Embeddings(cfg)
        dilations = [min(cfg["dilation_factor"] ** i, cfg["dilation_max"]) for i in range(cfg["depth"])]
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([Block(cfg, d) for d in dilations])
        self.norm = nn.LayerNorm(cfg["dim_embed"], eps=cfg["layer_norm_eps"])
        self.beat_classifier = Head(cfg, 1)
        self.downbeat_classifier = Head(cfg, 1)
        self.section_classifier = Head(cfg, 1)
        self.function_classifier = Head(cfg, cfg["data"]["num_labels"])

    def forward(self, spec):
        # spec: [K, T, F] for one track.
        h = self.embeddings(spec.unsqueeze(1))
        for layer in self.encoder.layers:
            h = layer(h)
        h = self.norm(h.unsqueeze(0))
        return self.beat_classifier(h)[0], self.downbeat_classifier(h)[0], self.section_classifier(h)[0], self.function_classifier(h)[0]


class Ensemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, spec):
        outs = [m(spec) for m in self.models]
        return tuple(torch.stack([o[i] for o in outs]).mean(0) for i in range(4))


def load_fold(name, cache):
    path = hf_hub_download(repo_id="taejunkim/allinone", filename=name, cache_dir=cache)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = AllInOne(cfg)
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=True), None
    model.eval()
    return model, cfg


def filterbank():
    """madmom's log filterbank for 2048-point frames at 44.1 kHz: [1024, 81]."""
    from madmom.audio.filters import LogarithmicFilterbank
    from madmom.audio.stft import fft_frequencies

    fb = LogarithmicFilterbank(fft_frequencies(1024, 44100), num_bands=12, fmin=30, fmax=17000, norm_filters=True, unique_filters=True)
    return np.asarray(fb, dtype=np.float32)


def check_frontend(fb):
    """The spectrogram recipe in the contract against madmom's own processors."""
    from madmom.audio.signal import FramedSignalProcessor, Signal
    from madmom.audio.spectrogram import FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor
    from madmom.audio.stft import ShortTimeFourierTransformProcessor
    from madmom.processors import SequentialProcessor

    rng = np.random.default_rng(1)
    n = 44100 * 3
    t = np.arange(n) / 44100.0
    audio = (0.3 * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.standard_normal(n)).astype(np.float32)
    proc = SequentialProcessor([
        FramedSignalProcessor(frame_size=2048, fps=100),
        ShortTimeFourierTransformProcessor(),
        FilteredSpectrogramProcessor(num_bands=12, fmin=30, fmax=17000, norm_filters=True),
        LogarithmicSpectrogramProcessor(mul=1, add=1),
    ])
    ref = np.asarray(proc(Signal((audio * 32767).astype(np.int16), sample_rate=44100, num_channels=1)))
    # Ours: the samples as floats, frames centred on t = i * 441, symmetric Hann, |FFT| without
    # the Nyquist bin, filterbank, log10(x + 1).
    x = (audio * 32767).astype(np.int16).astype(np.float64) / 32767.0
    frames = int(math.ceil(n / 441))
    win = np.hanning(2048)
    padded = np.concatenate([np.zeros(1024), x, np.zeros(2048)])
    mags = np.stack([np.abs(np.fft.rfft(padded[i * 441 : i * 441 + 2048] * win))[:1024] for i in range(frames)])
    ours = np.log10(mags @ fb.astype(np.float64) + 1.0)
    assert ref.shape == ours.shape, (ref.shape, ours.shape)
    err = np.abs(ref - ours).max()
    return float(err), ref.shape


def main():
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "allinone")
    out.mkdir(parents=True, exist_ok=True)
    cache = out / ".hf-cache"
    torch.manual_seed(0)

    # 1. Attention parity: vectorised gathers against the literal loop.
    report = {}
    for T, kernel, dilation in [(37, 5, 1), (37, 5, 2), (64, 5, 8), (5, 5, 1), (12, 5, 2)]:
        q, k, v = (torch.randn(2, 2, T, 6) for _ in range(3))
        rpb = torch.randn(2, 2 * kernel - 1)
        err = (na1d(q, k, v, rpb, kernel, dilation) - na1d_loop(q, k, v, rpb, kernel, dilation)).abs().max().item()
        report[f"na1d T{T} k{kernel} d{dilation}"] = err
        assert err < 1e-5, report
    q, k, v = (torch.randn(1, 2, 5, 50, 6) for _ in range(3))
    rpb = torch.randn(2, 9, 9)
    fast = na2d(q, k, v, rpb, 5, 1)
    report["na2d fast vs general"] = (fast - _na2d_general(q, k, v, rpb, 5, 1)).abs().max().item()
    assert report["na2d fast vs general"] < 1e-5, report
    print("na1d parity:", report)

    # 2. Models.
    models, cfg = [], None
    for name in FOLDS:
        m, cfg = load_fold(name, cache)
        models.append(m)
        print("loaded", name)
    ens = Ensemble(models).eval()

    # 3. Frontend.
    fb = filterbank()
    fb_err, fb_shape = check_frontend(fb)
    print("frontend parity against madmom:", fb_err, fb_shape)
    assert fb_err < 1e-5
    (out / "filterbank.bin").write_bytes(fb.tobytes())

    # 4. Export with a dynamic time axis.
    T0 = 10240
    spec = torch.randn(4, T0, 81) * 0.3 + 1.0
    with torch.no_grad():
        ref = ens(spec)
    onnx_path = out / "allinone.onnx"
    with torch.no_grad():
        torch.onnx.export(
            ens,
            (spec,),
            str(onnx_path),
            input_names=["spec"],
            output_names=["beat", "downbeat", "section", "function"],
            dynamic_axes={"spec": {1: "T"}, "beat": {0: "T"}, "downbeat": {0: "T"}, "section": {0: "T"}, "function": {1: "T"}},
            opset_version=17,
            dynamo=False,
            # Folding would bake the neighbourhood indices for the example length.
            do_constant_folding=False,
        )
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"spec": spec.numpy()})
    errs = [float(np.abs(g - r.numpy()).max()) for g, r in zip(got, ref)]
    print("onnx parity at T0:", errs)
    # Another length exercises the dynamic indices.
    spec2 = torch.randn(4, 11111, 81) * 0.3 + 1.0
    with torch.no_grad():
        ref2 = ens(spec2)
    got2 = sess.run(None, {"spec": spec2.numpy()})
    errs2 = [float(np.abs(g - r.numpy()).max()) for g, r in zip(got2, ref2)]
    print("onnx parity at T=12345:", errs2)
    assert max(errs + errs2) < 1e-3, (errs, errs2)

    sums = {}
    for f in ["allinone.onnx", "filterbank.bin"]:
        sums[f] = hashlib.sha256((out / f).read_bytes()).hexdigest()
    (out / "SHA256SUMS").write_text("".join(f"{h}  {f}\n" for f, h in sums.items()))
    (out / "config.json").write_text(json.dumps({
        "contract": "structure-allinone-81x100fps",
        "source": "https://github.com/mir-aidj/all-in-one",
        "weights": "https://huggingface.co/taejunkim/allinone (harmonix-fold0 … fold7, averaged)",
        "licence": "MIT",
        "fps": cfg["fps"], "hop": cfg["hop_size"], "frame": cfg["window_size"], "sample_rate": cfg["sample_rate"],
        "bins": cfg["dim_input"], "stems": ["bass", "drums", "other", "vocals"], "labels": LABELS,
        "min_frames": T0,
        "thresholds": {"beat": cfg["best_threshold_beat"], "downbeat": cfg["best_threshold_downbeat"]},
        "parity": {"na1d": report, "frontend_vs_madmom": fb_err, "onnx_vs_torch": errs + errs2},
    }, indent=2))
    print("wrote", onnx_path, onnx_path.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
