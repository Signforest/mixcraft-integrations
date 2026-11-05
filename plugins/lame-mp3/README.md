# lame-mp3: the MP3 encoder plug-in

MP3 files in MixCraft — recordings (`record.start {format: "mp3"}`), tracks converted for a USB
stick (`usb.transcode` with MP3 preferred, or for a player that reads no lossless format), and
`encode.file` — come from this plug-in: LAME's encoder library (libmp3lame 3.100) compiled to
WebAssembly. Owner decision 2026-11-05 (`docs/licensing.md` §3, `docs/plugin-candidates.md`).

## The licence boundary

- **MixCraft never links LAME.** The app (MIT OR Apache-2.0) loads `lame-mp3.wasm` at run time
  into its wasmi sandbox like any `encoder` plug-in (`docs/plugins.md`). Without the package,
  MP3 is simply not offered, and every place says so.
- **The package is LGPL** (LGPL-2.0-or-later from LAME, distributed under LGPL-2.1-or-later:
  `LICENSE` here holds the text, LAME's own `COPYING` is copied into each build). It is shipped
  separately (this folder in `mixcraft-integrations`, listed in MixCraft's official plug-in list),
  with `SOURCE.md` saying where the exact source is; users may swap in their own build.
- **No LAME source in this repository.** `build.sh` downloads `lame-3.100.tar.gz` and checks its
  SHA-256; only `libmp3lame/*.c` is compiled (the `lame` frontend is GPL and is not used).
- The files here (`src/mixcraft_lame.c`, `src/wasi_stubs.c`, `src/config.h`, `build.sh`,
  `plugin.toml`) are MixCraft's own, MIT OR Apache-2.0. Unlike the other plug-ins here this one is
  C, not a Cargo package.

## Building

```sh
plugins/lame-mp3/build.sh [work dir]  # default work dir: plugins/lame-mp3/build (gitignored)
```

Needs bash, curl, tar and sha256sum. On Linux x86_64 it fetches wasi-sdk 25.0
(clang, wasm-ld, wasi-libc; pinned by SHA-256); elsewhere set `WASI_SDK_PATH` to a wasi-sdk 25
install (on Windows: run it in WSL). It writes `lame-mp3.wasm`, `COPYING`, `SOURCE.md` and
`SHA256SUMS` into this folder, which is the package: commit them together with any source change.
Install with `plugin.get` (official list), `plugin.install {path}` or Preferences › Plug-ins ›
*Install from a folder…*, then turn it on (its terms state the LGPL).

The module imports only MixCraft's host API: wasi-libc's stdio references (pulled in by LAME's
default message handlers) are resolved inside the module by `src/wasi_stubs.c`, which fail every
call; `build.sh` refuses a module that still imports WASI.

## The ABI (`encoder` kind)

`alloc`, optional `encode_mode(mode, quality)` (0 CBR, 1 VBR at quality 0–9, 2 ABR),
`encode_begin(rate, channels, kbps)`, `encode(ptr, frames, rate, channels)` (interleaved `f32`,
4096 frames a call), `encode_end()`, optional `encode_header()` — the final Xing/LAME frame,
which the host writes over the placeholder frame at the start of the stream, so decoders trim
the encoder delay and padding (gapless) and know the length. ID3 tags are written by the host
afterwards with MixCraft's own tag code; LAME's own ID3 writing is off.

## Checking a build

```sh
# in a MixCraft checkout
MIXCRAFT_LAME_PLUGIN=/path/to/mixcraft-integrations/plugins/lame-mp3 \
  cargo test -p mixcraft-engine --lib real_lame -- --ignored --nocapture
```

encodes a 5 s 44.1 kHz stereo tone at 320 kb/s CBR and VBR through `encode.file` and decodes it
back: exactly the source's length (gapless), the tone in place, the tags there. Measured
2026-11-05 on a 2-core cloud VM (wasmi interpreter): about 1.3× real time for CBR 320 and 1.9×
for VBR at LAME's quality 5 (the glue's algorithm setting; quality 2 was half as fast).

## Not done yet

- Speed: the interpreter is the limit; a 6-minute track takes a few minutes on a slow machine.
- SHOUTcast / MP3 live streams (`record.broadcast`) through this encoder.
- Pinned wasi-sdk checksum for Linux x86_64 only; elsewhere `WASI_SDK_PATH`, unchecked.
