#!/usr/bin/env bash
# Build the LAME MP3 encoder plug-in (`lame-mp3`) for MixCraft.
#
# Fetches LAME and wasi-sdk at pinned versions, checks their SHA-256, compiles libmp3lame (the
# library only: no frontend, no decoder, no assembly) together with MixCraft's glue into one
# WebAssembly module that imports nothing but MixCraft's host API, and lays out the package:
#
#   lame-mp3.wasm   the module (libmp3lame + glue; LGPL as a whole)
#   COPYING         LAME's licence text (from the source tarball)
#   SOURCE.md       where the exact source is (LGPL source offer)
#   SHA256SUMS      plugin.toml, LICENSE, COPYING, SOURCE.md, lame-mp3.wasm
#
# all written into this folder (the package; commit them together with any source change).
#
# Usage: plugins/lame-mp3/build.sh [work dir]   (default: plugins/lame-mp3/build, gitignored)
# Needs: bash, curl, tar, sha256sum. Linux x86_64 or macOS
# (arm64/x86_64); on Windows run it in WSL. Nothing of it is part of the MixCraft build.
set -euo pipefail

LAME_VERSION=3.100
LAME_URL="https://downloads.sourceforge.net/project/lame/lame/${LAME_VERSION}/lame-${LAME_VERSION}.tar.gz"
LAME_SHA256=ddfe36cab873794038ae2c1210557ad34857a4b6bdc515785d1da9e175b1da1e

WASI_SDK_MAJOR=25
WASI_SDK_VERSION=25.0
case "$(uname -s)-$(uname -m)" in
  Linux-x86_64)   WASI_SDK_ARCH=x86_64-linux;  WASI_SDK_SHA256=52640dde13599bf127a95499e61d6d640256119456d1af8897ab6725bcf3d89c ;;
  *)              WASI_SDK_ARCH=""; WASI_SDK_SHA256="" ;;
esac

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
work="${1:-$here/build}"
mkdir -p "$work/downloads"

fetch() { # url sha256 file
  local url="$1" sum="$2" file="$3"
  if [ ! -f "$file" ]; then
    curl -fsSL --retry 3 -o "$file.part" "$url"
    mv "$file.part" "$file"
  fi
  if [ -n "$sum" ]; then
    echo "$sum  $file" | sha256sum -c - >/dev/null || { echo "checksum mismatch: $file" >&2; rm -f "$file"; exit 1; }
  else
    echo "note: no pinned checksum for $file on this platform; SHA-256 is $(sha256sum "$file" | cut -d' ' -f1)" >&2
  fi
}

# --- toolchain: wasi-sdk (clang + wasm-ld + wasi-libc), or WASI_SDK_PATH if set ---------------
if [ -n "${WASI_SDK_PATH:-}" ]; then
  sdk="$WASI_SDK_PATH"
else
  [ -n "$WASI_SDK_ARCH" ] || { echo "set WASI_SDK_PATH to a wasi-sdk ${WASI_SDK_VERSION} install on this platform" >&2; exit 1; }
  tarball="$work/downloads/wasi-sdk-${WASI_SDK_VERSION}-${WASI_SDK_ARCH}.tar.gz"
  fetch "https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-${WASI_SDK_MAJOR}/wasi-sdk-${WASI_SDK_VERSION}-${WASI_SDK_ARCH}.tar.gz" "$WASI_SDK_SHA256" "$tarball"
  sdk="$work/wasi-sdk-${WASI_SDK_VERSION}-${WASI_SDK_ARCH}"
  [ -d "$sdk" ] || tar -xzf "$tarball" -C "$work"
fi
cc="$sdk/bin/clang"
sysroot="$sdk/share/wasi-sysroot"

# --- LAME source ---------------------------------------------------------------------------
fetch "$LAME_URL" "$LAME_SHA256" "$work/downloads/lame-${LAME_VERSION}.tar.gz"
lame="$work/lame-${LAME_VERSION}"
[ -d "$lame" ] || tar -xzf "$work/downloads/lame-${LAME_VERSION}.tar.gz" -C "$work"

# --- compile -------------------------------------------------------------------------------
obj="$work/obj"
rm -rf "$obj" && mkdir -p "$obj"
cflags=(--target=wasm32-wasip1 "--sysroot=$sysroot" -O2 -DHAVE_CONFIG_H -I"$here/src" -I"$lame/include" -I"$lame/libmp3lame" -w)
for f in "$lame"/libmp3lame/*.c "$here/src/mixcraft_lame.c" "$here/src/wasi_stubs.c"; do
  "$cc" "${cflags[@]}" -c "$f" -o "$obj/$(basename "${f%.c}").o"
done
out="$work/lame-mp3.wasm"
"$cc" --target=wasm32-wasip1 "--sysroot=$sysroot" -O2 -nostartfiles \
  -Wl,--no-entry -Wl,--strip-all -Wl,-z,stack-size=1048576 -Wl,--initial-memory=4194304 \
  "$obj"/*.o -lm -o "$out"

# The module must import nothing but the `mixcraft` host API (the sandbox rejects anything else).
if LC_ALL=C grep -a -q "wasi_snapshot_preview1" "$out"; then
  echo "lame-mp3.wasm still imports WASI functions; extend src/wasi_stubs.c" >&2
  exit 1
fi

# --- package -------------------------------------------------------------------------------
# The package is this folder itself (as for every plug-in in this repository): the built module,
# LAME's licence text, the source offer and SHA256SUMS sit next to plugin.toml and the sources.
pkg="$here"
cp "$out" "$pkg/lame-mp3.wasm"
cp "$lame/COPYING" "$pkg/COPYING"
wasm_sum="$(sha256sum "$pkg/lame-mp3.wasm" | cut -d' ' -f1)"
cat > "$pkg/SOURCE.md" <<EOF2
# Source of lame-mp3.wasm

lame-mp3.wasm is libmp3lame ${LAME_VERSION} (LGPL) compiled to WebAssembly together with
MixCraft's glue (src/, MIT OR Apache-2.0) by build.sh with wasi-sdk ${WASI_SDK_VERSION}.

- LAME ${LAME_VERSION} source: ${LAME_URL}
  (SHA-256 ${LAME_SHA256}); unmodified, libmp3lame/*.c only.
- Glue and build script: plugins/lame-mp3 (src/, build.sh) in
  https://github.com/Signforest/mixcraft-integrations
- lame-mp3.wasm SHA-256: ${wasm_sum}

You may replace lame-mp3.wasm with your own build of a modified libmp3lame (same exports, see
src/mixcraft_lame.c): MixCraft loads whatever module this folder's plugin.toml names.
EOF2
( cd "$pkg" && sha256sum plugin.toml LICENSE COPYING SOURCE.md lame-mp3.wasm > SHA256SUMS )
echo "built $pkg/lame-mp3.wasm ($(wc -c < "$pkg/lame-mp3.wasm") bytes, sha256 $wasm_sum)"
