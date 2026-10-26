#!/bin/sh
# Download ONNX Runtime 1.20.1 from Microsoft's release and lay the libraries out as the
# `onnxruntime` release of this repository expects them.
set -eu
V=1.20.1
B=https://github.com/microsoft/onnxruntime/releases/download/v$V
mkdir -p out && cd out
curl -sLO $B/onnxruntime-win-x64-$V.zip
curl -sLO $B/onnxruntime-linux-x64-$V.tgz
curl -sLO $B/onnxruntime-osx-arm64-$V.tgz
curl -sLO $B/onnxruntime-osx-x86_64-$V.tgz
unzip -q -o onnxruntime-win-x64-$V.zip
for a in *.tgz; do tar xzf "$a"; done
cp onnxruntime-win-x64-$V/lib/onnxruntime.dll onnxruntime-win-x64.dll
cp onnxruntime-linux-x64-$V/lib/libonnxruntime.so.$V libonnxruntime-linux-x64.so
cp onnxruntime-osx-arm64-$V/lib/libonnxruntime.$V.dylib libonnxruntime-macos-arm64.dylib
cp onnxruntime-osx-x86_64-$V/lib/libonnxruntime.$V.dylib libonnxruntime-macos-x64.dylib
cp onnxruntime-win-x64-$V/LICENSE LICENSE
sha256sum onnxruntime-win-x64.dll libonnxruntime-linux-x64.so libonnxruntime-macos-arm64.dylib libonnxruntime-macos-x64.dylib LICENSE
