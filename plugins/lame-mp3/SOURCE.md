# Source of lame-mp3.wasm

lame-mp3.wasm is libmp3lame 3.100 (LGPL) compiled to WebAssembly together with
MixCraft's glue (src/, MIT OR Apache-2.0) by build.sh with wasi-sdk 25.0.

- LAME 3.100 source: https://downloads.sourceforge.net/project/lame/lame/3.100/lame-3.100.tar.gz
  (SHA-256 ddfe36cab873794038ae2c1210557ad34857a4b6bdc515785d1da9e175b1da1e); unmodified, libmp3lame/*.c only.
- Glue and build script: plugins/lame-mp3 (src/, build.sh) in
  https://github.com/Signforest/mixcraft-integrations
- lame-mp3.wasm SHA-256: 2e10b706710c512ce9772325e0c840528fa41411957440552d76eba2c6c209e6

You may replace lame-mp3.wasm with your own build of a modified libmp3lame (same exports, see
src/mixcraft_lame.c): MixCraft loads whatever module this folder's plugin.toml names.
