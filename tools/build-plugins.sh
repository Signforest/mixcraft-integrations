#!/usr/bin/env sh
# Build every plug-in under plugins/ for wasm32-unknown-unknown and refresh its SHA256SUMS.
set -eu
cd "$(dirname "$0")/.."
for dir in plugins/*/; do
  [ -f "$dir/Cargo.toml" ] || continue
  name=$(sed -n 's/^name = "\(.*\)"/\1/p' "$dir/Cargo.toml" | head -1 | tr '-' '_')
  module=$(sed -n 's/^module = "\(.*\)"/\1/p' "$dir/plugin.toml" | head -1)
  (cd "$dir" && cargo build --release --target wasm32-unknown-unknown --quiet)
  cp "$dir/target/wasm32-unknown-unknown/release/$name.wasm" "$dir/$module"
  (cd "$dir" && sha256sum plugin.toml TERMS.md "$module" > SHA256SUMS)
  echo "$dir -> $module ($(wc -c < "$dir/$module") bytes)"
done
