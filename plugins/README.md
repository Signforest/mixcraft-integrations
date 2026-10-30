# MixCraft plug-ins

Official plug-ins for [MixCraft](https://github.com/Signforest/mixcraft) (`docs/plugins.md` in
that repository describes the host, the kinds and the terms gate). Each folder is one package:
`plugin.toml`, the terms, the built `.wasm` module, its source and a `SHA256SUMS`.

Install: in MixCraft, `plugin.install {"path": "<folder or zip>"}` (the Preferences page comes
later), accept the terms, grant the hosts the plug-in asks for, switch it on.

| Folder | Kind | What | Hosts |
|---|---|---|---|
| `deezer-links` | `resolver` | Deezer track, album and playlist links in a request list → titles, artists, durations, ISRCs through Deezer's public API (no key, no account) | `api.deezer.com` |
| `spotify-links` | `resolver` | Spotify track, album and playlist links → titles, artists, durations read from Spotify's public embed pages (no key, no account). **Not an official API:** against Spotify's terms, breaks when the page changes, playlists cut at 100 tracks; its terms screen says so | `open.spotify.com` |

## Building

Every plug-in is a Rust `cdylib` for `wasm32-unknown-unknown` without WASI (`no_std` with a
bump allocator; the host offers no OS):

```sh
rustup target add wasm32-unknown-unknown
cd plugins/<name>
cargo build --release --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/<crate name>.wasm ./<module name in plugin.toml>
sha256sum plugin.toml TERMS.md *.wasm > SHA256SUMS
```

`tools/build-plugins.sh` does this for every folder. The checked-in `.wasm` is what MixCraft
users install; rebuild it in the same commit as any source change.

Tests run on the build machine: `cd plugins/<name> && cargo test` (where a plug-in has them;
`spotify-links` tests its page parsing against small made-up fixture pages).
