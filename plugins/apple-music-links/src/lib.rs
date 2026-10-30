//! MixCraft resolver plug-in: Apple Music links through the public web pages.
//!
//! The host hands `resolve` a JSON document `{"rows":[{"line", "link":{provider, kind, id,
//! url}, …}]}` with the link rows of a pasted request list. For every row whose provider is
//! `apple-music` this module fetches the public page on `music.apple.com` (the one a browser
//! shows without signing in), reads the `serialized-server-data` JSON it carries and answers
//! `{"rows":[…]}`: a song row gets `artist`, `title`, `duration_ms` (when the page has it) and
//! `url`; an album or playlist row gets `tracks: [...]`. Rows of other providers, and links
//! the page does not show, are left out of the answer (the host keeps them for the next
//! resolver).
//!
//! The page is not an API: Apple can change it at any time, which breaks this plug-in until
//! it is updated, and a long playlist's page lists only its first 100 songs. Reading it this
//! way is not something Apple's terms allow; `TERMS.md` says so before the user turns the
//! plug-in on.
//!
//! Host ABI (`docs/plugins.md` §4): imports `mixcraft.http_get`, `result_len`, `result_read`,
//! `emit`, `log`; exports `memory`, `alloc`, `resolve`. Built for `wasm32-unknown-unknown`
//! without WASI. The parsing below is plain Rust and is tested on the build machine with
//! `cargo test`.
#![cfg_attr(all(target_arch = "wasm32", not(test)), no_std)]
#![cfg_attr(not(any(test, target_arch = "wasm32")), allow(dead_code))]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

extern crate alloc;

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use serde::Deserialize;
use serde_json::{Map, Value, json};

/// Most tracks answered for one album or playlist (the host caps at 2,000).
const MAX_TRACKS: usize = 2_000;

#[derive(Deserialize)]
struct Input {
    #[serde(default)]
    rows: Vec<Row>,
}

#[derive(Deserialize)]
struct Row {
    line: u64,
    #[serde(default)]
    link: Option<Link>,
}

#[derive(Deserialize)]
struct Link {
    #[serde(default)]
    provider: String,
    #[serde(default)]
    kind: String,
    #[serde(default)]
    id: String,
    #[serde(default)]
    url: String,
}

/// Song and album ids are decimal; playlist ids are `pl.` and hex or base-62 letters.
fn apple_id<'a>(kind: &str, id: &'a str) -> Option<&'a str> {
    let ok = match kind {
        "playlist" => id.strip_prefix("pl.").is_some_and(|r| !r.is_empty() && r.len() <= 64 && r.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')),
        _ => !id.is_empty() && id.len() <= 20 && id.bytes().all(|b| b.is_ascii_digit()),
    };
    ok.then_some(id)
}

/// The storefront of the pasted link (`/de/…` → `de`), `us` when it has none.
fn storefront(url: &str) -> String {
    let rest = url.split_once("://").map_or(url, |(_, r)| r);
    let first = rest.split('/').nth(1).unwrap_or("");
    if first.len() == 2 && first.bytes().all(|b| b.is_ascii_lowercase()) { first.to_string() } else { "us".to_string() }
}

/// The JSON inside `<script … id="serialized-server-data" …>…</script>`.
fn page_data(html: &str) -> Option<Value> {
    let at = html.find("id=\"serialized-server-data\"")?;
    let rest = html.get(at..)?;
    let open = rest.find('>')? + 1;
    let body = rest.get(open..)?;
    let close = body.find("</script>")?;
    serde_json::from_str(body.get(..close)?).ok()
}

/// The page's sections: `data[0].data.sections`.
fn sections(page: &Value) -> Vec<Value> {
    page.pointer("/data/0/data/sections").and_then(Value::as_array).cloned().unwrap_or_default()
}

fn text(v: &Value, key: &str) -> String {
    v.get(key).and_then(Value::as_str).unwrap_or("").trim().to_string()
}

/// The titles of `subtitleLinks` (the artists), joined.
fn link_titles(v: &Value) -> String {
    v.get("subtitleLinks")
        .and_then(Value::as_array)
        .map(|a| a.iter().map(|l| text(l, "title")).filter(|t| !t.is_empty()).collect::<Vec<_>>().join(", "))
        .unwrap_or_default()
}

fn track_shape(title: String, artist: String, duration_ms: Option<u64>, url: String) -> Option<Map<String, Value>> {
    if title.is_empty() {
        return None;
    }
    let mut m = Map::new();
    m.insert("title".into(), Value::String(title));
    m.insert("artist".into(), Value::String(artist));
    if let Some(d) = duration_ms.filter(|d| *d > 0) {
        m.insert("duration_ms".into(), Value::from(d));
    }
    if url.starts_with("https://music.apple.com/") {
        m.insert("url".into(), Value::String(url));
    }
    Some(m)
}

/// What one page says about a link.
#[derive(Debug, PartialEq)]
enum Answer {
    Track(Map<String, Value>),
    /// The tracks, and the count the page's header gives (more than listed: cut).
    Tracks(Vec<Value>, Option<u64>),
}

/// An album's or playlist's page: the header (artist for albums, the song count) and the
/// `trackLockup` rows.
fn read_list(page: &Value) -> Option<Answer> {
    let secs = sections(page);
    let header = secs.iter().find(|s| text(s, "itemKind") == "containerDetailHeaderLockup").and_then(|s| s.pointer("/items/0")).cloned().unwrap_or(Value::Null);
    let album_artist = link_titles(&header);
    let count = header.get("trackCount").and_then(Value::as_u64);
    let mut tracks = Vec::new();
    for s in secs.iter().filter(|s| text(s, "itemKind") == "trackLockup") {
        for it in s.get("items").and_then(Value::as_array).into_iter().flatten() {
            if text(it.get("contentDescriptor").unwrap_or(&Value::Null), "kind") == "musicVideo" {
                continue;
            }
            let artist = Some(link_titles(it)).filter(|a| !a.is_empty()).unwrap_or_else(|| album_artist.clone());
            let url = it.pointer("/contentDescriptor/url").and_then(Value::as_str).unwrap_or("").to_string();
            if let Some(m) = track_shape(text(it, "title"), artist, it.get("duration").and_then(Value::as_u64), url) {
                tracks.push(Value::Object(m));
            }
        }
    }
    tracks.truncate(MAX_TRACKS);
    (!tracks.is_empty()).then_some(Answer::Tracks(tracks, count))
}

/// A song: the `songDetailItemLockup` with this `songId` (song pages), else the album page's
/// `trackLockup` whose id is the song's (album links with `?i=`).
fn read_song(page: &Value, id: &str) -> Option<Answer> {
    let secs = sections(page);
    for s in &secs {
        for it in s.get("items").and_then(Value::as_array).into_iter().flatten() {
            if text(it, "songId") == id {
                let url = format!("https://music.apple.com/song/{id}");
                return track_shape(text(it, "name"), text(it, "artistName"), None, url).map(Answer::Track);
            }
        }
    }
    let Some(Answer::Tracks(tracks, _)) = read_list(page) else { return None };
    tracks
        .into_iter()
        .find(|t| t.get("url").and_then(Value::as_str).is_some_and(|u| u.ends_with(&format!("?i={id}"))))
        .and_then(|t| t.as_object().cloned())
        .map(Answer::Track)
}

/// The page to fetch for one row, or `None` when the row is not ours.
fn page_url(link: &Link) -> Option<(String, &str)> {
    if link.provider != "apple-music" {
        return None;
    }
    let kind = match link.kind.as_str() {
        k @ ("track" | "album" | "playlist") => k,
        _ => return None,
    };
    let id = apple_id(kind, &link.id)?;
    let store = storefront(&link.url);
    // The slug is ignored by the site ("x" serves the same page).
    let url = match kind {
        "playlist" => format!("https://music.apple.com/{store}/playlist/x/{id}"),
        "album" => format!("https://music.apple.com/{store}/album/x/{id}"),
        _ => format!("https://music.apple.com/{store}/song/x/{id}"),
    };
    Some((url, kind))
}

/// Resolve the input document with `fetch` (URL → page body) and `note` (log line).
fn run(input: &[u8], mut fetch: impl FnMut(&str) -> Option<String>, mut note: impl FnMut(&str)) -> Value {
    let input: Input = match serde_json::from_slice(input) {
        Ok(i) => i,
        Err(e) => {
            note(&format!("input is not the rows document: {e}"));
            return json!({ "rows": [] });
        }
    };
    let mut rows = Vec::new();
    for row in input.rows.iter().take(10_000) {
        let Some(link) = &row.link else { continue };
        let Some((url, kind)) = page_url(link) else { continue };
        let Some(html) = fetch(&url) else { continue };
        let Some(page) = page_data(&html) else {
            note(&format!("{url}: no page data (the page changed?)"));
            continue;
        };
        let answer = if kind == "track" { read_song(&page, &link.id) } else { read_list(&page) };
        match answer {
            Some(Answer::Track(mut m)) => {
                m.insert("line".into(), Value::from(row.line));
                rows.push(Value::Object(m));
            }
            Some(Answer::Tracks(tracks, count)) => {
                if let Some(n) = count.filter(|n| *n as usize > tracks.len()) {
                    note(&format!("{url}: the page lists {} of {n} songs", tracks.len()));
                }
                rows.push(json!({ "line": row.line, "tracks": tracks }));
            }
            None => note(&format!("{url}: no {kind} on the page (private, removed, or the page changed)")),
        }
    }
    json!({ "rows": rows })
}

#[cfg(all(target_arch = "wasm32", not(test)))]
mod host {
    use alloc::format;
    use alloc::string::{String, ToString};
    use alloc::vec;

    #[link(wasm_import_module = "mixcraft")]
    unsafe extern "C" {
        fn log(ptr: *const u8, len: i32);
        fn emit(ptr: *const u8, len: i32);
        fn http_get(url_ptr: *const u8, url_len: i32, accept_ptr: *const u8, accept_len: i32) -> i32;
        fn result_len() -> i32;
        fn result_read(ptr: *mut u8, cap: i32) -> i32;
    }

    /// A bump allocator over a static arena: the host writes inputs here, the module keeps its
    /// own heap in the same arena. The host instantiates the module per `match.resolve` call.
    struct Bump;

    const ARENA_BYTES: usize = 48 * 1024 * 1024;
    static mut ARENA: [u8; ARENA_BYTES] = [0; ARENA_BYTES];
    static mut NEXT: usize = 0;

    unsafe impl core::alloc::GlobalAlloc for Bump {
        unsafe fn alloc(&self, layout: core::alloc::Layout) -> *mut u8 {
            // SAFETY: wasm is single-threaded; the arena is a static the module alone owns.
            unsafe {
                let align = layout.align().max(8);
                let start = (NEXT + align - 1) & !(align - 1);
                let end = match start.checked_add(layout.size()) {
                    Some(e) if e <= ARENA_BYTES => e,
                    _ => return core::ptr::null_mut(),
                };
                NEXT = end;
                core::ptr::addr_of_mut!(ARENA).cast::<u8>().add(start)
            }
        }

        unsafe fn dealloc(&self, _: *mut u8, _: core::alloc::Layout) {}
    }

    #[global_allocator]
    static GLOBAL: Bump = Bump;

    #[panic_handler]
    fn on_panic(_: &core::panic::PanicInfo) -> ! {
        // `panic = "abort"`: a trap ends the call; the host reports "faulted".
        core::arch::wasm32::unreachable()
    }

    /// The host writes inputs into memory the module hands out.
    #[unsafe(no_mangle)]
    pub extern "C" fn alloc(len: i32) -> i32 {
        let Ok(size) = usize::try_from(len) else { return 0 };
        let Ok(layout) = core::alloc::Layout::from_size_align(size.max(1), 8) else { return 0 };
        // SAFETY: the layout is valid; a null result (arena full) is reported as 0.
        let p = unsafe { core::alloc::GlobalAlloc::alloc(&GLOBAL, layout) };
        i32::try_from(p as usize).unwrap_or(0)
    }

    fn log_line(s: &str) {
        // SAFETY: the pointer and length describe a live string.
        unsafe { log(s.as_ptr(), i32::try_from(s.len()).unwrap_or(i32::MAX)) }
    }

    /// `GET` an embed page from a granted host; `None` on refusal, failure or a non-2xx status.
    fn get_page(url: &str) -> Option<String> {
        const ACCEPT: &str = "text/html";
        // SAFETY: both buffers are live for the call.
        let status = unsafe { http_get(url.as_ptr(), i32::try_from(url.len()).ok()?, ACCEPT.as_ptr(), i32::try_from(ACCEPT.len()).ok()?) };
        if !(200..300).contains(&status) {
            log_line(&format!("{url}: status {status}"));
            return None;
        }
        // SAFETY: result_len/result_read copy the pending body into a buffer we own.
        let len = usize::try_from(unsafe { result_len() }).ok()?;
        let mut buf = vec![0u8; len];
        let got = unsafe { result_read(buf.as_mut_ptr(), i32::try_from(len).ok()?) };
        buf.truncate(usize::try_from(got).ok()?.min(len));
        Some(String::from_utf8_lossy(&buf).to_string())
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn resolve(ptr: *const u8, len: i32) {
        let Ok(n) = usize::try_from(len) else { return };
        // SAFETY: the host wrote `len` bytes at `ptr` through `alloc`.
        let bytes = unsafe { core::slice::from_raw_parts(ptr, n) };
        let answer = super::run(bytes, get_page, log_line).to_string();
        // SAFETY: the pointer and length describe a live string.
        unsafe { emit(answer.as_ptr(), i32::try_from(answer.len()).unwrap_or(i32::MAX)) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PLAYLIST: &str = include_str!("../tests/fixtures/playlist.html");
    const ALBUM: &str = include_str!("../tests/fixtures/album.html");
    const SONG: &str = include_str!("../tests/fixtures/song.html");
    const NOT_FOUND: &str = include_str!("../tests/fixtures/not-found.html");

    fn pages(url: &str) -> Option<String> {
        let page = match url {
            "https://music.apple.com/de/playlist/x/pl.u-fixture01" => PLAYLIST,
            "https://music.apple.com/us/album/x/2000" => ALBUM,
            "https://music.apple.com/us/song/x/3001" => SONG,
            "https://music.apple.com/us/album/x/2002" => ALBUM,
            "https://music.apple.com/us/song/x/2002" => ALBUM,
            "https://music.apple.com/gb/playlist/x/pl.gone" => NOT_FOUND,
            _ => return None,
        };
        Some(page.to_string())
    }

    fn resolve(input: Value) -> (Value, Vec<String>, Vec<String>) {
        let mut fetched = Vec::new();
        let mut notes = Vec::new();
        let out = run(
            input.to_string().as_bytes(),
            |u| {
                fetched.push(u.to_string());
                pages(u)
            },
            |n| notes.push(n.to_string()),
        );
        (out, fetched, notes)
    }

    fn link_row(line: u64, kind: &str, id: &str, url: &str) -> Value {
        json!({ "line": line, "link": { "provider": "apple-music", "kind": kind, "id": id, "url": url }, "artist": "", "title": "" })
    }

    #[test]
    fn a_playlist_answers_its_songs_skips_videos_and_notes_a_cut_list() {
        let (out, fetched, notes) =
            resolve(json!({ "rows": [link_row(1, "playlist", "pl.u-fixture01", "https://music.apple.com/de/playlist/fixture-hits/pl.u-fixture01")] }));
        assert_eq!(fetched, ["https://music.apple.com/de/playlist/x/pl.u-fixture01"], "the storefront of the link is kept");
        let tracks = out["rows"][0]["tracks"].as_array().cloned().unwrap_or_default();
        assert_eq!(tracks.len(), 2, "the music video is left out: {tracks:?}");
        assert_eq!(tracks[0]["artist"], "Band One, Guest Two");
        assert_eq!(tracks[0]["duration_ms"], 201000);
        assert_eq!(tracks[1]["artist"], "Earth, Sky & Sea");
        assert_eq!(tracks[1]["url"], "https://music.apple.com/de/album/second/1003?i=1004");
        assert!(notes.iter().any(|n| n.contains("2 of 120")), "{notes:?}");
    }

    #[test]
    fn album_songs_take_the_album_artist() {
        let (out, _, notes) = resolve(json!({ "rows": [link_row(4, "album", "2000", "https://music.apple.com/us/album/fixture-album/2000")] }));
        assert_eq!(
            out["rows"][0]["tracks"][1],
            json!({ "title": "Side B", "artist": "Album Artist", "duration_ms": 600293, "url": "https://music.apple.com/us/album/side-b/2000?i=2002" })
        );
        assert!(notes.is_empty(), "{notes:?}");
    }

    #[test]
    fn songs_come_from_the_song_page_or_the_album_row() {
        let (out, _, _) = resolve(json!({ "rows": [
            link_row(2, "track", "3001", "https://music.apple.com/us/song/first-light/3001"),
            link_row(3, "track", "2002", "https://music.apple.com/us/album/fixture-album/2000?i=2002"),
        ] }));
        assert_eq!(out["rows"][0], json!({ "line": 2, "title": "First Light", "artist": "Example Artist", "url": "https://music.apple.com/song/3001" }));
        assert_eq!(out["rows"][1]["title"], "Side B");
        assert_eq!(out["rows"][1]["artist"], "Album Artist");
        assert_eq!(out["rows"][1]["line"], 3);
    }

    #[test]
    fn missing_pages_other_providers_and_odd_ids_are_left_out() {
        let (out, fetched, notes) = resolve(json!({ "rows": [
            link_row(1, "playlist", "pl.gone", "https://music.apple.com/gb/playlist/gone/pl.gone"),
            link_row(2, "track", "12ab", "https://music.apple.com/us/song/x/12ab"),
            link_row(3, "playlist", "../etc", "https://music.apple.com/us/playlist/x/../etc"),
            json!({ "line": 4, "link": { "provider": "spotify", "kind": "track", "id": "1", "url": "" } }),
            json!({ "line": 5, "artist": "Words", "title": "Only" }),
        ] }));
        assert_eq!(out, json!({ "rows": [] }));
        assert_eq!(fetched, ["https://music.apple.com/gb/playlist/x/pl.gone"]);
        assert_eq!(notes.len(), 1, "{notes:?}");
    }

    #[test]
    fn the_storefront_falls_back_to_us() {
        assert_eq!(storefront("https://music.apple.com/de/album/x/1"), "de");
        assert_eq!(storefront("https://music.apple.com/album/x/1"), "us");
        assert_eq!(storefront("https://itunes.apple.com/US/album/x/1"), "us");
        assert_eq!(storefront("not a link"), "us");
    }

    #[test]
    fn bad_input_answers_no_rows() {
        let mut notes = Vec::new();
        assert_eq!(run(b"not json", |_| None, |n| notes.push(n.to_string())), json!({ "rows": [] }));
        assert_eq!(notes.len(), 1);
    }
}
