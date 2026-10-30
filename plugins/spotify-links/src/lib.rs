//! MixCraft resolver plug-in: Spotify links through Spotify's public embed pages.
//!
//! The host hands `resolve` a JSON document `{"rows":[{"line", "link":{provider, kind, id,
//! url}, …}]}` with the link rows of a pasted request list. For every row whose provider is
//! `spotify` this module fetches the public embed page `open.spotify.com/embed/<kind>/<id>`
//! (the page a website shows for an embedded Spotify player; no account, no key), reads the
//! `__NEXT_DATA__` JSON the page carries and answers `{"rows":[…]}`: a track row gets `artist`,
//! `title`, `duration_ms` and `url`; an album or playlist row gets `tracks: [...]`. Rows of
//! other providers, and links Spotify does not show, are left out of the answer (the host keeps
//! them for the next resolver).
//!
//! The embed page is not an API: Spotify can change it at any time, which breaks this plug-in
//! until it is updated, and it lists at most the first 100 tracks of a playlist. Reading it
//! this way is not something Spotify's terms allow; `TERMS.md` says so before the user turns
//! the plug-in on.
//!
//! Host ABI (`docs/plugins.md` §4): imports `mixcraft.http_get`, `result_len`, `result_read`,
//! `emit`, `log`; exports `memory`, `alloc`, `resolve`. Built for `wasm32-unknown-unknown`
//! without WASI: no files, no clock, no threads. The parsing below is plain Rust and is tested
//! on the build machine with `cargo test`.
#![cfg_attr(all(target_arch = "wasm32", not(test)), no_std)]
#![cfg_attr(not(any(test, target_arch = "wasm32")), allow(dead_code))]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

extern crate alloc;

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use serde::Deserialize;
use serde_json::{Map, Value, json};

/// Most tracks answered for one album or playlist (the host caps at 2,000; the embed page
/// lists at most 100).
const MAX_TRACKS: usize = 2_000;
/// The embed page's own cap on a playlist's track list.
const EMBED_TRACK_CAP: usize = 100;

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
}

/// Spotify ids are base-62 (22 characters today); anything else is not sent.
fn spotify_id(id: &str) -> Option<&str> {
    (!id.is_empty() && id.len() <= 64 && id.bytes().all(|b| b.is_ascii_alphanumeric())).then_some(id)
}

/// The JSON inside `<script id="__NEXT_DATA__" …>…</script>`.
fn next_data(html: &str) -> Option<Value> {
    let at = html.find("id=\"__NEXT_DATA__\"")?;
    let rest = html.get(at..)?;
    let open = rest.find('>')? + 1;
    let body = rest.get(open..)?;
    let close = body.find("</script>")?;
    serde_json::from_str(body.get(..close)?).ok()
}

/// The page's entity: `props.pageProps.state.data.entity` (absent on a "page not found").
fn entity(page: &Value) -> Option<&Value> {
    page.pointer("/props/pageProps/state/data/entity").filter(|e| e.is_object())
}

fn text(v: &Value, key: &str) -> String {
    v.get(key).and_then(Value::as_str).unwrap_or("").trim().to_string()
}

/// The page joins artists with a comma and a no-break space; MixCraft's matcher reads ", ".
fn artists_text(s: &str) -> String {
    s.replace(",\u{a0}", ", ").replace('\u{a0}', " ").trim().to_string()
}

/// `spotify:track:<id>` → `https://open.spotify.com/track/<id>`.
fn track_url(uri: &str) -> Option<String> {
    let id = uri.strip_prefix("spotify:track:")?;
    spotify_id(id).map(|id| format!("https://open.spotify.com/track/{id}"))
}

/// The answer's track shape from its parts; `None` without a title.
fn track_shape(title: String, artist: String, duration_ms: Option<u64>, url: Option<String>) -> Option<Map<String, Value>> {
    if title.is_empty() {
        return None;
    }
    let mut m = Map::new();
    m.insert("title".into(), Value::String(title));
    m.insert("artist".into(), Value::String(artist));
    if let Some(d) = duration_ms.filter(|d| *d > 0) {
        m.insert("duration_ms".into(), Value::from(d));
    }
    if let Some(u) = url {
        m.insert("url".into(), Value::String(u));
    }
    Some(m)
}

/// One entry of an album's or playlist's `trackList` (podcast episodes are skipped).
fn list_entry(t: &Value) -> Option<Value> {
    let kind = text(t, "entityType");
    if !kind.is_empty() && kind != "track" {
        return None;
    }
    let url = track_url(&text(t, "uri"));
    track_shape(text(t, "title"), artists_text(&text(t, "subtitle")), t.get("duration").and_then(Value::as_u64), url).map(Value::Object)
}

/// What one embed page says about a link.
#[derive(Debug, PartialEq)]
enum Answer {
    /// A track page: the track itself.
    Track(Map<String, Value>),
    /// An album or playlist page: its tracks, and whether the page's list looks cut at the
    /// embed cap.
    Tracks(Vec<Value>, bool),
}

/// Read an embed page for a link of `kind`; `None` when the page holds nothing usable.
fn read_page(html: &str, kind: &str) -> Option<Answer> {
    let page = next_data(html)?;
    let e = entity(&page)?;
    match kind {
        "track" => {
            if text(e, "type") != "track" {
                return None;
            }
            let artist = e
                .get("artists")
                .and_then(Value::as_array)
                .map(|a| a.iter().map(|x| text(x, "name")).filter(|n| !n.is_empty()).collect::<Vec<_>>().join(", "))
                .filter(|a| !a.is_empty())
                .unwrap_or_else(|| artists_text(&text(e, "subtitle")));
            let url = track_url(&text(e, "uri"));
            track_shape(text(e, "name"), artist, e.get("duration").and_then(Value::as_u64), url).map(Answer::Track)
        }
        "album" | "playlist" => {
            if text(e, "type") != kind {
                return None;
            }
            let list = e.get("trackList").and_then(Value::as_array)?;
            let mut tracks: Vec<Value> = list.iter().filter_map(list_entry).collect();
            tracks.truncate(MAX_TRACKS);
            if tracks.is_empty() {
                return None;
            }
            Some(Answer::Tracks(tracks, kind == "playlist" && list.len() >= EMBED_TRACK_CAP))
        }
        _ => None,
    }
}

/// The work for one input row: the URL to fetch, or `None` when the row is not ours.
fn embed_url(row: &Row) -> Option<(String, &str)> {
    let link = row.link.as_ref()?;
    if link.provider != "spotify" {
        return None;
    }
    let id = spotify_id(&link.id)?;
    let kind = match link.kind.as_str() {
        k @ ("track" | "album" | "playlist") => k,
        _ => return None,
    };
    Some((format!("https://open.spotify.com/embed/{kind}/{id}"), kind))
}

/// Resolve the input document with `fetch` (URL → page body) and `note` (log line); returns
/// the answer document. Kept apart from the host calls so it can be tested.
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
        let Some((url, kind)) = embed_url(row) else { continue };
        let Some(html) = fetch(&url) else { continue };
        match read_page(&html, kind) {
            Some(Answer::Track(mut m)) => {
                m.insert("line".into(), Value::from(row.line));
                rows.push(Value::Object(m));
            }
            Some(Answer::Tracks(tracks, cut)) => {
                if cut {
                    note(&format!("{url}: the embed page lists only the first {EMBED_TRACK_CAP} tracks of a playlist"));
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

    const TRACK: &str = include_str!("../tests/fixtures/track.html");
    const PLAYLIST: &str = include_str!("../tests/fixtures/playlist.html");
    const ALBUM: &str = include_str!("../tests/fixtures/album.html");
    const NOT_FOUND: &str = include_str!("../tests/fixtures/not-found.html");

    fn pages(url: &str) -> Option<String> {
        let page = match url {
            "https://open.spotify.com/embed/track/0000000000000000000001" => TRACK,
            "https://open.spotify.com/embed/playlist/0000000000000000000002" => PLAYLIST,
            "https://open.spotify.com/embed/album/0000000000000000000003" => ALBUM,
            "https://open.spotify.com/embed/playlist/0000000000000000000004" => NOT_FOUND,
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

    fn link_row(line: u64, provider: &str, kind: &str, id: &str) -> Value {
        json!({ "line": line, "link": { "provider": provider, "kind": kind, "id": id, "url": "" }, "artist": "", "title": "" })
    }

    #[test]
    fn a_track_page_answers_artist_title_duration_and_url() {
        let (out, _, notes) = resolve(json!({ "rows": [link_row(3, "spotify", "track", "0000000000000000000001")] }));
        assert_eq!(
            out,
            json!({ "rows": [{ "line": 3, "title": "First Light", "artist": "Example Artist, Second Voice", "duration_ms": 213573, "url": "https://open.spotify.com/track/0000000000000000000001" }] })
        );
        assert!(notes.is_empty(), "{notes:?}");
    }

    #[test]
    fn a_playlist_page_answers_its_tracks_and_skips_episodes() {
        let (out, _, _) = resolve(json!({ "rows": [link_row(1, "spotify", "playlist", "0000000000000000000002")] }));
        let tracks = out["rows"][0]["tracks"].as_array().cloned().unwrap_or_default();
        assert_eq!(out["rows"][0]["line"], 1);
        assert_eq!(tracks.len(), 3, "the episode is left out: {tracks:?}");
        assert_eq!(tracks[0]["artist"], "Band One, Guest Two", "the no-break space after the comma becomes a space");
        assert_eq!(tracks[1]["artist"], "Earth, Sky & Sea", "an artist name with a comma stays whole");
        assert_eq!(tracks[2]["title"], "Third");
        assert_eq!(tracks[2]["url"], "https://open.spotify.com/track/000000000000000000000c");
        assert!(tracks[2].get("duration_ms").is_none(), "a zero duration is left out");
    }

    #[test]
    fn an_album_page_answers_its_tracks() {
        let (out, _, _) = resolve(json!({ "rows": [link_row(7, "spotify", "album", "0000000000000000000003")] }));
        assert_eq!(out["rows"][0]["tracks"][1]["title"], "Side B");
        assert_eq!(out["rows"][0]["tracks"][1]["duration_ms"], 600293);
    }

    #[test]
    fn missing_pages_other_providers_and_odd_ids_are_left_out() {
        let (out, fetched, notes) = resolve(json!({ "rows": [
            link_row(1, "spotify", "playlist", "0000000000000000000004"),
            link_row(2, "deezer", "track", "3135556"),
            link_row(3, "spotify", "track", "../../etc"),
            link_row(4, "spotify", "artist", "0000000000000000000001"),
            link_row(5, "spotify", "track", "0000000000000000000009"),
            { "line": 6, "artist": "Words", "title": "Only" },
        ] }));
        assert_eq!(out, json!({ "rows": [] }));
        assert_eq!(
            fetched,
            ["https://open.spotify.com/embed/playlist/0000000000000000000004", "https://open.spotify.com/embed/track/0000000000000000000009"],
            "only well-formed Spotify track, album and playlist ids are fetched"
        );
        assert_eq!(notes.len(), 1, "the page-not-found is logged: {notes:?}");
    }

    #[test]
    fn a_track_page_is_not_read_as_a_playlist() {
        assert_eq!(read_page(TRACK, "playlist"), None);
        assert_eq!(read_page(PLAYLIST, "track"), None);
        assert_eq!(read_page("<html>no data</html>", "track"), None);
        assert_eq!(read_page("<script id=\"__NEXT_DATA__\" type=\"application/json\">{not json</script>", "track"), None);
    }

    #[test]
    fn a_full_playlist_page_notes_the_embed_cap() {
        let list: Vec<Value> = (0..EMBED_TRACK_CAP)
            .map(|i| json!({ "uri": format!("spotify:track:{i:022}"), "title": format!("T{i}"), "subtitle": "A", "duration": 1000, "entityType": "track" }))
            .collect();
        let doc = json!({ "props": { "pageProps": { "state": { "data": { "entity": { "type": "playlist", "trackList": list } } } } } });
        let html = format!("<script id=\"__NEXT_DATA__\" type=\"application/json\">{doc}</script>");
        let answer = read_page(&html, "playlist");
        assert!(matches!(&answer, Some(Answer::Tracks(t, true)) if t.len() == EMBED_TRACK_CAP), "{answer:?}");
    }

    #[test]
    fn bad_input_answers_no_rows() {
        let mut notes = Vec::new();
        let out = run(b"not json", |_| None, |n| notes.push(n.to_string()));
        assert_eq!(out, json!({ "rows": [] }));
        assert_eq!(notes.len(), 1);
    }
}
