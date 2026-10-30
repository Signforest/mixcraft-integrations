//! MixCraft resolver plug-in: YouTube and YouTube Music links.
//!
//! The host hands `resolve` a JSON document `{"rows":[{"line", "link":{provider, kind, id,
//! url}, …}]}` with the link rows of a pasted request list. For every row whose provider is
//! `youtube`:
//!
//! - a **video** link asks YouTube's oEmbed endpoint (`www.youtube.com/oembed`, documented, no
//!   key) for the video's title and channel;
//! - a **playlist** link (YouTube or YouTube Music, `list=`) loads the public playlist page
//!   (`www.youtube.com/playlist?list=…`) and reads the `ytInitialData` JSON in it: each video's
//!   title, channel and length. The page lists the first 100 videos; more need the site's own
//!   script, so a longer playlist is cut there (the plug-in logs it).
//!
//! A video title is not "artist, title": [`split_title`] takes "Artist - Title (Official
//! Video)" apart, drops the video noise in brackets, notices when the channel's name sits
//! after the dash instead, and falls back to the channel (without " - Topic") as the artist.
//!
//! The playlist page is not an API: YouTube can change it at any time, which breaks playlists
//! until the plug-in is updated, and reading it this way is not something YouTube's terms
//! allow; `TERMS.md` says so before the user turns the plug-in on.
//!
//! Host ABI (`docs/plugins.md` §4): imports `mixcraft.http_get`, `result_len`, `result_read`,
//! `emit`, `log`; exports `memory`, `alloc`, `resolve`. Built for `wasm32-unknown-unknown`
//! without WASI. The parsing below is plain Rust and is tested with `cargo test`.
#![cfg_attr(all(target_arch = "wasm32", not(test)), no_std)]
#![cfg_attr(not(any(test, target_arch = "wasm32")), allow(dead_code))]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

extern crate alloc;

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use serde::Deserialize;
use serde_json::{Map, Value, json};

/// Most tracks answered for one playlist (the page itself lists at most 100).
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
}

/// Video ids are 11 characters of `A-Z a-z 0-9 - _`; playlist ids are longer runs of the same.
fn yt_id<'a>(kind: &str, id: &'a str) -> Option<&'a str> {
    let chars = id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_');
    let len = match kind {
        "track" => id.len() == 11,
        _ => (2..=64).contains(&id.len()),
    };
    (chars && len).then_some(id)
}

fn text(v: &Value, key: &str) -> String {
    v.get(key).and_then(Value::as_str).unwrap_or("").trim().to_string()
}

/// Words that mark a bracketed part of a video title as video noise, not part of the song's
/// title ("(Official Video)", "[Lyrics]", "(Video Oficial)", "(4K Remaster)").
const NOISE: &[&str] = &[
    "official",
    "oficial",
    "officiel",
    "offiziell",
    "video",
    "vídeo",
    "clip",
    "audio",
    "lyric",
    "lyrics",
    "letra",
    "visualizer",
    "visualiser",
    "hd",
    "hq",
    "4k",
    "mv",
    "m/v",
    "music video",
    "full song",
    "explicit",
];

/// Lowercase letters and digits only, for comparing names.
fn squash(s: &str) -> String {
    s.chars().filter(|c| c.is_alphanumeric()).flat_map(char::to_lowercase).collect()
}

/// Drop bracketed parts that are video noise; keep the rest ("(feat. X)", "(Extended Mix)").
fn strip_noise(title: &str) -> String {
    let mut out = String::new();
    let mut rest = title;
    while let Some(open) = rest.find(['(', '[']) {
        let close_char = if rest[open..].starts_with('(') { ')' } else { ']' };
        let Some(len) = rest[open + 1..].find(close_char) else { break };
        let inner = &rest[open + 1..open + 1 + len];
        let lower = inner.to_lowercase();
        let noise =
            lower.split(|c: char| !c.is_alphanumeric() && c != '/').any(|w| NOISE.contains(&w)) || NOISE.iter().any(|n| n.contains(' ') && lower.contains(n));
        out.push_str(&rest[..open]);
        if !noise {
            out.push_str(&rest[open..open + len + 2]);
        }
        rest = &rest[open + len + 2..];
    }
    out.push_str(rest);
    out.split_whitespace().collect::<Vec<_>>().join(" ").trim_matches(|c: char| c == '-' || c == '|' || c.is_whitespace()).to_string()
}

/// Only video words ("VIDEO OFICIAL", "Official Audio").
fn is_noise(part: &str) -> bool {
    let lower = part.to_lowercase();
    let mut words = lower.split(|c: char| !c.is_alphanumeric() && c != '/').filter(|w| !w.is_empty()).peekable();
    words.peek().is_some() && words.all(|w| NOISE.contains(&w))
}

/// The channel's name as an artist: without " - Topic" (YouTube Music's artist channels) and
/// a trailing "VEVO" / "Official".
fn channel_artist(channel: &str) -> String {
    let c = channel.trim();
    let c = c.strip_suffix(" - Topic").unwrap_or(c);
    let c = c.strip_suffix("VEVO").unwrap_or(c);
    let c = c.strip_suffix(" Official").unwrap_or(c);
    c.trim().to_string()
}

/// Unbracketed video words at the end of a title ("… Official Performance Video").
const TRAILING_NOISE: &[&str] = &[
    "official performance video",
    "official music video",
    "official lyric video",
    "official visualizer",
    "official video",
    "official audio",
    "lyric video",
    "music video",
    "visualizer",
];

fn strip_trailing_noise(title: &str) -> String {
    let mut t = title.trim().to_string();
    loop {
        let lower = t.to_lowercase();
        let Some(n) = TRAILING_NOISE.iter().find(|n| lower.ends_with(*n)) else { break };
        // Byte lengths match: the phrases are ASCII and lowercasing ASCII keeps lengths.
        let Some(cut) = t.len().checked_sub(n.len()).filter(|c| t.is_char_boundary(*c)) else { break };
        t.truncate(cut);
        t = t.trim_end_matches(|c: char| c.is_whitespace() || c == '-' || c == '|').to_string();
    }
    t
}

/// The first quoted part of a title (‘…’, “…”, "…"): (before, inside, after).
fn quoted(s: &str) -> Option<(&str, &str, &str)> {
    // Not the plain apostrophe: "Guns N' Roses", "Sweet Child O' Mine".
    for (open, close) in [('‘', '’'), ('“', '”'), ('"', '"')] {
        let Some(a) = s.find(open) else { continue };
        let inner_start = a + open.len_utf8();
        let Some(len) = s[inner_start..].find(close) else { continue };
        let inner = &s[inner_start..inner_start + len];
        if inner.trim().is_empty() {
            continue;
        }
        return Some((&s[..a], inner, &s[inner_start + len + close.len_utf8()..]));
    }
    None
}

/// A video title and its channel → (artist, title).
fn split_title(video_title: &str, channel: &str) -> (String, String) {
    let clean = strip_trailing_noise(&strip_noise(video_title));
    let artist_from_channel = channel_artist(channel);
    // `Artist 'Title' …`: the quoted part is the title when no dash comes before it.
    if let Some((before, inner, after)) = quoted(&clean) {
        let has_sep = |t: &str| [" - ", " – ", " — ", " | "].iter().any(|sep| t.contains(sep));
        let before = before.trim().trim_end_matches(['-', '–', '—', ':']).trim();
        if !before.is_empty() && !has_sep(before) && !has_sep(after) {
            return (before.to_string(), inner.trim().to_string());
        }
    }
    for sep in [" - ", " – ", " — ", " | "] {
        if let Some((a, b)) = clean.split_once(sep) {
            let (a, b) = (a.trim(), b.trim());
            if a.is_empty() || b.is_empty() {
                continue;
            }
            // "Title - Video Oficial": the other side is only video words.
            if is_noise(b) {
                return (artist_from_channel, a.to_string());
            }
            // "Title - Artist": the channel's name is after the dash, not before it.
            let ch = squash(&artist_from_channel);
            if !ch.is_empty() && squash(b).contains(&ch) && !squash(a).contains(&ch) {
                return (b.to_string(), a.to_string());
            }
            return (a.to_string(), b.to_string());
        }
    }
    (artist_from_channel, clean)
}

/// "3:55" or "1:02:03" → milliseconds.
fn clock_ms(s: &str) -> Option<u64> {
    let mut total = 0u64;
    let mut parts = 0;
    for p in s.trim().split(':') {
        total = total * 60 + p.parse::<u64>().ok()?;
        parts += 1;
    }
    (2..=3).contains(&parts).then_some(total * 1000)
}

fn track_shape(video_title: &str, channel: &str, duration_ms: Option<u64>, video_id: &str) -> Option<Map<String, Value>> {
    let (artist, title) = split_title(video_title, channel);
    if title.is_empty() {
        return None;
    }
    let mut m = Map::new();
    m.insert("title".into(), Value::String(title));
    m.insert("artist".into(), Value::String(artist));
    if let Some(d) = duration_ms.filter(|d| *d > 0) {
        m.insert("duration_ms".into(), Value::from(d));
    }
    if yt_id("track", video_id).is_some() {
        m.insert("url".into(), Value::String(format!("https://www.youtube.com/watch?v={video_id}")));
    }
    Some(m)
}

/// The oEmbed answer for one video.
fn read_oembed(body: &str, id: &str) -> Option<Map<String, Value>> {
    let v: Value = serde_json::from_str(body).ok()?;
    track_shape(&text(&v, "title"), &text(&v, "author_name"), None, id)
}

/// The JSON after `var ytInitialData = ` up to `;</script>`.
fn initial_data(html: &str) -> Option<Value> {
    const KEY: &str = "var ytInitialData = ";
    let at = html.find(KEY)? + KEY.len();
    let rest = html.get(at..)?;
    let end = rest.find(";</script>")?;
    serde_json::from_str(rest.get(..end)?).ok()
}

/// Every object under `key` anywhere in `v`, in document order.
fn collect<'a>(v: &'a Value, key: &str, out: &mut Vec<&'a Value>) {
    match v {
        Value::Object(m) => {
            if let Some(x) = m.get(key) {
                out.push(x);
                return;
            }
            m.values().for_each(|x| collect(x, key, out));
        }
        Value::Array(a) => a.iter().for_each(|x| collect(x, key, out)),
        _ => {}
    }
}

/// Every string in `v` (for the header's counts).
fn strings<'a>(v: &'a Value, out: &mut Vec<&'a str>) {
    match v {
        Value::String(s) => out.push(s),
        Value::Object(m) => m.values().for_each(|x| strings(x, out)),
        Value::Array(a) => a.iter().for_each(|x| strings(x, out)),
        _ => {}
    }
}

/// The leading number of "182 videos" / "84 unavailable videos are hidden".
fn leading_number(s: &str) -> Option<u64> {
    let digits: String = s.trim().chars().take_while(|c| c.is_ascii_digit() || *c == ',' || *c == '.').filter(char::is_ascii_digit).collect();
    digits.parse().ok()
}

/// A playlist page: its videos, and how many the playlist has that the page could show
/// (its count less the hidden unavailable ones).
fn read_playlist(html: &str) -> Option<(Vec<Value>, Option<u64>)> {
    let data = initial_data(html)?;
    let mut lockups = Vec::new();
    collect(&data, "lockupViewModel", &mut lockups);
    let mut tracks = Vec::new();
    for l in lockups {
        if text(l, "contentType") != "LOCKUP_CONTENT_TYPE_VIDEO" {
            continue;
        }
        let id = text(l, "contentId");
        let md = l.pointer("/metadata/lockupMetadataViewModel").unwrap_or(&Value::Null);
        let title = md.pointer("/title/content").and_then(Value::as_str).unwrap_or("").trim().to_string();
        let channel = md.pointer("/metadata/contentMetadataViewModel/metadataRows/0/metadataParts/0/text/content").and_then(Value::as_str).unwrap_or("");
        let mut badges = Vec::new();
        collect(l.get("contentImage").unwrap_or(&Value::Null), "thumbnailBadgeViewModel", &mut badges);
        let duration = badges.iter().find_map(|b| clock_ms(&text(b, "text")));
        if let Some(m) = track_shape(&title, channel, duration, &id) {
            tracks.push(Value::Object(m));
        }
    }
    tracks.truncate(MAX_TRACKS);
    let mut header = Vec::new();
    collect(&data, "pageHeaderViewModel", &mut header);
    let mut texts = Vec::new();
    header.iter().for_each(|h| strings(h, &mut texts));
    let count = texts.iter().find(|t| t.ends_with(" videos") || t.ends_with(" video")).and_then(|t| leading_number(t));
    let mut all = Vec::new();
    strings(&data, &mut all);
    let hidden = all.iter().find(|t| t.contains("unavailable video") && t.ends_with("hidden")).and_then(|t| leading_number(t)).unwrap_or(0);
    Some((tracks, count.map(|c| c.saturating_sub(hidden))))
}

/// The address to fetch for one row, or `None` when the row is not ours.
fn page_url(link: &Link) -> Option<(String, &str)> {
    if link.provider != "youtube" {
        return None;
    }
    match link.kind.as_str() {
        "track" => {
            let id = yt_id("track", &link.id)?;
            Some((format!("https://www.youtube.com/oembed?format=json&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3D{id}"), "track"))
        }
        "playlist" => {
            let id = yt_id("playlist", &link.id)?;
            // `ucbcb=1` skips the cookie-consent page some countries get first; `hl=en`
            // keeps the header's counts in English.
            Some((format!("https://www.youtube.com/playlist?list={id}&hl=en&ucbcb=1"), "playlist"))
        }
        _ => None,
    }
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
        let Some(body) = fetch(&url) else { continue };
        if kind == "track" {
            match read_oembed(&body, &link.id) {
                Some(mut m) => {
                    m.insert("line".into(), Value::from(row.line));
                    rows.push(Value::Object(m));
                }
                None => note(&format!("{url}: no video data")),
            }
            continue;
        }
        match read_playlist(&body) {
            Some((tracks, count)) if !tracks.is_empty() => {
                if let Some(n) = count.filter(|n| *n as usize > tracks.len()) {
                    note(&format!("{url}: the page lists {} of {n} videos", tracks.len()));
                }
                rows.push(json!({ "line": row.line, "tracks": tracks }));
            }
            Some(_) => note(&format!("{url}: no videos on the page (private, removed, or the page changed)")),
            None => note(&format!("{url}: no playlist data (a consent or sign-in page, or the page changed)")),
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
    const CONSENT: &str = include_str!("../tests/fixtures/consent.html");
    const OEMBED: &str = include_str!("../tests/fixtures/oembed.json");

    fn pages(url: &str) -> Option<String> {
        let page = match url {
            "https://www.youtube.com/playlist?list=PLfixture01&hl=en&ucbcb=1" => PLAYLIST,
            "https://www.youtube.com/playlist?list=PLconsent&hl=en&ucbcb=1" => CONSENT,
            "https://www.youtube.com/oembed?format=json&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DFGBhQbmPwH8" => OEMBED,
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

    fn link_row(line: u64, kind: &str, id: &str) -> Value {
        json!({ "line": line, "link": { "provider": "youtube", "kind": kind, "id": id, "url": "" }, "artist": "", "title": "" })
    }

    fn split(t: &str, c: &str) -> (String, String) {
        split_title(t, c)
    }

    #[test]
    fn video_titles_become_artist_and_title() {
        let s = |a: &str, b: &str| (a.to_string(), b.to_string());
        assert_eq!(split("Rick Astley - Never Gonna Give You Up (Official Video) (4K Remaster)", "Rick Astley"), s("Rick Astley", "Never Gonna Give You Up"));
        assert_eq!(
            split("BELLAKEO (Video Oficial) - Peso Pluma, Anitta", "Peso Pluma"),
            s("Peso Pluma, Anitta", "BELLAKEO"),
            "title first: the channel is after the dash"
        );
        assert_eq!(split("Finesse2Tymes - Crazy [Official Music Video]", "Finesse2Tymes"), s("Finesse2Tymes", "Crazy"));
        assert_eq!(split("Arcángel - FN8 ( Video Lyric )", "Arcángel"), s("Arcángel", "FN8"));
        assert_eq!(
            split("Phoenix (feat. Alexa Harley)", "Hybrid Minds - Topic"),
            s("Hybrid Minds", "Phoenix (feat. Alexa Harley)"),
            "a Topic channel is the artist"
        );
        assert_eq!(split("One More Time", "DaftPunkVEVO"), s("DaftPunk", "One More Time"));
        assert_eq!(
            split("정국 (Jung Kook), USHER ‘Standing Next to You - USHER Remix’ Official Performance Video", "BANGTANTV"),
            s("정국 (Jung Kook), USHER", "Standing Next to You - USHER Remix"),
            "a quoted title wins over a dash inside it"
        );
        assert_eq!(split("NLE Choppa - Shotta Flow 7 “FINAL”", "NLE Choppa"), s("NLE Choppa", "Shotta Flow 7 “FINAL”"), "quotes after the dash stay");
        assert_eq!(split("Fixture Band - Song Official Music Video", "Fixture Band"), s("Fixture Band", "Song"));
        assert_eq!(split("BENDECIDOS - VIDEO OFICIAL", "Fixture Label"), s("Fixture Label", "BENDECIDOS"));
        assert_eq!(split("El Alfa “El Jefe” X Donaty - TORETTO", "El Alfa"), s("El Alfa “El Jefe” X Donaty", "TORETTO"), "quotes before the dash stay");
        assert_eq!(split("Guns N' Roses - Sweet Child O' Mine (Official Music Video)", "Guns N' Roses"), s("Guns N' Roses", "Sweet Child O' Mine"));
        assert_eq!(split("Song (Extended Mix) [HD]", "Label Records"), s("Label Records", "Song (Extended Mix)"), "versions stay, noise goes");
    }

    #[test]
    fn a_playlist_page_answers_its_videos_with_lengths_and_notes_a_cut_list() {
        let (out, fetched, notes) = resolve(json!({ "rows": [link_row(1, "playlist", "PLfixture01")] }));
        assert_eq!(fetched, ["https://www.youtube.com/playlist?list=PLfixture01&hl=en&ucbcb=1"]);
        let tracks = out["rows"][0]["tracks"].as_array().cloned().unwrap_or_default();
        assert_eq!(tracks.len(), 4, "the nested playlist is left out: {tracks:?}");
        assert_eq!(
            tracks[0],
            json!({ "title": "Never Gonna Give You Up", "artist": "Rick Astley", "duration_ms": 214000, "url": "https://www.youtube.com/watch?v=aaaaaaaaaa1" })
        );
        assert_eq!(tracks[1]["artist"], "Peso Pluma, Anitta");
        assert_eq!(tracks[2]["artist"], "Hybrid Minds");
        assert_eq!(tracks[3]["duration_ms"], 3_723_000);
        assert!(notes.iter().any(|n| n.contains("4 of 110")), "130 videos less 20 hidden: {notes:?}");
    }

    #[test]
    fn a_video_link_asks_oembed() {
        let (out, _, notes) = resolve(json!({ "rows": [link_row(2, "track", "FGBhQbmPwH8")] }));
        assert_eq!(out["rows"][0], json!({ "line": 2, "title": "One More Time", "artist": "Daft Punk", "url": "https://www.youtube.com/watch?v=FGBhQbmPwH8" }));
        assert!(notes.is_empty(), "{notes:?}");
    }

    #[test]
    fn consent_pages_other_providers_and_odd_ids_are_left_out() {
        let (out, fetched, notes) = resolve(json!({ "rows": [
            link_row(1, "playlist", "PLconsent"),
            link_row(2, "track", "short"),
            link_row(3, "playlist", "PL/../x"),
            json!({ "line": 4, "link": { "provider": "spotify", "kind": "track", "id": "1" } }),
            json!({ "line": 5, "artist": "Words", "title": "Only" }),
        ] }));
        assert_eq!(out, json!({ "rows": [] }));
        assert_eq!(fetched, ["https://www.youtube.com/playlist?list=PLconsent&hl=en&ucbcb=1"]);
        assert!(notes.len() == 1 && notes[0].contains("consent"), "{notes:?}");
    }

    #[test]
    fn clocks_and_counts_read() {
        assert_eq!(clock_ms("3:55"), Some(235_000));
        assert_eq!(clock_ms("1:02:03"), Some(3_723_000));
        assert_eq!(clock_ms("LIVE"), None);
        assert_eq!(clock_ms("55"), None);
        assert_eq!(leading_number("1,234 videos"), Some(1234));
    }

    #[test]
    fn bad_input_answers_no_rows() {
        let mut notes = Vec::new();
        assert_eq!(run(b"not json", |_| None, |n| notes.push(n.to_string())), json!({ "rows": [] }));
        assert_eq!(notes.len(), 1);
    }
}
