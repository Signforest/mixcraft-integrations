//! MixCraft resolver plug-in: Deezer links.
//!
//! The host hands `resolve` a JSON document `{"rows":[{"line", "link":{provider, kind, id,
//! url}, …}]}` with the link rows of a pasted request list. For every row whose provider is
//! `deezer` this module asks Deezer's public API (`api.deezer.com`, no key) and answers
//! `{"rows":[…]}`: a track row gets `artist`, `title`, `isrc` and `duration_ms`; an album or
//! playlist row gets `tracks: [...]`. Rows of other providers, and rows Deezer does not know,
//! are left out of the answer (the host keeps them for the next resolver).
//!
//! Host ABI (`docs/plugins.md` §4): imports `mixcraft.http_get`, `result_len`, `result_read`,
//! `emit`, `log`; exports `memory`, `alloc`, `resolve`. Built for `wasm32-unknown-unknown`
//! without WASI: no files, no clock, no threads.
#![no_std]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

extern crate alloc;

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use serde::Deserialize;
use serde_json::{Map, Value, json};

#[link(wasm_import_module = "mixcraft")]
unsafe extern "C" {
    fn log(ptr: *const u8, len: i32);
    fn emit(ptr: *const u8, len: i32);
    fn http_get(url_ptr: *const u8, url_len: i32, accept_ptr: *const u8, accept_len: i32) -> i32;
    fn result_len() -> i32;
    fn result_read(ptr: *mut u8, cap: i32) -> i32;
}

/// A bump allocator over a static arena: the host writes inputs here, the module keeps its own
/// heap in the same arena. Reset per call is unnecessary; the host instantiates the module
/// per `match.resolve` call.
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

fn emit_bytes(b: &[u8]) {
    // SAFETY: the pointer and length describe a live slice.
    unsafe { emit(b.as_ptr(), i32::try_from(b.len()).unwrap_or(i32::MAX)) }
}

/// `GET` a JSON document from a granted host; `None` on refusal, failure or a non-2xx status.
fn get_json(url: &str) -> Option<Value> {
    const ACCEPT: &str = "application/json";
    // SAFETY: both buffers are live for the call.
    let status = unsafe { http_get(url.as_ptr(), i32::try_from(url.len()).ok()?, ACCEPT.as_ptr(), i32::try_from(ACCEPT.len()).ok()?) };
    if !(200..300).contains(&status) {
        log_line(&format!("{url}: status {status}"));
        return None;
    }
    // SAFETY: result_len/result_read copy the pending body into a buffer we own.
    let len = unsafe { result_len() };
    let len = usize::try_from(len).ok()?;
    let mut buf = vec![0u8; len];
    let got = unsafe { result_read(buf.as_mut_ptr(), i32::try_from(len).ok()?) };
    let got = usize::try_from(got).ok()?.min(len);
    buf.truncate(got);
    serde_json::from_slice(&buf).ok()
}

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

/// Deezer ids are decimal numbers; anything else is not sent.
fn numeric_id(id: &str) -> Option<&str> {
    (!id.is_empty() && id.len() <= 20 && id.bytes().all(|b| b.is_ascii_digit())).then_some(id)
}

fn text(v: &Value, key: &str) -> String {
    v.get(key).and_then(Value::as_str).unwrap_or("").trim().to_string()
}

/// One Deezer track object → the answer's track shape.
fn track_json(t: &Value) -> Option<Value> {
    let title = text(t, "title");
    if title.is_empty() {
        return None;
    }
    let artist = t.get("artist").map(|a| text(a, "name")).unwrap_or_default();
    let mut m = Map::new();
    m.insert("title".into(), Value::String(title));
    m.insert("artist".into(), Value::String(artist));
    let isrc = text(t, "isrc");
    if !isrc.is_empty() {
        m.insert("isrc".into(), Value::String(isrc));
    }
    if let Some(d) = t.get("duration").and_then(Value::as_f64).filter(|d| d.is_finite() && *d > 0.0) {
        m.insert("duration_ms".into(), Value::from((d * 1000.0) as u64));
    }
    if let Some(id) = t.get("id").and_then(Value::as_u64) {
        m.insert("url".into(), Value::String(format!("https://www.deezer.com/track/{id}")));
    }
    Some(Value::Object(m))
}

/// Most tracks read for one album or playlist (Deezer pages by 50; the host caps at 2,000).
const MAX_TRACKS: usize = 2_000;
/// Most pages fetched per playlist, within the host's request budget.
const MAX_PAGES: usize = 40;

/// All pages of `/<kind>/<id>/tracks`.
fn list_tracks(kind: &str, id: &str) -> Vec<Value> {
    let mut out = Vec::new();
    let mut url = format!("https://api.deezer.com/{kind}/{id}/tracks?limit=50&index=0");
    for _ in 0..MAX_PAGES {
        let Some(page) = get_json(&url) else { break };
        if let Some(e) = page.get("error") {
            log_line(&format!("{kind} {id}: {}", text(e, "message")));
            break;
        }
        let data = page.get("data").and_then(Value::as_array).cloned().unwrap_or_default();
        if data.is_empty() {
            break;
        }
        out.extend(data.iter().filter_map(track_json));
        if out.len() >= MAX_TRACKS {
            out.truncate(MAX_TRACKS);
            break;
        }
        match page.get("next").and_then(Value::as_str) {
            // Only follow Deezer's own host; the grant refuses anything else anyway.
            Some(n) if n.starts_with("https://api.deezer.com/") => url = n.to_string(),
            _ => break,
        }
    }
    out
}

#[unsafe(no_mangle)]
pub extern "C" fn resolve(ptr: *const u8, len: i32) {
    let Ok(n) = usize::try_from(len) else { return };
    // SAFETY: the host wrote `len` bytes at `ptr` through `alloc`.
    let bytes = unsafe { core::slice::from_raw_parts(ptr, n) };
    let input: Input = match serde_json::from_slice(bytes) {
        Ok(i) => i,
        Err(e) => {
            log_line(&format!("input is not the rows document: {e}"));
            emit_bytes(b"{\"rows\":[]}");
            return;
        }
    };
    let mut rows = Vec::new();
    for row in input.rows.iter().take(10_000) {
        let Some(link) = &row.link else { continue };
        if link.provider != "deezer" {
            continue;
        }
        let Some(id) = numeric_id(&link.id) else { continue };
        match link.kind.as_str() {
            "track" => {
                let Some(t) = get_json(&format!("https://api.deezer.com/track/{id}")) else { continue };
                if t.get("error").is_some() {
                    log_line(&format!("track {id}: {}", t.get("error").map(|e| text(e, "message")).unwrap_or_default()));
                    continue;
                }
                let Some(Value::Object(mut m)) = track_json(&t) else { continue };
                m.insert("line".into(), Value::from(row.line));
                rows.push(Value::Object(m));
            }
            kind @ ("album" | "playlist") => {
                let tracks = list_tracks(kind, id);
                if tracks.is_empty() {
                    continue;
                }
                rows.push(json!({ "line": row.line, "tracks": tracks }));
            }
            _ => {}
        }
    }
    let answer = json!({ "rows": rows });
    emit_bytes(answer.to_string().as_bytes());
}
