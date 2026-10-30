# Apple Music links — what this plug-in does

When you paste a request list that holds Apple Music links (`music.apple.com/…/song/…`,
`…/album/…`, `…/album/…?i=…` for one song, `…/playlist/…`), MixCraft hands those links to this
plug-in.

For each link the plug-in loads the public page on **music.apple.com** — the page a browser
shows without signing in — and reads the song data in it: a song's title and artists; an
album's or a public playlist's songs with their durations. MixCraft then looks for those
tracks in your library.

## Read this before you turn it on

- **This is not an official API.** Apple's keyless options do not list a playlist's songs;
  the web page is meant for people, and Apple's terms do not permit reading its pages with
  software. You turn this plug-in on at your own risk; the MixCraft project gives no
  assurance that this use is allowed.
- **It can stop working at any time.** When Apple changes the page, the plug-in finds nothing
  until it is updated. It never guesses: a link it cannot read stays unresolved.
- **Long playlists are cut at 100 songs.** The page lists at most the first 100 songs of a
  playlist; the plug-in log says so when a playlist has more. For a longer playlist, export
  it with a tool you trust as a CSV and paste or open that instead.
- Private playlists and your own library are not readable, and no ISRCs are given (matching
  uses artist, title and duration). Music videos in a playlist are skipped.
- The link's country (`/de/`, `/us/` …) is kept, so titles come as that store shows them.

## What leaves your computer

- Only the Apple Music ids and the country from the links you pasted, as requests to
  **music.apple.com**.
- No account, key, cookie or login is used or stored.
- Nothing about your library, your files or you is sent.
- Every request is listed in MixCraft's plug-in log (Preferences › Plug-ins › Log).
- The answers are used once to match your library; they are not stored beyond the playlist
  MixCraft builds from them. The plug-in never touches audio.

Apple Music is a trademark of Apple Inc. This plug-in is not made, endorsed or supported by
Apple.
