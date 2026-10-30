# Spotify links — what this plug-in does

When you paste a request list that holds Spotify links (`open.spotify.com/track/…`,
`open.spotify.com/album/…`, `open.spotify.com/playlist/…`, or `spotify:track:…` URIs),
MixCraft hands those links to this plug-in.

For each link the plug-in loads Spotify's public **embed page** — the page a website shows
when it embeds a Spotify player, `https://open.spotify.com/embed/<kind>/<id>` — and reads the
track data in it: a track's title, artists and duration; an album's or a public playlist's
tracks. MixCraft then looks for those tracks in your library.

## Read this before you turn it on

- **This is not an official API.** Spotify offers no keyless way to read a playlist. The embed
  page is meant for showing a player, and Spotify's Terms of Use and Developer Terms do not
  permit reading its pages with software. You turn this plug-in on at your own risk; the
  MixCraft project gives no assurance that this use is allowed.
- **It can stop working at any time.** When Spotify changes the page, the plug-in finds
  nothing until it is updated. It never guesses: a link it cannot read stays unresolved.
- **Playlists are cut at 100 tracks.** The embed page lists at most the first 100 tracks of a
  playlist; the plug-in log says so when a playlist reaches that limit. For a longer playlist,
  export it with a tool you trust as a CSV and paste or open that instead.
- Private playlists are not readable, and no ISRCs are given (matching uses artist, title and
  duration).

## What leaves your computer

- Only the Spotify ids from the links you pasted, as requests to **open.spotify.com**.
- No account, key, cookie or login is used or stored.
- Nothing about your library, your files or you is sent.
- Every request is listed in MixCraft's plug-in log (Preferences › Plug-ins).
- The answers are used once to match your library; they are not stored beyond the playlist
  MixCraft builds from them. The plug-in never touches audio.

Spotify is a trademark of Spotify AB. This plug-in is not made, endorsed or supported by
Spotify.
