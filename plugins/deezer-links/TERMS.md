# Deezer links — what this plug-in does

When you paste a request list that holds Deezer links (`deezer.com/track/…`,
`deezer.com/album/…`, `deezer.com/playlist/…`), MixCraft hands those links to this plug-in.

The plug-in asks **api.deezer.com**, Deezer's public API, for each link: a track's title,
artist, duration and ISRC; an album's or a public playlist's tracks (title, artist, duration,
and the ISRC where Deezer lists it). Private playlists are not readable this way.

- No account, key or login is used or stored.
- Only the Deezer ids from the links you pasted are sent. Nothing about your library, your
  files or you leaves the computer.
- Every request is listed in MixCraft's plug-in log (Preferences › Plug-ins).
- The answers are used once to match your library; they are not stored beyond the
  playlist MixCraft builds from them.

Deezer's developer guidelines (https://developers.deezer.com/guidelines) apply to the data
this plug-in reads. They forbid storing Deezer's audio (this plug-in never touches audio).
