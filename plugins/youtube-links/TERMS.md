# YouTube links — what this plug-in does

When you paste a request list that holds YouTube or YouTube Music links
(`youtube.com/watch?v=…`, `youtu.be/…`, `music.youtube.com/watch?v=…`, any link with
`list=…`), MixCraft hands those links to this plug-in.

- **A video link:** the plug-in asks YouTube's **oEmbed** service
  (`www.youtube.com/oembed`, a documented, keyless service for showing link previews) for the
  video's title and channel.
- **A playlist link:** the plug-in loads the public playlist page
  (`www.youtube.com/playlist?list=…`) and reads each video's title, channel and length from
  it.

A video title is not an artist and a title, so the plug-in works them out: "Artist - Title
(Official Video)" becomes *Artist* and *Title* (the bracketed video words are dropped, versions
such as "(Extended Mix)" or "(feat. …)" are kept); when the channel's name comes after the
dash, the two are swapped; a title without a dash takes the channel's name (without
" - Topic") as the artist. This guesses right for most music videos and wrong for some; the
matcher shows what it could not place.

## Read this before you turn it on

- **The playlist page is not an official API.** YouTube's terms do not permit reading its
  pages with software, and the page can change at any time — then playlists find nothing
  until the plug-in is updated (video links keep working through oEmbed). You turn this
  plug-in on at your own risk; the MixCraft project gives no assurance that this use is
  allowed.
- **Playlists are cut at 100 videos.** The page lists the first 100; the rest is loaded by
  YouTube's own script, which the plug-in does not run. The plug-in log says when a playlist
  has more. For a longer playlist, paste its videos' links or a list exported with a tool you
  trust.
- Private playlists, "Liked videos", "Watch later" and mixes made for you (`list=RD…`) are not
  readable. Unavailable videos are skipped.
- In some countries YouTube shows a cookie-consent page first; the plug-in asks for the page
  without it (`ucbcb=1`). If YouTube stops honouring that, the log says "a consent or sign-in
  page".

## What leaves your computer

- Only the video and playlist ids from the links you pasted, as requests to
  **www.youtube.com**.
- No account, key, cookie or login is used or stored.
- Nothing about your library, your files or you is sent.
- Every request is listed in MixCraft's plug-in log (Preferences › Plug-ins › Log).
- The answers are used once to match your library; they are not stored beyond the playlist
  MixCraft builds from them. The plug-in never touches audio — downloading is a separate,
  opt-in MixCraft feature (Library › Downloads).

YouTube is a trademark of Google LLC. This plug-in is not made, endorsed or supported by
Google.
