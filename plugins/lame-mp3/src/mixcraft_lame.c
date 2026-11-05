/* MixCraft encoder plug-in ABI over libmp3lame (docs/plugins.md §4, kind `encoder`).
 *
 * This file is MixCraft's own glue (MIT OR Apache-2.0). Linked with libmp3lame (LGPL), the
 * resulting lame-mp3.wasm is a work under the LGPL and ships as its own package, never inside
 * MixCraft (docs/licensing.md §3).
 *
 * Exports (all called by the host, one instance per file, never on the audio thread):
 *   alloc(len) -> ptr                  scratch memory for the host's input
 *   encode_mode(mode, quality)         optional, before encode_begin: 0 CBR, 1 VBR, 2 ABR;
 *                                      quality 0 (best) .. 9 (VBR quality)
 *   encode_begin(rate, channels, kbps) set up; emits nothing
 *   encode(ptr, frames, rate, channels) interleaved f32 in, MP3 frames out (emit)
 *   encode_end()                       flush the last frames (emit)
 *   encode_header()                    the final Xing/LAME header frame (emit): the host writes
 *                                      it over the placeholder at the start of the stream, so
 *                                      decoders know the encoder delay and padding (gapless)
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0
 */
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#include "lame.h"

#define IMPORT(name) __attribute__((import_module("mixcraft"), import_name(name)))
#define EXPORT(name) __attribute__((export_name(name)))

IMPORT("emit") void mc_emit(const void *ptr, int len);
IMPORT("log") void mc_log(const char *ptr, int len);

static unsigned char *scratch;
static size_t scratch_cap;
static unsigned char *out;
static size_t out_cap;
static lame_t gf;
static int rate_mode;    /* 0 CBR, 1 VBR, 2 ABR */
static int vbr_quality = 2;

static void say(const char *s) { mc_log(s, (int)strlen(s)); }

/* LAME's messages go to the plug-in log, never to a file descriptor. */
static void report(const char *format, va_list ap) {
    (void)ap;
    if (format) say(format);
}
static void quiet(const char *format, va_list ap) {
    (void)format;
    (void)ap;
}

static int ensure(unsigned char **buf, size_t *cap, size_t len) {
    if (*cap >= len) return 1;
    unsigned char *p = realloc(*buf, len);
    if (!p) return 0;
    *buf = p;
    *cap = len;
    return 1;
}

EXPORT("alloc") void *mc_alloc(int len) {
    if (len < 0 || !ensure(&scratch, &scratch_cap, (size_t)len + 16)) __builtin_trap();
    return scratch;
}

EXPORT("encode_mode") void mc_encode_mode(int mode, int quality) {
    rate_mode = (mode >= 0 && mode <= 2) ? mode : 0;
    vbr_quality = quality < 0 ? 0 : (quality > 9 ? 9 : quality);
}

EXPORT("encode_begin") void mc_encode_begin(int rate, int channels, int kbps) {
    if (gf) lame_close(gf);
    gf = lame_init();
    if (!gf || rate <= 0 || channels < 1 || channels > 2) {
        say("lame: bad parameters");
        __builtin_trap();
    }
    lame_set_errorf(gf, report);
    lame_set_debugf(gf, quiet);
    lame_set_msgf(gf, quiet);
    lame_set_in_samplerate(gf, rate);
    lame_set_num_channels(gf, channels);
    lame_set_mode(gf, channels == 1 ? MONO : JOINT_STEREO);
    lame_set_quality(gf, 5);
    lame_set_bWriteVbrTag(gf, 1);
    lame_set_write_id3tag_automatic(gf, 0);
    if (kbps < 8) kbps = 8;
    if (kbps > 320) kbps = 320;
    switch (rate_mode) {
    case 1:
        lame_set_VBR(gf, vbr_mtrh);
        lame_set_VBR_quality(gf, (float)vbr_quality);
        break;
    case 2:
        lame_set_VBR(gf, vbr_abr);
        lame_set_VBR_mean_bitrate_kbps(gf, kbps);
        break;
    default:
        lame_set_VBR(gf, vbr_off);
        lame_set_brate(gf, kbps);
        break;
    }
    if (lame_init_params(gf) < 0) {
        say("lame: lame_init_params refused the settings");
        __builtin_trap();
    }
}

EXPORT("encode") void mc_encode(const float *pcm, int frames, int rate, int channels) {
    (void)rate;
    if (!gf || frames < 0) __builtin_trap();
    if (frames == 0) return;
    size_t need = (size_t)frames * 5 / 4 + 7200;
    if (!ensure(&out, &out_cap, need)) __builtin_trap();
    int n;
    if (channels == 1) {
        n = lame_encode_buffer_ieee_float(gf, pcm, pcm, frames, out, (int)out_cap);
    } else {
        n = lame_encode_buffer_interleaved_ieee_float(gf, pcm, frames, out, (int)out_cap);
    }
    if (n < 0) {
        say("lame: encoding failed");
        __builtin_trap();
    }
    if (n > 0) mc_emit(out, n);
}

EXPORT("encode_end") void mc_encode_end(void) {
    if (!gf) __builtin_trap();
    if (!ensure(&out, &out_cap, 7200)) __builtin_trap();
    int n = lame_encode_flush(gf, out, (int)out_cap);
    if (n > 0) mc_emit(out, n);
}

EXPORT("encode_header") void mc_encode_header(void) {
    if (!gf) return;
    if (!ensure(&out, &out_cap, 4096)) __builtin_trap();
    size_t n = lame_get_lametag_frame(gf, out, out_cap);
    if (n > 0 && n <= out_cap) mc_emit(out, (int)n);
    lame_close(gf);
    gf = 0;
}
