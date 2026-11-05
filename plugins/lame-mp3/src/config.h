/* MixCraft LAME plug-in: build configuration for libmp3lame on wasm32-wasip1.
 *
 * Written for this plug-in (not taken from LAME's configure output): LAME's own `configure`
 * does not know the WebAssembly target, so build.sh passes this header with -DHAVE_CONFIG_H.
 * Only libmp3lame is compiled; no decoder (mpglib), no frontend, no assembly, no SSE.
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0
 */
#ifndef LAME_CONFIG_H
#define LAME_CONFIG_H

#define STDC_HEADERS 1
#define HAVE_ERRNO_H 1
#define HAVE_FCNTL_H 1
#define HAVE_INTTYPES_H 1
#define HAVE_LIMITS_H 1
#define HAVE_STDINT_H 1
#define HAVE_STDLIB_H 1
#define HAVE_STRING_H 1
#define HAVE_STRINGS_H 1
#define HAVE_SYS_TYPES_H 1
#define HAVE_SYS_STAT_H 1

#define HAVE_INT8_T 1
#define HAVE_INT16_T 1
#define HAVE_INT32_T 1
#define HAVE_INT64_T 1
#define HAVE_UINT8_T 1
#define HAVE_UINT16_T 1
#define HAVE_UINT32_T 1
#define HAVE_UINT64_T 1

#define HAVE_IEEE754_FLOAT32_T 0
#define HAVE_IEEE754_FLOAT64_T 0
typedef float ieee754_float32_t;
typedef double ieee754_float64_t;

#define SIZEOF_SHORT 2
#define SIZEOF_INT 4
#define SIZEOF_LONG 4
#define SIZEOF_LONG_LONG 8
#define SIZEOF_FLOAT 4
#define SIZEOF_DOUBLE 8
#define SIZEOF_UNSIGNED_SHORT 2
#define SIZEOF_UNSIGNED_INT 4
#define SIZEOF_UNSIGNED_LONG 4
#define SIZEOF_UNSIGNED_LONG_LONG 8

#define LAME_LIBRARY_BUILD 1
#define NOANALYSIS 1
#define TAKEHIRO_IEEE754_HACK 1
#define USE_FAST_LOG 1

#define PACKAGE "lame"
#define PACKAGE_NAME "lame"
#define PACKAGE_VERSION "3.100"
#define VERSION "3.100"

#endif
