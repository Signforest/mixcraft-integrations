/* The MixCraft sandbox offers no WASI: a module importing anything but the `mixcraft` host API
 * is rejected (docs/plugins.md §4). wasi-libc's stdio (pulled in by LAME's default message
 * handlers, which the glue replaces before any message can be printed) still references four
 * WASI calls; defining them here resolves those references inside the module, so the finished
 * lame-mp3.wasm imports nothing from WASI. Every call fails: there are no files, no console and
 * no exit, only a trap.
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0
 */
#include <stdint.h>

#define WASI_EBADF 8

int32_t __imported_wasi_snapshot_preview1_fd_write(int32_t fd, int32_t iovs, int32_t iovs_len, int32_t nwritten) {
    (void)fd;
    (void)iovs;
    (void)iovs_len;
    (void)nwritten;
    return WASI_EBADF;
}

int32_t __imported_wasi_snapshot_preview1_fd_close(int32_t fd) {
    (void)fd;
    return WASI_EBADF;
}

int32_t __imported_wasi_snapshot_preview1_fd_seek(int32_t fd, int64_t offset, int32_t whence, int32_t newoffset) {
    (void)fd;
    (void)offset;
    (void)whence;
    (void)newoffset;
    return WASI_EBADF;
}

_Noreturn void __imported_wasi_snapshot_preview1_proc_exit(int32_t code) {
    (void)code;
    __builtin_trap();
}
