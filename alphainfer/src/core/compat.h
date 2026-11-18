// SPDX-License-Identifier: Apache-2.0
// POSIX surface for the MSVC build. AlphaInfer is written against Linux; on Windows this header (force-included into
// every aicore source, and pulled in by the stub headers in alphainfer/win/) supplies the calls it uses with their POSIX
// shapes, so the engine sources compile unchanged. Linux and other POSIX builds never see anything here.
#pragma once

#if defined(_WIN32)

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#ifndef _CRT_NONSTDC_NO_DEPRECATE
#define _CRT_NONSTDC_NO_DEPRECATE
#endif

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <direct.h>
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#include <sys/types.h>

typedef std::intptr_t ssize_t;

// struct stat / stat() / fstat() with 64-bit sizes: the 32-bit defaults would truncate a model file past 2 GiB.
// <sys/stat.h> is included above, so its own declarations are not renamed.
#define stat  _stat64
#define fstat _fstat64

// 64-bit stdio seeks
#define fseeko _fseeki64
#define ftello _ftelli64

// mkdir(path, mode): Windows has no permission bits
#define mkdir(p, m) _mkdir(p)

// No O_DIRECT on Windows: the callers fall back to buffered reads, which is correct, just slower to cold-load.
#ifndef O_DIRECT
#define O_DIRECT 0
#endif

// setenv / unsetenv with POSIX semantics (overwrite == 0 keeps an existing value)
int setenv(const char * name, const char * value, int overwrite);
int unsetenv(const char * name);

// pread on a CRT file descriptor (the offset is 64-bit; the file position is not used)
ssize_t pread(int fd, void * buf, size_t n, int64_t offset);

// mmap of a whole file, read-only (PROT_READ + MAP_SHARED + fd), and anonymous read-write memory (MAP_ANONYMOUS, fd -1);
// munmap releases what mmap returned, whole. madvise is accepted and ignored.
#define PROT_NONE     0
#define PROT_READ     1
#define PROT_WRITE    2
#define MAP_SHARED    1
#define MAP_PRIVATE   2
#define MAP_ANONYMOUS 0x20
#define MAP_ANON      MAP_ANONYMOUS
#define MAP_FAILED    ((void *) -1)
#define MADV_NORMAL    0
#define MADV_DONTNEED  4
#define MADV_HUGEPAGE  14
void * mmap(void * addr, size_t len, int prot, int flags, int fd, int64_t offset);
int    munmap(void * addr, size_t len);
inline int madvise(void *, size_t, int) { return 0; }

// posix_memalign: memory from it must be released with ai_aligned_free (free() on Windows would corrupt the heap)
int  posix_memalign(void ** out, size_t align, size_t n);
void ai_aligned_free(void * p);

// opendir / readdir / closedir: the entry names of one directory
struct dirent {
    char d_name[260];
};
struct DIR;
DIR *    opendir(const char * path);
dirent * readdir(DIR * d);
int      closedir(DIR * d);

#else  // POSIX

#include <cstdlib>
inline void ai_aligned_free(void * p) { free(p); }

#endif

#include <string>

namespace ai {
// Full path of the running executable ("" when unknown)
std::string exe_path();
// Physical memory in KiB: total (available = false) or available now (true); 0 when unknown.
// Linux: /proc/meminfo MemTotal / MemAvailable. Windows: GlobalMemoryStatusEx.
long long mem_kb(bool available);
}  // namespace ai
