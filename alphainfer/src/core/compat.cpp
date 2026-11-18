// SPDX-License-Identifier: Apache-2.0
#include "core/compat.h"

#if defined(_WIN32)

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cerrno>
#include <cstring>
#include <malloc.h>

int setenv(const char * name, const char * value, int overwrite) {
    if (!name || !*name || !value) { errno = EINVAL; return -1; }
    if (!overwrite && getenv(name)) return 0;
    return _putenv_s(name, value) == 0 ? 0 : -1;
}

int unsetenv(const char * name) {
    if (!name || !*name) { errno = EINVAL; return -1; }
    return _putenv_s(name, "") == 0 ? 0 : -1;   // an empty value removes the variable on Windows
}

ssize_t pread(int fd, void * buf, size_t n, int64_t offset) {
    const HANDLE h = (HANDLE) _get_osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE) { errno = EBADF; return -1; }
    // ReadFile takes 32-bit counts: read at most 1 GiB per call, the callers loop on short reads
    const DWORD want = (DWORD) (n > (1u << 30) ? (1u << 30) : n);
    OVERLAPPED  ov {};
    ov.Offset     = (DWORD) ((uint64_t) offset & 0xffffffffu);
    ov.OffsetHigh = (DWORD) ((uint64_t) offset >> 32);
    DWORD got = 0;
    if (!ReadFile(h, buf, want, &got, &ov)) {
        if (GetLastError() == ERROR_HANDLE_EOF) return 0;
        errno = EIO;
        return -1;
    }
    return (ssize_t) got;
}

void * mmap(void * addr, size_t len, int prot, int flags, int fd, int64_t offset) {
    (void) addr;
    if (len == 0) { errno = EINVAL; return MAP_FAILED; }
    if (flags & MAP_ANONYMOUS) {
        void * p = VirtualAlloc(nullptr, len, MEM_RESERVE | MEM_COMMIT, (prot & PROT_WRITE) ? PAGE_READWRITE : PAGE_READONLY);
        return p ? p : MAP_FAILED;
    }
    if (prot & PROT_WRITE) { errno = ENOTSUP; return MAP_FAILED; }   // only read-only file views are used
    const HANDLE h = (HANDLE) _get_osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE) { errno = EBADF; return MAP_FAILED; }
    const uint64_t end = (uint64_t) offset + len;
    HANDLE m = CreateFileMappingA(h, nullptr, PAGE_READONLY, (DWORD) (end >> 32), (DWORD) (end & 0xffffffffu), nullptr);
    if (!m) { errno = ENOMEM; return MAP_FAILED; }
    void * p = MapViewOfFile(m, FILE_MAP_READ, (DWORD) ((uint64_t) offset >> 32), (DWORD) ((uint64_t) offset & 0xffffffffu), len);
    CloseHandle(m);   // the view keeps the mapping alive
    return p ? p : MAP_FAILED;
}

int munmap(void * addr, size_t len) {
    (void) len;
    MEMORY_BASIC_INFORMATION mi {};
    if (!VirtualQuery(addr, &mi, sizeof(mi))) { errno = EINVAL; return -1; }
    if (mi.Type == MEM_MAPPED) return UnmapViewOfFile(addr) ? 0 : -1;
    return VirtualFree(mi.AllocationBase, 0, MEM_RELEASE) ? 0 : -1;
}

int posix_memalign(void ** out, size_t align, size_t n) {
    *out = _aligned_malloc(n ? n : 1, align);
    return *out ? 0 : ENOMEM;
}

void ai_aligned_free(void * p) { _aligned_free(p); }

struct DIR {
    HANDLE           h = INVALID_HANDLE_VALUE;
    WIN32_FIND_DATAA fd {};
    bool             first = true;
    dirent           e {};
};

DIR * opendir(const char * path) {
    DIR * d = new DIR;
    d->h = FindFirstFileA((std::string(path) + "\\*").c_str(), &d->fd);
    if (d->h == INVALID_HANDLE_VALUE) { delete d; return nullptr; }
    return d;
}

dirent * readdir(DIR * d) {
    if (!d) return nullptr;
    if (!d->first && !FindNextFileA(d->h, &d->fd)) return nullptr;
    d->first = false;
    strncpy(d->e.d_name, d->fd.cFileName, sizeof(d->e.d_name) - 1);
    d->e.d_name[sizeof(d->e.d_name) - 1] = 0;
    return &d->e;
}

int closedir(DIR * d) {
    if (!d) return -1;
    FindClose(d->h);
    delete d;
    return 0;
}

namespace ai {
std::string exe_path() {
    char buf[MAX_PATH * 4] = { 0 };
    const DWORD n = GetModuleFileNameA(nullptr, buf, (DWORD) sizeof(buf));
    return n > 0 && n < sizeof(buf) ? std::string(buf, n) : std::string();
}
long long mem_kb(bool available) {
    MEMORYSTATUSEX m;
    m.dwLength = sizeof(m);
    if (!GlobalMemoryStatusEx(&m)) return 0;
    return (long long) ((available ? m.ullAvailPhys : m.ullTotalPhys) / 1024);
}
}  // namespace ai

#else  // POSIX

#include <unistd.h>
#include <cstdio>
#include <cstring>

namespace ai {
std::string exe_path() {
    char buf[4096] = { 0 };
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    return n > 0 ? std::string(buf, (size_t) n) : std::string();
}
long long mem_kb(bool available) {
    FILE * f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    const char * want = available ? "MemAvailable:" : "MemTotal:";
    char      line[256], key[64];
    long long kb = 0, v = 0;
    while (fgets(line, sizeof(line), f)) {   // "Key:   value [kB]" (some lines have no unit)
        if (sscanf(line, "%63s %lld", key, &v) == 2 && strcmp(key, want) == 0) { kb = v; break; }
    }
    fclose(f);
    return kb;
}
}  // namespace ai

#endif
