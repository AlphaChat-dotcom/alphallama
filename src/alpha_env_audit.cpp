#include "alpha_env_audit.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(__linux__)

#include <link.h>
#include <elf.h>
#include <unistd.h>

extern char ** environ;

namespace {

struct ro_seg { const unsigned char * base; size_t len; };

// Read-only PT_LOAD segments of every loaded object. String literals live here; .data/.bss are skipped
// because a writable match would not prove a getenv literal.
std::vector<ro_seg> * g_segs = nullptr;

int collect_segments(struct dl_phdr_info * info, size_t, void *) {
    for (int i = 0; i < info->dlpi_phnum; ++i) {
        const ElfW(Phdr) * p = &info->dlpi_phdr[i];
        if (p->p_type != PT_LOAD)  continue;
        if (!(p->p_flags & PF_R))  continue;   // must be readable to scan
        if (p->p_flags & PF_W)     continue;   // skip writable segments
        if (p->p_filesz == 0)      continue;   // file-backed bytes only (never the bss tail)
        g_segs->push_back({ (const unsigned char *)(info->dlpi_addr + p->p_vaddr), (size_t) p->p_filesz });
    }
    return 0;
}

// Plain search -- avoids depending on memmem's feature-test macros.
bool seg_contains(const ro_seg & s, const char * needle, size_t nlen) {
    if (nlen == 0 || s.len < nlen) return false;
    const unsigned char first = (unsigned char) needle[0];
    const unsigned char * end = s.base + (s.len - nlen);
    for (const unsigned char * p = s.base; p <= end; ++p) {
        p = (const unsigned char *) memchr(p, first, (size_t)(end - p) + 1);
        if (!p) break;
        if (memcmp(p, needle, nlen) == 0) return true;
    }
    return false;
}

} // namespace

void alpha_env_audit(void) {
    if (getenv("ALPHA_NO_ENV_AUDIT")) return;

    std::vector<ro_seg> segs;
    g_segs = &segs;
    dl_iterate_phdr(collect_segments, nullptr);
    g_segs = nullptr;
    if (segs.empty()) return;   // cannot introspect -- stay silent rather than warn wrongly

    std::vector<std::string> orphans;
    for (char ** e = environ; e && *e; ++e) {
        const char * eq = strchr(*e, '=');
        if (!eq) continue;
        const size_t nlen = (size_t)(eq - *e);
        if (nlen < 7 || strncmp(*e, "ALPHA_", 6) != 0) continue;

        std::string name(*e, nlen);
        if (name == "ALPHA_NO_ENV_AUDIT") continue;

        // Require the trailing NUL so ALPHA_MTP is not "found" inside ALPHA_MTP_DBG.
        const std::string needle = name + '\0';
        bool found = false;
        for (const auto & s : segs) {
            if (seg_contains(s, needle.data(), needle.size())) { found = true; break; }
        }
        if (!found) orphans.push_back(name);
    }

    if (orphans.empty()) return;
    fprintf(stderr, "\n[alpha-env-audit] WARNING: %zu ALPHA_* variable(s) are set but NO loaded code reads them.\n",
            orphans.size());
    fprintf(stderr, "[alpha-env-audit] These are SILENT NO-OPS -- the setting has no effect whatsoever:\n");
    for (const auto & n : orphans) fprintf(stderr, "[alpha-env-audit]     %s\n", n.c_str());
    fprintf(stderr, "[alpha-env-audit] Most often the flag belongs to a feature DISABLED at build time\n");
    fprintf(stderr, "[alpha-env-audit]   (e.g. -DALPHA_FLASHINFER=OFF / -DALPHA_DENSE_L0=OFF), so its\n");
    fprintf(stderr, "[alpha-env-audit]   code was preprocessed out, or its object was never pulled from the static lib.\n");
    fprintf(stderr, "[alpha-env-audit] Otherwise: a typo, or a flag removed from the build.\n");
    fprintf(stderr, "[alpha-env-audit] (If built with GGML_BACKEND_DL=ON a backend may load later and read it.)\n");
    fprintf(stderr, "[alpha-env-audit] (silence with ALPHA_NO_ENV_AUDIT=1)\n\n");
}

#else   // non-Linux: no dl_iterate_phdr

void alpha_env_audit(void) {}

#endif
