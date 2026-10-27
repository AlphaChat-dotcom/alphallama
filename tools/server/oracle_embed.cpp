#include "oracle_embed.h"

#ifdef ALPHA_ORACLE_EMBEDDED

// Platform-specific access to embedded oracle data

#ifdef _WIN32
// Windows: oracle is embedded as a RCDATA resource via .rc file
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

// Resource ID must match oracle_embed.rc
#define ORACLE_RESOURCE_ID 101

static const uint8_t * s_data = nullptr;
static size_t          s_size = 0;
static bool            s_init = false;

static void ensure_init() {
    if (s_init) return;
    s_init = true;

    HRSRC hRes = FindResourceA(NULL, MAKEINTRESOURCEA(ORACLE_RESOURCE_ID), RT_RCDATA);
    if (!hRes) return;

    HGLOBAL hData = LoadResource(NULL, hRes);
    if (!hData) return;

    s_data = static_cast<const uint8_t *>(LockResource(hData));
    s_size = static_cast<size_t>(SizeofResource(NULL, hRes));
}

bool oracle_embed::available() { ensure_init(); return s_data != nullptr && s_size > 0; }
const uint8_t * oracle_embed::data() { ensure_init(); return s_data; }
size_t oracle_embed::size() { ensure_init(); return s_size; }

#else
// Unix (Linux/macOS): oracle is embedded via .incbin in assembly.
// The linker exposes start/end symbols.
extern "C" {
    extern const uint8_t _oracle_embed_start[];
    extern const uint8_t _oracle_embed_end[];
}

bool oracle_embed::available() {
    return (&_oracle_embed_end[0] - &_oracle_embed_start[0]) > 0;
}

const uint8_t * oracle_embed::data() {
    return _oracle_embed_start;
}

size_t oracle_embed::size() {
    return static_cast<size_t>(&_oracle_embed_end[0] - &_oracle_embed_start[0]);
}

#endif // _WIN32

#else // !ALPHA_ORACLE_EMBEDDED

// Not embedded: stubs
bool oracle_embed::available() { return false; }
const uint8_t * oracle_embed::data() { return nullptr; }
size_t oracle_embed::size() { return 0; }

#endif // ALPHA_ORACLE_EMBEDDED
