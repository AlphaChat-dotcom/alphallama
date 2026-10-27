#pragma once
// Encrypted model pack (.apack) reader API.
// Binary format for distributing expert profiles + portfolio configs.

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define APACK_MAGIC       0x4B415041  // "APAK"
#define APACK_VERSION     1
#define APACK_FP_SIZE     32          // SHA-256 fingerprint
#define APACK_NONCE_SIZE  12          // AES-GCM nonce
#define APACK_TAG_SIZE    16          // AES-GCM auth tag
#define APACK_MAX_PROFILES 16
#define APACK_MAX_LAYERS   64
#define APACK_EXPERTS_PER  33

// TLV section types
#define APACK_SEC_META      0
#define APACK_SEC_PROFILE   1
#define APACK_SEC_PORTFOLIO 2

// Single expert profile (per model family)
typedef struct {
    char     name[24];                                   // e.g. "factual", "code"
    int      n_layers;
    int      experts[APACK_MAX_LAYERS][APACK_EXPERTS_PER]; // top expert IDs per layer
} apack_profile;

// Decoded pack contents
typedef struct {
    bool           valid;
    int            version;
    uint8_t        fingerprint[APACK_FP_SIZE];
    int            n_profiles;
    apack_profile  profiles[APACK_MAX_PROFILES];
    char           portfolio_json[4096];  // optional portfolio override (JSON)
    int            portfolio_len;
} apack_contents;

// Compute model fingerprint: SHA-256(arch:n_expert:n_expert_used:n_layer:n_embd[:base_hash]).
// base_hash (optional) binds the pack to a specific base checkpoint: pass a witness-set hash of the base
// weights so a pack built against one checkpoint refuses to load onto another with the same
// arch/n_expert/... When base_hash is nullptr the fingerprint omits the suffix.
void apack_compute_fingerprint(const char * arch, int n_expert, int n_expert_used,
                               int n_layer, int n_embd, uint8_t fp[APACK_FP_SIZE],
                               const uint8_t * base_hash = nullptr, size_t base_hash_len = 0);

// Convert fingerprint to hex string (65 bytes including null terminator)
void apack_fingerprint_hex(const uint8_t fp[APACK_FP_SIZE], char hex[65]);

// SHA-256 helper (thin public wrapper over the internal implementation) for callers that need to hash
// arbitrary bytes, e.g. a base-checkpoint witness hash. out = 32 bytes.
void apack_sha256(const uint8_t * data, size_t len, uint8_t out[32]);

// Load and decrypt an .apack file. Returns contents with valid=true on success.
apack_contents apack_load(const char * path, const uint8_t expected_fp[APACK_FP_SIZE]);

// Resolve pack cache directory (~/.alphachat/packs/). Returns false if can't resolve.
bool apack_cache_dir(char * buf, size_t buf_size);

// Try to find a cached .apack for the given fingerprint. Returns path or empty.
// Checks ~/.alphachat/packs/{fingerprint_hex}.apack
bool apack_find_cached(const uint8_t fp[APACK_FP_SIZE], char * path, size_t path_size);

// Sync: fetch pack from remote if not cached. Returns true if pack is now available.
// Uses opaque URL path and generic User-Agent. TLS-only. Single attempt, no retry.
// If fetch fails, returns false silently (caller falls back to embedded data).
bool apack_sync(const uint8_t fp[APACK_FP_SIZE]);

#ifdef __cplusplus
}
#endif
