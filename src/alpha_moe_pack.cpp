// Encrypted model pack (.apack): reader and sync implementation.
// AES-256-GCM decryption with HKDF key derivation.
// Platform crypto: BCrypt (Windows), OpenSSL/dlopen (Linux), CommonCrypto (macOS).
// HTTP sync: opaque URL, generic UA, TLS-only, single attempt.

#include "alpha_moe_pack.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#include <windows.h>
#include <bcrypt.h>
#include <shlobj.h>
#include <winhttp.h>
#pragma comment(lib, "bcrypt")
#pragma comment(lib, "winhttp")
#else
#include <dlfcn.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <sys/ptrace.h>
#ifdef __linux__
#include <sys/prctl.h>
#include <sys/mman.h>
#endif
#endif

#include <string>
#include <fstream>

// Runtime hardening: anti-debug, memory scrub, dump prevention

// Secure memory wipe that the compiler cannot optimize away
static void _scrub(void * ptr, size_t len) {
#ifdef _WIN32
    SecureZeroMemory(ptr, len);
#elif defined(__STDC_LIB_EXT1__)
    memset_s(ptr, len, 0, len);
#else
    volatile unsigned char * p = (volatile unsigned char *)ptr;
    while (len--) *p++ = 0;
#endif
}

// Constant-time comparison, avoids a timing side channel on the fingerprint check
static bool _ct_eq(const uint8_t * a, const uint8_t * b, size_t len) {
    volatile uint8_t diff = 0;
    for (size_t i = 0; i < len; i++) diff |= a[i] ^ b[i];
    return diff == 0;
}

// Platform hardening, called once before the first decrypt
static bool _hardened = false;
static void _harden(void) {
    if (_hardened) return;
    _hardened = true;
#ifdef _WIN32
    // Prevent error dialogs / crash dump popups
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
#else
    // Disable core dumps (prevents key extraction from crash dumps)
    struct rlimit rl = {0, 0};
    setrlimit(RLIMIT_CORE, &rl);
#ifdef __linux__
    // Mark process as non-dumpable (blocks ptrace attach, /proc/pid/mem reads)
    prctl(PR_SET_DUMPABLE, 0);
#endif
#endif
}

// SHA-256 (FIPS 180-4), self-contained

static const uint32_t sha256_k[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

#define RR(x,n) (((x)>>(n))|((x)<<(32-(n))))
#define CH(x,y,z) (((x)&(y))^((~(x))&(z)))
#define MAJ(x,y,z) (((x)&(y))^((x)&(z))^((y)&(z)))
#define EP0(x) (RR(x,2)^RR(x,13)^RR(x,22))
#define EP1(x) (RR(x,6)^RR(x,11)^RR(x,25))
#define SIG0(x) (RR(x,7)^RR(x,18)^((x)>>3))
#define SIG1(x) (RR(x,17)^RR(x,19)^((x)>>10))

static void sha256_transform(uint32_t state[8], const uint8_t block[64]) {
    uint32_t w[64], a,b,c,d,e,f,g,h,t1,t2;
    for (int i = 0; i < 16; i++)
        w[i] = ((uint32_t)block[i*4]<<24)|((uint32_t)block[i*4+1]<<16)|
               ((uint32_t)block[i*4+2]<<8)|block[i*4+3];
    for (int i = 16; i < 64; i++)
        w[i] = SIG1(w[i-2]) + w[i-7] + SIG0(w[i-15]) + w[i-16];
    a=state[0]; b=state[1]; c=state[2]; d=state[3];
    e=state[4]; f=state[5]; g=state[6]; h=state[7];
    for (int i = 0; i < 64; i++) {
        t1 = h + EP1(e) + CH(e,f,g) + sha256_k[i] + w[i];
        t2 = EP0(a) + MAJ(a,b,c);
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
    state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
}

static void sha256(const uint8_t * data, size_t len, uint8_t hash[32]) {
    uint32_t state[8] = {
        0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
        0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19
    };
    uint8_t block[64];
    size_t i = 0;
    for (; i + 64 <= len; i += 64) sha256_transform(state, data + i);
    size_t rem = len - i;
    memcpy(block, data + i, rem);
    block[rem++] = 0x80;
    if (rem > 56) { memset(block + rem, 0, 64 - rem); sha256_transform(state, block); rem = 0; }
    memset(block + rem, 0, 56 - rem);
    uint64_t bits = (uint64_t)len * 8;
    for (int j = 7; j >= 0; j--) block[56 + (7-j)] = (uint8_t)(bits >> (j*8));
    sha256_transform(state, block);
    for (int j = 0; j < 8; j++) {
        hash[j*4]   = (uint8_t)(state[j]>>24); hash[j*4+1] = (uint8_t)(state[j]>>16);
        hash[j*4+2] = (uint8_t)(state[j]>>8);  hash[j*4+3] = (uint8_t)(state[j]);
    }
}

// HMAC-SHA256 + HKDF-SHA256 (RFC 5869)

static void hmac_sha256(const uint8_t * key, size_t key_len,
                        const uint8_t * data, size_t data_len,
                        uint8_t out[32]) {
    uint8_t k_pad[64], o_pad[64];
    uint8_t k_ipad_data[64 + 1024]; // max data size
    uint8_t k_hash[32];

    if (key_len > 64) { sha256(key, key_len, k_hash); key = k_hash; key_len = 32; }
    memset(k_pad, 0, 64);
    memcpy(k_pad, key, key_len);

    // ipad
    uint8_t inner[64 + 4096];
    for (int i = 0; i < 64; i++) inner[i] = k_pad[i] ^ 0x36;
    memcpy(inner + 64, data, data_len);
    uint8_t inner_hash[32];
    sha256(inner, 64 + data_len, inner_hash);

    // opad
    uint8_t outer[64 + 32];
    for (int i = 0; i < 64; i++) outer[i] = k_pad[i] ^ 0x5c;
    memcpy(outer + 64, inner_hash, 32);
    sha256(outer, 96, out);

    (void)k_ipad_data; (void)o_pad;
}

static void hkdf_sha256(const uint8_t * ikm, size_t ikm_len,
                        const uint8_t * salt, size_t salt_len,
                        const uint8_t * info, size_t info_len,
                        uint8_t * okm, size_t okm_len) {
    // Extract
    uint8_t prk[32];
    hmac_sha256(salt, salt_len, ikm, ikm_len, prk);
    // Expand (single block, okm_len <= 32)
    uint8_t t_input[256];
    memcpy(t_input, info, info_len);
    t_input[info_len] = 0x01;
    uint8_t t[32];
    hmac_sha256(prk, 32, t_input, info_len + 1, t);
    size_t copy = okm_len < 32 ? okm_len : 32;
    memcpy(okm, t, copy);
}

// Forward declarations (defined later in the file)
static bool _http_fetch(const char * host, const char * path, const char * out_path);
static bool _ensure_dir(const char * dir);

// Master key: server-side fetch with embedded fallback.
//
// Key acquisition chain:
// 1. Check cached device key at ~/.alphachat/.dk (32 bytes, from previous server fetch)
// 2. If not cached, fetch from key server: GET /s/{device_hash} -> 32 bytes
// 3. Cache the result for offline use
// 4. Fallback: XOR-scattered fragments in binary (for offline-first or server-down)
//
// The server returns HKDF input material, not the actual pack key.
// The pack key is derived: HKDF(server_material, model_fingerprint, "apack-v1")
// This means the server never knows which model you're loading.

// Device ID: SHA-256 of machine-specific data (hostname + username + MAC hint)
static void _device_id(char hex[65]) {
    char buf[512] = {};
#ifdef _WIN32
    DWORD sz = sizeof(buf)/2;
    GetComputerNameA(buf, &sz);
    sz = sizeof(buf)/2;
    GetUserNameA(buf + strlen(buf), &sz);
#else
    gethostname(buf, sizeof(buf)/2);
    const char * user = getenv("USER");
    if (user) strncat(buf, user, sizeof(buf) - strlen(buf) - 1);
#endif
    uint8_t h[32];
    sha256((const uint8_t *)buf, strlen(buf), h);
    _scrub(buf, sizeof(buf));
    static const char hx[] = "0123456789abcdef";
    for (int i = 0; i < 32; i++) { hex[i*2] = hx[h[i]>>4]; hex[i*2+1] = hx[h[i]&0xf]; }
    hex[64] = '\0';
}

// Try to read cached device key
static bool _read_dk(uint8_t key[32]) {
    char dir[512];
    if (!apack_cache_dir(dir, sizeof(dir))) return false;
    // Go up one level from packs/ to .alphachat/
    char * last_sep = strrchr(dir, '/');
    if (!last_sep) last_sep = strrchr(dir, '\\');
    if (last_sep) *last_sep = '\0';

    char path[512];
    snprintf(path, sizeof(path), "%s/.dk", dir);
    FILE * f = fopen(path, "rb");
    if (!f) return false;
    bool ok = (fread(key, 1, 32, f) == 32);
    fclose(f);
    return ok;
}

// Write cached device key
static void _write_dk(const uint8_t key[32]) {
    char dir[512];
    if (!apack_cache_dir(dir, sizeof(dir))) return;
    char * last_sep = strrchr(dir, '/');
    if (!last_sep) last_sep = strrchr(dir, '\\');
    if (last_sep) *last_sep = '\0';

    char path[512];
    snprintf(path, sizeof(path), "%s/.dk", dir);
    FILE * f = fopen(path, "wb");
    if (!f) return;
    fwrite(key, 1, 32, f);
    fclose(f);
}

// Key server host (assembled at runtime)
static const char * _ks_host() {
    static char h[64] = {};
    if (!h[0]) { snprintf(h, sizeof(h), "%s.%s.%s", "k", "alphachat", "ai"); }
    return h;
}

// Fetch key material from server
static bool _fetch_dk(uint8_t key[32]) {
    char dev_hex[65];
    _device_id(dev_hex);
    char url_path[128];
    snprintf(url_path, sizeof(url_path), "/s/%s", dev_hex);

    // Use same HTTP fetch infrastructure but to key server
    // Write to temp file, read 32 bytes
    char dir[512];
    if (!apack_cache_dir(dir, sizeof(dir))) return false;
    char * last_sep = strrchr(dir, '/');
    if (!last_sep) last_sep = strrchr(dir, '\\');
    if (last_sep) *last_sep = '\0';
    _ensure_dir(dir);

    char tmp[512];
    snprintf(tmp, sizeof(tmp), "%s/.dk.tmp", dir);

    bool fetched = _http_fetch(_ks_host(), url_path, tmp);
    if (!fetched) { remove(tmp); return false; }

    FILE * f = fopen(tmp, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz != 32) { fclose(f); remove(tmp); return false; }
    bool ok = (fread(key, 1, 32, f) == 32);
    fclose(f);
    remove(tmp);
    return ok;
}

// Embedded fallback (XOR-scattered), used when the server is unreachable and nothing is cached
static const uint8_t _kf_a[32] = {
    0xa3,0x7b,0x12,0x9f,0xe4,0x56,0x81,0xcd, 0x3f,0xb8,0x62,0x0d,0x95,0x4a,0xf7,0x28,
    0x6e,0xd1,0x43,0x87,0xbc,0x50,0x1e,0xa9, 0x74,0x0f,0xe3,0x96,0x5b,0xc2,0x38,0xdd
};
static const uint8_t _kf_b[32] = {
    0x51,0xc4,0x8e,0x33,0x76,0xa2,0xdf,0x15, 0xe8,0x4d,0x91,0xba,0x07,0xf6,0x63,0xc0,
    0x2b,0x89,0xfe,0x14,0x5d,0x73,0xa8,0x46, 0x0c,0xd7,0x39,0xe5,0xb2,0x60,0x1f,0x94
};
static const uint8_t _kf_c[32] = {
    0x87,0x19,0x5c,0xe1,0x32,0xf0,0xae,0x68, 0x4b,0x25,0xd3,0x77,0xc9,0x0e,0xa4,0x5f,
    0xb1,0x46,0x2d,0x93,0xe7,0x0a,0xc6,0x5b, 0xd8,0x64,0x1a,0x83,0xf9,0x3e,0xa7,0x42
};

// Key acquisition: cached server key -> server fetch -> embedded fallback
static void _get_mk(uint8_t key[32]) {
    // 1. Try cached device key (from previous server fetch)
    if (_read_dk(key)) return;

    // 2. Try fetching from key server
    if (_fetch_dk(key)) {
        _write_dk(key); // cache for offline use
        return;
    }

    // 3. Fallback: embedded XOR fragments (always works, less secure)
    for (int i = 0; i < 32; i++)
        key[i] = _kf_a[i] ^ _kf_b[i] ^ _kf_c[i];
}

// AES-256-GCM decryption, platform-specific

#ifdef _WIN32

static bool aes256gcm_decrypt(const uint8_t * key, const uint8_t * nonce,
                              const uint8_t * ct, size_t ct_len,
                              const uint8_t * aad, size_t aad_len,
                              const uint8_t * tag,
                              uint8_t * pt) {
    BCRYPT_ALG_HANDLE hAlg = NULL;
    BCRYPT_KEY_HANDLE hKey = NULL;
    bool ok = false;

    if (BCryptOpenAlgorithmProvider(&hAlg, BCRYPT_AES_ALGORITHM, NULL, 0) != 0) return false;
    if (BCryptSetProperty(hAlg, BCRYPT_CHAINING_MODE, (PUCHAR)BCRYPT_CHAIN_MODE_GCM,
                          sizeof(BCRYPT_CHAIN_MODE_GCM), 0) != 0) goto done;
    if (BCryptGenerateSymmetricKey(hAlg, &hKey, NULL, 0, (PUCHAR)key, 32, 0) != 0) goto done;

    {
        BCRYPT_AUTHENTICATED_CIPHER_MODE_INFO authInfo;
        BCRYPT_INIT_AUTH_MODE_INFO(authInfo);
        authInfo.pbNonce = (PUCHAR)nonce;
        authInfo.cbNonce = APACK_NONCE_SIZE;
        authInfo.pbTag = (PUCHAR)tag;
        authInfo.cbTag = APACK_TAG_SIZE;
        authInfo.pbAuthData = (PUCHAR)aad;
        authInfo.cbAuthData = (ULONG)aad_len;

        ULONG pt_len = 0;
        NTSTATUS s = BCryptDecrypt(hKey, (PUCHAR)ct, (ULONG)ct_len, &authInfo,
                                   NULL, 0, pt, (ULONG)ct_len, &pt_len, 0);
        ok = (s == 0 && pt_len == ct_len);
    }

done:
    if (hKey) BCryptDestroyKey(hKey);
    if (hAlg) BCryptCloseAlgorithmProvider(hAlg, 0);
    return ok;
}

#else // Linux/macOS: dlopen OpenSSL

typedef struct evp_cipher_ctx_st EVP_CIPHER_CTX;
typedef struct evp_cipher_st EVP_CIPHER;
typedef int (*pEVP_DecryptInit_ex)(EVP_CIPHER_CTX*,const EVP_CIPHER*,void*,const unsigned char*,const unsigned char*);
typedef int (*pEVP_DecryptUpdate)(EVP_CIPHER_CTX*,unsigned char*,int*,const unsigned char*,int);
typedef int (*pEVP_DecryptFinal_ex)(EVP_CIPHER_CTX*,unsigned char*,int*);
typedef EVP_CIPHER_CTX* (*pEVP_CIPHER_CTX_new)(void);
typedef void (*pEVP_CIPHER_CTX_free)(EVP_CIPHER_CTX*);
typedef int (*pEVP_CIPHER_CTX_ctrl)(EVP_CIPHER_CTX*,int,int,void*);
typedef const EVP_CIPHER* (*pEVP_aes_256_gcm)(void);

static bool aes256gcm_decrypt(const uint8_t * key, const uint8_t * nonce,
                              const uint8_t * ct, size_t ct_len,
                              const uint8_t * aad, size_t aad_len,
                              const uint8_t * tag,
                              uint8_t * pt) {
    void * ssl = dlopen("libcrypto.so", RTLD_LAZY);
    if (!ssl) ssl = dlopen("libcrypto.so.3", RTLD_LAZY);
    if (!ssl) ssl = dlopen("libcrypto.so.1.1", RTLD_LAZY);
#ifdef __APPLE__
    if (!ssl) ssl = dlopen("libcrypto.dylib", RTLD_LAZY);
#endif
    if (!ssl) return false;

    auto fn_new   = (pEVP_CIPHER_CTX_new)dlsym(ssl, "EVP_CIPHER_CTX_new");
    auto fn_free  = (pEVP_CIPHER_CTX_free)dlsym(ssl, "EVP_CIPHER_CTX_free");
    auto fn_init  = (pEVP_DecryptInit_ex)dlsym(ssl, "EVP_DecryptInit_ex");
    auto fn_upd   = (pEVP_DecryptUpdate)dlsym(ssl, "EVP_DecryptUpdate");
    auto fn_fin   = (pEVP_DecryptFinal_ex)dlsym(ssl, "EVP_DecryptFinal_ex");
    auto fn_ctrl  = (pEVP_CIPHER_CTX_ctrl)dlsym(ssl, "EVP_CIPHER_CTX_ctrl");
    auto fn_gcm   = (pEVP_aes_256_gcm)dlsym(ssl, "EVP_aes_256_gcm");

    if (!fn_new || !fn_free || !fn_init || !fn_upd || !fn_fin || !fn_ctrl || !fn_gcm) {
        dlclose(ssl);
        return false;
    }

    EVP_CIPHER_CTX * ctx = fn_new();
    if (!ctx) { dlclose(ssl); return false; }

    bool ok = false;
    #define EVP_CTRL_GCM_SET_IVLEN 0x9
    #define EVP_CTRL_GCM_SET_TAG   0x11

    if (fn_init(ctx, fn_gcm(), NULL, NULL, NULL) != 1) goto cleanup;
    if (fn_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, APACK_NONCE_SIZE, NULL) != 1) goto cleanup;
    if (fn_init(ctx, NULL, NULL, key, nonce) != 1) goto cleanup;

    // AAD
    if (aad_len > 0) {
        int tmp = 0;
        if (fn_upd(ctx, NULL, &tmp, aad, (int)aad_len) != 1) goto cleanup;
    }

    // Decrypt
    {
        int out_len = 0;
        if (fn_upd(ctx, pt, &out_len, ct, (int)ct_len) != 1) goto cleanup;

        // Set tag (must be a mutable buffer, OpenSSL may write to it)
        uint8_t tag_buf[APACK_TAG_SIZE];
        memcpy(tag_buf, tag, APACK_TAG_SIZE);
        if (fn_ctrl(ctx, EVP_CTRL_GCM_SET_TAG, APACK_TAG_SIZE, tag_buf) != 1) goto cleanup;

        int final_len = 0;
        ok = (fn_fin(ctx, pt + out_len, &final_len) == 1);
        _scrub(tag_buf, APACK_TAG_SIZE);
    }

cleanup:
    fn_free(ctx);
    dlclose(ssl);
    return ok;
}

#endif // _WIN32

// Public API

void apack_compute_fingerprint(const char * arch, int n_expert, int n_expert_used,
                               int n_layer, int n_embd, uint8_t fp[APACK_FP_SIZE],
                               const uint8_t * base_hash, size_t base_hash_len) {
    char buf[256];
    int len = snprintf(buf, sizeof(buf), "%s:%d:%d:%d:%d",
                       arch ? arch : "", n_expert, n_expert_used, n_layer, n_embd);
    if (len < 0) len = 0;
    if (len > (int)sizeof(buf)) len = (int)sizeof(buf);
    if (base_hash && base_hash_len > 0) {
        // Append ":<base_hash bytes>" to the SHA input, which binds the fingerprint to the base checkpoint.
        uint8_t in[sizeof(buf) + 1 + 64];
        size_t bl = base_hash_len > 64 ? 64 : base_hash_len;
        memcpy(in, buf, (size_t)len);
        in[len] = ':';
        memcpy(in + len + 1, base_hash, bl);
        sha256(in, (size_t)len + 1 + bl, fp);
    } else {
        sha256((const uint8_t *)buf, (size_t)len, fp);   // no base hash: shape-only fingerprint, compatible with older packs
    }
}

void apack_sha256(const uint8_t * data, size_t len, uint8_t out[32]) {
    sha256(data, len, out);
}

void apack_fingerprint_hex(const uint8_t fp[APACK_FP_SIZE], char hex[65]) {
    static const char h[] = "0123456789abcdef";
    for (int i = 0; i < APACK_FP_SIZE; i++) {
        hex[i*2]   = h[fp[i] >> 4];
        hex[i*2+1] = h[fp[i] & 0xf];
    }
    hex[64] = '\0';
}

apack_contents apack_load(const char * path, const uint8_t expected_fp[APACK_FP_SIZE]) {
    apack_contents c = {};
    c.valid = false;

    // Apply runtime hardening on first call
    _harden();

    FILE * f = fopen(path, "rb");
    if (!f) return c;

    // Read header: magic(4) + version(2) + reserved(2) + fingerprint(32) + nonce(12) + payload_size(4)
    uint8_t hdr[56];
    if (fread(hdr, 1, 56, f) != 56) { fclose(f); return c; }

    uint32_t magic = hdr[0] | (hdr[1]<<8) | (hdr[2]<<16) | (hdr[3]<<24);
    uint16_t ver   = hdr[4] | (hdr[5]<<8);
    if (magic != APACK_MAGIC || ver != APACK_VERSION) { fclose(f); return c; }

    const uint8_t * file_fp = hdr + 8;
    const uint8_t * nonce   = hdr + 40;
    uint32_t ct_len = hdr[52] | (hdr[53]<<8) | (hdr[54]<<16) | (hdr[55]<<24);

    // Sanity check payload size (max 1MB; profiles are ~100KB)
    if (ct_len > 1024 * 1024) { fclose(f); return c; }

    // Constant-time fingerprint comparison (prevents timing side-channel)
    if (expected_fp && !_ct_eq(file_fp, expected_fp, APACK_FP_SIZE)) { fclose(f); return c; }

    // Read ciphertext + tag
    uint8_t * ct = (uint8_t *)malloc(ct_len + APACK_TAG_SIZE);
    if (!ct) { fclose(f); return c; }
    if (fread(ct, 1, ct_len + APACK_TAG_SIZE, f) != ct_len + APACK_TAG_SIZE) {
        _scrub(ct, ct_len + APACK_TAG_SIZE);
        free(ct); fclose(f); return c;
    }
    fclose(f);

    // Derive key (scrub all intermediates)
    uint8_t mk[32];
    _get_mk(mk);
    uint8_t dk[32];
    hkdf_sha256(mk, 32, file_fp, APACK_FP_SIZE, (const uint8_t *)"apack-v1", 8, dk, 32);
    _scrub(mk, 32);

    // Allocate plaintext buffer and lock it in RAM (prevent swap-out)
    uint8_t * pt = (uint8_t *)malloc(ct_len);
    if (!pt) { _scrub(dk, 32); _scrub(ct, ct_len + APACK_TAG_SIZE); free(ct); return c; }
#if defined(__linux__)
    mlock(pt, ct_len); // best-effort: prevent swapping decrypted data to disk
#elif defined(_WIN32)
    VirtualLock(pt, ct_len);
#endif

    bool decrypted = aes256gcm_decrypt(dk, nonce, ct, ct_len,
                                        file_fp, APACK_FP_SIZE, // AAD = fingerprint
                                        ct + ct_len, // tag is after ciphertext
                                        pt);
    _scrub(dk, 32);
    _scrub(ct, ct_len + APACK_TAG_SIZE);
    free(ct);
    if (!decrypted) { _scrub(pt, ct_len); free(pt); return c; }

    // Parse TLV sections
    memcpy(c.fingerprint, file_fp, APACK_FP_SIZE);
    c.version = ver;
    c.n_profiles = 0;
    c.portfolio_len = 0;

    size_t pos = 0;
    while (pos + 8 <= ct_len) {
        uint16_t sec_type = pt[pos] | (pt[pos+1]<<8);
        uint16_t name_len = pt[pos+2] | (pt[pos+3]<<8);
        pos += 4;
        if (pos + name_len + 4 > ct_len) break;

        char name[64] = {};
        size_t nc = name_len < 63 ? name_len : 63;
        memcpy(name, pt + pos, nc);
        pos += name_len;

        uint32_t data_len = pt[pos] | (pt[pos+1]<<8) | (pt[pos+2]<<16) | (pt[pos+3]<<24);
        pos += 4;
        if (pos + data_len > ct_len) break;

        const uint8_t * data = pt + pos;
        pos += data_len;

        if (sec_type == APACK_SEC_PROFILE && c.n_profiles < APACK_MAX_PROFILES) {
            apack_profile * p = &c.profiles[c.n_profiles];
            snprintf(p->name, sizeof(p->name), "%s", name);

            if (data_len >= 4) {
                p->n_layers = data[0] | (data[1]<<8);
                int n_exp   = data[2] | (data[3]<<8);
                if (n_exp > APACK_EXPERTS_PER) n_exp = APACK_EXPERTS_PER;
                if (p->n_layers > APACK_MAX_LAYERS) p->n_layers = APACK_MAX_LAYERS;

                size_t off = 4;
                for (int il = 0; il < p->n_layers && off + n_exp * 4 <= data_len; il++) {
                    for (int e = 0; e < n_exp; e++) {
                        int32_t v = (int32_t)(data[off] | (data[off+1]<<8) |
                                              (data[off+2]<<16) | (data[off+3]<<24));
                        p->experts[il][e] = v;
                        off += 4;
                    }
                    // Fill remaining slots with -1
                    for (int e = n_exp; e < APACK_EXPERTS_PER; e++)
                        p->experts[il][e] = -1;
                }
            }
            c.n_profiles++;
        } else if (sec_type == APACK_SEC_PORTFOLIO) {
            size_t copy = data_len < sizeof(c.portfolio_json) - 1 ? data_len : sizeof(c.portfolio_json) - 1;
            memcpy(c.portfolio_json, data, copy);
            c.portfolio_json[copy] = '\0';
            c.portfolio_len = (int)copy;
        }
    }

    // Scrub decrypted plaintext before freeing
    _scrub(pt, ct_len);
#if defined(__linux__)
    munlock(pt, ct_len);
#elif defined(_WIN32)
    VirtualUnlock(pt, ct_len);
#endif
    free(pt);
    c.valid = (c.n_profiles > 0);
    return c;
}

bool apack_cache_dir(char * buf, size_t buf_size) {
#ifdef _WIN32
    char home[MAX_PATH];
    if (SUCCEEDED(SHGetFolderPathA(NULL, CSIDL_PROFILE, NULL, 0, home))) {
        snprintf(buf, buf_size, "%s/.alphachat/packs", home);
        return true;
    }
    return false;
#else
    const char * home = getenv("HOME");
    if (!home) return false;
    snprintf(buf, buf_size, "%s/.alphachat/packs", home);
    return true;
#endif
}

bool apack_find_cached(const uint8_t fp[APACK_FP_SIZE], char * path, size_t path_size) {
    char dir[512];
    if (!apack_cache_dir(dir, sizeof(dir))) return false;
    char hex[65];
    apack_fingerprint_hex(fp, hex);
    snprintf(path, path_size, "%s/%s.apack", dir, hex);
    FILE * f = fopen(path, "rb");
    if (f) { fclose(f); return true; }
    return false;
}

// HTTP sync: fetch the pack from the CDN if not cached.
//
// Security design:
// - URL: /d/{sha256_hex}, an opaque path with no extension or descriptive segments
// - User-Agent: generic browser string (not "alphachat")
// - TLS only (HTTPS)
// - Single attempt, no retry (prevents timing analysis)
// - Response validated by AES-GCM auth tag (tamper-proof)
// - No log messages reveal purpose (silent success/failure)
// - Cache dir uses opaque hex filename, no metadata sidecar
//
// What a network observer sees: HTTPS GET to CDN for a SHA-256 named blob.
// Indistinguishable from any content-addressed storage fetch.

// Sync endpoint, obfuscated against casual string extraction
// Actual URL: https://d.alphachat.ai/d/{hash}
// Stored as two parts to avoid a single searchable string
static const char * _sync_host() {
    static char h[64] = {};
    if (!h[0]) { snprintf(h, sizeof(h), "%s.%s.%s", "d", "alphachat", "ai"); }
    return h;
}

static bool _ensure_dir(const char * dir) {
#ifdef _WIN32
    CreateDirectoryA(dir, NULL);
    // Also create parent if needed
    std::string s(dir);
    size_t pos = s.rfind('/');
    if (pos == std::string::npos) pos = s.rfind('\\');
    if (pos != std::string::npos) {
        std::string parent = s.substr(0, pos);
        CreateDirectoryA(parent.c_str(), NULL);
        CreateDirectoryA(dir, NULL);
    }
    return true;
#else
    // mkdir -p equivalent
    std::string s(dir);
    for (size_t i = 1; i < s.size(); i++) {
        if (s[i] == '/') {
            s[i] = '\0';
            mkdir(s.c_str(), 0755);
            s[i] = '/';
        }
    }
    mkdir(s.c_str(), 0755);
    return true;
#endif
}

#ifdef _WIN32

static bool _http_fetch(const char * host, const char * path, const char * out_path) {
    // WinHTTP: HTTPS GET with generic User-Agent
    wchar_t whost[128], wpath[256];
    MultiByteToWideChar(CP_UTF8, 0, host, -1, whost, 128);
    MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath, 256);

    HINTERNET hSession = WinHttpOpen(L"Mozilla/5.0", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                                     WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (!hSession) return false;

    // Timeouts: 15s connect, 30s send/receive
    WinHttpSetTimeouts(hSession, 15000, 15000, 30000, 30000);

    HINTERNET hConnect = WinHttpConnect(hSession, whost, INTERNET_DEFAULT_HTTPS_PORT, 0);
    if (!hConnect) { WinHttpCloseHandle(hSession); return false; }

    HINTERNET hRequest = WinHttpOpenRequest(hConnect, L"GET", wpath, NULL,
                                            WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES,
                                            WINHTTP_FLAG_SECURE);
    if (!hRequest) { WinHttpCloseHandle(hConnect); WinHttpCloseHandle(hSession); return false; }

    bool ok = false;
    if (WinHttpSendRequest(hRequest, WINHTTP_NO_ADDITIONAL_HEADERS, 0, NULL, 0, 0, 0) &&
        WinHttpReceiveResponse(hRequest, NULL)) {
        DWORD status = 0, sz = sizeof(status);
        WinHttpQueryHeaders(hRequest, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                           NULL, &status, &sz, NULL);
        if (status == 200) {
            std::ofstream out(out_path, std::ios::binary);
            if (out.is_open()) {
                DWORD avail = 0, read = 0;
                char buf[8192];
                while (WinHttpQueryDataAvailable(hRequest, &avail) && avail > 0) {
                    DWORD to_read = avail < sizeof(buf) ? avail : sizeof(buf);
                    if (WinHttpReadData(hRequest, buf, to_read, &read) && read > 0) {
                        out.write(buf, read);
                    } else break;
                }
                out.close();
                ok = true;
            }
        }
    }

    WinHttpCloseHandle(hRequest);
    WinHttpCloseHandle(hConnect);
    WinHttpCloseHandle(hSession);
    return ok;
}

#else // Linux/macOS: libcurl via dlopen

typedef void CURL;
typedef int CURLcode;
typedef int CURLoption;

static bool _http_fetch(const char * host, const char * path, const char * out_path) {
    void * curl_lib = dlopen("libcurl.so", RTLD_LAZY);
    if (!curl_lib) curl_lib = dlopen("libcurl.so.4", RTLD_LAZY);
#ifdef __APPLE__
    if (!curl_lib) curl_lib = dlopen("libcurl.dylib", RTLD_LAZY);
#endif
    if (!curl_lib) return false;

    auto fn_init   = (CURL*(*)())dlsym(curl_lib, "curl_easy_init");
    auto fn_setopt = (CURLcode(*)(CURL*,CURLoption,...))dlsym(curl_lib, "curl_easy_setopt");
    auto fn_perform = (CURLcode(*)(CURL*))dlsym(curl_lib, "curl_easy_perform");
    auto fn_cleanup = (void(*)(CURL*))dlsym(curl_lib, "curl_easy_cleanup");
    auto fn_getinfo = (CURLcode(*)(CURL*,int,...))dlsym(curl_lib, "curl_easy_getinfo");

    if (!fn_init || !fn_setopt || !fn_perform || !fn_cleanup) {
        dlclose(curl_lib);
        return false;
    }

    char url[512];
    snprintf(url, sizeof(url), "https://%s%s", host, path);

    CURL * ch = fn_init();
    if (!ch) { dlclose(curl_lib); return false; }

    FILE * fp = fopen(out_path, "wb");
    if (!fp) { fn_cleanup(ch); dlclose(curl_lib); return false; }

    #define CURLOPT_URL          10002
    #define CURLOPT_WRITEDATA    10001
    #define CURLOPT_USERAGENT    10018
    #define CURLOPT_TIMEOUT      13
    #define CURLOPT_CONNECTTIMEOUT 78
    #define CURLOPT_FOLLOWLOCATION 52
    #define CURLOPT_FAILONERROR    45
    #define CURLINFO_RESPONSE_CODE 0x200002

    fn_setopt(ch, (CURLoption)CURLOPT_URL, url);
    fn_setopt(ch, (CURLoption)CURLOPT_WRITEDATA, fp);
    fn_setopt(ch, (CURLoption)CURLOPT_USERAGENT, "Mozilla/5.0");
    fn_setopt(ch, (CURLoption)CURLOPT_TIMEOUT, (long)30);
    fn_setopt(ch, (CURLoption)CURLOPT_CONNECTTIMEOUT, (long)15);
    fn_setopt(ch, (CURLoption)CURLOPT_FOLLOWLOCATION, (long)1);
    fn_setopt(ch, (CURLoption)CURLOPT_FAILONERROR, (long)1);

    CURLcode res = fn_perform(ch);
    fclose(fp);

    bool ok = (res == 0);
    if (!ok) {
        remove(out_path); // clean up partial download
    }

    fn_cleanup(ch);
    dlclose(curl_lib);
    return ok;
}

#endif // _WIN32

bool apack_sync(const uint8_t fp[APACK_FP_SIZE]) {
    // Already cached?
    char path[512];
    if (apack_find_cached(fp, path, sizeof(path))) return true;

    // Build opaque URL path: /d/{hash} (no extension, no descriptive segments)
    char hex[65];
    apack_fingerprint_hex(fp, hex);
    char url_path[128];
    snprintf(url_path, sizeof(url_path), "/d/%s", hex);

    // Ensure cache directory exists
    char dir[512];
    if (!apack_cache_dir(dir, sizeof(dir))) return false;
    _ensure_dir(dir);

    // Build temp path (download to .tmp, rename on success for atomicity)
    char tmp_path[512];
    snprintf(tmp_path, sizeof(tmp_path), "%s/%s.tmp", dir, hex);
    snprintf(path, sizeof(path), "%s/%s.apack", dir, hex);

    // Single fetch attempt, no retry
    bool fetched = _http_fetch(_sync_host(), url_path, tmp_path);
    if (!fetched) {
        remove(tmp_path);
        return false;
    }

    // Validate: try to decrypt before accepting
    apack_contents test = apack_load(tmp_path, fp);
    if (!test.valid) {
        remove(tmp_path); // corrupted or tampered
        return false;
    }

    // Atomic rename
    remove(path); // remove stale if exists
    rename(tmp_path, path);
    return true;
}
