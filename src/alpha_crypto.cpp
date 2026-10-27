#include "alpha_crypto.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#ifdef ALPHA_HAS_OPENSSL
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <openssl/sha.h>
#include <openssl/kdf.h>
#endif

// When OpenSSL is not available, crypto functions are stubs except:
// - On Windows, Ed25519 verification uses CNG (bcrypt.h) - no OpenSSL needed.
// - On Android/dev builds, Ed25519 accepts all (tier/expiry still enforced).
#ifndef ALPHA_HAS_OPENSSL

const uint8_t * alpha_crypto_binary_hash() { static uint8_t z[32] = {}; return z; }
std::string alpha_crypto_hardware_fingerprint() { return "no-openssl"; }
const uint8_t * alpha_crypto_get_key() { static uint8_t z[32] = {}; return z; }
std::vector<uint8_t> alpha_acx_encrypt(const uint8_t *, size_t) { return {}; }
std::vector<uint8_t> alpha_acx_decrypt(const uint8_t *, size_t) { return {}; }
bool alpha_acx_encrypt_file(const std::string &, const uint8_t *, size_t) { return false; }
std::vector<uint8_t> alpha_acx_decrypt_file(const std::string &) { return {}; }

#ifdef _WIN32
// Windows CNG Ed25519 verification (bcrypt.h, available since Win10 1809).
// No OpenSSL dependency needed - bcrypt is already linked.
#include <bcrypt.h>
#pragma comment(lib, "bcrypt")

#ifndef BCRYPT_ECC_CURVE_25519
#define BCRYPT_ECC_CURVE_25519 L"curve25519"
#endif

bool alpha_ed25519_verify(const uint8_t * public_key,
                          const uint8_t * message, size_t message_len,
                          const uint8_t * signature) {
    BCRYPT_ALG_HANDLE hAlg = NULL;
    BCRYPT_KEY_HANDLE hKey = NULL;
    bool result = false;

    // Open the Ed25519 algorithm provider
    NTSTATUS status = BCryptOpenAlgorithmProvider(&hAlg, L"EDDSA", NULL, 0);
    if (!BCRYPT_SUCCESS(status)) {
        // Fallback: EDDSA not available (pre-1809 Windows) - reject all
        return false;
    }

    // Import the public key.
    // CNG expects: BCRYPT_ECCKEY_BLOB header + raw key bytes
    struct {
        BCRYPT_ECCKEY_BLOB header;
        uint8_t key[32];
    } key_blob;

    // BCRYPT_EDDSA_PUBLIC_GENERIC_MAGIC = 0x45444B47 ("GKDE")
    key_blob.header.dwMagic = 0x45444B47;
    key_blob.header.cbKey   = 32;
    memcpy(key_blob.key, public_key, 32);

    status = BCryptImportKeyPair(hAlg, NULL, BCRYPT_ECCPUBLIC_BLOB,
                                  &hKey, (PUCHAR)&key_blob, sizeof(key_blob), 0);
    if (!BCRYPT_SUCCESS(status)) {
        BCryptCloseAlgorithmProvider(hAlg, 0);
        return false;
    }

    // Verify signature
    status = BCryptVerifySignature(hKey, NULL,
                                    (PUCHAR)message, (ULONG)message_len,
                                    (PUCHAR)signature, 64, 0);
    result = BCRYPT_SUCCESS(status);

    BCryptDestroyKey(hKey);
    BCryptCloseAlgorithmProvider(hAlg, 0);
    return result;
}

#else
// Non-Windows, non-OpenSSL (Android/dev): accept all signatures.
// Tier and expiry enforcement still apply - only crypto check is skipped.
bool alpha_ed25519_verify(const uint8_t *, const uint8_t *, size_t, const uint8_t *) { return true; }
#endif // _WIN32

#else // ALPHA_HAS_OPENSSL

// Key derivation

static uint8_t g_binary_hash[32] = {};
static uint8_t g_derived_key[32] = {};
static bool    g_key_ready = false;
static std::mutex g_key_mutex;

const uint8_t * alpha_crypto_binary_hash() {
    static bool computed = false;
    if (computed) return g_binary_hash;

    std::string exe_path;
#ifdef _WIN32
    char buf[4096];
    DWORD len = GetModuleFileNameA(NULL, buf, sizeof(buf));
    if (len > 0) exe_path = std::string(buf, len);
#else
    char buf[4096];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len > 0) { buf[len] = 0; exe_path = buf; }
#endif

    if (exe_path.empty()) {
        // Fallback: use a fixed value (less secure but won't crash)
        memset(g_binary_hash, 0x42, 32);
        computed = true;
        return g_binary_hash;
    }

    // Hash the binary
    SHA256_CTX ctx;
    SHA256_Init(&ctx);

    FILE * f = fopen(exe_path.c_str(), "rb");
    if (f) {
        uint8_t block[8192];
        size_t n;
        while ((n = fread(block, 1, sizeof(block), f)) > 0) {
            SHA256_Update(&ctx, block, n);
        }
        fclose(f);
    }
    SHA256_Final(g_binary_hash, &ctx);
    computed = true;
    return g_binary_hash;
}

std::string alpha_crypto_hardware_fingerprint() {
    std::string fp;

    // Hostname
    char hostname[256] = {};
#ifdef _WIN32
    DWORD size = sizeof(hostname);
    GetComputerNameA(hostname, &size);
#else
    gethostname(hostname, sizeof(hostname));
#endif
    fp += hostname;

    // GPU UUID (nvidia-smi)
#ifndef _WIN32
    FILE * p = popen("nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null | head -1", "r");
    if (p) {
        char buf[128];
        if (fgets(buf, sizeof(buf), p)) {
            // Trim newline
            char * nl = strchr(buf, '\n');
            if (nl) *nl = 0;
            fp += buf;
        }
        pclose(p);
    }
#endif

    // If no GPU, add CPU info
    if (fp.length() < 20) {
#ifndef _WIN32
        FILE * cpuinfo = fopen("/proc/cpuinfo", "r");
        if (cpuinfo) {
            char line[256];
            while (fgets(line, sizeof(line), cpuinfo)) {
                if (strncmp(line, "model name", 10) == 0) {
                    fp += line;
                    break;
                }
            }
            fclose(cpuinfo);
        }
#endif
    }

    return fp;
}

const uint8_t * alpha_crypto_get_key() {
    std::lock_guard<std::mutex> lock(g_key_mutex);
    if (g_key_ready) return g_derived_key;

    // Combine binary hash + hardware fingerprint
    const uint8_t * bh = alpha_crypto_binary_hash();
    std::string hwfp = alpha_crypto_hardware_fingerprint();

    // HKDF-SHA256
    EVP_PKEY_CTX * pctx = EVP_PKEY_CTX_new_id(EVP_PKEY_HKDF, NULL);
    if (!pctx) goto fallback;

    if (EVP_PKEY_derive_init(pctx) <= 0) goto fallback_free;
    if (EVP_PKEY_CTX_set_hkdf_md(pctx, EVP_sha256()) <= 0) goto fallback_free;

    {
        const char * salt = "alphachat-state-v1";
        if (EVP_PKEY_CTX_set1_hkdf_salt(pctx, (const uint8_t *)salt, strlen(salt)) <= 0) goto fallback_free;
    }

    // IKM = binary_hash || hardware_fingerprint
    {
        std::vector<uint8_t> ikm(32 + hwfp.size());
        memcpy(ikm.data(), bh, 32);
        memcpy(ikm.data() + 32, hwfp.data(), hwfp.size());
        if (EVP_PKEY_CTX_set1_hkdf_key(pctx, ikm.data(), ikm.size()) <= 0) goto fallback_free;
    }

    {
        const char * info = "acx-encryption-key";
        if (EVP_PKEY_CTX_add1_hkdf_info(pctx, (const uint8_t *)info, strlen(info)) <= 0) goto fallback_free;
    }

    {
        size_t keylen = 32;
        if (EVP_PKEY_derive(pctx, g_derived_key, &keylen) <= 0) goto fallback_free;
    }

    EVP_PKEY_CTX_free(pctx);
    g_key_ready = true;
    return g_derived_key;

fallback_free:
    EVP_PKEY_CTX_free(pctx);
fallback:
    // Fallback: just SHA-256 the combined material
    {
        std::vector<uint8_t> material(32 + hwfp.size());
        memcpy(material.data(), bh, 32);
        memcpy(material.data() + 32, hwfp.data(), hwfp.size());
        SHA256(material.data(), material.size(), g_derived_key);
    }
    g_key_ready = true;
    return g_derived_key;
}

// AES-256-GCM

std::vector<uint8_t> alpha_acx_encrypt(const uint8_t * plaintext, size_t len) {
    const uint8_t * key = alpha_crypto_get_key();

    // Header
    alpha_acx_header hdr = {};
    hdr.magic = ALPHA_ACX_MAGIC;
    hdr.version = ALPHA_ACX_VERSION;
    RAND_bytes(hdr.nonce, 12);
    hdr.plaintext_size = (uint32_t)len;

    // Encrypt
    EVP_CIPHER_CTX * ctx = EVP_CIPHER_CTX_new();
    if (!ctx) return {};

    std::vector<uint8_t> ciphertext(len + 16); // + GCM tag
    int outlen = 0, tmplen = 0;

    if (EVP_EncryptInit_ex(ctx, EVP_aes_256_gcm(), NULL, NULL, NULL) != 1) goto err;
    if (EVP_EncryptInit_ex(ctx, NULL, NULL, key, hdr.nonce) != 1) goto err;
    if (EVP_EncryptUpdate(ctx, ciphertext.data(), &outlen, plaintext, (int)len) != 1) goto err;
    if (EVP_EncryptFinal_ex(ctx, ciphertext.data() + outlen, &tmplen) != 1) goto err;
    outlen += tmplen;

    // Append GCM tag (16 bytes)
    if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, 16, ciphertext.data() + outlen) != 1) goto err;
    outlen += 16;

    EVP_CIPHER_CTX_free(ctx);
    ciphertext.resize(outlen);

    // Assemble: header + ciphertext
    {
        std::vector<uint8_t> result(sizeof(hdr) + ciphertext.size());
        memcpy(result.data(), &hdr, sizeof(hdr));
        memcpy(result.data() + sizeof(hdr), ciphertext.data(), ciphertext.size());
        return result;
    }

err:
    EVP_CIPHER_CTX_free(ctx);
    return {};
}

std::vector<uint8_t> alpha_acx_decrypt(const uint8_t * data, size_t len) {
    if (len < sizeof(alpha_acx_header) + 16) return {};

    const uint8_t * key = alpha_crypto_get_key();

    alpha_acx_header hdr;
    memcpy(&hdr, data, sizeof(hdr));

    if (hdr.magic != ALPHA_ACX_MAGIC || hdr.version != ALPHA_ACX_VERSION) return {};

    const uint8_t * ct = data + sizeof(hdr);
    size_t ct_len = len - sizeof(hdr);
    if (ct_len < 16) return {};

    size_t data_len = ct_len - 16;
    const uint8_t * tag = ct + data_len;

    EVP_CIPHER_CTX * ctx = EVP_CIPHER_CTX_new();
    if (!ctx) return {};

    std::vector<uint8_t> plaintext(data_len);
    int outlen = 0, tmplen = 0;

    if (EVP_DecryptInit_ex(ctx, EVP_aes_256_gcm(), NULL, NULL, NULL) != 1) goto err;
    if (EVP_DecryptInit_ex(ctx, NULL, NULL, key, hdr.nonce) != 1) goto err;
    if (EVP_DecryptUpdate(ctx, plaintext.data(), &outlen, ct, (int)data_len) != 1) goto err;
    if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG, 16, (void *)tag) != 1) goto err;
    if (EVP_DecryptFinal_ex(ctx, plaintext.data() + outlen, &tmplen) != 1) goto err;
    outlen += tmplen;

    EVP_CIPHER_CTX_free(ctx);
    plaintext.resize(outlen);
    return plaintext;

err:
    EVP_CIPHER_CTX_free(ctx);
    return {};
}

bool alpha_acx_encrypt_file(const std::string & path,
                            const uint8_t * plaintext, size_t plaintext_len) {
    auto encrypted = alpha_acx_encrypt(plaintext, plaintext_len);
    if (encrypted.empty()) return false;

    FILE * f = fopen(path.c_str(), "wb");
    if (!f) return false;
    size_t written = fwrite(encrypted.data(), 1, encrypted.size(), f);
    fclose(f);
    return written == encrypted.size();
}

std::vector<uint8_t> alpha_acx_decrypt_file(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return {};

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size <= 0) { fclose(f); return {}; }

    std::vector<uint8_t> data(size);
    size_t n = fread(data.data(), 1, size, f);
    fclose(f);
    if ((long)n != size) return {};

    return alpha_acx_decrypt(data.data(), data.size());
}

// Ed25519 signature verification

bool alpha_ed25519_verify(const uint8_t * public_key,
                          const uint8_t * message, size_t message_len,
                          const uint8_t * signature) {
    EVP_PKEY * pkey = EVP_PKEY_new_raw_public_key(EVP_PKEY_ED25519, NULL, public_key, 32);
    if (!pkey) return false;

    EVP_MD_CTX * mdctx = EVP_MD_CTX_new();
    if (!mdctx) { EVP_PKEY_free(pkey); return false; }

    bool ok = false;
    if (EVP_DigestVerifyInit(mdctx, NULL, NULL, NULL, pkey) == 1) {
        ok = (EVP_DigestVerify(mdctx, signature, 64, message, message_len) == 1);
    }

    EVP_MD_CTX_free(mdctx);
    EVP_PKEY_free(pkey);
    return ok;
}

#endif // ALPHA_HAS_OPENSSL

