#pragma once

#include <cstdint>
#include <string>
#include <vector>

// AES-256-GCM encryption for checkpoint/state files

// .acx file format: ACX1(4) + version(4) + nonce(12) + plaintext_size(4) + ciphertext(N+16)
#define ALPHA_ACX_MAGIC  0x31584341  // "ACX1"
#define ALPHA_ACX_VERSION 1

struct alpha_acx_header {
    uint32_t magic;
    uint32_t version;
    uint8_t  nonce[12];
    uint32_t plaintext_size;
};

// Encrypt plaintext -> .acx file. Key derived from binary hash + hardware fingerprint.
bool alpha_acx_encrypt_file(const std::string & path,
                            const uint8_t * plaintext, size_t plaintext_len);

// Decrypt .acx file -> plaintext. Returns empty vector on failure (wrong key, tampered, etc.)
std::vector<uint8_t> alpha_acx_decrypt_file(const std::string & path);

// Encrypt to memory buffer (for streaming/append use).
std::vector<uint8_t> alpha_acx_encrypt(const uint8_t * plaintext, size_t len);

// Decrypt from memory buffer.
std::vector<uint8_t> alpha_acx_decrypt(const uint8_t * data, size_t len);

// Key derivation

// Get the derived key (cached after first call).
// key = HKDF-SHA256(binary_hash || hardware_fingerprint, salt="alphachat-state-v1")
const uint8_t * alpha_crypto_get_key();

// SHA-256 hash of the running binary (/proc/self/exe or equivalent).
// Returns 32-byte hash. Cached after first call.
const uint8_t * alpha_crypto_binary_hash();

// Hardware fingerprint string (GPU UUID + hostname hash).
std::string alpha_crypto_hardware_fingerprint();

// Ed25519 signature verification

// Verify an Ed25519 signature. public_key = 32 bytes, signature = 64 bytes.
bool alpha_ed25519_verify(const uint8_t * public_key,
                          const uint8_t * message, size_t message_len,
                          const uint8_t * signature);

