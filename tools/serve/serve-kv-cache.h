// SPDX-License-Identifier: Apache-2.0
#pragma once
// Redis KV cache: cross-session prefix reuse
//
// Caches paged KV cache data in Redis keyed by prompt token hash.
// On cache hit, restores KV to GPU and skips prefill entirely.
// A repeated system prompt of a few thousand tokens then needs no prefill.

#ifdef ALPHA_HAS_REDIS
#if __has_include(<hiredis/hiredis.h>)
#include <hiredis/hiredis.h>
#elif __has_include("hiredis.h")
#include "hiredis.h"
#else
#undef ALPHA_HAS_REDIS
#endif
#endif

#include "llama.h"
#include <vector>
#include <cstdint>
#include <cstring>
#include <cstdio>

namespace alphaserve {

struct KVCacheManager {
#ifdef ALPHA_HAS_REDIS
    redisContext * redis = nullptr;
#else
    void * redis = nullptr;
#endif
    int ttl_seconds = 3600;
    uint64_t model_hash = 0;  // distinguishes models

    bool connect(const char * host, int port) {
#ifdef ALPHA_HAS_REDIS
        struct timeval tv = {5, 0};
        redis = redisConnectWithTimeout(host, port, tv);
        if (!redis || redis->err) {
            fprintf(stderr, "[kv-cache] Redis connect failed: %s\n",
                    redis ? redis->errstr : "alloc error");
            if (redis) { redisFree(redis); redis = nullptr; }
            return false;
        }
        fprintf(stderr, "[kv-cache] Redis connected: %s:%d (TTL=%ds)\n", host, port, ttl_seconds);
        return true;
#else
        (void)host; (void)port;
        fprintf(stderr, "[kv-cache] Redis not compiled (ALPHA_REDIS=OFF)\n");
        return false;
#endif
    }

    void set_model_hash(const char * model_path) {
        // FNV-1a hash of model path
        model_hash = 14695981039346656037ULL;
        for (const char * p = model_path; *p; p++) {
            model_hash ^= (uint8_t)*p;
            model_hash *= 1099511628211ULL;
        }
    }

    // Hash prompt tokens for cache key
    static uint64_t hash_tokens(const llama_token * tokens, int n) {
        uint64_t h = 14695981039346656037ULL;
        for (int i = 0; i < n; i++) {
            h ^= (uint32_t)tokens[i];
            h *= 1099511628211ULL;
        }
        return h;
    }

    // Save KV blob to Redis
    bool save(const llama_token * tokens, int n_tokens,
              const uint8_t * kv_data, size_t kv_size) {
#ifdef ALPHA_HAS_REDIS
        if (!redis) return false;
        uint64_t tok_hash = hash_tokens(tokens, n_tokens);
        char key[128];
        snprintf(key, sizeof(key), "kv:%llx:%llx:%d",
                 (unsigned long long)model_hash, (unsigned long long)tok_hash, n_tokens);

        redisReply * r = (redisReply *)redisCommand(redis,
            "SETEX %s %d %b", key, ttl_seconds, kv_data, kv_size);
        if (!r || r->type == REDIS_REPLY_ERROR) {
            if (r) freeReplyObject(r);
            return false;
        }
        freeReplyObject(r);
        fprintf(stderr, "[kv-cache] saved %d tokens (%.1f MB) key=%s\n",
                n_tokens, (double)kv_size / (1024*1024), key);
        return true;
#else
        (void)tokens; (void)n_tokens; (void)kv_data; (void)kv_size;
        return false;
#endif
    }

    // Load KV blob from Redis. Returns {data, n_cached_tokens} or empty.
    struct CacheResult {
        std::vector<uint8_t> data;
        int n_tokens = 0;
    };

    CacheResult load(const llama_token * tokens, int n_tokens) {
#ifdef ALPHA_HAS_REDIS
        if (!redis) return {};
        uint64_t tok_hash = hash_tokens(tokens, n_tokens);
        char key[128];
        snprintf(key, sizeof(key), "kv:%llx:%llx:%d",
                 (unsigned long long)model_hash, (unsigned long long)tok_hash, n_tokens);

        redisReply * r = (redisReply *)redisCommand(redis, "GET %s", key);
        if (!r || r->type != REDIS_REPLY_STRING || r->len == 0) {
            if (r) freeReplyObject(r);
            return {};
        }
        CacheResult result;
        result.data.assign((uint8_t *)r->str, (uint8_t *)r->str + r->len);
        result.n_tokens = n_tokens;
        freeReplyObject(r);
        fprintf(stderr, "[kv-cache] HIT %d tokens (%.1f MB) key=%s\n",
                n_tokens, (double)result.data.size() / (1024*1024), key);
        return result;
#else
        (void)tokens; (void)n_tokens;
        return {};
#endif
    }

    void disconnect() {
#ifdef ALPHA_HAS_REDIS
        if (redis) { redisFree(redis); redis = nullptr; }
#endif
    }

    ~KVCacheManager() { disconnect(); }
};

} // namespace alphaserve
