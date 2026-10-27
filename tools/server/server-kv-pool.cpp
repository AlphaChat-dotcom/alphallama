#include "server-kv-pool.h"

#ifdef ALPHA_HAS_REDIS
#include <hiredis/hiredis.h>
#endif

// Redis L3 operations

void kv_pool::save_to_redis(const std::string & key, const uint8_t * data, size_t size, int32_t n_tokens) {
#ifdef ALPHA_HAS_REDIS
    if (!redis) return;

    // Store KV data as binary blob: kvpool:{key}
    std::string redis_key = "kvpool:" + key;

    // Use SETEX with binary-safe SET
    redisReply * reply = (redisReply *)redisCommand(redis,
        "SET %s %b EX %d", redis_key.c_str(), data, size, ttl_sec);
    if (reply) {
        freeReplyObject(reply);
    }

    // Store metadata: kvpool_meta:{key} = n_tokens
    std::string meta_key = "kvpool_meta:" + key;
    reply = (redisReply *)redisCommand(redis,
        "SETEX %s %d %d", meta_key.c_str(), ttl_sec, n_tokens);
    if (reply) {
        freeReplyObject(reply);
    }

    SRV_INF("kv_pool: saved to L3 (Redis) '%s' (%zu bytes, TTL=%ds)\n",
            key.c_str(), size, ttl_sec);
#else
    (void)key; (void)data; (void)size; (void)n_tokens;
#endif
}

bool kv_pool::load_from_redis(const std::string & key, std::vector<uint8_t> & out_data, int32_t & out_n_tokens) {
#ifdef ALPHA_HAS_REDIS
    if (!redis) return false;

    std::string redis_key = "kvpool:" + key;

    redisReply * reply = (redisReply *)redisCommand(redis, "GET %s", redis_key.c_str());
    if (!reply || reply->type != REDIS_REPLY_STRING || reply->len == 0) {
        if (reply) freeReplyObject(reply);
        return false;
    }

    out_data.assign(
        reinterpret_cast<const uint8_t *>(reply->str),
        reinterpret_cast<const uint8_t *>(reply->str) + reply->len
    );
    freeReplyObject(reply);

    // Get metadata
    std::string meta_key = "kvpool_meta:" + key;
    reply = (redisReply *)redisCommand(redis, "GET %s", meta_key.c_str());
    if (reply && reply->type == REDIS_REPLY_STRING) {
        out_n_tokens = std::atoi(reply->str);
    } else {
        out_n_tokens = 0;
    }
    if (reply) freeReplyObject(reply);

    return true;
#else
    (void)key; (void)out_data; (void)out_n_tokens;
    return false;
#endif
}

void kv_pool::remove_from_redis(const std::string & key) {
#ifdef ALPHA_HAS_REDIS
    if (!redis) return;

    std::string redis_key = "kvpool:" + key;
    std::string meta_key = "kvpool_meta:" + key;

    redisReply * reply = (redisReply *)redisCommand(redis, "DEL %s %s",
        redis_key.c_str(), meta_key.c_str());
    if (reply) freeReplyObject(reply);
#else
    (void)key;
#endif
}
