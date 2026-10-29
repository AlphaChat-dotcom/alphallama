// SPDX-License-Identifier: Apache-2.0
/* Draft-vocab delta gate: prefill cost and draft-head behaviour across a sequence of requests on one engine.
 *
 *   ai_api_draftvocab_test <model.gguf> <n_gen> <prompt1.i32> [<prompt2.i32> ...]
 *
 * Each prompt (token ids, alphainfer --dump-tokens) is a request on stream 0, as a server sends it: reset the stream,
 * one ai_engine_prefill_chunk(last=1) with the whole prompt, then MTP decode to n_gen tokens. Per request it prints the
 * prefill wall time, the MTP rounds and an FNV hash of the generated tokens.
 *
 * The draft head's rows are re-chosen per request from the prompt's scripts, and only the rows that changed are
 * rewritten. The draft head decides only acceptance, so in exact mode the tokens never depend on it but the round
 * count does: run the same sequence on two library builds and compare (rounds, hash) per request. They must be
 * identical; the prefill times are the measurement. Order the prompts so the scripts change (en, zh, en, ko, en) or
 * the delta path is never exercised.
 */
#include "api/ai_engine.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int32_t * read_i32(const char * path, int * n) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    int32_t * v = (int32_t *) malloc((size_t) sz + 4);
    *n = (int) fread(v, 4, (size_t) sz / 4, f);
    fclose(f);
    return v;
}

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.gguf> <n_gen> <prompt1.i32> [<prompt2.i32> ...]\n", argv[0]);
        return 2;
    }
    const int n_gen = atoi(argv[2]);
    ai_engine_params p;
    memset(&p, 0, sizeof p);
    p.model_path           = argv[1];
    p.ctx_cells_per_stream = getenv("AI_API_CTX") ? atoi(getenv("AI_API_CTX")) : 16384;
    p.n_streams            = getenv("AI_API_STREAMS") ? atoi(getenv("AI_API_STREAMS")) : 1;
    p.kv_type_is_q4_0      = getenv("AI_API_KV") ? atoi(getenv("AI_API_KV")) : -1;   /* -1 = the library's auto */
    p.mtp_k                = 3;
    p.verbose              = getenv("AI_API_VERBOSE") != NULL;
    const char * err = NULL;
    ai_engine * e = ai_engine_create(&p, &err);
    if (!e) { fprintf(stderr, "create failed: %s\n", err ? err : "?"); return 3; }
    const int cap = ai_engine_max_tokens_per_round(e);
    int32_t * ot = (int32_t *) malloc(sizeof(int32_t) * (size_t) cap);
    int32_t * os = (int32_t *) malloc(sizeof(int32_t) * (size_t) cap);
    for (int r = 3; r < argc; ++r) {
        int n = 0;
        int32_t * prompt = read_i32(argv[r], &n);
        if (r > 3 && ai_engine_reset_stream(e, 0) != AI_ENGINE_OK) { printf("reset: %s\n", ai_engine_last_error(e)); return 1; }
        int32_t pend = -1;
        const double t0 = now_ms();
        if (ai_engine_prefill_chunk(e, 0, prompt, n, 1, &pend) != AI_ENGINE_OK) { printf("prefill: %s\n", ai_engine_last_error(e)); return 1; }
        const double t1 = now_ms();
        uint64_t h = 1469598103934665603ULL;
        h = (h ^ (uint64_t) (uint32_t) pend) * 1099511628211ULL;
        int got = 1, rounds = 0;
        int64_t pos = n;
        while (got < n_gen) {
            int32_t tok = pend, pp = (int32_t) pos, sid = 0;
            int out_n = 0;
            if (ai_engine_decode_mtp(e, &tok, &pp, &sid, 1, ot, os, cap, &out_n) != AI_ENGINE_OK) {
                printf("decode: %s\n", ai_engine_last_error(e)); return 1;
            }
            rounds++;
            for (int k = 0; k < out_n && got < n_gen; ++k) { h = (h ^ (uint64_t) (uint32_t) ot[k]) * 1099511628211ULL; got++; }
            pend = ot[out_n - 1];
            pos += out_n;
        }
        const double t2 = now_ms();
        const char * base = strrchr(argv[r], '/');
        printf("req %d %-14s prompt %5d | prefill %8.1f ms | decode %4d tok in %4d rounds, %7.1f ms | hash %016llx\n",
               r - 2, base ? base + 1 : argv[r], n, t1 - t0, got, rounds, t2 - t1, (unsigned long long) h);
        free(prompt);
    }
    free(ot); free(os);
    ai_engine_destroy(e);
    return 0;
}
