/* Acceptance test for DFlash2 through the C API (AI_DFLASH2=<drafter.gguf> at create).
 *
 *   AI_DFLASH2=<drafter.gguf> ai_api_df2_test <model.gguf> <prompt.i32> <cli.i32>
 *
 * prompt.i32: the prompt's token ids (alphainfer --dump-tokens); cli.i32: what the CLI generated for it with the same
 * drafter (alphainfer --tokens-in prompt.i32 --dflash2 <drafter.gguf> --gen-out cli.i32) at the same --ctx/--kv-type.
 *
 * Checks token identity with the CLI's --dflash2 run: the C API must configure the engine as the CLI does
 * (drafter, k from its block size, the fast verify policy) and its chunked prefill must equal a whole prefill.
 *   1. a 2-stream create with AI_DFLASH2 is refused (one drafter context), with a message.
 *   2. mtp_info: K == 7 (the drafter's block size - 1), exact == 0; max_tokens_per_round == 8.
 *   3. whole prefill + decode_mtp: tokens == the CLI's, for the CLI's whole length.
 *   4. chunked prefill (7-token pieces): out_first == -1 until is_last; then tokens == the CLI's.
 *   5. reset_stream abandons a buffered partial prompt: a fresh chunked prefill after it still == the CLI's.
 * Can't-fail row: the CLI reference must contain at least 2 distinct tokens, or identity proves nothing.
 */
#include "api/ai_engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int checks = 0, fails = 0;

static void ck(int cond, const char * what, const char * detail) {
    checks++;
    printf("  [%s] %s%s%s\n", cond ? "ok" : "FAIL", what, (!cond && detail) ? "  -- " : "", (!cond && detail) ? detail : "");
    if (!cond) fails++;
}

static int32_t * read_i32(const char * path, int * n) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    int32_t * v = (int32_t *) malloc((size_t) sz + 4);
    *n = (int) (fread(v, 4, (size_t) sz / 4, f));
    fclose(f);
    return v;
}

static ai_engine * create(const char * model, int n_streams, const char ** err) {
    ai_engine_params p;
    memset(&p, 0, sizeof p);
    p.model_path           = model;
    p.ctx_cells_per_stream = getenv("AI_API_CTX") ? atoi(getenv("AI_API_CTX")) : 4096;
    p.n_streams            = n_streams;
    p.kv_type_is_q4_0      = getenv("AI_API_KV") ? atoi(getenv("AI_API_KV")) : 0;
    p.mtp_k                = 3;   /* overridden by the drafter's block size */
    p.verbose              = getenv("AI_API_VERBOSE") != NULL;
    return ai_engine_create(&p, err);
}

/* decode stream 0 from its pending token until want tokens (first included); returns tokens collected or -1 */
static int decode_to(ai_engine * e, int32_t first, int plen, int32_t * got, int want) {
    const int cap = ai_engine_max_tokens_per_round(e);
    int32_t * ot = (int32_t *) malloc(sizeof(int32_t) * (size_t) cap);
    int32_t * os = (int32_t *) malloc(sizeof(int32_t) * (size_t) cap);
    int32_t pend = first, pos = plen, sid = 0;
    int n = 0;
    got[n++] = first;
    for (int round = 0; round < 4000 && n < want; ++round) {
        int out_n = -1;
        if (ai_engine_decode_mtp(e, &pend, &pos, &sid, 1, ot, os, cap, &out_n) != AI_ENGINE_OK) {
            printf("      round %d: %s\n", round, ai_engine_last_error(e));
            free(ot); free(os);
            return -1;
        }
        for (int k = 0; k < out_n; ++k) {
            if (os[k] != 0) { printf("      bad stream id %d\n", os[k]); free(ot); free(os); return -1; }
            if (n < want + 8) got[n++] = ot[k];
            pend = ot[k];
        }
        pos += out_n;
    }
    free(ot); free(os);
    return n;
}

static int first_diff(const int32_t * a, const int32_t * b, int n) {
    for (int i = 0; i < n; ++i) if (a[i] != b[i]) return i;
    return -1;
}

/* chunked prefill in pieces of `piece` tokens; returns the first token, or -2 on error */
static int32_t prefill_chunked(ai_engine * e, const int32_t * p, int n, int piece, int * bad_first) {
    int32_t first = -1;
    *bad_first = 0;
    for (int off = 0; off < n; off += piece) {
        const int t = n - off < piece ? n - off : piece;
        const int last = off + t == n;
        if (ai_engine_prefill_chunk(e, 0, p + off, t, last, &first) != AI_ENGINE_OK) {
            printf("      prefill_chunk at %d: %s\n", off, ai_engine_last_error(e));
            return -2;
        }
        if (!last && first != -1) *bad_first = 1;
    }
    return first;
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: AI_DFLASH2=<drafter.gguf> %s <model.gguf> <prompt.i32> <cli.i32>\n", argv[0]);
        return 2;
    }
    if (!getenv("AI_DFLASH2")) { fprintf(stderr, "set AI_DFLASH2 to the drafter gguf\n"); return 2; }
    const char * model = argv[1];
    int pn, cn;
    int32_t * pr  = read_i32(argv[2], &pn);
    int32_t * cli = read_i32(argv[3], &cn);
    char d[200];
    int distinct = 0;
    for (int i = 1; i < cn; ++i) if (cli[i] != cli[0]) { distinct = 1; break; }
    ck(cn >= 8 && distinct, "can't-fail row: the CLI reference has >= 8 tokens, not all equal", NULL);

    printf("== 1. two streams with AI_DFLASH2 are refused\n");
    {
        const char * err = NULL;
        ai_engine * e = create(model, 2, &err);
        ck(e == NULL && err && strstr(err, "AI_DFLASH2"), "create(n_streams 2) -> NULL with an AI_DFLASH2 message", err);
        if (e) ai_engine_destroy(e);
    }
    const char * err = NULL;
    ai_engine * e = create(model, 1, &err);
    if (!e) { printf("  [FAIL] create(n_streams 1): %s\n", err ? err : "?"); return 1; }

    printf("== 2. mtp_info\n");
    {
        int k = -1, exact = -1;
        ai_engine_mtp_info(e, &k, &exact);
        snprintf(d, sizeof d, "mtp_k %d exact %d", k, exact);
        ck(k == 7 && exact == 0, "K == 7 from the drafter, FAST verify", d);
        ck(ai_engine_max_tokens_per_round(e) == 8, "max_tokens_per_round == 8", NULL);
    }
    int32_t * got = (int32_t *) calloc((size_t) cn + 16, 4);

    printf("== 3. whole prefill\n");
    {
        int32_t first = -1;
        ck(ai_engine_prefill(e, 0, pr, pn, &first) == AI_ENGINE_OK, "prefill", ai_engine_last_error(e));
        const int n = decode_to(e, first, pn, got, cn);
        const int fd = n >= cn ? first_diff(got, cli, cn) : -2;
        snprintf(d, sizeof d, "first difference at token %d of %d (collected %d)", fd, cn, n);
        ck(fd == -1, "whole prefill: C API tokens == CLI --dflash2 tokens", d);
    }
    printf("== 4. chunked prefill, 7-token pieces\n");
    {
        int bad = 0;
        const int32_t first = prefill_chunked(e, pr, pn, 7, &bad);
        ck(first >= 0 && !bad, "out_first == -1 until is_last, then a token", NULL);
        memset(got, 0, ((size_t) cn + 16) * 4);
        const int n = first >= 0 ? decode_to(e, first, pn, got, cn) : -1;
        const int fd = n >= cn ? first_diff(got, cli, cn) : -2;
        snprintf(d, sizeof d, "first difference at token %d of %d", fd, cn);
        ck(fd == -1, "chunked prefill: tokens == CLI", d);
    }
    printf("== 5. reset_stream abandons a buffered partial prompt\n");
    {
        int32_t f = 0;
        ck(ai_engine_prefill_chunk(e, 0, pr, pn > 5 ? 5 : pn, 0, &f) == AI_ENGINE_OK && f == -1, "partial piece buffered", NULL);
        ck(ai_engine_reset_stream(e, 0) == AI_ENGINE_OK, "reset_stream", ai_engine_last_error(e));
        int bad = 0;
        const int32_t first = prefill_chunked(e, pr, pn, 11, &bad);
        memset(got, 0, ((size_t) cn + 16) * 4);
        const int n = first >= 0 ? decode_to(e, first, pn, got, cn) : -1;
        const int fd = n >= cn ? first_diff(got, cli, cn) : -2;
        snprintf(d, sizeof d, "first difference at token %d of %d", fd, cn);
        ck(fd == -1, "after reset + a fresh chunked prefill: tokens == CLI", d);
    }
    ai_engine_destroy(e);
    free(got);
    printf("%s: %d/%d checks passed\n", fails ? "FAILED" : "PASSED", checks - fails, checks);
    return fails ? 1 : 0;
}
