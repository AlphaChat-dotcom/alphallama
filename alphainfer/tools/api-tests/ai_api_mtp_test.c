/* Acceptance gate for MTP through the C API: ai_engine_decode_mtp, ai_engine_max_tokens_per_round,
 * ai_engine_mtp_info, and the shared verify policy (engine/mtp_policy.cpp) applied by ai_engine_create.
 *
 *   ai_api_mtp_test <model.gguf> <promptA.i32> <cliA.i32> [<promptB.i32> <cliB.i32>]
 *
 * promptX.i32: the prompt's token ids (alphainfer --dump-tokens); cliX.i32: what the CLI generated for it with the
 * same MTP settings (alphainfer --tokens-in promptX.i32 --mtp 3 --gen-out cliX.i32). Env AI_MTP_MODE applies to both.
 *
 * The point is token identity with the CLI. The C API drives the same Engine::decode_round through decode_step,
 * so any difference means the C path configured the engine differently or attributes tokens wrongly. rc == 0 alone would prove neither.
 *
 *   1. mtp_info: the effective K and the exact flag match the policy (MoE default: K <= 3, exact).
 *   2. capacity below the bound -> EINVAL before the round: the stream's position must not move.
 *   3. ai_engine_decode on an MTP engine -> ENOTSUP (its signature cannot bound several tokens per stream).
 *   4. one stream: the tokens equal the CLI's, for the CLI's whole length.
 *   5. two streams, prompts A and B (if given): each stream equals the CLI's own 2-stream batch of A and B
 *      (--prompt-list), not a single-stream run: exact mode is MTP == plain at the same batch size, not batch
 *      invariance. The stream ids returned per token are what make that checkable. Can't-fail row: A's and B's
 *      CLI outputs must differ, or a swapped attribution would pass.
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

static ai_engine * create(const char * model, int n_streams, int mtp_k) {
    ai_engine_params p;
    memset(&p, 0, sizeof p);
    p.model_path           = model;
    /* AI_API_CTX / AI_API_KV: run at another config, e.g. a large context and q4_0 (1) -- the CLI
     * reference must then use the same --ctx/--kv-type (and AI_F_MIN when the engine has 2 slots) */
    p.ctx_cells_per_stream = getenv("AI_API_CTX") ? atoi(getenv("AI_API_CTX")) : 4096;
    p.n_streams            = n_streams;
    p.kv_type_is_q4_0      = getenv("AI_API_KV") ? atoi(getenv("AI_API_KV")) : 0;   /* 0 = f16, the CLI's default here */
    p.mtp_k                = mtp_k;
    p.verbose              = getenv("AI_API_VERBOSE") != NULL;
    const char * err = NULL;
    ai_engine * e = ai_engine_create(&p, &err);
    if (!e) { fprintf(stderr, "create failed: %s\n", err ? err : "?"); exit(3); }
    return e;
}

/* Run streams 0..ns-1 from their prompts until each has at least want[s] tokens (first token included).
 * got[s][..] receives each stream's tokens in order; returns 0 on success. */
static int run(ai_engine * e, int ns, int32_t ** prompts, const int * plen, int32_t ** got, const int * want) {
    int32_t pend[2];
    int64_t pos[2];
    int     n[2] = {0, 0};
    for (int s = 0; s < ns; ++s) {
        if (ai_engine_prefill(e, s, prompts[s], plen[s], &pend[s]) != AI_ENGINE_OK) {
            printf("      prefill %d: %s\n", s, ai_engine_last_error(e));
            return 1;
        }
        pos[s] = plen[s];
        got[s][n[s]++] = pend[s];
    }
    const int cap = ai_engine_max_tokens_per_round(e);
    int32_t * ot = (int32_t *) malloc(sizeof(int32_t) * (size_t) cap);
    int32_t * os = (int32_t *) malloc(sizeof(int32_t) * (size_t) cap);
    int open[2] = {1, 1};
    for (int round = 0; round < 4000; ++round) {
        /* a stream that has its tokens ends, as a server ends a request: reset its slot and stop naming it. The CLI
         * reference marks such a stream done the same way, so the remaining stream's rounds compact to one wide
         * in both runs. */
        int done = 1, nn = 0;
        int32_t tok[2], pp[2], sid[2];
        for (int s = 0; s < ns; ++s) {
            if (open[s] && n[s] >= want[s]) {
                if (ns > 1 && ai_engine_reset_stream(e, s) != AI_ENGINE_OK) {
                    printf("      reset %d: %s\n", s, ai_engine_last_error(e));
                    free(ot); free(os);
                    return 1;
                }
                open[s] = 0;
            }
            if (!open[s]) continue;
            done = 0;
            tok[nn] = pend[s]; pp[nn] = (int32_t) pos[s]; sid[nn] = s; ++nn;
        }
        if (done) break;
        int out_n = -1;
        const int rc = ai_engine_decode_mtp(e, tok, pp, sid, nn, ot, os, cap, &out_n);
        if (rc != AI_ENGINE_OK) {
            printf("      round %d rc=%d err=%s\n", round, rc, ai_engine_last_error(e));
            free(ot); free(os);
            return 1;
        }
        int per[2] = {0, 0};
        for (int k = 0; k < out_n; ++k) {
            const int s = os[k];
            if (s < 0 || s >= ns) { printf("      bad stream id %d\n", s); free(ot); free(os); return 1; }
            if (n[s] < want[s] + 8) got[s][n[s]++] = ot[k];
            pend[s] = ot[k];   /* the last token a stream emits is its new pending token */
            per[s]++;
        }
        for (int s = 0; s < ns; ++s) pos[s] += per[s];
    }
    free(ot); free(os);
    return 0;
}

static int first_diff(const int32_t * a, const int32_t * b, int n) {
    for (int i = 0; i < n; ++i) if (a[i] != b[i]) return i;
    return -1;
}

int main(int argc, char ** argv) {
    if (argc != 4 && argc != 6) {
        fprintf(stderr, "usage: %s <model.gguf> <promptA.i32> <cliA.i32> [<promptB.i32> <cliB.i32>]\n", argv[0]);
        return 2;
    }
    const char * model = argv[1];
    int pa_n, ca_n, pb_n = 0, cb_n = 0;
    int32_t * pa = read_i32(argv[2], &pa_n);
    int32_t * ca = read_i32(argv[3], &ca_n);
    int32_t * pb = NULL, * cb = NULL;
    if (argc == 6) { pb = read_i32(argv[4], &pb_n); cb = read_i32(argv[5], &cb_n); }
    char d[160];

    /* AI_API_TWO_ONLY=1: only the two-stream block. Exact mode promises MTP == plain decode at the same batch size,
     * not batch invariance, so the two-stream references must come from the CLI running the same two prompts as
     * one 2-stream batch (--prompt-list), and the single-stream references from single-stream CLI runs. */
    const int two_only = getenv("AI_API_TWO_ONLY") != NULL && atoi(getenv("AI_API_TWO_ONLY")) != 0;
    if (!two_only) {
    printf("== one stream, mtp_k 3\n");
    {
        ai_engine * e = create(model, 1, 3);
        int k = -1, exact = -1;
        ai_engine_mtp_info(e, &k, &exact);
        snprintf(d, sizeof d, "mtp_k %d exact %d (AI_MTP_MODE=%s)", k, exact, getenv("AI_MTP_MODE") ? getenv("AI_MTP_MODE") : "default");
        ck(k == 3, "mtp_info: effective K is 3", d);
        printf("      %s\n", d);
        ck(ai_engine_max_tokens_per_round(e) == 4, "max_tokens_per_round == n_streams x (K + 1) = 4", NULL);

        int32_t first;
        ck(ai_engine_prefill(e, 0, pa, pa_n, &first) == AI_ENGINE_OK, "prefill A", ai_engine_last_error(e));
        int32_t tok = first, pp = pa_n, sid = 0, ot[8], os[8];
        int out_n = -1;
        int rc = ai_engine_decode_mtp(e, &tok, &pp, &sid, 1, ot, os, 3, &out_n);
        ck(rc == AI_ENGINE_EINVAL && out_n == 0, "capacity 3 < bound 4 -> EINVAL, nothing written", ai_engine_last_error(e));
        int32_t samp[4];
        int ns = -1;
        rc = ai_engine_decode(e, &tok, &pp, &sid, 1, NULL, samp, &ns);
        ck(rc == AI_ENGINE_ENOTSUP, "ai_engine_decode on an MTP engine -> ENOTSUP", ai_engine_last_error(e));
        /* the refused calls must not have advanced the stream: a correct position still decodes */
        rc = ai_engine_decode_mtp(e, &tok, &pp, &sid, 1, ot, os, 8, &out_n);
        ck(rc == AI_ENGINE_OK && out_n >= 1, "position unchanged by the refused calls (next round accepted)", ai_engine_last_error(e));
        ai_engine_destroy(e);
    }
    {
        ai_engine * e = create(model, 1, 3);
        int32_t * gA = (int32_t *) calloc((size_t) ca_n + 16, 4);
        int32_t * prompts[1] = { pa };
        int32_t * got[1] = { gA };
        int plen[1] = { pa_n }, want[1] = { ca_n };
        ck(run(e, 1, prompts, plen, got, want) == 0, "stream 0 ran to the CLI's length", NULL);
        const int fd = first_diff(gA, ca, ca_n);
        snprintf(d, sizeof d, "first difference at token %d of %d", fd, ca_n);
        ck(fd < 0, "one stream: C API tokens == CLI tokens", d);
        printf("      %d tokens compared\n", ca_n);
        free(gA);
        ai_engine_destroy(e);
    }
    {
        /* The common serving case: a 2-slot engine serving one request, on slot 1 (so the compacted round must use
         * region 1, not 0). The idle slot is never named; the tokens must equal the single-stream CLI's. */
        printf("== one request on a 2-slot engine (slot 1), mtp_k 3\n");
        ai_engine * e = create(model, 2, 3);
        int32_t * gA = (int32_t *) calloc((size_t) ca_n + 16, 4);
        int32_t pend;
        ck(ai_engine_prefill(e, 1, pa, pa_n, &pend) == AI_ENGINE_OK, "prefill slot 1 only", ai_engine_last_error(e));
        int64_t pos = pa_n;
        int n = 0;
        gA[n++] = pend;
        const int cap = ai_engine_max_tokens_per_round(e);
        int32_t * ot = (int32_t *) malloc(sizeof(int32_t) * (size_t) cap);
        int32_t * os = (int32_t *) malloc(sizeof(int32_t) * (size_t) cap);
        int ok = 1;
        while (n < ca_n && ok) {
            int32_t tok = pend, pp = (int32_t) pos, sid = 1;
            int out_n = -1;
            if (ai_engine_decode_mtp(e, &tok, &pp, &sid, 1, ot, os, cap, &out_n) != AI_ENGINE_OK) {
                printf("      %s\n", ai_engine_last_error(e));
                ok = 0;
                break;
            }
            for (int k = 0; k < out_n; ++k) {
                if (os[k] != 1) { printf("      token attributed to stream %d\n", os[k]); ok = 0; }
                if (n < ca_n + 8) gA[n++] = ot[k];
                pend = ot[k];
            }
            pos += out_n;
        }
        ck(ok, "decoded with the idle slot unnamed", NULL);
        const int fd = first_diff(gA, ca, ca_n);
        snprintf(d, sizeof d, "first difference at token %d of %d", fd, ca_n);
        ck(fd < 0, "one request on slot 1 of 2: tokens == single-stream CLI", d);
        int32_t t0 = 0, p0 = 0, s0 = 0;
        int on = -1;
        ck(ai_engine_decode_mtp(e, &t0, &p0, &s0, 1, ot, os, cap, &on) != AI_ENGINE_OK && on == 0,
           "naming the idle slot is refused, nothing emitted", ai_engine_last_error(e));
        free(ot); free(os); free(gA);
        ai_engine_destroy(e);
    }
    }   /* !two_only */
    if (pb) {
        printf("== two streams, mtp_k 3: A on stream 0, B on stream 1\n");
        const int m = ca_n < cb_n ? ca_n : cb_n;
        ck(first_diff(ca, cb, m) >= 0, "can't-fail row: CLI outputs for A and B differ", NULL);
        ai_engine * e = create(model, 2, 3);
        int32_t * gA = (int32_t *) calloc((size_t) ca_n + 16, 4);
        int32_t * gB = (int32_t *) calloc((size_t) cb_n + 16, 4);
        int32_t * prompts[2] = { pa, pb };
        int32_t * got[2] = { gA, gB };
        int plen[2] = { pa_n, pb_n }, want[2] = { ca_n, cb_n };
        ck(run(e, 2, prompts, plen, got, want) == 0, "both streams ran", NULL);
        int fa = first_diff(gA, ca, ca_n), fb = first_diff(gB, cb, cb_n);
        snprintf(d, sizeof d, "A first diff %d / %d, B first diff %d / %d", fa, ca_n, fb, cb_n);
        ck(fa < 0 && fb < 0, "two streams: each stream == the CLI's 2-stream batch for that stream", d);
        free(gA); free(gB);
        ai_engine_destroy(e);
    }
    printf("%d/%d checks passed\n", checks - fails, checks);
    return fails ? 1 : 0;
}
