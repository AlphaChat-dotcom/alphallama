/* Test for ai_engine_prefill_chunk (AI_PREFILL_CHUNK): a prompt fed in pieces, with another stream decoding between
 * the pieces, must produce exactly the tokens of a one-shot prefill. References are the single-stream CLI's tokens
 * (--dump-tokens / --gen-out), as for ai_api_mtp_test; run the CLI at the same --ctx/--kv-type/--chunk 1024 and
 * AI_F_MIN.
 *   usage: ai_api_chunk_test <model.gguf> <promptA.i32> <cliA.i32> <promptB.i32> <cliB.i32>
 *   A must be long (several pieces, e.g. a 3k+ token document) and B short (e.g. one q20 question, 50 tokens).
 *   1. solo: A on slot 0 fed in uneven pieces (700, then 1500s, plus a 0-token call), decoded alone == cliA
 *   2. interleaved, the server pattern: B prefilled one-shot on slot 1 and decoded to completion in rounds run
 *      between A's pieces (slot 0 mid-prefill throughout), then A's last piece, then A decoded alone.
 *      B == cliB (its rounds compact to slot 1 alone) and A == cliA.
 *   3. contract: out_first == -1 for every non-last piece. */
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

static ai_engine * create(const char * model) {
    ai_engine_params p;
    memset(&p, 0, sizeof p);
    p.model_path           = model;
    p.ctx_cells_per_stream = getenv("AI_API_CTX") ? atoi(getenv("AI_API_CTX")) : 4096;
    p.n_streams            = 2;
    p.kv_type_is_q4_0      = getenv("AI_API_KV") ? atoi(getenv("AI_API_KV")) : 0;
    p.mtp_k                = 3;
    p.verbose              = getenv("AI_API_VERBOSE") != NULL;
    const char * err = NULL;
    ai_engine * e = ai_engine_create(&p, &err);
    if (!e) { fprintf(stderr, "create failed: %s\n", err ? err : "?"); exit(3); }
    return e;
}

static int first_diff(const int32_t * a, const int32_t * b, int n) {
    for (int i = 0; i < n; ++i) if (a[i] != b[i]) return i;
    return -1;
}

/* One MTP round over the streams in live[]; appends each stream's tokens to got[s] (capped at want[s]) and updates
 * pend/pos. Returns 0 on success. */
static int round_once(ai_engine * e, const int * live, int32_t * pend, int64_t * pos, int32_t ** got, int * n,
                      const int * want, int32_t * ot, int32_t * os, int cap) {
    int32_t tok[2], pp[2], sid[2];
    int nn = 0;
    for (int s = 0; s < 2; ++s) {
        if (!live[s]) continue;
        tok[nn] = pend[s]; pp[nn] = (int32_t) pos[s]; sid[nn] = s; ++nn;
    }
    if (nn == 0) return 0;
    int out_n = -1;
    if (ai_engine_decode_mtp(e, tok, pp, sid, nn, ot, os, cap, &out_n) != AI_ENGINE_OK) {
        printf("      decode_mtp: %s\n", ai_engine_last_error(e));
        return 1;
    }
    for (int k = 0; k < out_n; ++k) {
        const int s = os[k];
        if (n[s] < want[s]) got[s][n[s]] = ot[k];
        n[s]++;
        pend[s] = ot[k];
        pos[s] += 1;
    }
    return 0;
}

/* feed prompt p (np tokens) to slot `s` in pieces: 700, then 1500s, with a 0-token call after the first piece.
 * Between pieces, call between(ctx). Returns the first token. */
typedef void (*between_fn)(void * ctx);
static int feed(ai_engine * e, int s, const int32_t * p, int np, between_fn between, void * ctx, int * nonlast_ok,
                int32_t * first) {
    int off = 0, piece = 0;
    *nonlast_ok = 1;
    while (1) {
        const int want = piece == 0 ? 700 : 1500;
        int len = np - off < want ? np - off : want;
        const int last = off + len == np;
        int32_t of = 12345;
        if (ai_engine_prefill_chunk(e, s, p + off, len, last, &of) != AI_ENGINE_OK) {
            printf("      prefill_chunk: %s\n", ai_engine_last_error(e));
            return 1;
        }
        off += len;
        if (last) { *first = of; return 0; }
        if (of != -1) *nonlast_ok = 0;
        if (piece == 0) {   /* a 0-token call mid-prompt must be accepted and change nothing */
            int32_t z = 12345;
            if (ai_engine_prefill_chunk(e, s, p + off, 0, 0, &z) != AI_ENGINE_OK || z != -1) *nonlast_ok = 0;
        }
        if (between) between(ctx);
        ++piece;
    }
}

struct bctx {
    ai_engine * e; int live[2]; int32_t pend[2]; int64_t pos[2]; int32_t * got[2]; int n[2]; int want[2];
    int32_t * ot; int32_t * os; int cap; int err; int rounds_between;
};

static void decode_b_some(void * vp) {
    struct bctx * c = (struct bctx *) vp;
    for (int r = 0; r < 25 && c->live[1] && !c->err; ++r) {   /* up to 25 rounds of B per gap */
        int live[2] = { 0, 1 };
        if (round_once(c->e, live, c->pend, c->pos, c->got, c->n, c->want, c->ot, c->os, c->cap)) { c->err = 1; return; }
        c->rounds_between++;
        if (c->n[1] >= c->want[1]) {
            if (ai_engine_reset_stream(c->e, 1) != AI_ENGINE_OK) { c->err = 1; return; }
            c->live[1] = 0;
        }
    }
}

int main(int argc, char ** argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s <model.gguf> <promptA.i32> <cliA.i32> <promptB.i32> <cliB.i32>\n", argv[0]);
        return 2;
    }
    int pa_n, ca_n, pb_n, cb_n;
    int32_t * pa = read_i32(argv[2], &pa_n);
    int32_t * ca = read_i32(argv[3], &ca_n);
    int32_t * pb = read_i32(argv[4], &pb_n);
    int32_t * cb = read_i32(argv[5], &cb_n);
    char d[200];
    ai_engine * e = create(argv[1]);
    const int cap = ai_engine_max_tokens_per_round(e);
    int32_t * ot = (int32_t *) malloc(sizeof(int32_t) * (size_t) cap);
    int32_t * os = (int32_t *) malloc(sizeof(int32_t) * (size_t) cap);
    printf("prompt A %d tokens (%d pieces), prompt B %d tokens\n", pa_n, 1 + (pa_n > 700 ? (pa_n - 700 + 1499) / 1500 : 0), pb_n);

    printf("== 1. solo: A chunked on slot 0, decoded alone\n");
    {
        struct bctx c; memset(&c, 0, sizeof c);
        c.got[0] = (int32_t *) calloc((size_t) ca_n + 16, 4); c.want[0] = ca_n;
        int nl = 0; int32_t first = -2;
        const int rc = feed(e, 0, pa, pa_n, NULL, NULL, &nl, &first);
        ck(rc == 0, "chunked prefill of A ran", ai_engine_last_error(e));
        ck(nl, "out_first == -1 on every non-last piece (and a 0-token piece is accepted)", NULL);
        c.pend[0] = first; c.pos[0] = pa_n; c.got[0][c.n[0]++] = first; c.live[0] = 1;
        while (c.n[0] < c.want[0]) {
            if (round_once(e, c.live, c.pend, c.pos, c.got, c.n, c.want, ot, os, cap)) break;
        }
        const int f = first_diff(c.got[0], ca, ca_n);
        snprintf(d, sizeof d, "first difference at token %d of %d", f, ca_n);
        ck(f < 0, "solo: chunked A == one-shot CLI tokens", d);
        ai_engine_reset_stream(e, 0);
        free(c.got[0]);
    }

    printf("== 2. interleaved: B decodes on slot 1 between A's pieces on slot 0\n");
    {
        struct bctx c; memset(&c, 0, sizeof c);
        c.e = e; c.ot = ot; c.os = os; c.cap = cap;
        c.got[0] = (int32_t *) calloc((size_t) ca_n + 16, 4); c.want[0] = ca_n;
        c.got[1] = (int32_t *) calloc((size_t) cb_n + 16, 4); c.want[1] = cb_n;
        int32_t fb = -2;
        ck(ai_engine_prefill(e, 1, pb, pb_n, &fb) == AI_ENGINE_OK, "B prefilled one-shot on slot 1", ai_engine_last_error(e));
        c.pend[1] = fb; c.pos[1] = pb_n; c.got[1][c.n[1]++] = fb; c.live[1] = 1;
        int nl = 0; int32_t first = -2;
        const int rc = feed(e, 0, pa, pa_n, decode_b_some, &c, &nl, &first);
        ck(rc == 0 && !c.err, "A fed in pieces with B's rounds in between", ai_engine_last_error(e));
        snprintf(d, sizeof d, "%d B rounds ran between A's pieces", c.rounds_between);
        ck(c.rounds_between > 0, "B really decoded while A was mid-prefill", d);
        ck(nl, "out_first == -1 on every non-last piece", NULL);
        /* B must finish while A is mid-prefill: once A is live, a B-only round is refused (every open stream must be
         * named) and a 2-wide round is not the single-stream reference. A test-design failure, reported as such. */
        ck(!c.live[1], "B finished before A's last piece (test design: use a longer A or a shorter B otherwise)", NULL);
        const int fbd = first_diff(c.got[1], cb, cb_n);
        snprintf(d, sizeof d, "first difference at token %d of %d", fbd, cb_n);
        ck(fbd < 0, "B (decoded between A's pieces) == one-shot CLI tokens", d);
        c.pend[0] = first; c.pos[0] = pa_n; c.got[0][c.n[0]++] = first; c.live[0] = 1;
        while (c.n[0] < c.want[0] && !c.err) {
            if (round_once(e, c.live, c.pend, c.pos, c.got, c.n, c.want, ot, os, cap)) break;
        }
        const int fa = first_diff(c.got[0], ca, ca_n);
        snprintf(d, sizeof d, "first difference at token %d of %d", fa, ca_n);
        ck(fa < 0, "A (prefilled in pieces around B's rounds) == one-shot CLI tokens", d);
        free(c.got[0]); free(c.got[1]);
    }
    printf("%d/%d checks passed\n", checks - fails, checks);
    ai_engine_destroy(e);
    return fails ? 1 : 0;
}
