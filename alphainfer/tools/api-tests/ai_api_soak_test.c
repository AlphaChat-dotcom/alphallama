/* Soak test for a 2-slot MTP server on the C API (the serving path: AlphaServe -> ai_engine_*).
 *
 *   ai_api_soak_test <model.gguf> <tokens_a.i32> [<tokens_b.i32> ...]
 *
 * tokens_X.i32: long pre-tokenized real text (alphainfer --dump-tokens). Each request is a slice of one of them at a
 * random offset, so prompts are real text of random length. Env: AI_SOAK_N (requests, default 160), AI_SOAK_CTX (cells
 * per slot, default 49152), AI_SOAK_SEED (default 1).
 *
 * Pass 1 (serve): two clients issue requests back to back with random idle gaps, as a server would see them. Prompt
 * lengths: 50% 64-512, 35% 2k-8k, 15% 20k-44k tokens; generation budgets 16-256 tokens. A request ends when it has its
 * budget (mid-round, like an EOG): the extra tokens are dropped and the slot is reset. Idle slots are never named.
 * Pass 2 (oracle): every request that ran alone (no round of it was shared with the other slot) is replayed on a fresh
 * 1-stream engine and must match token for token. That is what catches state leaking through slot reuse, round
 * compaction, the gate's carried verify width, or calibration -- exact mode makes it a bitwise property.
 * Shared requests are batch-variant by design (plain decode is too), so they are checked for completion only.
 * Free VRAM is sampled through the run and must not fall.
 */
#include "api/ai_engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

typedef struct {
    int src, off, plen, budget;   /* prompt = src tokens [off, off + plen) */
    int shared;                   /* 1 if any decode round of it had the other slot live */
    int done, n_out;
    int32_t * out;
} Req;

static uint32_t rng = 1;
static uint32_t rnd(void) { rng = rng * 1664525u + 1013904223u; return rng >> 8; }
static int rrange(int lo, int hi) { return lo + (int) (rnd() % (uint32_t) (hi - lo + 1)); }

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

static ai_engine * create(const char * model, int ns, int ctx) {
    ai_engine_params p;
    memset(&p, 0, sizeof p);
    p.model_path           = model;
    p.ctx_cells_per_stream = ctx;
    p.n_streams            = ns;
    /* Both engines must hold the same KV type, or the oracle compares two precisions: with auto, the 2-slot server
     * engine can pick q4_0 at 2 x 49152 (q8_0 does not fit) while the 1-stream reference picks q8_0 when its
     * calibrated workspace leaves room. Default q4_0, the auto choice at this config. AI_SOAK_KV overrides
     * (-1 auto, 0 f16, 1 q4_0, 2 q8_0). */
    p.kv_type_is_q4_0      = getenv("AI_SOAK_KV") ? atoi(getenv("AI_SOAK_KV")) : 1;
    p.mtp_k                = 3;
    p.verbose              = getenv("AI_SOAK_VERBOSE") != NULL;
    const char * err = NULL;
    ai_engine * e = ai_engine_create(&p, &err);
    if (!e) { fprintf(stderr, "create failed: %s\n", err ? err : "?"); exit(3); }
    return e;
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <model.gguf> <tokens.i32>...\n", argv[0]); return 2; }
    const char * model = argv[1];
    const int nsrc = argc - 2;
    int32_t ** src = (int32_t **) calloc((size_t) nsrc, sizeof(int32_t *));
    int * srcn = (int *) calloc((size_t) nsrc, sizeof(int));
    for (int i = 0; i < nsrc; ++i) src[i] = read_i32(argv[2 + i], &srcn[i]);
    const int N   = getenv("AI_SOAK_N") ? atoi(getenv("AI_SOAK_N")) : 160;
    const int CTX = getenv("AI_SOAK_CTX") ? atoi(getenv("AI_SOAK_CTX")) : 49152;
    rng = getenv("AI_SOAK_SEED") ? (uint32_t) atoi(getenv("AI_SOAK_SEED")) : 1u;

    Req * rq = (Req *) calloc((size_t) N, sizeof(Req));
    for (int i = 0; i < N; ++i) {
        const int u = rrange(0, 99);
        int plen = u < 50 ? rrange(64, 512) : u < 85 ? rrange(2048, 8192) : rrange(20480, 45056);
        const int budget = rrange(16, 256);
        if (plen + budget + 16 > CTX) plen = CTX - budget - 16;
        if (getenv("AI_SOAK_MAXP") && plen > atoi(getenv("AI_SOAK_MAXP"))) plen = atoi(getenv("AI_SOAK_MAXP"));   /* diagnostic */
        int s = rrange(0, nsrc - 1);
        if (plen > srcn[s] - 1) plen = srcn[s] - 1;
        rq[i].src = s; rq[i].plen = plen; rq[i].budget = budget;
        rq[i].off = rrange(0, srcn[s] - plen);
        rq[i].out = (int32_t *) malloc(sizeof(int32_t) * (size_t) (budget + 8));
    }

    /* pass 1: serve */
    printf("== pass 1: %d requests on a 2-slot engine, ctx %d per slot, MTP3 (mode %s)\n", N, CTX,
           getenv("AI_MTP_MODE") ? getenv("AI_MTP_MODE") : "default");
    ai_engine * e = create(model, 2, CTX);
    int k = -1, exact = -1;
    ai_engine_mtp_info(e, &k, &exact);
    printf("   mtp_k %d exact %d\n", k, exact);
    const int cap = ai_engine_max_tokens_per_round(e);
    int32_t * ot = (int32_t *) malloc(sizeof(int32_t) * (size_t) cap);
    int32_t * os = (int32_t *) malloc(sizeof(int32_t) * (size_t) cap);
    int active[2] = {-1, -1}, idle[2] = {0, 0};
    int32_t pend[2];
    int64_t pos[2];
    int next = 0, finished = 0, rounds = 0, shared_rounds = 0, errors = 0;
    long long emitted = 0;
    double t_dec = 0, t_pre = 0;
    int64_t vram0 = -1, vram_min = 1 << 30, vram_last = -1;
    while (finished < N && errors == 0) {
        for (int s = 0; s < 2; ++s) {
            if (active[s] >= 0) continue;
            if (idle[s] > 0) { idle[s]--; continue; }
            if (next >= N) continue;
            Req * r = &rq[next];
            int32_t first;
            const double tp = now_s();
            const int prc = ai_engine_prefill(e, s, src[r->src] + r->off, r->plen, &first);
            t_pre += now_s() - tp;
            if (prc != AI_ENGINE_OK) {
                printf("   [FAIL] prefill req %d slot %d: %s\n", next, s, ai_engine_last_error(e));
                errors++;
                break;
            }
            r->out[r->n_out++] = first;
            pend[s] = first;
            pos[s] = r->plen;
            active[s] = next++;
        }
        if (errors) break;
        int32_t tok[2], pp[2], sid[2];
        int nn = 0;
        for (int s = 0; s < 2; ++s) {
            if (active[s] < 0) continue;
            tok[nn] = pend[s]; pp[nn] = (int32_t) pos[s]; sid[nn] = s; ++nn;
        }
        if (nn == 0) continue;   /* both idle this tick; the countdowns move on */
        if (nn == 2) { rq[active[0]].shared = 1; rq[active[1]].shared = 1; shared_rounds++; }
        int out_n = -1;
        const double td = now_s();
        const int drc = ai_engine_decode_mtp(e, tok, pp, sid, nn, ot, os, cap, &out_n);
        t_dec += now_s() - td;
        emitted += out_n > 0 ? out_n : 0;
        if (drc != AI_ENGINE_OK) {
            printf("   [FAIL] decode round %d: %s\n", rounds, ai_engine_last_error(e));
            errors++;
            break;
        }
        rounds++;
        int per[2] = {0, 0};
        for (int i = 0; i < out_n; ++i) {
            const int s = os[i];
            if (s < 0 || s > 1 || active[s] < 0) { printf("   [FAIL] token for idle/invalid slot %d\n", s); errors++; break; }
            Req * r = &rq[active[s]];
            if (r->n_out < r->budget) r->out[r->n_out++] = ot[i];   /* tokens past the budget are dropped, as on EOG */
            pend[s] = ot[i];
            per[s]++;
        }
        for (int s = 0; s < 2; ++s) {
            if (active[s] < 0) continue;
            pos[s] += per[s];
            Req * r = &rq[active[s]];
            if (r->n_out >= r->budget) {
                if (ai_engine_reset_stream(e, s) != AI_ENGINE_OK) {
                    printf("   [FAIL] reset slot %d: %s\n", s, ai_engine_last_error(e));
                    errors++;
                }
                r->done = 1;
                finished++;
                active[s] = -1;
                /* a server's clients come and go: mostly back to back, sometimes a long pause, so some requests
                 * run alone (the oracle's population) and the 2-wide <-> 1-wide transitions are exercised */
                idle[s] = rrange(0, 99) < 60 ? rrange(0, 6) : rrange(200, 900);
                if (finished % 20 == 0) {
                    vram_last = ai_engine_free_vram_mib(e);
                    if (vram0 < 0) vram0 = vram_last;
                    if (vram_last < vram_min) vram_min = vram_last;
                    printf("   %3d/%d done, %d rounds (%d shared), free VRAM %lld MiB\n", finished, N, rounds,
                           shared_rounds, (long long) vram_last);
                    fflush(stdout);
                }
            }
        }
    }
    free(ot); free(os);
    ai_engine_destroy(e);
    int alone = 0, long_alone = 0;
    for (int i = 0; i < N; ++i) if (rq[i].done && !rq[i].shared) { alone++; long_alone += rq[i].plen >= 20480; }
    printf("   pass 1: %d/%d finished, %d errors, %d rounds (%d shared), %d requests ran alone (%d long)\n", finished, N,
           errors, rounds, shared_rounds, alone, long_alone);
    printf("   decode: %lld tokens in %.1f s = %.1f tok/s aggregate (%.2f ms/round); prefill %.1f s\n", emitted, t_dec,
           t_dec > 0 ? emitted / t_dec : 0.0, rounds ? 1000.0 * t_dec / rounds : 0.0, t_pre);
    printf("   free VRAM: first sample %lld, minimum %lld, last %lld MiB\n", (long long) vram0, (long long) vram_min,
           (long long) vram_last);

    /* pass 2: oracle */
    printf("== pass 2: replay every alone request on a FRESH 1-stream engine\n");
    e = create(model, 1, CTX);
    const int cap1 = ai_engine_max_tokens_per_round(e);
    ot = (int32_t *) malloc(sizeof(int32_t) * (size_t) cap1);
    os = (int32_t *) malloc(sizeof(int32_t) * (size_t) cap1);
    int32_t * ref = (int32_t *) malloc(sizeof(int32_t) * 300);
    int checked = 0, matched = 0;
    for (int i = 0; i < N && errors == 0; ++i) {
        Req * r = &rq[i];
        if (!r->done || r->shared) continue;
        int nref = 0;
        int32_t p0;
        if (ai_engine_reset_stream(e, 0) != AI_ENGINE_OK && checked > 0) { /* first call: no batch open yet */ }
        if (ai_engine_prefill(e, 0, src[r->src] + r->off, r->plen, &p0) != AI_ENGINE_OK) {
            printf("   [FAIL] oracle prefill req %d: %s\n", i, ai_engine_last_error(e));
            errors++;
            break;
        }
        ref[nref++] = p0;
        int64_t ps = r->plen;
        while (nref < r->budget) {
            int32_t t = p0, pp = (int32_t) ps, sid = 0;
            int on = -1;
            if (ai_engine_decode_mtp(e, &t, &pp, &sid, 1, ot, os, cap1, &on) != AI_ENGINE_OK) {
                printf("   [FAIL] oracle decode req %d: %s\n", i, ai_engine_last_error(e));
                errors++;
                break;
            }
            for (int j = 0; j < on; ++j) { if (nref < r->budget) ref[nref++] = ot[j]; p0 = ot[j]; }
            ps += on;
        }
        checked++;
        int d = -1;
        for (int j = 0; j < r->budget; ++j) if (ref[j] != r->out[j]) { d = j; break; }
        if (d < 0) matched++;
        else printf("   [FAIL] req %d (prompt %d tok, budget %d): differs from the fresh engine at token %d\n", i, r->plen,
                    r->budget, d);
    }
    ai_engine_destroy(e);
    printf("   pass 2: %d/%d alone requests identical to a fresh single-stream engine\n", matched, checked);
    const int ok = errors == 0 && finished == N && matched == checked && checked > 0 && vram_min >= vram0 - 64;
    printf("%s\n", ok ? "SOAK PASS" : "SOAK FAIL");
    return ok ? 0 : 1;
}
