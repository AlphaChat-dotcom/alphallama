/* Acceptance gate for ai_engine_prefill / ai_engine_decode / ai_engine_reset_stream.
 *
 * Pure C, so the "C so the ABI does not depend on either side's flags" promise is exercised rather
 * than asserted.
 *
 * The slot-reuse row is the main check; rc == 0 alone would prove little. reset_stream must make a
 * second request on a reused slot correct, and the only way to know that is to compare it against the
 * same request on a fresh engine. So:
 *
 *   handle A:  prefill(P1) -> N tokens = T1
 *              reset_stream(0)
 *              prefill(P2) -> N tokens = T2_reused
 *   handle B:  prefill(P2) -> N tokens = T2_fresh      (fresh process state, same prompt)
 *
 *   PASS  iff  T2_reused == T2_fresh
 *   AND   iff  T1 != T2_reused   <- can't-fail row: if two different prompts produced the same
 *                                   tokens, the equality above would hold for free and prove nothing.
 *
 * The two handles run sequentially (A destroyed before B is created) because two resident copies of a
 * 21 GiB model do not fit on a 24 GB GPU.
 */
#include "api/ai_engine.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define N_GEN 12

static int checks = 0;
static int fails  = 0;

static void ck(int cond, const char * what, const char * detail) {
    checks++;
    if (cond) {
        printf("  [ok]   %s\n", what);
    } else {
        fails++;
        printf("  [FAIL] %s%s%s\n", what, detail ? "  -- " : "", detail ? detail : "");
    }
}

/* P1 and P2 are deliberately different so the can't-fail row has teeth. vocab is 248320, bos 248044. */
static const int32_t P1[] = { 248044,   100,   200,   300,   400 };
static const int32_t P2[] = { 248044,  9000,  9100,  9200,  9300 };

/* Drive n_gen rounds on stream 0, starting from `first` at position `pos0`. Returns tokens written. */
static int run_gen(ai_engine * e, int32_t first, int64_t pos0, int32_t * out, int n_gen, const char * tag) {
    int32_t cur = first;
    int64_t pos = pos0;
    int     n   = 0;
    for (int i = 0; i < n_gen; ++i) {
        int32_t tok = cur;
        int32_t pp  = (int32_t) pos;
        int32_t sid = 0;
        int32_t sampled[8];
        int     n_sampled = -1;
        const int rc = ai_engine_decode(e, &tok, &pp, &sid, 1, NULL, sampled, &n_sampled);
        if (rc != AI_ENGINE_OK || n_sampled != 1) {
            printf("      %s: decode round %d rc=%d n_sampled=%d err=%s\n",
                   tag, i, rc, n_sampled, ai_engine_last_error(e));
            return n;
        }
        out[n++] = sampled[0];
        cur = sampled[0];
        pos += 1;
    }
    return n;
}

int main(void) {
    ai_engine_params p;
    memset(&p, 0, sizeof p);
    p.model_path           = "/data/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf";
    p.ctx_cells_per_stream = 4096;
    p.n_streams            = 1;
    p.expert_slots         = 0;
    p.ws_mib               = 192;
    p.host_embd            = 0;
    p.kv_type_is_q4_0      = 1;
    p.mtp_k                = 0;
    p.verbose              = 0;
    /* AI_API_KV / AI_API_WS / AI_API_CTX override the three sizing fields, so the same gate also runs the
     * auto path (kv -1, ws 0, ctx -1) through the C ABI. */
    if (getenv("AI_API_KV"))  p.kv_type_is_q4_0      = atoi(getenv("AI_API_KV"));
    if (getenv("AI_API_WS"))  p.ws_mib               = atoi(getenv("AI_API_WS"));
    if (getenv("AI_API_CTX")) p.ctx_cells_per_stream = atoll(getenv("AI_API_CTX"));
    if (getenv("AI_API_VERBOSE")) p.verbose = 1;

    int32_t t1[N_GEN], t2_reused[N_GEN], t2_fresh[N_GEN];
    int     n1 = 0, n2r = 0, n2f = 0;
    int32_t first1 = 0, first2 = 0, first2b = 0;

    /* handle A: request 1, reset, request 2 */
    printf("######## handle A: request 1, reset_stream, request 2 on the SAME slot ########\n");
    {
        const char * err = NULL;
        ai_engine *  e   = ai_engine_create(&p, &err);
        ck(e != NULL, "create", err);
        if (!e) { return 1; }

        /* decode before any prefill must be refused, not run on an unprepared slot */
        {
            int32_t tok = 100, pp = 0, sid = 0, sampled[8];
            int     ns  = -1;
            const int rc = ai_engine_decode(e, &tok, &pp, &sid, 1, NULL, sampled, &ns);
            ck(rc == AI_ENGINE_EINVAL, "decode before prefill is EINVAL", ai_engine_last_error(e));
        }

        ck(ai_engine_prefill(e, 0, P1, (int) (sizeof P1 / sizeof P1[0]), &first1) == AI_ENGINE_OK,
           "prefill P1", ai_engine_last_error(e));
        ck(first1 >= 0 && first1 < 248320, "prefill P1 returned an in-range token", NULL);

        n1 = run_gen(e, first1, (int64_t) (sizeof P1 / sizeof P1[0]), t1, N_GEN, "A/req1");
        ck(n1 == N_GEN, "request 1 produced N_GEN tokens", NULL);

        /* the position assert, exercised deliberately: it must fire and name both values */
        {
            int32_t tok = t1[N_GEN - 1], sid = 0, sampled[8];
            int32_t bad = (int32_t) (sizeof P1 / sizeof P1[0]) + N_GEN + 7;   /* deliberately wrong */
            int     ns  = -1;
            const int rc = ai_engine_decode(e, &tok, &bad, &sid, 1, NULL, sampled, &ns);
            const char * m = ai_engine_last_error(e);
            ck(rc == AI_ENGINE_EINVAL, "wrong position is EINVAL, not plausible output", m);
            ck(strstr(m, "position mismatch") != NULL, "error says position mismatch", m);
            ck(strstr(m, "engine has") != NULL && strstr(m, "caller sent") != NULL,
               "error names BOTH positions", m);
        }

        /* a refused call must not have advanced or corrupted the slot */
        {
            int32_t tok = t1[N_GEN - 1], sid = 0, sampled[8];
            int32_t pp  = (int32_t) (sizeof P1 / sizeof P1[0]) + N_GEN;
            int     ns  = -1;
            const int rc = ai_engine_decode(e, &tok, &pp, &sid, 1, NULL, sampled, &ns);
            ck(rc == AI_ENGINE_OK && ns == 1, "slot still usable after the refused call",
               ai_engine_last_error(e));
        }

        /* a sampling constraint must be refused rather than silently dropped */
        {
            int32_t tok = 100, pp = 0, sid = 0, sampled[8];
            uint8_t mask = 1;
            int     ns   = -1;
            const int rc = ai_engine_decode(e, &tok, &pp, &sid, 1, &mask, sampled, &ns);
            ck(rc == AI_ENGINE_ENOTSUP, "logits_mask is ENOTSUP, not ignored", ai_engine_last_error(e));
        }

        /* more tokens than streams cannot be honoured by a batch-rigid round */
        {
            int32_t tok[2] = { 100, 101 }, pp[2] = { 0, 0 }, sid[2] = { 0, 1 }, sampled[8];
            int     ns     = -1;
            const int rc = ai_engine_decode(e, tok, pp, sid, 2, NULL, sampled, &ns);
            ck(rc == AI_ENGINE_EINVAL, "n_tokens > n_streams is EINVAL", ai_engine_last_error(e));
        }

        ck(ai_engine_reset_stream(e, 0) == AI_ENGINE_OK, "reset_stream(0)", ai_engine_last_error(e));
        ck(ai_engine_reset_stream(e, 5) == AI_ENGINE_EINVAL, "reset_stream(out of range) is EINVAL",
           ai_engine_last_error(e));

        ck(ai_engine_prefill(e, 0, P2, (int) (sizeof P2 / sizeof P2[0]), &first2) == AI_ENGINE_OK,
           "prefill P2 on the reused slot", ai_engine_last_error(e));
        n2r = run_gen(e, first2, (int64_t) (sizeof P2 / sizeof P2[0]), t2_reused, N_GEN, "A/req2");
        ck(n2r == N_GEN, "request 2 on the reused slot produced N_GEN tokens", NULL);

        ck(ai_engine_capture_state(e) == AI_CAPTURE_OWN, "capture state is OWN", NULL);
        {
            const int64_t f = ai_engine_free_vram_mib(e);
            char buf[64];
            snprintf(buf, sizeof buf, "free_vram_mib = %lld", (long long) f);
            ck(f > 0 && f < 24576, "free_vram_mib in a sane range", buf);
        }
        ai_engine_destroy(e);
    }

    /* handle B: request 2 on a fresh engine */
    printf("######## handle B: the SAME request 2 on a fresh engine ########\n");
    {
        const char * err = NULL;
        ai_engine *  e   = ai_engine_create(&p, &err);
        ck(e != NULL, "create (fresh)", err);
        if (!e) { return 1; }
        ck(ai_engine_prefill(e, 0, P2, (int) (sizeof P2 / sizeof P2[0]), &first2b) == AI_ENGINE_OK,
           "prefill P2 (fresh)", ai_engine_last_error(e));
        n2f = run_gen(e, first2b, (int64_t) (sizeof P2 / sizeof P2[0]), t2_fresh, N_GEN, "B/req2");
        ck(n2f == N_GEN, "fresh request 2 produced N_GEN tokens", NULL);
        ai_engine_destroy(e);
    }

    /* verdict */
    printf("######## VERDICT ########\n");
    ck(first2 == first2b, "prefill's first token matches between reused and fresh slot", NULL);

    int same_reuse = (n2r == N_GEN && n2f == N_GEN);
    for (int i = 0; i < N_GEN && same_reuse; ++i) {
        if (t2_reused[i] != t2_fresh[i]) { same_reuse = 0; }
    }
    int differ_prompts = (n1 == N_GEN && n2r == N_GEN);
    int any_diff = 0;
    for (int i = 0; i < N_GEN && differ_prompts; ++i) {
        if (t1[i] != t2_reused[i]) { any_diff = 1; }
    }
    differ_prompts = differ_prompts && any_diff;

    printf("    req1 (P1)        :");
    for (int i = 0; i < n1; ++i) { printf(" %d", t1[i]); }
    printf("\n    req2 reused slot :");
    for (int i = 0; i < n2r; ++i) { printf(" %d", t2_reused[i]); }
    printf("\n    req2 fresh engine:");
    for (int i = 0; i < n2f; ++i) { printf(" %d", t2_fresh[i]); }
    printf("\n");

    ck(differ_prompts, "CAN'T-FAIL ROW: P1 and P2 produce DIFFERENT tokens "
                       "(otherwise the reuse comparison is vacuous)", NULL);
    ck(same_reuse, "SLOT REUSE IS CORRECT: reused-slot request 2 == fresh-engine request 2", NULL);

    printf("\n  %d/%d checks passed\n", checks - fails, checks);
    if (fails == 0 && differ_prompts && same_reuse) {
        printf("  => PASS: caller-driven prefill/decode/reset_stream work, and slot reuse is correct.\n");
        return 0;
    }
    printf("  => FAIL: %d check(s) failed.\n", fails);
    return 1;
}
