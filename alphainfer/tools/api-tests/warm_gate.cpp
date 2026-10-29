// SPDX-License-Identifier: Apache-2.0
/* Test for eager profile instantiation.
 *
 * The property that matters is not that warm_profiles returns a number but that nothing
 * instantiates after startup. So the measurement is tm.rounds_recorded across the serving phase, read
 * from the counter rather than parsed out of logs.
 *
 *   arm A (no warm):  serve -> records R_A profiles lazily, mid-serving
 *   arm B (warmed):   warm  -> records N, then serve -> must record 0
 *
 * Can't-fail row: R_A must be > 0. If the unwarmed arm also records nothing during serving, then
 * "0 during serving" is true for free and proves nothing. It is checked before any pass is reported.
 *
 * Warming must also not change what the engine emits: arm A's tokens must equal arm B's.
 *
 * The two engines run sequentially in nested scopes, since two resident copies of the model may not
 * fit in VRAM.
 */
#include "engine/engine.h"
#include "engine/tokenizer.h"

#include <cstdio>
#include <utility>
#include <vector>

static const char * MODEL = "/data/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf";
static const int    N_GEN = 24;

static ai::EngineConfig make_cfg() {
    ai::EngineConfig cfg;
    cfg.model_path = MODEL;
    cfg.max_ctx    = 4096;          /* ladder: 256 512 1024 2048 4096 -> 5 buckets */
    cfg.max_batch  = 1;
    cfg.kv_type    = ai::Q4_0;
    cfg.ws_bytes   = (size_t) 192 << 20;
    cfg.verbose    = false;
    return cfg;
}

/* Drive the caller-driven path. Returns tokens; *recorded receives profiles recorded during serving. */
static std::vector<int32_t> serve(ai::Engine & eng, const std::vector<int32_t> & prompt,
                                 int64_t * recorded) {
    ai::Timing tm;
    eng.open_streams(1);
    const int64_t before = tm.rounds_recorded;
    const int32_t first  = eng.prefill_one(0, prompt, tm);
    std::vector<int32_t> got;
    got.push_back(first);
    int64_t pos = (int64_t) prompt.size();
    int32_t cur = first;
    while ((int) got.size() < N_GEN) {
        std::vector<std::pair<int, int32_t>> out;
        std::vector<ai::Engine::DecodeIn>    in;
        in.push_back(ai::Engine::DecodeIn{ 0, pos, cur });
        eng.decode_step(in, out, tm);
        for (size_t i = 0; i < out.size(); ++i) { got.push_back(out[i].second); }
        cur = out.back().second;
        pos += 1;
    }
    *recorded = tm.rounds_recorded - before;
    return got;
}

int main() {
    ai::Tokenizer tok;
    if (!tok.load(MODEL)) { printf("tokenizer load failed\n"); return 1; }
    const std::vector<int32_t> prompt = tok.encode(
        ai::Tokenizer::chat_prompt("Explain in two sentences why memory bandwidth limits the speed of "
                                   "transformer inference on a GPU.", false));
    printf("  prompt: %zu tokens, ctx 4096\n\n", prompt.size());

    std::vector<int32_t> tok_a, tok_b;
    int64_t rec_a = -1, rec_b = -1;
    int     warmed = -1;

    /* ---- arm A: no warming. Profiles are recorded lazily, mid-serving. ---- */
    {
        ai::Engine eng;
        if (!eng.init(make_cfg())) { printf("init A failed\n"); return 1; }
        tok_a = serve(eng, prompt, &rec_a);
        printf("  arm A (no warm)  : %zu tokens, %lld profile(s) recorded DURING serving\n",
               tok_a.size(), (long long) rec_a);
    }

    /* ---- arm B: warm at startup, then serve. ---- */
    {
        ai::Engine eng;
        if (!eng.init(make_cfg())) { printf("init B failed\n"); return 1; }
        ai::Timing wtm;
        eng.open_streams(1);
        warmed = eng.warm_profiles(wtm);
        tok_b  = serve(eng, prompt, &rec_b);
        printf("  arm B (warmed)   : %zu tokens, %d warmed at init, %lld recorded DURING serving\n",
               tok_b.size(), warmed, (long long) rec_b);
    }

    printf("\n######## VERDICT ########\n");
    int fails = 0;

    /* can't-fail row first: without it, arm B's zero is free */
    if (rec_a > 0) {
        printf("  [ok]   CAN'T-FAIL ROW: the unwarmed arm DID instantiate mid-serving (%lld), so the\n"
               "         counter can detect instantiation at all\n", (long long) rec_a);
    } else {
        printf("  [FAIL] CAN'T-FAIL ROW: the unwarmed arm recorded %lld during serving. The counter\n"
               "         cannot detect instantiation here, so arm B proves nothing.\n", (long long) rec_a);
        fails++;
    }

    if (warmed >= 5) {
        printf("  [ok]   warm_profiles recorded %d profiles (ladder for ctx 4096 is 5)\n", warmed);
    } else {
        printf("  [FAIL] warm_profiles recorded %d, expected the full 5-bucket ladder\n", warmed);
        fails++;
    }

    if (rec_b == 0) {
        printf("  [ok]   zero instantiates during serving after warming\n");
    } else {
        printf("  [FAIL] %lld profile(s) still instantiated during serving after warming\n",
               (long long) rec_b);
        fails++;
    }

    if (tok_a == tok_b) {
        printf("  [ok]   warming changed NO output (arm A tokens == arm B tokens)\n");
    } else {
        printf("  [FAIL] warming CHANGED the output\n");
        printf("         A:");
        for (size_t i = 0; i < tok_a.size(); ++i) { printf(" %d", tok_a[i]); }
        printf("\n         B:");
        for (size_t i = 0; i < tok_b.size(); ++i) { printf(" %d", tok_b[i]); }
        printf("\n");
        fails++;
    }

    /* strength gate: an equality over a degenerate sequence carries little information */
    int distinct = 0;
    for (size_t i = 0; i < tok_a.size(); ++i) {
        int seen = 0;
        for (size_t j = 0; j < i; ++j) { if (tok_a[j] == tok_a[i]) { seen = 1; break; } }
        distinct += seen ? 0 : 1;
    }
    printf("  reference has %d distinct tokens over %zu\n", distinct, tok_a.size());
    if (distinct < 5) {
        printf("  [FAIL] too few distinct tokens for the equality to mean much\n");
        fails++;
    }
    printf("  decoded: ");
    for (size_t i = 0; i < tok_a.size(); ++i) { printf("%s", tok.piece(tok_a[i]).c_str()); }
    printf("\n\n");

    if (fails == 0) {
        printf("  => PASS: every profile is instantiated at startup and none during serving, and\n");
        printf("     warming is output-neutral.\n");
        return 0;
    }
    printf("  => FAIL: %d check(s) failed.\n", fails);
    return 1;
}
