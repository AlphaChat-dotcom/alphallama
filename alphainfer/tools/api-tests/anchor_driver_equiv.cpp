/* Absolute anchor for the caller-driven decode path.
 *
 * The C acceptance test compares a reused slot against a fresh engine, which is a relative test: if
 * the caller-driven path were broken in a way that affected both arms equally, the comparison would
 * still pass, and a "decode echoes its input" bug would look like plausible output for arbitrary ids.
 *
 * This test drives the same token sequence through both paths in one process and requires identical
 * output:
 *
 *   ref = eng.generate_batch(prompt, N)             <- the engine's own loop
 *   got = open_streams / prefill_one / decode_step  <- the caller-driven path
 */
#include "engine/engine.h"
#include "engine/tokenizer.h"

#include <cstdio>
#include <utility>
#include <vector>

int main() {
    ai::EngineConfig cfg;
    cfg.model_path = "/data/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf";
    cfg.max_ctx    = 4096;
    cfg.max_batch  = 1;
    cfg.kv_type    = ai::Q4_0;
    cfg.ws_bytes   = (size_t) 192 << 20;
    cfg.verbose    = false;

    ai::Engine eng;
    if (!eng.init(cfg)) {
        printf("init failed\n");
        return 1;
    }

    /* A real text prompt, not arbitrary ids: arbitrary ids tend to make the reference repeat one token,
     * and two paths agreeing on a constant sequence carries little information. A coherent prompt
     * gives a varied reference, and the strength check below refuses to report a pass if it does not. */
    ai::Tokenizer tok;
    if (!tok.load(cfg.model_path)) {
        printf("tokenizer load failed\n");
        return 1;
    }
    const std::vector<int32_t> prompt = tok.encode(
        ai::Tokenizer::chat_prompt("Explain in two sentences why memory bandwidth limits the speed of "
                                   "transformer inference on a GPU.", false));
    const int N = 24;
    printf("  prompt: %zu tokens\n", prompt.size());

    /* path 1: the engine's own loop (the reference) */
    ai::Timing t1;
    std::vector<std::vector<int32_t>> outs =
        eng.generate_batch(std::vector<std::vector<int32_t>>{ prompt }, N,
                           [](int, int32_t) { return true; }, &t1);
    const std::vector<int32_t> ref = outs.at(0);

    /* path 2: caller-driven, same tokens */
    ai::Timing t2;
    eng.open_streams(1);
    const int32_t first = eng.prefill_one(0, prompt, t2);
    std::vector<int32_t> got;
    got.push_back(first);                      /* generate_batch emits the prefill token too */
    int64_t pos = (int64_t) prompt.size();
    int32_t cur = first;
    int     guard = 0;
    while (got.size() < ref.size() && guard++ < 4 * N) {
        std::vector<std::pair<int, int32_t>> out;
        std::vector<ai::Engine::DecodeIn>    in;
        in.push_back(ai::Engine::DecodeIn{ 0, pos, cur });
        eng.decode_step(in, out, t2);
        if (out.empty()) {
            printf("decode_step emitted nothing at pos %lld\n", (long long) pos);
            return 1;
        }
        for (size_t i = 0; i < out.size(); ++i) {
            got.push_back(out[i].second);
        }
        cur = out.back().second;
        pos += 1;
    }

    printf("  generate_batch (%zu):", ref.size());
    for (size_t i = 0; i < ref.size(); ++i) { printf(" %d", ref[i]); }
    printf("\n  caller-driven  (%zu):", got.size());
    for (size_t i = 0; i < got.size(); ++i) { printf(" %d", got[i]); }
    printf("\n\n");

    if (got.size() != ref.size()) {
        printf("  => FAIL: length %zu vs %zu\n", got.size(), ref.size());
        return 1;
    }
    size_t first_diff = ref.size();
    for (size_t i = 0; i < ref.size(); ++i) {
        if (ref[i] != got[i]) { first_diff = i; break; }
    }
    if (first_diff != ref.size()) {
        printf("  => FAIL: paths diverge at index %zu (ref %d, caller-driven %d)\n",
               first_diff, ref[first_diff], got[first_diff]);
        return 1;
    }
    /* Strength check on the test itself: equality over a near-constant reference is not reported as a
     * pass, so require the reference to be varied. */
    int distinct = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        int seen = 0;
        for (size_t j = 0; j < i; ++j) { if (ref[j] == ref[i]) { seen = 1; break; } }
        distinct += seen ? 0 : 1;
    }
    printf("  reference has %d distinct tokens over %zu\n", distinct, ref.size());
    printf("  decoded: ");
    for (size_t i = 0; i < ref.size(); ++i) { printf("%s", tok.piece(ref[i]).c_str()); }
    printf("\n\n");
    if (distinct < 5) {
        printf("  => INCONCLUSIVE: the reference emits only %d distinct tokens, so identical output\n"
               "     carries too little information to call this a pass. Not reporting one.\n", distinct);
        return 1;
    }
    printf("  => PASS: the caller-driven path is IDENTICAL to generate_batch over a VARIED reference\n");
    printf("     (%d distinct tokens). Chained with Stage 1 (pre-binary == post generate_batch), the\n",
           distinct);
    printf("     caller-driven path reproduces the original engine exactly.\n");
    return 0;
}
