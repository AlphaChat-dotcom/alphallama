// The decode-level teacher-forced comparator (AI_TF_*), shared by every Program's engine; the dumps are diffed by
// tools/tf_compare.py.
#pragma once

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ai {

// Decode-level teacher-forced comparator (debug; batch 1). Answers whether an MTP verify row is bit-identical to
// plain decode at the same position. Generated text cannot say -- where two greedy texts first differ is one sample
// of a chaotic process -- and --score at chunk > 1 is no proxy, since prefill chunks are not width-invariant while
// decode rounds use their own invariance-tuned kernels.
//   AI_TF_DUMP=<file>   append one record per decoded position: i64 pos, i32 token fed, i32 argmax, i64 V, f32 V.
//                       Plain: every round. MTP: verify rows 0..accepted (the rows whose input is committed).
//   AI_TF_FORCE=<file>  a plain run's dump: before each MTP round the drafts become the tokens plain decode fed
//                       at those positions, so every verify row sits on the same position and input as a plain
//                       step until the first row whose argmax differs.
// tools/tf_compare.py diffs two dumps by position.
struct TfState {
    FILE *                         dump = nullptr;
    std::vector<std::pair<int64_t, int32_t>> force;   // (pos, token fed), sorted
};
inline TfState & tf_state() {
    static TfState t = [] {
        TfState s;
        if (const char * p = getenv("AI_TF_DUMP")) s.dump = fopen(p, "wb");
        if (const char * p = getenv("AI_TF_FORCE")) {
            FILE * f = fopen(p, "rb");
            if (!f) throw std::runtime_error(std::string("AI_TF_FORCE: cannot open ") + p);
            int64_t pos, V;
            int32_t tok, am;
            while (fread(&pos, 8, 1, f) == 1 && fread(&tok, 4, 1, f) == 1 && fread(&am, 4, 1, f) == 1 &&
                   fread(&V, 8, 1, f) == 1) {
                s.force.emplace_back(pos, tok);
                fseek(f, (long) (V * 4), SEEK_CUR);
            }
            fclose(f);
            std::sort(s.force.begin(), s.force.end());
            fprintf(stderr, "[ai-tf] forcing drafts from %zu plain positions\n", s.force.size());
        }
        return s;
    }();
    return t;
}
inline bool tf_force_token(int64_t pos, int32_t * tok) {
    const auto & f = tf_state().force;
    auto it = std::lower_bound(f.begin(), f.end(), std::make_pair(pos, INT32_MIN));
    if (it == f.end() || it->first != pos) return false;
    *tok = it->second;
    return true;
}
inline void tf_write(int64_t pos, int32_t tok, int32_t am, const float * row, int64_t V) {
    FILE * d = tf_state().dump;
    fwrite(&pos, 8, 1, d);
    fwrite(&tok, 4, 1, d);
    fwrite(&am, 4, 1, d);
    fwrite(&V, 8, 1, d);
    fwrite(row, 4, (size_t) V, d);
}

}  // namespace ai
