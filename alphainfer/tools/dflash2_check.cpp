// SPDX-License-Identifier: Apache-2.0
// DFlash2 isolation test: AlphaInfer's drafter forward against the z-lab reference, on identical inputs.
//
//   ai-dflash2-check <dflash2.gguf> <target-io.gguf> <oracle prompt dir> [rounds]
//
// The oracle dir holds what the reference saw: prompt.i32 + gen.i32 (the target's greedy sequence), taps.bin (the
// target's taps at every position) and what it produced: oracle_sched.txt ("start ctx_rows produced d1..d7" per
// round), oracle_h.f32 (first rounds' final-normed hidden, [7][E]) and oracle_cand.i32 (their top-16 ids,
// sorted). This tool follows the oracle's schedule exactly (the same context rows at the same positions, the
// same anchors), so every round is a paired comparison, and reports:
//   * hidden: relative L2 error and cosine vs the reference (first rounds)
//   * candidates: top-16 set overlap
//   * drafts: per-position agreement, and tokens/round each side would get against the true sequence
// target-io.gguf: the target's token_embd.weight and output.weight only (the drafter borrows both).
#include "core/device.h"
#include "core/weights.h"
#include "models/dflash2.h"
#include "ops/ops.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace ai;

template <class T> static std::vector<T> read_vec(const std::string & p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) {
        fprintf(stderr, "cannot read %s\n", p.c_str());
        exit(2);
    }
    std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<T>    v(b.size() / sizeof(T));
    memcpy(v.data(), b.data(), v.size() * sizeof(T));
    return v;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <dflash2.gguf> <target-io.gguf> <oracle prompt dir> [rounds]\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[3];
    const int max_rounds = argc > 4 ? atoi(argv[4]) : 1 << 30;

    auto prompt = read_vec<int32_t>(dir + "/prompt.i32");
    auto gen    = read_vec<int32_t>(dir + "/gen.i32");
    std::vector<int32_t> seq(prompt);
    seq.insert(seq.end(), gen.begin(), gen.end());
    auto           tb = read_vec<char>(dir + "/taps.bin");
    int64_t        cols, rows;
    memcpy(&cols, tb.data(), 8);
    memcpy(&rows, tb.data() + 8, 8);
    if (rows != (int64_t) seq.size()) {
        fprintf(stderr, "REFUSING: taps rows %lld != sequence %zu\n", (long long) rows, seq.size());
        return 2;
    }
    struct Round { int64_t start, ctx_rows, produced; std::vector<int32_t> drafts; };
    std::vector<Round> sched;
    {
        std::ifstream f(dir + "/oracle_sched.txt");
        std::string   line;
        while (std::getline(f, line)) {
            std::istringstream is(line);
            Round              r;
            is >> r.start >> r.ctx_rows >> r.produced;
            int32_t t;
            while (is >> t) r.drafts.push_back(t);
            if (!r.drafts.empty()) sched.push_back(r);
        }
    }
    auto oh = read_vec<float>(dir + "/oracle_h.f32");
    std::vector<int32_t> oc;   // v2 only (v1 drafts are the argmax; no candidate set)
    if (std::ifstream(dir + "/oracle_cand.i32").good()) oc = read_vec<int32_t>(dir + "/oracle_cand.i32");
    if (sched.empty()) {
        fprintf(stderr, "REFUSING: empty oracle schedule\n");
        return 2;
    }

    Device dev;
    if (!dev.init(0)) return 1;
    WeightSet tio;
    if (!tio.open(argv[2]) || !tio.load(dev, 4ULL << 30, false)) return 1;
    const Tensor & embed  = tio.get("token_embd.weight");
    const Tensor & head_w = tio.has("output.weight") ? tio.get("output.weight") : embed;
    DFlash2 df;
    if (!df.load(dev, argv[1], false)) return 1;
    df.alloc(dev, rows + 64);
    if (cols != df.tap_width()) {
        fprintf(stderr, "REFUSING: taps cols %lld != drafter tap width %lld\n", (long long) cols,
                (long long) df.tap_width());
        return 2;
    }
    // all tap rows resident: the schedule appends views of them
    Tensor taps;
    taps.type  = F32;
    taps.ne[0] = cols;
    taps.ne[1] = rows;
    set_contiguous_strides(taps);
    taps.buf  = dev.alloc_buffer(taps.nbytes() + dev.alignment(), "taps");
    taps.data = dev.buffer_base(taps.buf);
    dev.init_workspace(1ULL << 30);
    dev.finalize();
    dev.upload(taps, tb.data() + 16, 0, taps.nbytes());
    auto rows_at = [&](int64_t r0, int64_t n) {
        return taps.view(cols, n, 1, 1, taps.nb[1], taps.nb[1] * n, taps.nb[1] * n, taps.nb[1] * (size_t) r0);
    };

    const int64_t E = df.tap_width() / (int64_t) df.target_layers().size(), nd = df.block_size() - 1;
    const int64_t n_h = (int64_t) oh.size() / (nd * E);
    static const bool trace = getenv("DF2_TRACE") != nullptr;   // sync + print after each stage (fault localisation)
    auto stage = [&](const char * what) {
        if (trace) {
            dev.synchronize();
            fprintf(stderr, "[trace] %s ok\n", what);
        }
    };
    df.ctx_append(dev, rows_at(0, sched[0].start), 0);
    stage("prefill ctx_append");
    int64_t agree_all = 0, rounds = 0, ours_tok = 0, ref_tok = 0;
    std::vector<int64_t> agree_pos((size_t) nd, 0);
    for (size_t ri = 0; ri < sched.size() && (int) ri < max_rounds; ++ri) {
        const Round & r = sched[ri];
        if (df.ctx_rows() != r.start) {
            fprintf(stderr, "schedule drift: ctx rows %lld vs start %lld\n", (long long) df.ctx_rows(),
                    (long long) r.start);
            return 1;
        }
        const bool dbg = (int64_t) ri < n_h;
        auto       d   = df.draft(dev, seq[(size_t) r.start], r.start, embed, head_w, dbg);
        stage("draft");
        if (dbg) {
            double num = 0, den = 0, dot = 0, nn = 0;
            const float * ref = oh.data() + (size_t) (ri * nd * E);
            for (int64_t i = 0; i < nd * E; ++i) {
                const double a = d.h[(size_t) i], b = ref[i];
                num += (a - b) * (a - b); den += b * b; dot += a * b; nn += a * a;
            }
            int overlap = 0;
            for (int64_t t = 0; t < nd && df.variant() == 2 && !oc.empty(); ++t) {
                const int32_t *   rc = oc.data() + (size_t) ((ri * nd + t) * 16);
                std::set<int32_t> s(rc, rc + 16);
                for (int c = 0; c < 16; ++c) overlap += s.count(d.cand[(size_t) (t * 16 + c)]);
            }
            printf("round %zu: hidden rel-L2 %.4f cos %.6f | top-16 overlap %d/%lld\n", ri, sqrt(num / den),
                   dot / sqrt(den * nn), overlap, (long long) (16 * nd));
        }
        bool all = true;
        int  acc_ours = 0, acc_ref = 0;
        bool run_o = true, run_r = true;
        for (int64_t t = 0; t < nd; ++t) {
            const int32_t truth = r.start + 1 + t < (int64_t) seq.size() ? seq[(size_t) (r.start + 1 + t)] : -1;
            const bool    same  = d.tokens[(size_t) t] == r.drafts[(size_t) t];
            agree_pos[(size_t) t] += same;
            all = all && same;
            run_o = run_o && d.tokens[(size_t) t] == truth;
            run_r = run_r && r.drafts[(size_t) t] == truth;
            acc_ours += run_o;
            acc_ref += run_r;
        }
        if (acc_ref + 1 != r.produced) {
            fprintf(stderr, "oracle bookkeeping mismatch at round %zu: acc %d produced %lld\n", ri, acc_ref,
                    (long long) r.produced);
            return 1;
        }
        agree_all += all;
        ours_tok += acc_ours + 1;
        ref_tok += acc_ref + 1;
        ++rounds;
        df.ctx_append(dev, rows_at(r.start, r.produced), r.start);
        stage("ctx_append");
    }
    printf("rounds %lld: all-7 drafts identical %.1f%% | per-position agreement", (long long) rounds,
           100.0 * agree_all / rounds);
    for (int64_t t = 0; t < nd; ++t) printf(" %.3f", (double) agree_pos[(size_t) t] / rounds);
    printf("\ntokens/round on the oracle's schedule: reference %.3f, AlphaInfer %.3f (%+.2f%%)\n",
           (double) ref_tok / rounds, (double) ours_tok / rounds, 100.0 * ((double) ours_tok / ref_tok - 1.0));
    return 0;
}
