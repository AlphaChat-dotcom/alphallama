// SPDX-License-Identifier: Apache-2.0
// ai-bench-fixed - per-call cost and read rate of the one-token paths, by weight type.
//
// Two questions it answers:
//  1. what does a single MUL_MAT (or MUL_MAT_ID) call cost before it reads any weights, and
//  2. how fast does each GGUF weight type stream at T=1 (the decode regime)?
//
// Method: synthetic weights over a grid of (cols, rows) per type, with enough copies that one
// timed cycle spans >= --span-mib (so reads come from HBM, not L2). Each copy is its own captured
// one-op round, the way decode runs; a cycle launches every copy and synchronises once. An empty
// round (a 1-element ADD) is timed the same way and subtracted, so the figures exclude the host's
// launch of the round itself. --sync-each reproduces ai-bench-gemv's method (one sync per call).
//
// Geometry and math switches of the MMVQ kernel, read once from the environment and printed with each result
// (ggml in this repository does not read them, so here they only label the run):
//   AI_MMVQ_T1_SMALLK=0|1   one grid block per row | rows_per_block = nwarps  (unset: ggml's choice)
//   AI_MMVQ_T1_MODE=1|2     skip the weight reads | return at kernel entry     (unset: normal)
//
//   ai-bench-fixed [-n cycles] [--types q8_0,q4_K,...] [--cols ...] [--rows ...]
//                  [--experts N --topk K]   (expert path: one MUL_MAT_ID over a [cols,rows,N] bank)
#include "core/device.h"
#include "core/tensor.h"
#include "ops/ops.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace ai;

static std::vector<int64_t> parse_list(const std::string & s) {
    std::vector<int64_t> v;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            v.push_back(atoll(item.c_str()));
        }
    }
    return v;
}

static int type_of(const std::string & n) {
    if (n == "q8_0") return Q8_0;
    if (n == "q4_0") return Q4_0;
    if (n == "q4_1") return Q4_1;
    if (n == "q4_K") return Q4_K;
    if (n == "q5_K") return Q5_K;
    if (n == "q6_K") return Q6_K;
    return -1;
}

// random payload with finite scales, so the kernels do real work on real values
static void fill_blocks(std::vector<uint8_t> & buf, int type, std::mt19937 & rng) {
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto & b : buf) {
        b = (uint8_t) byte(rng);
    }
    const uint16_t d    = 0x23D7;   // ~0.01
    const uint16_t dmin = 0x1C00;   // ~0.0049
    const size_t   ts   = type_size(type);
    for (size_t off = 0; off + ts <= buf.size(); off += ts) {
        uint8_t * p = buf.data() + off;
        switch (type) {
            case Q8_0: case Q4_0: memcpy(p, &d, 2); break;
            case Q4_1: memcpy(p, &d, 2); memcpy(p + 2, &dmin, 2); break;
            case Q4_K: case Q5_K: memcpy(p, &d, 2); memcpy(p + 2, &dmin, 2); break;
            case Q6_K: memcpy(p + 208, &d, 2); break;   // ql[128] qh[64] scales[16] d
            default: break;
        }
    }
}

static Tensor carve(Device & dev, int buf, size_t & off, int type, int64_t n0, int64_t n1, int64_t n2 = 1) {
    Tensor t;
    t.type  = type;
    t.ne[0] = n0;
    t.ne[1] = n1;
    t.ne[2] = n2;
    set_contiguous_strides(t);
    t.buf  = buf;
    t.data = (char *) dev.buffer_base(buf) + off;
    off += (dev.alloc_size(t) + dev.alignment() - 1) / dev.alignment() * dev.alignment();
    return t;
}

// least squares for y = a + b*rows + c*MiB
static void fit3(const std::vector<double> & u, const std::vector<double> & v, const std::vector<double> & y,
                 double & a, double & b, double & c) {
    double S[3][4] = {{0}};
    for (size_t i = 0; i < y.size(); ++i) {
        const double r[3] = {1.0, u[i], v[i]};
        for (int p = 0; p < 3; ++p) {
            for (int q = 0; q < 3; ++q) {
                S[p][q] += r[p] * r[q];
            }
            S[p][3] += r[p] * y[i];
        }
    }
    for (int p = 0; p < 3; ++p) {
        int piv = p;
        for (int q = p + 1; q < 3; ++q) {
            if (std::fabs(S[q][p]) > std::fabs(S[piv][p])) piv = q;
        }
        for (int k = 0; k < 4; ++k) std::swap(S[p][k], S[piv][k]);
        const double dd = S[p][p];
        if (std::fabs(dd) < 1e-30) { a = b = c = NAN; return; }
        for (int k = 0; k < 4; ++k) S[p][k] /= dd;
        for (int q = 0; q < 3; ++q) {
            if (q == p) continue;
            const double f = S[q][p];
            for (int k = 0; k < 4; ++k) S[q][k] -= f * S[p][k];
        }
    }
    a = S[0][3];
    b = S[1][3];
    c = S[2][3];
}

int main(int argc, char ** argv) {
    int  cycles = 100, copies = 4, min_span_mib = 256, max_copies = 256, experts = 0, topk = 8, tokens = 1;
    int  overlap = 0;   // % of each column's experts shared with the previous column
    bool sync_each = false;
    std::vector<int64_t> cols_list = {512, 2048, 4096};
    std::vector<int64_t> rows_list = {16, 64, 256, 512, 1024, 2048, 4096, 8192, 12288};
    std::vector<std::string> type_names = {"q8_0"};
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i];
        if (a == "-n") cycles = atoi(argv[i + 1]);
        else if (a == "--copies") copies = atoi(argv[i + 1]);
        else if (a == "--cols") cols_list = parse_list(argv[i + 1]);
        else if (a == "--rows") rows_list = parse_list(argv[i + 1]);
        else if (a == "--span-mib") min_span_mib = atoi(argv[i + 1]);
        else if (a == "--max-copies") max_copies = atoi(argv[i + 1]);
        else if (a == "--sync-each") sync_each = atoi(argv[i + 1]) != 0;
        else if (a == "--experts") experts = atoi(argv[i + 1]);
        else if (a == "--topk") topk = atoi(argv[i + 1]);
        else if (a == "--tokens") tokens = std::max(1, atoi(argv[i + 1]));
        else if (a == "--overlap") overlap = atoi(argv[i + 1]);
        else if (a == "--types") {
            type_names.clear();
            std::stringstream ss(argv[i + 1]);
            std::string t;
            while (std::getline(ss, t, ',')) {
                if (!t.empty()) type_names.push_back(t);
            }
        }
    }
    const char * sk = getenv("AI_MMVQ_T1_SMALLK");
    const char * md = getenv("AI_MMVQ_T1_MODE");
    const std::string arm = std::string("smallk=") + (sk ? sk : "auto") + " mode=" + (md ? md : "0") +
                            (experts > 0 ? " mmid" : "") + (tokens > 1 ? " T=" + std::to_string(tokens) + " ov" + std::to_string(overlap) : "");

    op::set_capture(op::Capture::Own);
    Device dev;
    if (!dev.init(0)) {
        return 1;
    }

    struct Shape { int type; int64_t cols, rows; int n; std::vector<Tensor> w; std::vector<Tensor> ids; };
    std::vector<Shape> shapes;
    size_t total = 0, ids_bytes = 0;
    for (const std::string & tn : type_names) {
        const int ty = type_of(tn);
        if (ty < 0) {
            fprintf(stderr, "unknown type %s\n", tn.c_str());
            return 2;
        }
        for (int64_t c : cols_list) {
            for (int64_t r : rows_list) {
                if (c % block_size(ty) != 0) {
                    continue;
                }
                Tensor probe;
                probe.type  = ty;
                probe.ne[0] = c;
                probe.ne[1] = r;
                probe.ne[2] = experts > 0 ? experts : 1;
                set_contiguous_strides(probe);
                const size_t per = (dev.alloc_size(probe) + dev.alignment() - 1) / dev.alignment() * dev.alignment();
                // bytes a call actually reads (expert path reads topk of the bank)
                const size_t read = experts > 0 ? per / (size_t) experts * (size_t) topk : per;
                const int n = std::min(max_copies, std::max(copies, (int) (((size_t) min_span_mib << 20) / read + 1)));
                total     += per * (size_t) n;
                ids_bytes += (size_t) topk * 4 * (size_t) tokens * (size_t) n + 64;
                shapes.push_back({ty, c, r, n, {}, {}});
            }
        }
    }
    const int wbuf = dev.alloc_buffer(total + 16 * dev.alignment(), "bench.weights");
    size_t off = 0;
    for (auto & s : shapes) {
        for (int k = 0; k < s.n; ++k) {
            s.w.push_back(carve(dev, wbuf, off, s.type, s.cols, s.rows, experts > 0 ? experts : 1));
        }
    }
    const int xbuf = dev.alloc_buffer(64 * 1024 + ids_bytes + 32 * dev.alignment(), "bench.x");
    size_t xoff = 0;
    std::vector<Tensor> xs;
    for (int64_t c : cols_list) {
        xs.push_back(carve(dev, xbuf, xoff, F32, c, tokens));
    }
    if (experts > 0) {
        for (auto & s : shapes) {
            for (int k = 0; k < s.n; ++k) {
                s.ids.push_back(carve(dev, xbuf, xoff, I32, topk, tokens));
            }
        }
    }
    Tensor one_a = carve(dev, xbuf, xoff, F32, 1, 1);
    Tensor one_b = carve(dev, xbuf, xoff, F32, 1, 1);
    dev.init_workspace(256ULL << 20);
    dev.finalize();

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto & s : shapes) {
        std::vector<uint8_t> host(s.w[0].nbytes());
        fill_blocks(host, s.type, rng);   // one pattern per shape; timing depends on addresses
        for (auto & wt : s.w) {
            dev.upload(wt, host.data(), 0, host.size());
        }
        for (size_t k = 0; k < s.ids.size(); ++k) {
            // --overlap: how many of each column's experts the previous column also used
            std::vector<int32_t> sel((size_t) topk * (size_t) tokens);
            const int shared = std::max(0, std::min(topk, topk * overlap / 100));
            for (int t = 0; t < tokens; ++t) {
                for (int j = 0; j < topk; ++j) {
                    const size_t idx = (size_t) t * (size_t) topk + (size_t) j;
                    sel[idx] = j < shared ? (int32_t) ((k * 13 + j * 7 + 3) % (size_t) experts)
                                          : (int32_t) ((k * 13 + (t + 1) * 29 + j * 7 + 3) % (size_t) experts);
                }
            }
            dev.upload(s.ids[k], sel.data(), 0, sel.size() * 4);
        }
    }
    for (size_t i = 0; i < xs.size(); ++i) {
        std::vector<float> v((size_t) cols_list[i] * (size_t) tokens);
        for (auto & e : v) e = nd(rng);
        dev.upload(xs[i], v.data(), 0, v.size() * 4);
    }
    const float onev = 1.0f;
    dev.upload(one_a, &onev, 0, 4);
    dev.upload(one_b, &onev, 0, 4);

    const auto time_rounds = [&](std::vector<std::unique_ptr<op::Round>> & rounds) {
        for (int warm = 0; warm < 3; ++warm) {
            for (auto & r : rounds) r->launch();
        }
        dev.synchronize();
        std::vector<double> per;
        const int cyc = std::max(3, (int) ((size_t) cycles * 4 / std::max<size_t>(4, rounds.size())));
        for (int c = 0; c < cyc; ++c) {
            if (sync_each) {   // ai-bench-gemv's method: one sync per call
                for (auto & r : rounds) {
                    const auto t1 = std::chrono::steady_clock::now();
                    r->launch();
                    dev.synchronize();
                    per.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t1).count());
                }
                continue;
            }
            const auto t0 = std::chrono::steady_clock::now();
            for (auto & r : rounds) r->launch();
            dev.synchronize();
            per.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() /
                          (double) rounds.size());
        }
        std::sort(per.begin(), per.end());
        return per[per.size() / 2];
    };

    double base = 0;
    {
        dev.ws_reset(0);
        std::vector<std::unique_ptr<op::Round>> rounds;
        for (int k = 0; k < 4; ++k) {
            auto r = std::make_unique<op::Round>(dev);
            r->begin();
            op::add(dev, one_a, one_b);
            r->end();
            rounds.push_back(std::move(r));
        }
        base = time_rounds(rounds);
    }

    printf("# ai-bench-fixed  arm: %s%s  span >=%d MiB (<=%d copies)  cycles %d  empty round %.2f us/call (subtracted)\n",
           arm.c_str(), sync_each ? "  [sync after every call]" : "", min_span_mib, max_copies, cycles, base);
    if (experts > 0) {
        printf("# expert path: MUL_MAT_ID over [cols, rows, %d], %d experts selected per call\n", experts, topk);
    }
    printf("# columns per call (T) = %d; \"read MiB\" counts each weight byte once per call\n", tokens);
    printf("%-6s %-24s %6s %6s %9s %6s | %9s %9s %8s\n", "type", "arm", "cols", "rows", "read MiB", "copies",
           "us/call", "net us", "GB/s");
    std::string cur_type;
    std::vector<double> fu, fv, fy;
    const auto flush_fit = [&]() {
        if (fy.size() < 3 || cur_type.empty()) {
            return;
        }
        double a = 0, b = 0, c = 0;
        fit3(fu, fv, fy, a, b, c);
        printf("# fit %-5s %s: net us/call = %.3f + %.5f * rows + %.4f * MiB   (%.0f GB/s marginal)\n",
               cur_type.c_str(), arm.c_str(), a, b, c, c > 0 ? 1048576.0 / (c * 1e-6) / 1e9 : 0.0);
        fu.clear(); fv.clear(); fy.clear();
    };
    for (auto & s : shapes) {
        if (dtype_name(s.type) != cur_type) {
            flush_fit();
            cur_type = dtype_name(s.type);
        }
        size_t xi = 0;
        while (cols_list[xi] != s.cols) ++xi;
        dev.ws_reset(0);
        std::vector<std::unique_ptr<op::Round>> rounds;
        for (size_t k = 0; k < s.w.size(); ++k) {
            auto r = std::make_unique<op::Round>(dev);
            r->begin();
            if (experts > 0) {
                op::linear_id(dev, s.w[k], xs[xi], s.ids[k]);
            } else {
                op::linear(dev, s.w[k], xs[xi]);
            }
            r->end();
            rounds.push_back(std::move(r));
        }
        const double us   = time_rounds(rounds);
        const double net  = us - base;
        const double mib  = s.w[0].nbytes() / 1048576.0 / (experts > 0 ? (double) experts / topk : 1.0);
        printf("%-6s %-24s %6lld %6lld %9.2f %6d | %9.2f %9.2f %8.0f\n", dtype_name(s.type), arm.c_str(),
               (long long) s.cols, (long long) s.rows, mib, s.n, us, net,
               net > 0 ? mib * 1048576.0 / (net * 1e-6) / 1e9 : 0.0);
        fflush(stdout);
        fu.push_back((double) s.rows);
        fv.push_back(mib);
        fy.push_back(net);
    }
    flush_fit();
    return 0;
}
