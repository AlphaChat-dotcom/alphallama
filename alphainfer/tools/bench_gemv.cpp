// SPDX-License-Identifier: Apache-2.0
// ai-bench-gemv - one-token projection latency on the model's real weights.
//
// Each case is one projection role across every layer that has it. Each (arm, layer) is recorded
// as its own round and replayed under the engine's capture, cycling through the layers so every
// launch reads weights the L2 cache has not just seen (the model's real access pattern; a single
// matrix replayed in a loop measures L2, not HBM). Inputs alternate every launch and the last
// output is checked against an exact FP64 product, so a replay that skipped work would show.
//
//   ai-bench-gemv -m model.gguf [-n 300]
#include "core/device.h"
#include "core/weights.h"
#include "kernels/kernels.h"
#include "ops/ops.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace ai;

static double fp16(uint16_t h) {
    const int s = h >> 15, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    double v = e == 0 ? std::ldexp((double) m, -24) : std::ldexp((double) (m | 0x400), e - 25);
    return s ? -v : v;
}

// f32 -> fp16, round to nearest even. For the fp16-activation arms. Normal range only is
// exercised (the bench's x is N(0,1)); subnormals and overflow are handled but not relied on.
static uint16_t to_fp16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int32_t  exp  = (int32_t) ((x >> 23) & 0xff) - 127 + 15;
    uint32_t       mant = x & 0x7fffffu;
    if (exp >= 31) {
        return (uint16_t) (sign | 0x7c00u);
    }
    if (exp <= 0) {
        if (exp < -10) {
            return (uint16_t) sign;
        }
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t) (14 - exp);
        uint32_t       h     = mant >> shift;
        const uint32_t rem   = mant & ((1u << shift) - 1u);
        const uint32_t half  = 1u << (shift - 1u);
        if (rem > half || (rem == half && (h & 1u))) ++h;
        return (uint16_t) (sign | h);
    }
    uint32_t h = sign | ((uint32_t) exp << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
    return (uint16_t) h;
}

static std::vector<double> exact_q8(const std::vector<uint8_t> & w, int64_t rows, int64_t cols,
                                    const std::vector<float> & x) {
    std::vector<double> y((size_t) rows, 0.0);
    const int64_t nb = cols / 32;
    for (int64_t r = 0; r < rows; ++r) {
        const uint8_t * row = w.data() + r * nb * 34;
        double acc = 0;
        for (int64_t b = 0; b < nb; ++b) {
            uint16_t h;
            memcpy(&h, row + b * 34, 2);
            const double d = fp16(h);
            for (int j = 0; j < 32; ++j) {
                acc += d * (double) (int8_t) row[b * 34 + 2 + j] * (double) x[(size_t) (b * 32 + j)];
            }
        }
        y[(size_t) r] = acc;
    }
    return y;
}

static double nmse(const std::vector<float> & a, const std::vector<double> & ref) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        num += (a[i] - ref[i]) * (a[i] - ref[i]);
        den += ref[i] * ref[i];
    }
    return num / std::max(den, 1e-300);
}

struct Role {
    const char *     suffix;
    std::vector<int> layers;
};

int main(int argc, char ** argv) {
    std::string model;
    int n = 300;
    int T = 1;   // activation columns per call: 1 = one decode token; 2 = production np=2; k+1 = an MTP verify
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i];
        if (a == "-m") model = argv[i + 1];
        else if (a == "-n") n = atoi(argv[i + 1]);
        else if (a == "-t") T = atoi(argv[i + 1]);
    }
    if (T < 1 || T > 8) {
        fprintf(stderr, "-t must be 1..8 (MMVQ's and the planar kernel's column range)\n");
        return 1;
    }
    std::vector<int> gdn, attn, all;
    for (int l = 0; l < 40; ++l) {
        ((l + 1) % 4 ? gdn : attn).push_back(l);
        all.push_back(l);
    }
    const std::vector<Role> roles = {
        {"attn_qkv.weight", gdn},   {"attn_gate.weight", gdn},  {"ssm_out.weight", gdn},
        {"attn_q.weight", attn},    {"attn_k.weight", attn},    {"attn_output.weight", attn},
        {"ffn_up_shexp.weight", all}, {"ffn_down_shexp.weight", all},
    };
    const auto name_of = [](int l, const char * suffix) {
        return "blk." + std::to_string(l) + "." + suffix;
    };

    op::set_capture(op::Capture::Own);
    Device dev;
    if (!dev.init(0)) {
        return 1;
    }
    WeightSet w;
    const WeightSet::HostFilter keep_on_host = [&](const std::string & name) {
        if (name == "output.weight") {
            return false;
        }
        for (const Role & r : roles) {
            for (int l : r.layers) {
                if (name == name_of(l, r.suffix)) {
                    return false;
                }
            }
        }
        return true;
    };
    if (!w.open(model) || !w.load(dev, 4ULL << 30, false, keep_on_host)) {
        return 1;
    }
    dev.init_workspace(512ULL << 20);
    dev.finalize();

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    printf("T = %d activation column(s) per call\n", T);
    printf("%-24s %-5s %5s %6s %7s %3s | %7s %9s %8s | %s\n", "role", "type", "cols", "rows", "MiB", "L",
           "arm", "us/call", "GB/s", "NMSE vs exact (vs stale input)");
    struct Job { std::string role; std::vector<std::string> names; };
    std::vector<Job> jobs;
    for (const Role & r : roles) {
        Job j{r.suffix, {}};
        for (int l : r.layers) {
            j.names.push_back(name_of(l, r.suffix));
        }
        jobs.push_back(j);
    }
    jobs.push_back({"output.weight", {"output.weight"}});

    for (const Job & job : jobs) {
        const Tensor * w0 = w.find(job.names[0]);
        if (!w0) {
            continue;
        }
        const int64_t cols = w0->ne[0], rows = w0->ne[1];
        const bool q8 = w0->type == Q8_0;
        dev.ws_reset(0);
        Tensor x   = dev.ws_alloc(F32, cols, T);
        Tensor x16 = dev.ws_alloc(F16, cols, T);   // the same activations rounded to fp16
        const size_t after_x = dev.ws_mark();
        std::vector<float> xa((size_t) (cols * T)), xb((size_t) (cols * T));
        for (auto & v : xa) v = nd(rng);
        for (auto & v : xb) v = nd(rng);
        std::vector<uint16_t> xa16(xa.size()), xb16(xb.size());
        for (size_t i = 0; i < xa.size(); ++i) {
            xa16[i] = to_fp16(xa[i]);
            xb16[i] = to_fp16(xb[i]);
        }

        // The planar arms run after "ggml" (the ggml path) and "native", because they repack each layer's bytes
        // in place (same byte count) and the first two arms need GGUF's interleaved layout.
        const bool planar = q8 && kern::gemv_q8_planar_ok(cols);
        std::vector<std::string> arm_names{"ggml"};
        if (q8 && T == 1) arm_names.push_back("native");   // the native prototype is single-token
        if (planar) {
            // fp16 x reading GGUF's own layout -- must run before the planar repack below
            for (int s : {1, 2, 4}) {
                if (kern::gemv_q8_planar_split_ok(cols, s)) {
                    arm_names.push_back("gsplit" + std::to_string(s));
                }
            }
            if (T == 1) arm_names.push_back("planar1");   // reference: shared-x, single column only
            for (int s : {1, 2, 4}) {         // split-K, f32 x straight from L1/L2
                if (kern::gemv_q8_planar_split_ok(cols, s)) {
                    arm_names.push_back("split" + std::to_string(s));
                }
            }
            for (int s : {1, 2, 4}) {         // the same kernel reading fp16 x
                if (kern::gemv_q8_planar_split_ok(cols, s)) {
                    arm_names.push_back("hsplit" + std::to_string(s));
                }
            }
        }
        // Host copies of the original GGUF bytes, taken before the repack. The planar arms' FP64 reference
        // must come from these: computing it from the repacked device bytes would reinterpret codes as
        // scales, and could agree with a kernel that makes the same misreading.
        std::vector<std::vector<uint8_t>> original(job.names.size());
        bool repacked = false;
        for (size_t a = 0; a < arm_names.size(); ++a) {
            const bool is_g      = arm_names[a].rfind("gsplit", 0) == 0;   // fp16 x, GGUF layout
            const bool is_h      = arm_names[a].rfind("hsplit", 0) == 0;   // fp16 activations
            const bool is_split  = arm_names[a].rfind("split", 0) == 0 || is_h;
            const bool is_planar = arm_names[a].rfind("planar", 0) == 0 || is_split;   // all read planar bytes
            const int  digit     = (is_planar || is_g) ? arm_names[a].back() - '0' : 0;
            const int  rpw       = is_split ? -digit : digit;   // negative selects split-K in the launcher
            if (is_planar && !repacked) {
                for (size_t li = 0; li < job.names.size(); ++li) {
                    const Tensor & wt = w.get(job.names[li]);
                    original[li].resize(wt.nbytes());
                    dev.download(wt, original[li].data(), 0, wt.nbytes());
                    std::vector<uint8_t> pl(wt.nbytes());
                    kern::q8_repack_planar(original[li].data(), pl.data(), wt.ne[1], wt.ne[0]);
                    dev.upload(wt, pl.data(), 0, pl.size());
                }
                dev.synchronize();
                repacked = true;
            }
            dev.ws_reset(after_x);   // keep x; this arm records from here
            std::vector<std::unique_ptr<op::Round>> rounds;
            std::vector<Tensor> outs;
            for (const auto & nm : job.names) {
                const Tensor & wt = w.get(nm);
                auto r = std::make_unique<op::Round>(dev);
                r->begin();
                if (is_g) {
                    outs.push_back(op::linear_q8_gguf_f16x(dev, wt, x16, digit));
                } else if (is_planar) {
                    outs.push_back(op::linear_q8_planar(dev, wt, is_h ? x16 : x, rpw));
                } else {
                    outs.push_back(a == 0 ? op::linear(dev, wt, x) : op::linear_q8_native(dev, wt, x, 1));
                }
                r->end();
                rounds.push_back(std::move(r));
            }
            const size_t L = rounds.size();
            // warm and capture every round, then time cycles over the layers
            for (int warm = 0; warm < 3; ++warm) {
                for (auto & r : rounds) {
                    r->launch();
                }
            }
            dev.synchronize();
            std::vector<double> us;
            size_t last = 0;
            int it = 0;
            const int cycles = std::max(1, n / (int) L);
            for (int cyc = 0; cyc < cycles; ++cyc) {
                for (size_t l = 0; l < L; ++l, ++it) {
                    if (is_h || is_g) {
                        const std::vector<uint16_t> & hi = (it % 2) ? xb16 : xa16;
                        dev.upload(x16, hi.data(), 0, hi.size() * 2);
                    } else {
                        const std::vector<float> & xi = (it % 2) ? xb : xa;
                        dev.upload(x, xi.data(), 0, xi.size() * 4);
                    }
                    dev.synchronize();
                    const auto t0 = std::chrono::steady_clock::now();
                    rounds[l]->launch();
                    dev.synchronize();
                    us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
                    last = l;
                }
            }
            const bool last_b = ((it - 1) % 2) != 0;
            std::sort(us.begin(), us.end());
            const double med = us[us.size() / 2];
            std::string acc = "-";
            if (q8) {
                const Tensor & wt = w.get(job.names[last]);
                std::vector<uint8_t> wh(wt.nbytes());
                if (is_planar) {
                    wh = original[last];   // GGUF bytes from before the repack -- never the planar ones
                } else {
                    dev.download(wt, wh.data(), 0, wh.size());
                }
                // FP64 reference per activation column; output column t sits at t*rows.
                const std::vector<float> & xl = last_b ? xb : xa;
                const std::vector<float> & xs = last_b ? xa : xb;
                std::vector<double> ref, stale;
                for (int t = 0; t < T; ++t) {
                    const std::vector<float> ct(xl.begin() + (size_t) t * cols, xl.begin() + (size_t) (t + 1) * cols);
                    const std::vector<float> cs(xs.begin() + (size_t) t * cols, xs.begin() + (size_t) (t + 1) * cols);
                    const auto rt = exact_q8(wh, rows, cols, ct);
                    const auto rs = exact_q8(wh, rows, cols, cs);
                    ref.insert(ref.end(), rt.begin(), rt.end());
                    stale.insert(stale.end(), rs.begin(), rs.end());
                }
                std::vector<float> yh((size_t) (rows * T));
                dev.download(outs[last], yh.data(), 0, yh.size() * 4);
                char b[64];
                snprintf(b, sizeof(b), "%.2e (%.2e)", nmse(yh, ref), nmse(yh, stale));
                acc = b;
            }
            printf("%-24s %-5s %5lld %6lld %7.2f %3zu | %7s %9.1f %8.1f | %s\n", a == 0 ? job.role.c_str() : "",
                   a == 0 ? dtype_name(w0->type) : "", (long long) cols, (long long) rows,
                   w0->nbytes() / 1048576.0, L, arm_names[a].c_str(), med, w0->nbytes() / (med * 1e-6) / 1e9,
                   acc.c_str());
            fflush(stdout);
        }
    }
    return 0;
}
