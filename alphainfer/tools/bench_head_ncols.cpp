// ai-bench-head-ncols: is the LM head amortised across verify rows, or paid per row?
//
// op::head is a single ggml_mul_mat over [E, n], so the op is batched; the question is whether the kernel
// ggml picks below it reads the weight matrix once per call or once per column. The tool computes the
// implied bandwidth both ways and reports which one stays flat near the hardware ceiling:
//
//     GB/s @1read  = weight_bytes           / time     flat  => amortised   (one read, n free)
//     GB/s @nreads = weight_bytes * ncols   / time     flat  => per column  (n reads)
//
// Two sanity checks run before any verdict:
//   * implied GB/s at ncols=1 must not exceed the device's theoretical peak. The head is ~400 MiB, far past
//     L2, so a figure above peak means work was skipped or cached.
//   * the output must change when the input changes; a replay that did nothing would read as amortised.
#include "core/device.h"
#include "core/tensor.h"
#include "core/weights.h"
#include "ops/ops.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace ai;
using clk = std::chrono::steady_clock;

static double ms_between(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

int main(int argc, char ** argv) {
    std::string model;
    int    reps     = 60;
    int    max_cols = 8;
    double peak_gbs = 936.0;   // RTX 3090 theoretical memory bandwidth
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "-m") model = argv[i + 1];
        else if (a == "-r") reps = atoi(argv[i + 1]);
        else if (a == "-c") max_cols = atoi(argv[i + 1]);
        else if (a == "--peak") peak_gbs = atof(argv[i + 1]);
    }
    if (model.empty()) {
        printf("usage: ai-bench-head-ncols -m model.gguf [-r reps] [-c max_cols] [--peak GBs]\n");
        return 2;
    }

    op::set_capture(op::Capture::Own);
    Device dev;
    if (!dev.init(0)) {
        printf("device init failed\n");
        return 1;
    }
    // Keep everything on the host except the two tensors the head needs, so this costs ~400 MiB of VRAM
    // and can run without stopping a resident service.
    WeightSet w;
    const WeightSet::HostFilter keep_on_host = [](const std::string & name) {
        return !(name == "output.weight" || name == "output_norm.weight");
    };
    if (!w.open(model) || !w.load(dev, 4ULL << 30, false, keep_on_host)) {
        printf("weight load failed\n");
        return 1;
    }
    const Tensor * ow = w.find("output.weight");
    const Tensor * on = w.find("output_norm.weight");
    if (!ow || !on) {
        printf("REFUSING: output.weight or output_norm.weight not found\n");
        return 2;
    }
    const int64_t E = ow->ne[0], V = ow->ne[1];
    const size_t  wb = row_size(ow->type, E) * (size_t) V;
    printf("head: output.weight [%lld, %lld] %s = %.1f MiB   norm %s\n",
           (long long) E, (long long) V, dtype_name(ow->type), wb / 1048576.0, dtype_name(on->type));
    printf("device peak assumed %.0f GB/s; reps %d per column count\n\n", peak_gbs, reps);

    // 256 MiB is ample: the head's output is V x ncols floats (~8 MiB at ncols=8). Kept small so the
    // ~400 MiB of weights plus workspace can coexist with a resident service if needed.
    dev.init_workspace(256ULL << 20);
    dev.finalize();

    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    printf("%5s %10s %12s %12s %10s\n", "ncols", "us/call", "GB/s @1read", "GB/s @nreads", "out-changes");
    std::vector<double> us;
    std::vector<int>    changed;
    for (int n = 1; n <= max_cols; ++n) {
        dev.ws_reset(0);
        Tensor x = dev.ws_alloc(F32, E, n);
        std::vector<float> xa((size_t) (E * n)), xb((size_t) (E * n));
        for (auto & v : xa) v = nd(rng);
        for (auto & v : xb) v = nd(rng) + 3.0f;   // deliberately different distribution

        // correctness/liveness: the result must differ between two inputs, or a no-op replay would
        // read as perfectly amortised
        const size_t mark = dev.ws_mark();
        std::vector<float> ya, yb;
        for (int pass = 0; pass < 2; ++pass) {
            dev.ws_reset(mark);
            dev.upload(x, pass ? xb.data() : xa.data(), 0, (size_t) (E * n) * 4);
            Tensor y = op::head(dev, x, *on, *ow, 1e-6f);
            dev.synchronize();
            std::vector<float> host((size_t) V);
            dev.download(y, host.data(), 0, (size_t) V * 4);   // first column only
            (pass ? yb : ya) = host;
        }
        int diff = 0;
        for (size_t i = 0; i < ya.size() && i < yb.size(); ++i) {
            if (ya[i] != yb[i]) { ++diff; }
        }

        // timed loop: fresh workspace each call, input re-uploaded so nothing is constant-folded
        dev.ws_reset(mark);
        dev.upload(x, xa.data(), 0, (size_t) (E * n) * 4);
        for (int i = 0; i < 5; ++i) {                  // warm
            dev.ws_reset(mark);
            (void) op::head(dev, x, *on, *ow, 1e-6f);
        }
        dev.synchronize();
        const auto t0 = clk::now();
        for (int i = 0; i < reps; ++i) {
            dev.ws_reset(mark);
            (void) op::head(dev, x, *on, *ow, 1e-6f);
        }
        dev.synchronize();
        const auto t1 = clk::now();
        const double per_us = ms_between(t0, t1) * 1000.0 / reps;
        const double gbs1 = wb / (per_us * 1e-6) / 1e9;
        const double gbsn = gbs1 * n;
        us.push_back(per_us);
        changed.push_back(diff);
        printf("%5d %10.1f %12.1f %12.1f %10s\n", n, per_us, gbs1, gbsn,
               diff > 0 ? "yes" : "NO");
    }

    printf("\n---- VERDICT ----\n");
    bool live = true;
    for (size_t i = 0; i < changed.size(); ++i) {
        if (changed[i] == 0) { live = false; }
    }
    if (!live) {
        printf("  FAIL: the output did not change with the input at some column count. The op is not doing\n");
        printf("     the work, and a no-op would read as perfectly amortised. Nothing below is readable.\n");
        return 1;
    }
    printf("  liveness OK: the output changed with the input at every column count.\n");

    const double gbs1_at1 = wb / (us[0] * 1e-6) / 1e9;
    if (gbs1_at1 > peak_gbs) {
        printf("  FAIL: at ncols=1 the implied bandwidth is %.0f GB/s, above the %.0f GB/s peak. The head is\n",
               gbs1_at1, peak_gbs);
        printf("     %.0f MiB so it cannot be cached; this means work was skipped. Not readable.\n",
               wb / 1048576.0);
        return 1;
    }
    printf("  plausibility OK: ncols=1 implies %.0f GB/s against a %.0f GB/s peak (%.0f%%).\n",
           gbs1_at1, peak_gbs, 100 * gbs1_at1 / peak_gbs);

    const size_t last = us.size() - 1;
    const int    nlast = (int) last + 1;
    const double ratio = us[last] / us[0];
    printf("\n  time at ncols=%d is %.2fx the time at ncols=1 (per-column would be %.2fx, amortised 1.00x)\n",
           nlast, ratio, (double) nlast);
    const double d_per_col = (us[last] - us[0]) / (double) (nlast - 1);
    printf("  marginal cost per extra column: %.1f us (a full extra weight read would be %.1f us)\n",
           d_per_col, us[0]);
    if (ratio > 1.0 + 0.6 * (nlast - 1)) {
        printf("\n  => the head is paid per column: the per-row-head mechanism is confirmed, and\n");
        printf("     amortising the verify head would cut the verify cost.\n");
    } else if (ratio < 1.25) {
        printf("\n  => the head is amortised: the per-row-head hypothesis does not hold, and the\n");
        printf("     verify's excess cost is somewhere else (per-row attention is the next candidate).\n");
    } else {
        printf("\n  => partial: neither fully amortised nor fully per-column: the marginal column costs\n");
        printf("     %.0f%% of a full weight read. Size the lever from that fraction, not from the\n",
               100 * d_per_col / us[0]);
        printf("     full-head assumption, which would overstate it.\n");
    }
    return 0;
}
