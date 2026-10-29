// SPDX-License-Identifier: Apache-2.0
// ai-doorbell-bench: the GPU <-> CPU handshake the offload window is built on.
//
// Per "layer": a kernel rings a doorbell in mapped host memory (seq = i), a CPU worker spinning on it answers
// (writes done = i and a payload), and a device spin kernel waits for done >= i before the next layer. This is the
// pattern of the DRE path in ggml-cuda (dre_signal_k / dre_wait_fill_k): no driver synchronisation inside the
// window. Measures, for L layers:
//   (a) the handshake inside one captured CUDA graph, replayed, us per handshake;
//   (b) the same chain launched directly (no graph);
//   (c) a host round trip per layer: cudaStreamSynchronize + host work + relaunch.
// Pass: (a) completes (spin kernels are legal under capture), and its per-handshake cost << (c).
//
//   ai-doorbell-bench [layers=48] [reps=200] [worker_cpu=-1]
#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

struct Bell {
    volatile unsigned * seq;    // device -> host request counter
    volatile unsigned * done;   // host -> device answer counter
    volatile float *    pay;    // host -> device payload (one float per layer)
};

__global__ void ring_k(Bell b, const unsigned * base, int layer, float * x) {
    const unsigned s = *base + (unsigned) layer + 1;
    __threadfence_system();
    *b.seq = s;
    __threadfence_system();
    (void) x;
}
__global__ void wait_k(Bell b, const unsigned * base, int layer, float * x) {
    const unsigned s = *base + (unsigned) layer + 1;
    while (*b.done < s) { __nanosleep(32); }
    __threadfence_system();
    x[layer] = b.pay[layer];
}
__global__ void bump_k(unsigned * base, int layers) { *base += (unsigned) layers; }

int main(int argc, char ** argv) {
    const int layers = argc > 1 ? atoi(argv[1]) : 48;
    const int reps   = argc > 2 ? atoi(argv[2]) : 200;
    const int wcpu   = argc > 3 ? atoi(argv[3]) : -1;
    CK(cudaSetDeviceFlags(cudaDeviceMapHost));
    Bell hb{}, db{};
    unsigned * hseq; unsigned * hdone; float * hpay;
    CK(cudaHostAlloc(&hseq, 64, cudaHostAllocMapped));
    CK(cudaHostAlloc(&hdone, 64, cudaHostAllocMapped));
    CK(cudaHostAlloc(&hpay, sizeof(float) * (size_t) layers, cudaHostAllocMapped));
    *hseq = 0; *hdone = 0;
    hb = { hseq, hdone, hpay };
    unsigned * dseq; unsigned * ddone; float * dpay;
    CK(cudaHostGetDevicePointer((void **) &dseq, hseq, 0));
    CK(cudaHostGetDevicePointer((void **) &ddone, hdone, 0));
    CK(cudaHostGetDevicePointer((void **) &dpay, hpay, 0));
    db = { dseq, ddone, dpay };
    unsigned * dbase; float * dx;
    CK(cudaMalloc(&dbase, sizeof(unsigned)));
    CK(cudaMemset(dbase, 0, sizeof(unsigned)));
    CK(cudaMalloc(&dx, sizeof(float) * (size_t) layers));
    cudaStream_t st;
    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));

    // CPU worker: answers every ring by copying seq into done (plus a payload), spinning
    std::atomic<bool> stop{ false };
    std::atomic<unsigned long long> answered{ 0 };
    std::thread worker([&] {
#if defined(__linux__)
        if (wcpu >= 0) { cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(wcpu, &cs); pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs); }
#endif
        unsigned last = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            const unsigned s = *hb.seq;
            if (s != last) {
                hpay[(s - 1) % (unsigned) layers] = (float) s;
                std::atomic_thread_fence(std::memory_order_seq_cst);
                *hb.done = s;
                last = s;
                answered.fetch_add(1, std::memory_order_relaxed);
            }
        }
    });

    auto chain = [&](cudaStream_t s) {
        for (int l = 0; l < layers; ++l) {
            ring_k<<<1, 1, 0, s>>>(db, dbase, l, dx);
            wait_k<<<1, 1, 0, s>>>(db, dbase, l, dx);
        }
        bump_k<<<1, 1, 0, s>>>(dbase, layers);
    };
    using clk = std::chrono::steady_clock;
    auto us = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::micro>(b - a).count(); };

    // (b) direct launches
    chain(st); CK(cudaStreamSynchronize(st));   // warm
    auto t0 = clk::now();
    for (int r = 0; r < reps; ++r) chain(st);
    CK(cudaStreamSynchronize(st));
    const double direct = us(t0, clk::now()) / (reps * layers);

    // (a) one captured graph per window, replayed
    cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
    chain(st);
    CK(cudaStreamEndCapture(st, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));
    CK(cudaGraphLaunch(ge, st)); CK(cudaStreamSynchronize(st));   // warm
    t0 = clk::now();
    for (int r = 0; r < reps; ++r) CK(cudaGraphLaunch(ge, st));
    CK(cudaStreamSynchronize(st));
    const double graph = us(t0, clk::now()) / (reps * layers);

    // (c) a host round trip per layer (launch, synchronize, host work, relaunch)
    t0 = clk::now();
    for (int r = 0; r < reps; ++r) {
        for (int l = 0; l < layers; ++l) {
            bump_k<<<1, 1, 0, st>>>(dbase, 0);
            CK(cudaStreamSynchronize(st));
            hpay[l] = (float) l;
        }
    }
    const double v1 = us(t0, clk::now()) / (reps * layers);

    stop = true;
    worker.join();
    printf("doorbell handshake, %d layers x %d reps (worker cpu %d), answered %llu:\n", layers, reps, wcpu,
           (unsigned long long) answered.load());
    printf("  (a) captured graph, replayed : %7.2f us per handshake (%.3f ms per %d-layer window)\n", graph,
           graph * layers / 1000.0, layers);
    printf("  (b) direct launches          : %7.2f us per handshake\n", direct);
    printf("  (c) sync + relaunch (v1)     : %7.2f us per layer\n", v1);
    return 0;
}
