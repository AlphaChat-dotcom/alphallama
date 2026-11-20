// SPDX-License-Identifier: Apache-2.0
#include "core/wddm_budget.h"

#include <cstdio>
#include <cstdlib>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <dxgi1_4.h>
#include <cuda_runtime.h>
#include <cstring>
#include <mutex>
#pragma comment(lib, "dxgi.lib")
#endif

namespace ai {

bool wddm_headroom(int cuda_device, size_t * headroom) {
#if defined(_WIN32)
    static const bool on = !getenv("AI_WDDM_BUDGET") || atoi(getenv("AI_WDDM_BUDGET")) != 0;
    if (!on || !headroom) return false;
    static std::mutex      mu;
    static IDXGIAdapter3 * ad     = nullptr;
    static int             ad_dev = -2;
    std::lock_guard<std::mutex> lk(mu);
    if (ad_dev != cuda_device) {   // the adapter with the CUDA device's LUID, looked up once per device
        ad_dev = cuda_device;
        if (ad) { ad->Release(); ad = nullptr; }
        cudaDeviceProp p {};
        if (cudaGetDeviceProperties(&p, cuda_device) != cudaSuccess) { (void) cudaGetLastError(); return false; }
        LUID luid;
        memcpy(&luid, p.luid, sizeof luid);
        IDXGIFactory1 * f = nullptr;
        if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void **) &f)) || !f) return false;
        IDXGIAdapter1 * a = nullptr;
        for (UINT i = 0; f->EnumAdapters1(i, &a) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_ADAPTER_DESC1 d;
            const bool match = SUCCEEDED(a->GetDesc1(&d)) && d.AdapterLuid.LowPart == luid.LowPart &&
                               d.AdapterLuid.HighPart == luid.HighPart;
            if (match) a->QueryInterface(__uuidof(IDXGIAdapter3), (void **) &ad);
            a->Release();
            if (match) break;
        }
        f->Release();
    }
    if (!ad) return false;
    DXGI_QUERY_VIDEO_MEMORY_INFO mi {};
    if (FAILED(ad->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mi))) return false;
    *headroom = mi.Budget > mi.CurrentUsage ? (size_t) (mi.Budget - mi.CurrentUsage) : 0;
    return true;
#else
    (void) cuda_device;
    (void) headroom;
    return false;
#endif
}

void wddm_cap_free(int cuda_device, size_t * free_b, const char * who) {
    size_t h = 0;
    if (!free_b || !wddm_headroom(cuda_device, &h) || h >= *free_b) return;
    static bool said = false;
    if (!said) {
        said = true;
        fprintf(stderr, "[%s] Windows video-memory budget: %.2f GiB of headroom for this process, below the card's %.2f GiB "
                        "free -- sizing from the budget (AI_WDDM_BUDGET=0 turns this off)\n",
                who ? who : "ai", h / 1073741824.0, *free_b / 1073741824.0);
    }
    *free_b = h;
}

}  // namespace ai
