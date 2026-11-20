// SPDX-License-Identifier: Apache-2.0
// Windows video-memory budget (WDDM). Windows gives each process a budget for local video memory and may withhold part
// of the card for other processes (the desktop, browsers, other GPU apps); cudaMemGetInfo reports the card's free memory,
// not this process's budget. Sizing slots, KV and workspace from the card's free figure can then overshoot what Windows
// lets this process keep resident, and past it Windows pages the process's allocations (much slower) instead of failing.
// wddm_cap_free() lowers a free figure to the budget headroom (budget - this process's current usage), from DXGI's
// QueryVideoMemoryInfo for the adapter with the CUDA device's LUID. AI_WDDM_BUDGET=0 turns it off. Not Windows: no-ops.
// (The Windows headers stay in wddm_budget.cpp: dxgi pulls in rpcndr.h, whose macros must not reach other files.)
#pragma once

#include <cstddef>

namespace ai {

// The bytes this process may still allocate on the CUDA device under its Windows budget; false = not Windows, switched
// off, or the adapter / query is not available.
bool wddm_headroom(int cuda_device, size_t * headroom);

// Lowers *free_b to the budget headroom when that is smaller; says so once per process (prefix `who`).
void wddm_cap_free(int cuda_device, size_t * free_b, const char * who);

}  // namespace ai
