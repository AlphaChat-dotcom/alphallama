// SPDX-License-Identifier: Apache-2.0
// Auto-tune for AlphaInfer 2 (Flash-Next): pick the engine's run settings from measured machine rates instead of
// fixed VRAM classes. A short calibration (PCIe host-to-device bandwidth, DRAM read bandwidth, CPU cache topology)
// is cached per machine; each setting is applied as an environment default (an explicit env var always wins) and
// logged as "[auto-tune] KNOB=value: reason". ALPHA_AUTOTUNE=0 turns it off, ALPHA_AUTOTUNE=remeasure redoes the
// calibration.
#pragma once

#include <cstddef>

namespace alphaserve {

struct autotune_rates {
    double pcie_gbs   = 0;    // pinned host -> device copy, GB/s (0 = not measured)
    double dram_gbs   = 0;    // multi-threaded host read, GB/s
    int    l3_domains = 0;    // CPU last-level cache domains (CCDs on AMD; 1 on most Intel desktop parts)
    bool   cached     = false;
};

// Measure (or read from the per-machine cache) the rates above. Safe to call before the engine exists.
autotune_rates autotune_measure();

// Apply the Flash-Next (qwen4exp) settings for `model_path` on this machine; vram_total is the GPU's memory in bytes.
// Returns false when auto-tune is off (ALPHA_AUTOTUNE=0) and nothing was set.
bool autotune_apply_qwen4exp(const char * model_path, size_t vram_total);

}  // namespace alphaserve
