// The process-wide verify configuration for an AlphaInfer run: MTP exact/fast, the verify pins, the K clamp and the
// confidence-gate default. Shared by every entry point (apps/alphainfer.cpp, the C API), so a default changed for
// the CLI also applies to AlphaServe.
//
// Call it before Engine::init: most of what it sets are environment switches the ggml backend reads once, into
// statics. It only fills in values the caller has not set (setenv(..., 0)), so an explicit env always wins.
#pragma once

#include "engine/engine.h"

namespace ai {

enum MtpMode {
    kMtpModeDefault = 0,   // exact on MoE models, fast on dense models
    kMtpModeFast    = 1,   // --mtp-fast
    kMtpModeExact   = 2,   // --mtp-exact
};

// Applies the policy to `cfg` (may lower cfg.mtp_k: exact verify is bit-identical only up to K = 3 on the default
// build) and to the process environment. Returns whether the exact verify is on. Safe with mtp_k == 0 (plain): it
// then sets only the plain-relevant defaults. Prints its decisions to stderr, as the app always did.
bool apply_mtp_policy(EngineConfig & cfg, int mode);

// AI_MTP_MODE=fast|exact|default for callers that cannot pass a mode (the C API, whose params struct is shared
// field-for-field with AlphaServe's copy of the header and must not change). Unset or unknown = default.
int mtp_mode_from_env();

}  // namespace ai
