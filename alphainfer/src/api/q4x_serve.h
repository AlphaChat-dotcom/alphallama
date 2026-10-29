// SPDX-License-Identifier: Apache-2.0
#pragma once

// Q4xServe: the C API's backend for qwen4exp (Qwen3.8-Flash-Next) models -- one stream, doorbell expert offload,
// layer-major prompt prefill, greedy decode with MTP drafts and the optional suffix copy. ai_engine_create picks it
// from the GGUF architecture; every ai_engine_* entry then forwards here (api/ai_engine.cpp).
//
// What the caller must know (the C API header's contract, narrowed):
//   - n_streams must be 1 (the QSA attention state is single-stream); concurrent requests queue in the server.
//   - Decode runs in FAST mode, never exact: verify widths above 8 rows and the doorbell kernels are not bit-identical
//     to plain decode, and the tail rule skips low-ranked missed experts (NLL-neutral in paired evaluation).
//     ai_engine_mtp_info reports exact = 0.
//   - The CPU expert pool pins the thread that calls ai_engine_create as the launch thread: create and decode on
//     the same thread.
//
// Configuration beyond ai_engine_params, read at create (the CLI's names, so a CLI config carries over):
//   AI_Q4X_MTP          the MTP sidecar GGUF; default: the only .gguf in <model dir>/MTP/
//   AI_Q4X_PROFILE      expert residency profile; default: ALPHA_DRE_PROFILE when set
//   AI_Q4X_SPEC_HYBRID  suffix copy: minimum match length (default 3; 0 = off), cost-gated on short matches
//   AI_Q4X_SPEC_COPY_K  suffix copy: drafts per copied window (default 15, at most kDoorbellT - 1)
//   AI_Q4X_SPEC_AGREE   suffix copy only when the MTP's first draft agrees (default 1)
//   AI_Q4X_VRAM_RESERVE_MB  VRAM kept free when expert_slots is auto (-1 or 0)
// and the runtime's own AI_Q4X_* knobs (doorbell, capture, RAM tier and admission default on, as the CLI runs them).

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ai {

class Q4xServe {
public:
    struct Config {
        std::string model_path;
        int64_t     ctx          = 8192;
        int         expert_slots = -1;   // <= 0: auto from the VRAM left
        int         mtp_k        = 0;    // 0 = plain decode
        int         ws_mib       = 0;    // 0 = 640
        bool        verbose      = false;
    };
    // Throws std::runtime_error with the cause on failure.
    explicit Q4xServe(const Config & c);
    ~Q4xServe();

    bool    has_mtp() const;
    int     mtp_k() const;           // drafts per round at most (MTP depth, or the copy window when larger)
    int64_t ctx_cells() const;
    int64_t n_embd() const;
    int64_t free_vram_mib() const;
    int64_t position() const;        // the pending token's position (prompt length after prefill + emitted)
    bool    prefilled() const { return prefilled_; }
    int64_t reused_tokens() const { return reused_; }   // prompt tokens the last prefill took from the kept state

    int32_t prefill(const int32_t * tokens, int n);            // returns the first token
    // prefill_chunk's buffering: append; at is_last prefill the whole prompt and return its first token (else -1)
    int32_t prefill_feed(const int32_t * tokens, int n, bool is_last);
    int32_t decode_plain(int32_t pending);                      // one token
    void    decode_spec(int32_t pending, std::vector<int32_t> & out);   // 1 .. mtp_k() + 1 tokens, the last pending
    void    reset();

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
    bool                  prefilled_ = false;
    int64_t               reused_    = 0;
    int32_t               ck_first_  = -1;   // the checkpointed prompt's first token
    std::vector<int32_t>  feed_;
    bool                  feed_open_ = false;
};

// The GGUF's general.architecture ("" when unreadable); ai_engine_create dispatches on it.
std::string gguf_arch(const std::string & path);

}  // namespace ai
