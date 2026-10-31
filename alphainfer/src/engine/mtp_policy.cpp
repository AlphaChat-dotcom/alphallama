// SPDX-License-Identifier: Apache-2.0
// See engine/mtp_policy.h. Shared by the C API and the CLI so both apply the same verify configuration.
#include "engine/mtp_policy.h"
#include "kernels/kernels.h"

#include "core/weights.h"

#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <string>
#include <cstring>

namespace ai {

int mtp_mode_from_env() {
    const char * e = getenv("AI_MTP_MODE");
    if (e == nullptr) return kMtpModeDefault;
    if (strcmp(e, "fast") == 0) return kMtpModeFast;
    if (strcmp(e, "exact") == 0) return kMtpModeExact;
    return kMtpModeDefault;
}

bool apply_mtp_policy(EngineConfig & cfg, int mode) {
    const bool mtp_exact = mode == kMtpModeExact;
    const bool mtp_fast  = mode == kMtpModeFast;
    // A verify row must be bit-identical to the plain decode of the same token, or MTP silently
    // changes the output. ggml's flash attention picks its tile geometry and KV split from Q->ne[1],
    // so a 2..16-row verify would run a different kernel than a 1-row decode; ALPHA_FA_BI_MAXQ makes
    // those widths decide as a single query would. MTP only (a wide co-batched decode keeps its batch
    // kernel), and a value the caller chose is never overridden.
    // ggml's small-M MMQ tier is tuned for dense models and turns itself off once graph_compute sees a
    // MUL_MAT_ID, which this engine never calls, so a MoE model would get the dense tuning at 4..8
    // columns. Only an MTP verify is that wide.
    // Dense models with --mtp (and without --mtp-exact) take the tier instead: a dense verify at 4..8
    // columns on MMVQ is ALU-bound on Ampere (1.5-2x a 1-column call per GEMV), and MMQ makes the width
    // nearly flat. It is not bit-identical to plain decode (MMQ accumulates in another order); --mtp-exact
    // keeps that guarantee.
    // Exact replay: MTP verify rows bit-identical to plain decode of the same token. This needs the expert
    // MUL_MAT_ID batched with the 1-token geometry (ALPHA_MMID_SPLIT) and the LM head kept off tensor-core
    // MMQ at >= 4 columns (ALPHA_OUTPUT_MMQ). Default for MoE models; --mtp-fast selects the fast path.
    // Dense models keep the fast small-M tiers by default and get exactness with --mtp-exact. Expert
    // offload has no split path, so it stays fast.
    const bool is_dense = cfg.mtp_k > 0 && gguf_is_dense(cfg.model_path);
    bool mtp_exact_on   = cfg.mtp_k > 0 && (mtp_exact || (!mtp_fast && !is_dense));
    // AI_WIDE_EXACT (default on for dense models in exact mode on a batch-invariant build): plain decode and the
    // exact verify run the width-invariant tensor-core kernels (q4_K/q5_K/iq4_xs on the int8 kernel with per-32
    // power-of-two activation sub-scales, q8_0 on the f16-activation kernel, q6_K and everything else on MMVQ pinned
    // to the 1-row shape), so the exact verify can widen to K = 7 (8 rows) and use n-gram lookup. Exact means
    // bit-identical to plain decode of this build; output differs from the MMVQ build by rounding. Plain dense decode
    // takes the same kernels, at a small cost per GEMV at M = 1. MoE models are unchanged (their wide exact verify is
    // expert-bound). Requires ggml built with the MMVQ pin (-DAI_BI_FORK=ON with -DALPHA_BATCH_INVARIANT_MMVQ=1);
    // without it exact above 4 rows is not exact. AI_WIDE_EXACT=0 disables; AI_EXACT_K (default 7) sets the widened K.
#ifdef AI_BI_FORK
    if (gguf_is_dense(cfg.model_path) && cfg.dflash2_path.empty() && cfg.expert_slots == 0 &&
        (cfg.mtp_k == 0 || mtp_exact_on) && !(getenv("AI_WIDE_EXACT") && atoi(getenv("AI_WIDE_EXACT")) == 0)) {
        setenv("ALPHA_SMALLM_TC", "46", 0);
        setenv("ALPHA_SMALLM_TC_MINCOLS", "1", 0);
        setenv("ALPHA_SMALLM_A16", "1", 0);
        setenv("ALPHA_SMALLM_TC_E", "1", 0);
        setenv("AI_EXACT_ANY_K", "1", 0);
        setenv("ALPHA_FA_BI_GRAIN", "8", 0);
        setenv("AI_GRAPH_MAX", "32", 0);
        if (cfg.mtp_k > 0) {
            const int k = getenv("AI_EXACT_K") ? std::max(1, std::min(7, atoi(getenv("AI_EXACT_K")))) : 7;
            if (cfg.mtp_k < k) cfg.mtp_k = k;
            setenv("AI_LOOKUP", "1", 0);
            setenv("AI_LOOKUP_NMIN", "4", 0);
        }
        fprintf(stderr, "[ai] AI_WIDE_EXACT: dense %s on the width-invariant TC kernels (ALPHA_SMALLM_TC=%s, TC_E=%s)%s\n",
                cfg.mtp_k > 0 ? "exact MTP" : "plain decode", getenv("ALPHA_SMALLM_TC"), getenv("ALPHA_SMALLM_TC_E"),
                cfg.mtp_k > 0 ? (std::string(", K = ") + std::to_string(cfg.mtp_k) + " + lookup").c_str() : "");
    }
#endif
    // AI_LOOKUP_ARM: per-model default for shadow-armed lookup. Dense models arm at 8 matched tokens; a MoE model's
    // wide rounds route to more distinct experts, so it arms at 32. An explicit value wins.
    if (getenv("AI_LOOKUP") && atoi(getenv("AI_LOOKUP")) != 0) {
        setenv("AI_LOOKUP_ARM", gguf_is_dense(cfg.model_path) ? "8" : "32", 0);
        fprintf(stderr, "[ai] lookup: shadow-armed at %s matched tokens (%s model)\n", getenv("AI_LOOKUP_ARM"),
                gguf_is_dense(cfg.model_path) ? "dense" : "MoE");
    }
    if (mtp_exact_on && cfg.expert_slots > 0) {
        fprintf(stderr, "[ai] MTP: exact verify is not available with expert offload (no pool-direct split path); "
                        "running the fast verify\n");
        mtp_exact_on = false;
    }
    if (mtp_exact_on && !cfg.dflash2_path.empty() && !(getenv("AI_EXACT_ANY_K") && atoi(getenv("AI_EXACT_ANY_K")))) {
        // A DFlash verify is block-size rows (8 for the v1 drafter); exact above 4 rows needs ggml built with
        // -DALPHA_BATCH_INVARIANT_MMVQ=1 and AI_EXACT_ANY_K=1 (see the K <= 3 note below).
        fprintf(stderr, "[ai] DFlash: exact verify needs the batch-invariant build + AI_EXACT_ANY_K=1; running the FAST "
                        "verify (output can drift from plain decode)\n");
        mtp_exact_on = false;
    }
    // Plain decode of a MoE model takes the same width-invariance pins as the exact verify, so a plain run and an
    // exact-MTP run share one numeric configuration. The pins act at 2..16 rows, which plain decode meets in a
    // prefill's last chunk (prompt length mod chunk in 2..16) and in multi-stream rounds; without them plain and
    // exact diverge from the first decoded position. Single-stream decode (M = 1) is unaffected.
    // AI_PLAIN_EXACT=0 disables.
    // Dense models take them too from Ada on (cc >= 8.9; on Blackwell the exact verify also takes the pins below,
    // ALPHA_FA_BI_PB among them). On sm_86 the dense plain path already matches exact without them; on sm_89 it does
    // not: a plain prefill's last chunk of 2..16 rows ran another attention geometry than the exact verify, and the
    // 27B diverged from exact at a near-tie on 1 prompt of 25 (identical with the pins, same plain speed).
    const bool plain_exact = cfg.mtp_k == 0 && cfg.expert_slots == 0 && cfg.dflash2_path.empty() &&
                             !(getenv("AI_PLAIN_EXACT") && atoi(getenv("AI_PLAIN_EXACT")) == 0) &&
                             (!gguf_is_dense(cfg.model_path) || kern::device_cc() >= 89);
    if (plain_exact) {
        setenv("ALPHA_FA_BI_MAXQ", "16", 0);
        setenv("ALPHA_MMVF_BI_MAXQ", "16", 0);
        setenv("ALPHA_MMID_SPLIT", "16", 0);
    }
    // ALPHA_FA_BI_PB: exact MTP and the exact-pinned plain decode on Blackwell (sm_12x) use a constant flash-attention
    // KV split (4). There decode takes ggml's vector kernel, whose split count follows the attention envelope, so a row
    // computed by plain decode (one envelope) and by a verify round (the next bucket) would differ near every bucket
    // boundary. The cost is within ~1-2%. Ampere (sm_86) takes another kernel and is untouched.
    if ((mtp_exact_on || plain_exact) && kern::device_cc_major() >= 12) {
        setenv("ALPHA_FA_BI_PB", "4", 0);
    }
    // ALPHA_GDN_TF32_1X: ggml's chunked GDN prefill scan multiplies in plain TF32 instead of 3xTF32 (10-bit mantissa
    // products, fp32 accumulation). Prefill only: decode and verify take other GDN paths, so MTP == plain still holds
    // bitwise. Paired multilingual perplexity shows no measurable quality change. MoE models only;
    // ALPHA_GDN_TF32_1X=0 restores 3xTF32.
    if (!gguf_is_dense(cfg.model_path)) {
        setenv("ALPHA_GDN_TF32_1X", "1", 0);
    }
    const bool dense_mmq_verify = cfg.mtp_k > 0 && !mtp_exact_on && is_dense;
    setenv("ALPHA_SMALLM_MMQ", dense_mmq_verify ? "8" : "0", 0);
    // ALPHA_SMALLM_TC: the register-dequant tensor-core kernel in front of that tier (q4_K/q5_K/iq4_xs at 2..8
    // columns; everything else still takes the tier). Column-position invariant, so MTP and DFlash2 emit identical
    // text. Same scope as the tier: dense, not --mtp-exact.
    // AI_Q8_TC (opt-in): Q8_0 dense projections on the same kernel at 1..8 columns, plain decode included
    // (MINCOLS=1), so exact MTP stays bit-identical to plain. Faster at M = 4..8, slower at M = 1.
    // Not with the dense MMQ tier above (that path owns ALPHA_SMALLM_TC on dense models).
    // AI_A16: the f16-activation tensor-core kernels for Q8_0 projections and the Q6_K LM head at 1..8 columns,
    // plain decode included: ALPHA_SMALLM_TC=48 (q8_0|q6_K), MINCOLS=1, ALPHA_SMALLM_A16=1. On by default for
    // cc >= 12, where it is a few percent faster per round than MMVQ and more accurate. Off below cc 12: on sm_86
    // MMVQ is at the bandwidth roofline and A16 is slower. AI_A16=0/1 overrides either way. Applied to exact and
    // plain alike (exact == plain needs the same kernels).
    const char * a16all = getenv("AI_A16");
    const bool a16_on = a16all ? atoi(a16all) != 0 : kern::device_cc_major() >= 12;
    if (a16_on && !dense_mmq_verify) {
        setenv("ALPHA_SMALLM_TC", "48", 0);
        setenv("ALPHA_SMALLM_TC_MINCOLS", "1", 0);
        setenv("ALPHA_SMALLM_A16", "1", 0);
        fprintf(stderr, "[ai] AI_A16: Q8_0 projections + Q6_K head on the f16-activation tensor-core kernels at 1..8 "
                        "columns (ALPHA_SMALLM_TC=%s, A16=%s)\n", getenv("ALPHA_SMALLM_TC"), getenv("ALPHA_SMALLM_A16"));
    }
    // ALPHA_FA_KVMAX_SMALL=0 on cc >= 12 MoE: skip flash_attn_mask_to_KV_max at decode / verify widths. It is a
    // one-CTA backward scan of the mask, and under the exact stream-K split the blocks past the live length run in
    // parallel anyway. Exact by construction (masked tiles add 0). sm_86 and dense models keep the scan. An explicit
    // ALPHA_FA_KVMAX_SMALL wins.
    if (kern::device_cc_major() >= 12 && !gguf_is_dense(cfg.model_path)) {
        setenv("ALPHA_FA_KVMAX_SMALL", "0", 0);
    }
    const char * q8tc = getenv("AI_Q8_TC");
    if (q8tc && atoi(q8tc) != 0 && !dense_mmq_verify) {
        setenv("ALPHA_SMALLM_TC", "32", 0);
        setenv("ALPHA_SMALLM_TC_MINCOLS", "1", 0);
        fprintf(stderr, "[ai] AI_Q8_TC: Q8_0 projections on the tensor-core small-M kernel at 1..8 columns "
                        "(ALPHA_SMALLM_TC=%s, MINCOLS=%s)\n", getenv("ALPHA_SMALLM_TC"), getenv("ALPHA_SMALLM_TC_MINCOLS"));
    }
    setenv("ALPHA_SMALLM_TC", dense_mmq_verify ? "1" : "0", 0);
    if (dense_mmq_verify) {
        fprintf(stderr, "[ai] dense model + MTP: verify matmuls on the small-M MMQ tier (ALPHA_SMALLM_MMQ=%s); "
                        "--mtp-exact for bit-identical-to-plain output\n", getenv("ALPHA_SMALLM_MMQ"));
    }
    // The LM head alone on tensor-core MMQ at 4..8 columns (the verify head; plain decode is M=1 and never
    // takes it). Needs the head tensor named "output.weight" (ops.cpp head()).
    setenv("ALPHA_OUTPUT_MMQ", (mtp_exact_on || plain_exact) ? "0" : "8", 0);
    cfg.exact_verify = mtp_exact_on;
    // Verify width cap with 2+ streams. Exact: always (more than 4 columns is not width-invariant). Fast: MoE only,
    // where a wide 2-stream verify reads ~2x the distinct experts. On a dense model a verify column is cheap and the
    // cap only throws accepted tokens away. AI_MTP_BCAP=0|1 overrides in fast mode.
    {
        const char * bc  = getenv("AI_MTP_BCAP");
        const bool   moe = !gguf_is_dense(cfg.model_path);
        cfg.stream_cap   = cfg.mtp_k > 0 && (mtp_exact_on || (bc ? atoi(bc) != 0 : moe));
    }
    // The staging registry resets once per graph-evaluation pass. In llama.cpp a pass is a token; here it is a
    // span and a round is ~41 of them, with the producer and its consumers in different spans, so the backend's
    // reset would drop every entry before it could be used. Host-owned lifetime leaves the clear() to the engine,
    // which calls it once per round.
    setenv("ALPHA_Q8_STAGE_HOST_LIFETIME", "1", 0);
    // AI_F_MIN: multi-slot engines (servers) take decode envelopes from 16384 up, 3 envelope buckets instead of 9, so
    // the captured-round working set fits the 16-profile graph budget. A lone short request pays a little for the
    // larger envelope. Single-stream engines keep the full ladder. On cc >= 12 the floor costs short requests much
    // more, so those GPUs keep the full ladder. AI_F_MIN=0 restores it; an explicit AI_F_MIN wins on every GPU.
    if (cfg.max_batch > 1) {
        const bool explicit_env = getenv("AI_F_MIN") != nullptr;
        const bool full_ladder  = kern::device_cc_major() >= 12;
        setenv("AI_F_MIN", full_ladder ? "0" : "16384", 0);
        fprintf(stderr, "[ai] envelope floor AI_F_MIN=%s (%s)\n", getenv("AI_F_MIN"),
                explicit_env ? "explicit" : full_ladder ? "cc >= 12: full ladder" : "multi-slot graph-budget floor");
    }
    // ALPHA_FA_SK_TPB: exact MTP and plain decode below Blackwell use a constant 32 KV tiles per stream-K block in the
    // MMA flash-attention kernel, which sm_86 decode takes (GQA, f16 KV). Its stock block count is capped by the SM
    // count, so above the cap the next envelope moves every block boundary; an exact verify round reaches the next
    // envelope a few positions before plain decode, so exact != plain from the first crossing on. A constant tile count
    // per block fixes this at negligible cost (TPB=1 would be much slower: one-tile blocks lose the prefetch overlap and
    // the fixup overflows L2). The kernel needs >= 2 blocks per tile at every envelope (one block normalises in-kernel,
    // more in the fixup), i.e. F >= 64 * 2 * 32 = 4096, so the envelope floor rises to 4096 where it is lower.
    // Plain decode keys on "no drafter" here, not plain_exact: the dense plain path on cc < 12 takes no pins but must
    // take this, or plain and exact split the KV differently. AI_EXACT_SK=0 opts out.
    // Blackwell (cc >= 12) has two causes of exact != plain at envelope crossings: the stream-K split above, and the
    // f16 decode kernel choice (vector below K = 8192, MMA above for gqa > 4). Per model:
    //   MoE:   ALPHA_FA_KSEL=1 (always MMA), TPB 8, floor 4096 (TPB 32 leaves models with few KV heads short of
    //          blocks; a floor of 8192 instead of the pin costs short requests).
    //   dense: TPB 32, floor 8192 (decode never below K = 8192, so never the vector kernel).
    // ALPHA_FA_KSEL=1 is also set below Blackwell: it acts only for cc >= Ada (inert on sm_86), and Ada has the same
    // vector -> MMA switch.
    const bool sk_tpb = (mtp_exact_on || (cfg.mtp_k == 0 && cfg.dflash2_path.empty())) &&
                        !(getenv("AI_EXACT_SK") && atoi(getenv("AI_EXACT_SK")) == 0);
    if (sk_tpb) {
        const bool bw    = kern::device_cc_major() >= 12;
        const bool dense = gguf_is_dense(cfg.model_path);
        const char * tpb   = bw && !dense ? "8" : "32";
        const long long fl = bw && dense ? 8192 : 4096;
        setenv("ALPHA_FA_SK_TPB", tpb, 0);
        if (!(bw && dense)) setenv("ALPHA_FA_KSEL", "1", 0);
        // AI_FA_Q_REGION_SPLIT: the same envelope dependence exists in AlphaInfer's own q8_0/q4_0 decode attention
        // (AI_FA_Q, sm_8x), which a quantised KV cache takes instead of ggml's flash attention, and TPB does not reach
        // it. Sized from the KV region instead (ops.cpp), the split is one constant per engine.
        setenv("AI_FA_Q_REGION_SPLIT", "1", 0);
        // ALPHA_FA_BI_GRAIN: exact MTP and plain decode take one 4-row flash-attention tile at 1..4 query rows, so a
        // K = 3 verify's 4 rows share each KV load instead of re-reading the KV once per row (the per-row 1-query tile of
        // ALPHA_FA_BI_MAXQ). A constant, not K + 1: plain and exact are separate engines and must pick the same tile.
        // The gain grows with context. Exact above K = 3 (AI_EXACT_ANY_K) exceeds the grain and is not covered.
        setenv("ALPHA_FA_BI_GRAIN", "4", 0);
        const char * fm = getenv("AI_F_MIN");
        if (!fm || atoll(fm) < fl) {
            char buf[32];
            snprintf(buf, sizeof buf, "%lld", fl);
            if (fm) fprintf(stderr, "[ai] AI_F_MIN=%s raised to %s for the constant stream-K split (exact == plain)\n", fm, buf);
            setenv("AI_F_MIN", buf, 1);
        }
        fprintf(stderr, "[ai] stream-K split ALPHA_FA_SK_TPB=%s, kernel pin ALPHA_FA_KSEL=%s, envelope floor AI_F_MIN=%s "
                        "(exact == plain across envelopes)\n",
                getenv("ALPHA_FA_SK_TPB"), getenv("ALPHA_FA_KSEL") ? getenv("ALPHA_FA_KSEL") : "0", getenv("AI_F_MIN"));
    }
    if (cfg.mtp_k > 0) {
        // A verify row must be bit-identical to the plain decode of the same token or MTP silently
        // changes the output. Flash attention is the cheap half (~1%).
        // AI_FAST_UNPIN (default 1), fast mode only: drop this pin and the MMVF one below. The pin gives each verify row
        // its own 1-query FA tile, so every row streams the whole KV envelope; fast mode does not promise MTP == plain,
        // and paired multilingual NLL shows no quality change. Faster at long context. Applies to DFlash2 too (a fast
        // verify). AI_FAST_UNPIN=0 restores the pins.
        static const bool fast_unpin = getenv("AI_FAST_UNPIN") ? atoi(getenv("AI_FAST_UNPIN")) != 0 : true;
        const char *      bi_pin     = (fast_unpin && !mtp_exact_on) ? "0" : "16";
        if (fast_unpin && !mtp_exact_on) {
            fprintf(stderr, "[ai] AI_FAST_UNPIN: fast MTP without the FA/MMVF batch-invariance pins\n");
        }
        setenv("ALPHA_FA_BI_MAXQ", bi_pin, 0);
        // ALPHA_MMVF_BI_MAXQ, the float half: ggml_cuda_should_use_mmvf keeps F32 src0 on the mmvf GEMV only for
        // ne11 <= 3 on Ampere, so a 4-row verify would drop to the cuBLAS GEMM, which tiles per M. The model's only
        // F32 matmuls are ssm_alpha/ssm_beta at the top of every DeltaNet block. mmvf's K-reduction is
        // block_size = f(K) alone, so pinning 2..8 rows to the 1-row decision makes it invariant.
        // Set explicitly because ggml's own default keys off ALPHA_MTP, which this engine never sets.
        setenv("ALPHA_MMVF_BI_MAXQ", bi_pin, 0);
        if (mtp_exact_on) {
            // The expert MUL_MAT_ID as one launch with the tokens on the sample axis, keeping the 1-token column
            // geometry.
            setenv("ALPHA_MMID_SPLIT", "16", 0);
            // Exact up to K = 3 (4 verify rows). At 5+ rows ggml's MMVQ calc_nwarps drops from 4 to 2 warps and its
            // K-reduction order changes with the width; the pin for that is compile-time only
            // (-DALPHA_BATCH_INVARIANT_MMVQ=1). K = 3 is close to the speed optimum for MoE models anyway.
            // AI_EXACT_ANY_K=1 declares ggml was built with that pin (MMVQ's warp split fixed to the 1-row shape at
            // every width), so the K <= 3 bound does not apply.
            static const bool any_k = getenv("AI_EXACT_ANY_K") && atoi(getenv("AI_EXACT_ANY_K")) != 0;
            if (cfg.mtp_k > 3 && !any_k) {
                fprintf(stderr, "[ai] MTP: exact verify is bit-identical up to K = 3; using K = 3 (--mtp-fast for K = %d)\n",
                        cfg.mtp_k);
                cfg.mtp_k = 3;
            }
            fprintf(stderr, "[ai] MTP: EXACT verify -- every verify row is bit-identical to plain decode of the same "
                            "token (expert split + LM head off MMQ)%s\n", mtp_exact ? "" : "; --mtp-fast for the old path");
            // AI_MTP_GATE, default 0.3 in exact mode: verify only the leading drafts whose cumulative draft confidence
            // stays >= 0.3. Every width <= 4 is exact, so the guarantee is unchanged. AI_MTP_GATE=0 turns it off.
            setenv("AI_MTP_GATE", "0.3", 0);
            if (atof(getenv("AI_MTP_GATE")) > 0.0) {
                fprintf(stderr, "[ai] MTP: confidence-gated verify width (AI_MTP_GATE=%s; 0 = always verify K)\n",
                        getenv("AI_MTP_GATE"));
            }
        } else if (mtp_fast || is_dense) {
            fprintf(stderr, "[ai] MTP: FAST verify -- output can drift from plain decode over long generations%s\n",
                    is_dense && !mtp_fast ? " (dense default; --mtp-exact for bit-identical)" : "");
        }
        // AI_MTP_GATE, default 0.2 in fast MTP mode on MoE models. Not on dense models: their fast verify is nearly
        // flat in width, so a gate has nothing to save. AI_MTP_GATE=0 turns it off.
        if (!mtp_exact_on && !is_dense && cfg.dflash2_path.empty()) {
            setenv("AI_MTP_GATE", "0.2", 0);
            if (atof(getenv("AI_MTP_GATE")) > 0.0) {
                fprintf(stderr, "[ai] MTP: confidence-gated verify width (AI_MTP_GATE=%s; 0 = always verify K)\n",
                        getenv("AI_MTP_GATE"));
            }
        }
    }
    return mtp_exact_on;
}

}  // namespace ai
