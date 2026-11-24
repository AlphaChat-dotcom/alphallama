#include "models.h"

#include "llama-impl.h"
#include "llama-kv-cache-dsv41.h"
#include "llama-engram.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>

// ggml.c (ALPHA): ggml_get_rows with an explicit result type
extern "C" struct ggml_tensor * ggml_get_rows_t(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, enum ggml_type type);

// DeepSeek-V4.1-Flash (arch deepseek41), text only.
//
// Hand port of upstream llama.cpp src/models/deepseek41.cpp (and the parts of deepseek4.cpp it builds on) onto this
// fork's graph helpers and its own memory module (llama-kv-cache-dsv41.h). The math follows upstream op for op:
// single-pass mHC (the pre-mix that collapses the streams lags one sublayer, no hc_head), attention over the raw
// sliding window plus the compressed rows of the layer's KV source selected by the top-k of its index source (with
// the two-level candidate block selection), Engram n-gram embeddings gated into the streams, MoE with a shared expert.
//
// Not ported: the fused ops (lightning indexer, top-k mask, hc pre / comb / post), the decode gather path (a decode
// step attends over window + all compressed rows with the unselected ones masked, as upstream does for large
// batches), image (embedding) inputs, the DSpark layer-input taps, control vectors.
//
// ALPHA_DSV41_HC_FUSED=1 uses this fork's DeepSeek-V4 mixer ops (one Sinkhorn kernel) instead of the op chain.

static constexpr int64_t DSV41_ENGRAM_DEAD = -1;

static bool dsv41_hc_fused() {
    static const bool on = [] {
        // default on: this fork's fused DeepSeek-V4 mixer ops, checked on CUDA (f45: the 8230-token code word and the
        // 2K summary right; the MoE layers of the 8K prompt 17.1 -> 15.4 s)
        const char * e = getenv("ALPHA_DSV41_HC_FUSED");
        return e ? atoi(e) != 0 : true;
    }();
    return on;
}

//
// Engram n-gram hash (engram.py NgramHashState.forward)
//

int llama_model::engram_index(int il) const {
    if (hparams.ple_n_heads == 0 || !hparams.is_ple(il)) {
        return -1;
    }
    int e = 0;
    for (int j = 0; j < il; ++j) {
        e += hparams.is_ple(j);
    }
    return e;
}

void llama_model::engram_hash(int e, const llama_ubatch & ubatch, const std::vector<llama_token> & prev, std::vector<int32_t> & out) const {
    const int64_t n_gram = hparams.ple_ngram_size;
    const int64_t n_prev = n_gram - 1;
    const int64_t n_head = hparams.ple_heads_per_ngram;
    const int64_t n_cols = hparams.ple_n_heads;

    const uint64_t * mult   = hparams.ple_layer_multipliers.data() + (size_t) e*n_gram;
    const uint32_t * primes = hparams.ple_head_vocab_sizes.data()  + (size_t) e*n_cols;
    const int64_t    pad    = engram_token_map[hparams.engram_pad_id];
    const uint32_t * offs   = hparams.ple_head_offsets.data()      + (size_t) e*n_cols;

    const auto cid = [&](llama_token t) -> int64_t {
        return t < 0 || t >= (llama_token) engram_token_map.size() ? DSV41_ENGRAM_DEAD : engram_token_map[t];
    };

    out.resize(ubatch.n_tokens*n_cols);
    std::vector<uint64_t> toks(n_gram);
    for (int64_t i = 0; i < ubatch.n_tokens; ++i) {
        // shift 0 is the token itself; once a shift is blocked (sequence start or dead token) every longer one is too
        int64_t c = !ubatch.token ? DSV41_ENGRAM_DEAD : cid(ubatch.token[i]);
        bool blocked = c == DSV41_ENGRAM_DEAD;
        toks[0] = blocked ? pad : c;
        for (int64_t s = 1; s < n_gram; ++s) {
            c = ubatch.pos[i] - s < 0 ? DSV41_ENGRAM_DEAD : cid(prev[i*n_prev + (n_prev - s)]);
            blocked = blocked || c == DSV41_ENGRAM_DEAD;
            toks[s] = blocked ? pad : c;
        }

        uint64_t rolling = toks[0]*mult[0];
        for (int64_t o = 1; o < n_gram; ++o) {
            rolling ^= toks[o]*mult[o];
            for (int64_t h = 0; h < n_head; ++h) {
                const int64_t col = (o - 1)*n_head + h;
                out[i*n_cols + col] = (int32_t) (rolling % primes[col] + offs[col]);
            }
        }
    }
}

//
// graph inputs
//

class llm_graph_input_dsv41 : public llm_graph_input_i {
public:
    llm_graph_input_dsv41(const llama_model & model, const llama_cparams & cparams, const llama_kv_cache_dsv41_context * mctx) :
        model(model), cparams(cparams), mctx(mctx) {}

    void set_input(const llama_ubatch * ubatch) override;
    bool can_reuse(const llm_graph_params & params) override;

    struct comp_input {
        ggml_tensor * kq_mask     = nullptr; // F32 [n_comp, n_tokens, 1, 1]
        ggml_tensor * read_idxs   = nullptr; // I32 [ratio*n_blocks]
        ggml_tensor * write_idxs  = nullptr; // I64 [n_blocks]
        ggml_tensor * write_pos   = nullptr; // I32 [n_blocks]
        ggml_tensor * persist_src = nullptr; // I32
        ggml_tensor * persist_dst = nullptr; // I64
    };

    ggml_tensor * k_idxs   = nullptr; // I64 [n_tokens]
    ggml_tensor * raw_mask = nullptr; // F32 [n_kv, n_tokens, 1, 1]

    comp_input comp[2]; // by ratio - 1; unused ratios stay empty

    std::vector<ggml_tensor *> engram_rows; // per Engram module: F32 [n_heads*head_dim, n_tokens]

    // +1e30 on the block with the newest compressed row of each query, so a partly filled block is always kept; the
    // reference uses +inf, but a finite pin keeps fully masked blocks at -inf instead of NaN
    ggml_tensor * cand_pin   = nullptr; // F32 [n_blk, n_tokens, 1, 1]
    uint32_t      cand_ratio = 0;

    ggml_tensor * group_ids = nullptr; // I32 [n_groups, n_tokens]

    uint32_t stream = 0;

    const llama_model & model;
    const llama_cparams cparams;
    const llama_kv_cache_dsv41_context * mctx;
};

template <typename T>
static void dsv41_set(ggml_tensor * dst, const std::vector<T> & src) {
    if (!dst || !dst->buffer) {
        return;
    }
    GGML_ASSERT(dst->ne[0] == (int64_t) src.size());
    ggml_backend_tensor_set(dst, src.data(), 0, src.size()*sizeof(T));
}

// ALPHA_ENGRAM_ASYNC (default on): the Engram rows of modules >= 1 are read by one background thread into their (host)
// input tensors; the scheduler's input wait hook joins it before it copies one of them
extern "C" void ggml_backend_sched_set_input_wait_fn(void (*fn)(const struct ggml_tensor *));
static struct dsv41_engram_async_t {
    std::thread                        th;
    std::vector<const ggml_tensor *>   inputs;
    void join() { if (th.joinable()) th.join(); inputs.clear(); }
    ~dsv41_engram_async_t() { if (th.joinable()) th.join(); }
} dsv41_engram_async;
static void dsv41_engram_wait(const struct ggml_tensor * t) {
    if (!dsv41_engram_async.th.joinable()) {
        return;
    }
    for (const ggml_tensor * x : dsv41_engram_async.inputs) {
        if (x == t) {
            dsv41_engram_async.join();
            return;
        }
    }
}

void llm_graph_input_dsv41::set_input(const llama_ubatch * ubatch) {
    const int64_t n_tokens = ubatch->n_tokens;
    const auto & hparams = model.hparams;

    // ALPHA_INPUT_TIMERS=1: stage times of this setup for prefill ubatches ([dsv41-in])
    static const bool tm = getenv("ALPHA_INPUT_TIMERS") != nullptr;
    const bool tm_on = tm && n_tokens > 256;
    double t_ms[8] = {};   // kidx, raw mask, plan 1, mask 1, plan 2, mask 2, engram, cand
    int64_t tq = ggml_time_us();
    const auto lap = [&](int i) { if (tm_on) { const int64_t t = ggml_time_us(); t_ms[i] += (t - tq) / 1000.0; tq = t; } };

    if (k_idxs && k_idxs->buffer) {
        mctx->set_input_k_idxs(k_idxs, ubatch);
    }
    lap(0);
    if (raw_mask && raw_mask->buffer) {
        mctx->set_input_kq_mask(raw_mask, ubatch, cparams.causal_attn);
    }
    lap(1);

    for (uint32_t r = 1; r <= 2; ++r) {
        const comp_input & ci = comp[r - 1];
        if (!ci.kq_mask) {
            continue;
        }
        const auto & plan = mctx->get_plan(r, *ubatch);

        dsv41_set(ci.read_idxs,   plan.read_idxs);
        dsv41_set(ci.write_idxs,  plan.write_idxs);
        dsv41_set(ci.write_pos,   plan.write_pos);
        dsv41_set(ci.persist_src, plan.persist_src);
        if (ci.persist_dst && ci.persist_dst->buffer) {
            std::vector<int64_t> dst(plan.persist_dst.begin(), plan.persist_dst.end());
            dsv41_set(ci.persist_dst, dst);
        }
        lap(r == 1 ? 2 : 4);

        if (ci.kq_mask->buffer) {
            GGML_ASSERT(ci.kq_mask->ne[0] == plan.n_comp && ci.kq_mask->ne[1] == n_tokens);
            GGML_ASSERT(ggml_backend_buffer_is_host(ci.kq_mask->buffer));
            float * data = (float *) ci.kq_mask->data;
            for (int64_t i = 0; i < n_tokens; ++i) {
                const int32_t n_visible = plan.n_visible[i];
                for (int64_t j = 0; j < plan.n_comp; ++j) {
                    data[i*plan.n_comp + j] = j < n_visible ? 0.0f : -INFINITY;
                }
            }
        }
        lap(r == 1 ? 3 : 5);
    }

    if (!engram_rows.empty()) {
        const llama_engram_table & table = *model.engram_table;
        const int64_t n_cols = hparams.ple_n_heads;

        std::vector<llama_token> prev;
        mctx->get_prev_tokens(*ubatch, hparams.ple_ngram_size - 1, prev);

        // hash all Engram layers first so the table gets one batched readahead before the first read
        std::vector<std::vector<int32_t>> ids(engram_rows.size());
        for (size_t e = 0; e < engram_rows.size(); ++e) {
            model.engram_hash((int) e, *ubatch, prev, ids[e]);
            table.prefetch_rows((int) e, ids[e].data(), ids[e].size());
        }
        // ai4: the engram modules concurrently, each converted straight into its input tensor when that is host memory
        static thread_local std::vector<std::vector<float>> rows(4);   // kept across prompts (only for device inputs)
        const size_t ne = std::min<size_t>(engram_rows.size(), rows.size());
        std::vector<char> on_host(ne, 0);   // char, not bool: the modules set their own entries concurrently
        auto read_module = [&](size_t e) {
            if (!engram_rows[e]->buffer) {
                return;
            }
            float * dst = nullptr;
            if (ggml_backend_buffer_is_host(engram_rows[e]->buffer)) {
                on_host[e] = 1;
                dst = (float *) engram_rows[e]->data;
            } else {
                rows[e].resize((size_t) n_tokens*n_cols*table.head_dim);
                dst = rows[e].data();
            }
            table.read_rows((int) e, ids[e].data(), ids[e].size(), dst);
        };
        static const bool eng_async = [] { const char * v = getenv("ALPHA_ENGRAM_ASYNC"); return !v || atoi(v) != 0; }();
        static bool hook_set = false;
        if (eng_async && !hook_set) {
            ggml_backend_sched_set_input_wait_fn(dsv41_engram_wait);
            hook_set = true;
        }
        dsv41_engram_async.join();   // the previous ubatch's reader, if the scheduler never asked for its inputs
        std::vector<size_t> async_e;
        std::vector<std::thread> eth;
        for (size_t e = 1; e < ne; ++e) {
            // only a pinned staging input (CUDA_Host) is copied by the scheduler, i.e. waited for; a CPU-only graph reads
            // its input in place
            if (eng_async && engram_rows[e]->buffer && ggml_backend_buffer_is_host(engram_rows[e]->buffer) &&
                strstr(ggml_backend_buffer_name(engram_rows[e]->buffer), "CUDA_Host")) {
                async_e.push_back(e);
                on_host[e] = 1;
            } else {
                eth.emplace_back(read_module, e);
            }
        }
        static const bool async_early = [] { const char * v = getenv("ALPHA_ENGRAM_ASYNC_EARLY"); return !v || atoi(v) != 0; }();   // default on (f71: 8K 646 vs 642)
        const auto start_async = [&]() {
            if (async_e.empty()) {
                return;
            }
            std::vector<std::vector<int32_t>> a_ids;
            std::vector<float *>              a_dst;
            std::vector<int>                  a_mod;
            for (size_t e : async_e) {
                a_ids.push_back(std::move(ids[e]));
                a_dst.push_back((float *) engram_rows[e]->data);
                a_mod.push_back((int) e);
                dsv41_engram_async.inputs.push_back(engram_rows[e]);
            }
            const llama_engram_table * tp = &table;
            dsv41_engram_async.th = std::thread([tp, a_ids = std::move(a_ids), a_dst, a_mod]() {
                for (size_t k = 0; k < a_mod.size(); ++k) {
                    tp->read_rows(a_mod[k], a_ids[k].data(), a_ids[k].size(), a_dst[k]);
                }
            });
        };
        if (async_early) {
            start_async();
        }
        if (ne > 0) {
            read_module(0);
        }
        for (auto & x : eth) {
            x.join();
        }
        if (!async_early) {
            start_async();
        }
        for (size_t e = 0; e < ne; ++e) {
            if (engram_rows[e]->buffer && !on_host[e]) {
                ggml_backend_tensor_set(engram_rows[e], rows[e].data(), 0, rows[e].size()*sizeof(float));
            }
        }
    }

    lap(6);
    if (cand_pin && cand_pin->buffer) {
        const auto & plan = mctx->get_plan(cand_ratio, *ubatch);
        const int64_t n_blk = cand_pin->ne[0];
        const int64_t bs    = hparams.dsv41_cand_block_size;
        std::vector<float> pin(n_blk*n_tokens, 0.0f);
        for (int64_t i = 0; i < n_tokens; ++i) {
            const int64_t nv = plan.n_visible[i];
            if (nv > 0) {
                GGML_ASSERT((nv - 1)/bs < n_blk);
                pin[i*n_blk + (nv - 1)/bs] = 1e30f;
            }
        }
        ggml_backend_tensor_set(cand_pin, pin.data(), 0, pin.size()*sizeof(float));
    }

    if (group_ids && group_ids->buffer) {
        const int64_t n_groups = group_ids->ne[0];
        std::vector<int32_t> data(n_groups*n_tokens);
        for (int64_t t = 0; t < n_tokens; ++t) {
            for (int64_t g = 0; g < n_groups; ++g) {
                data[t*n_groups + g] = (int32_t) g;
            }
        }
        ggml_backend_tensor_set(group_ids, data.data(), 0, data.size()*sizeof(int32_t));
    }
    lap(7);
    if (tm_on) {
        fprintf(stderr, "[dsv41-in] n=%lld kidx %.1f raw-mask %.1f | ratio1 plan %.1f mask %.1f | ratio2 plan %.1f mask %.1f | engram %.1f | cand %.1f ms\n",
                (long long) n_tokens, t_ms[0], t_ms[1], t_ms[2], t_ms[3], t_ms[4], t_ms[5], t_ms[6], t_ms[7]);
    }
}

bool llm_graph_input_dsv41::can_reuse(const llm_graph_params & params) {
    mctx = static_cast<const llama_kv_cache_dsv41_context *>(params.mctx);

    const int64_t n_tokens = params.ubatch.n_tokens;

    bool res = !mctx->is_full();

    res = res && k_idxs->ne[0] == n_tokens;
    res = res && raw_mask->ne[0] == (int64_t) mctx->get_n_kv() && raw_mask->ne[1] == n_tokens;
    res = res && stream == mctx->get_stream();
    if (!res) {
        return false;
    }

    for (uint32_t r = 1; r <= 2; ++r) {
        const comp_input & ci = comp[r - 1];
        if (!ci.kq_mask) {
            continue;
        }
        const auto & plan = mctx->get_plan(r, params.ubatch);
        res &= ci.kq_mask->ne[0] == plan.n_comp && ci.kq_mask->ne[1] == n_tokens;
        res &= ci.write_idxs->ne[0] == plan.n_blocks;
        if (ci.read_idxs) {
            res &= ci.read_idxs->ne[0] == (int64_t) plan.read_idxs.size();
            res &= ci.persist_src->ne[0] == (int64_t) plan.persist_src.size();
        }
        if (cand_pin && cand_ratio == r) {
            const int64_t bs = model.hparams.dsv41_cand_block_size;
            res &= cand_pin->ne[0] == (plan.n_comp + bs - 1)/bs && cand_pin->ne[1] == n_tokens;
        }
    }

    res &= engram_rows.empty() == (params.ubatch.token == nullptr || model.hparams.ple_n_heads == 0);
    for (auto * t : engram_rows) {
        res &= t->ne[1] == n_tokens;
    }
    if (group_ids) {
        res &= group_ids->ne[1] == n_tokens;
    }

    return res;
}

//
// mHC
//

static ggml_tensor * dsv41_view_1d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t i0) {
    return ggml_view_1d(ctx, t, ne0, ggml_row_size(t->type, i0));
}

static ggml_tensor * dsv41_view_2d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t ne1, int64_t i0) {
    return ggml_view_2d(ctx, t, ne0, ne1, t->nb[1], ggml_row_size(t->type, i0));
}

static ggml_tensor * dsv41_hc_affine(ggml_context * ctx, ggml_tensor * x, ggml_tensor * scale, ggml_tensor * base) {
    x = ggml_mul(ctx, x, scale);
    x = ggml_add(ctx, x, base);
    return x;
}

ggml_tensor * llm_build_deepseek41::build_hc_sinkhorn(ggml_tensor * comb) const {
    // comb is [dst_hc, src_hc, n_tokens]. Sinkhorn follows the reference:
    // row softmax over dst, one column normalization, then repeated row/column normalization.
    comb = ggml_soft_max(ctx0, comb);

    ggml_tensor * eps = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    eps = ggml_fill(ctx0, eps, hparams.hc_eps);

    comb = ggml_add(ctx0, comb, eps);

    auto norm_cols = [&]() {
        ggml_tensor * comb_src_dst = ggml_cont(ctx0, ggml_permute(ctx0, comb, 1, 0, 2, 3));
        ggml_tensor * col_sum = ggml_sum_rows(ctx0, comb_src_dst);
        col_sum = ggml_add(ctx0, col_sum, eps);
        col_sum = ggml_permute(ctx0, col_sum, 1, 0, 2, 3);
        comb = ggml_div(ctx0, comb, col_sum);
    };

    auto norm_rows = [&]() {
        ggml_tensor * row_sum = ggml_sum_rows(ctx0, comb);
        row_sum = ggml_add(ctx0, row_sum, eps);
        comb = ggml_div(ctx0, comb, row_sum);
    };

    norm_cols();
    for (uint32_t i = 1; i < hparams.hc_sinkhorn_iters; ++i) {
        norm_rows();
        norm_cols();
    }

    return comb;
}

llm_build_deepseek41::hc_mixes llm_build_deepseek41::build_hc_mixes(
        ggml_tensor * x,
        ggml_tensor * hc_fn,
        ggml_tensor * hc_scale,
        ggml_tensor * hc_base,
        int il) const {
    const int64_t hc         = hparams.hc_count;
    const int64_t hc_dim     = hc*n_embd;
    const int64_t hc_mix_dim = (2 + hc)*hc;
    const int64_t nt         = x->ne[2];

    GGML_ASSERT(hc_fn->ne[1] == hc_mix_dim);

    ggml_tensor * flat      = ggml_reshape_2d(ctx0, x, hc_dim, nt);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx0, flat, hparams.f_norm_rms_eps);
    ggml_tensor * mixes     = ggml_mul_mat(ctx0, hc_fn, flat_norm);
    cb(mixes, "hc_mixes", il);

    if (dsv41_hc_fused()) {
        // this fork's DeepSeek-V4 mixer: pre | post | comb rows of one tensor
        ggml_tensor * split = ggml_dsv4_hc_split_sinkhorn(ctx0, mixes, hc_scale, hc_base, (int) hc,
                (int) hparams.hc_sinkhorn_iters, hparams.hc_eps);
        ggml_tensor * pre  = ggml_cont(ctx0, ggml_view_2d(ctx0, split, hc,    nt, split->nb[1], 0));
        ggml_tensor * post = ggml_cont(ctx0, ggml_view_2d(ctx0, split, hc,    nt, split->nb[1],   hc * split->nb[0]));
        ggml_tensor * comb = ggml_cont(ctx0, ggml_view_2d(ctx0, split, hc*hc, nt, split->nb[1], 2*hc * split->nb[0]));
        comb = ggml_reshape_3d(ctx0, comb, hc, hc, nt);
        return { pre, post, comb };
    }

    ggml_tensor * scale_pre  = dsv41_view_1d(ctx0, hc_scale, 1, 0);
    ggml_tensor * scale_post = dsv41_view_1d(ctx0, hc_scale, 1, 1);
    ggml_tensor * scale_comb = dsv41_view_1d(ctx0, hc_scale, 1, 2);

    ggml_tensor * base_pre  = dsv41_view_1d(ctx0, hc_base, hc,    0);
    ggml_tensor * base_post = dsv41_view_1d(ctx0, hc_base, hc,    hc);
    ggml_tensor * base_comb = dsv41_view_1d(ctx0, hc_base, hc*hc, 2*hc);

    ggml_tensor * pre = dsv41_view_2d(ctx0, mixes, hc, nt, 0);
    pre = dsv41_hc_affine(ctx0, pre, scale_pre, base_pre);
    pre = ggml_sigmoid(ctx0, pre);
    pre = ggml_scale_bias(ctx0, pre, 1.0f, hparams.hc_eps);
    cb(pre, "hc_pre", il);

    ggml_tensor * post = dsv41_view_2d(ctx0, mixes, hc, nt, hc);
    post = dsv41_hc_affine(ctx0, post, scale_post, base_post);
    post = ggml_sigmoid(ctx0, post);
    post = ggml_scale(ctx0, post, 2.0f);
    cb(post, "hc_post", il);

    ggml_tensor * comb = dsv41_view_2d(ctx0, mixes, hc*hc, nt, 2*hc);
    comb = dsv41_hc_affine(ctx0, comb, scale_comb, base_comb);
    comb = ggml_reshape_3d(ctx0, comb, hc, hc, nt);
    comb = build_hc_sinkhorn(comb);
    cb(comb, "hc_comb", il);

    return { pre, post, comb };
}

ggml_tensor * llm_build_deepseek41::build_hc_pre(ggml_tensor * x, ggml_tensor * weights, int il) const {
    GGML_UNUSED(il);
    GGML_ASSERT(x->ne[0] == n_embd);
    GGML_ASSERT(x->ne[1] == hparams.hc_count);

    const int64_t hc = hparams.hc_count;
    const int64_t nt = x->ne[2];

    if (dsv41_hc_fused()) {
        return ggml_dsv4_hc_weighted_sum(ctx0, x, weights);
    }

    ggml_tensor * result = nullptr;
    for (int64_t ih = 0; ih < hc; ++ih) {
        ggml_tensor * xh  = ggml_view_2d(ctx0, x, n_embd, nt, x->nb[2], ih*x->nb[1]);
        ggml_tensor * wh  = ggml_view_2d(ctx0, weights, 1, nt, weights->nb[1], ih*weights->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx0, xh, wh);
        result = result ? ggml_add(ctx0, result, cur) : cur;
    }

    return result;
}

ggml_tensor * llm_build_deepseek41::build_hc_post(
        ggml_tensor * x,
        ggml_tensor * residual,
        ggml_tensor * post,
        ggml_tensor * comb,
        int il) const {
    GGML_UNUSED(il);
    GGML_ASSERT(x->ne[0] == n_embd);
    GGML_ASSERT(residual->ne[1] == hparams.hc_count);

    if (dsv41_hc_fused()) {
        return ggml_dsv4_hc_expand(ctx0, x, residual, post, comb);
    }

    const int64_t hc = hparams.hc_count;
    const int64_t nt = x->ne[1];

    ggml_tensor * out = nullptr;
    for (int64_t dst = 0; dst < hc; ++dst) {
        ggml_tensor * post_dst = ggml_view_2d(ctx0, post, 1, nt, post->nb[1], dst*post->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx0, x, post_dst);

        for (int64_t src = 0; src < hc; ++src) {
            ggml_tensor * res_src = ggml_view_2d(ctx0, residual, n_embd, nt, residual->nb[2], src*residual->nb[1]);
            ggml_tensor * comb_src_dst = ggml_view_2d(ctx0, comb, 1, nt, comb->nb[2],
                    dst*comb->nb[0] + src*comb->nb[1]);
            cur = ggml_add(ctx0, cur, ggml_mul(ctx0, res_src, comb_src_dst));
        }

        cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, nt);
        out = out ? ggml_concat(ctx0, out, cur, 1) : cur;
    }

    return out;
}

//
// attention
//

// rotary embedding on the last n_rot dims of x [n_dim, n_head, n]; compressed layers use their own base with YaRN,
// window-only layers the plain base (upstream: ggml_rope_ext + ggml_rope_set_offset)
ggml_tensor * llm_build_deepseek41::build_rope(ggml_tensor * x, ggml_tensor * pos, bool comp_rope, bool inverse) const {
    const int64_t n_dim  = x->ne[0];
    const int64_t n_rope = hparams.n_rot();
    const int64_t n_nope = n_dim - n_rope;

    const float   freq_base_l  = comp_rope ? hparams.dsv4_compress_rope_base : freq_base;
    const float   freq_scale_l = comp_rope ? freq_scale : 1.0f;
    const float   ext_factor_l = comp_rope ? ext_factor : 0.0f;
    // YaRN attention factor that cancels ggml's rope mscale (the reference applies none)
    const float   attn_factor_l = ext_factor_l == 0.0f ? 1.0f : 1.0f/(1.0f + 0.1f*logf(1.0f/freq_scale_l));
    const float   beta_fast_l  = comp_rope ? beta_fast : 0.0f;
    const float   beta_slow_l  = comp_rope ? beta_slow : 0.0f;
    const int32_t n_ctx_orig_l = comp_rope ? n_ctx_orig : 0;

    const auto rope = [&](ggml_tensor * t) {
        return inverse
            ? ggml_rope_ext_back(ctx0, t, pos, nullptr, n_rope, rope_type, n_ctx_orig_l, freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l)
            : ggml_rope_ext     (ctx0, t, pos, nullptr, n_rope, rope_type, n_ctx_orig_l, freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
    };

    GGML_ASSERT(ggml_n_dims(x) <= 3 && n_nope >= 0);
    if (n_nope == 0) {
        return rope(x);
    }

    ggml_tensor * x_nope = ggml_view_3d(ctx0, x, n_nope, x->ne[1], x->ne[2], x->nb[1], x->nb[2], 0);
    ggml_tensor * x_tail = ggml_view_3d(ctx0, x, n_rope, x->ne[1], x->ne[2], x->nb[1], x->nb[2], ggml_row_size(x->type, n_nope));
    x_tail = rope(ggml_cont(ctx0, x_tail));

    return ggml_concat(ctx0, x_nope, x_tail, 0);
}

ggml_tensor * llm_build_deepseek41::build_engram(ggml_tensor * x, ggml_tensor * rows, int il) const {
    const auto & layer = model.layers[il];
    const int64_t hc = hparams.hc_count;
    const int64_t nt = x->ne[2];
    const float   norm_rms_eps = hparams.f_norm_rms_eps;

    GGML_ASSERT(rows->ne[0] == (int64_t) (hparams.ple_head_dim*hparams.ple_n_heads) && rows->ne[1] == nt);

    ggml_tensor * kv = build_lora_mm(layer.engram_kv, rows);
    cb(kv, "engram_kv", il);

    ggml_tensor * key   = ggml_cont(ctx0, ggml_view_3d(ctx0, kv, n_embd, hc, nt, ggml_row_size(kv->type, n_embd), kv->nb[1], 0));
    ggml_tensor * value = ggml_cont(ctx0, ggml_view_3d(ctx0, kv, n_embd, 1,  nt, ggml_row_size(kv->type, n_embd), kv->nb[1],
            ggml_row_size(kv->type, hc*n_embd)));

    // quantized copy has no binary-op kernel
    ggml_tensor * egate = layer.engram_gate->type != GGML_TYPE_F32 ? ggml_cast(ctx0, layer.engram_gate, GGML_TYPE_F32) : layer.engram_gate;
    ggml_tensor * dot = ggml_mul(ctx0, ggml_mul(ctx0, ggml_rms_norm(ctx0, x, norm_rms_eps), ggml_rms_norm(ctx0, key, norm_rms_eps)), egate);
    dot = ggml_scale(ctx0, ggml_sum_rows(ctx0, dot), 1.0f/sqrtf((float) n_embd));
    cb(dot, "engram_dot", il);

    // sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot)); sign(0) = +1 as in copysign
    ggml_tensor * mag  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, dot), 1e-6f, INFINITY));
    ggml_tensor * sign = ggml_scale_bias(ctx0, ggml_step(ctx0, ggml_neg(ctx0, dot)), -2.0f, 1.0f);
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, mag, sign));
    cb(gate, "engram_gate", il);

    return ggml_add(ctx0, x, ggml_mul(ctx0, ggml_repeat_4d(ctx0, value, n_embd, hc, nt, 1), gate));
}

// kq_mask [n_comp, n_tokens, 1, 1] F32 with -inf everywhere but the top-k rows of each token
ggml_tensor * llm_build_deepseek41::build_top_k_mask(ggml_tensor * kq_mask, ggml_tensor * top_k, const char * name, int il) const {
    ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);
    kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3],
            kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

    ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1,
            top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
    zeros = ggml_fill(ctx0, zeros, 0.0f);

    ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);
    kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k,
            kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3],
            kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);

    kq_mask_top_k = ggml_add(ctx0, kq_mask_top_k, kq_mask);
    cb(kq_mask_top_k, name, il);

    return kq_mask_top_k;
}

ggml_tensor * llm_build_deepseek41::build_attention(
        llm_graph_input_dsv41 * inp,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il) {
    const auto & layer = model.layers[il];
    const auto * mctx41 = inp->mctx;

    const int64_t n_embd_head = hparams.n_embd_head_k();
    const int64_t n_groups    = hparams.dsv4_o_group_count;
    const int64_t o_lora_rank = hparams.dsv4_o_lora_rank;
    const int64_t n_idx_embd  = hparams.indexer_head_size;
    const int64_t n_idx_head  = hparams.indexer_n_head;
    const int64_t nt          = cur->ne[1];
    const float   kq_scale    = 1.0f/sqrtf(float(n_embd_head));
    const int64_t ratio       = hparams.compress_ratios[il];
    const bool    comp_rope   = ratio != 0;

    ggml_tensor * qr = build_norm(build_lora_mm(layer.wq_a, cur), layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(qr, "qr_norm", il);

    ggml_tensor * q = ggml_reshape_3d(ctx0, build_lora_mm(layer.wq_b, qr), n_embd_head, n_head, nt);
    q = build_rope(q, inp_pos, comp_rope, false);
    cb(q, "q", il);

    ggml_tensor * kv = build_norm(build_lora_mm(layer.wkv, cur), layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il);
    kv = build_rope(ggml_reshape_3d(ctx0, kv, n_embd_head, 1, nt), inp_pos, comp_rope, false);
    cb(kv, "kv", il);

    const int32_t kv_src   = hparams.dsv41_kv_src[il];
    const int32_t topk_src = hparams.dsv41_topk_src[il];

    if (ratio != 0 && kv_src == il) {
        // KV source: compress the tokens of the blocks this ubatch completes and store [latent | index key]
        const auto & ci = inp->comp[ratio - 1];
        const int64_t n_blocks = ci.write_idxs->ne[0];

        ggml_tensor * state_kv    = build_lora_mm(layer.attn_comp_wkv, cur);
        ggml_tensor * state_score = nullptr;

        ggml_tensor * comp = nullptr;
        if (ratio > 1) {
            state_score = build_lora_mm(layer.attn_comp_wgate, cur);

            // gather source: [state ring of the stream | tokens of this ubatch]
            ggml_tensor * src_kv    = ggml_concat(ctx0, mctx41->get_state_kv   (ctx0, il), state_kv,    1);
            ggml_tensor * src_score = ggml_concat(ctx0, mctx41->get_state_score(ctx0, il), state_score, 1);

            ggml_tensor * v = ggml_get_rows(ctx0, src_kv,    ci.read_idxs);
            ggml_tensor * s = ggml_get_rows(ctx0, src_score, ci.read_idxs);
            v = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, v, n_embd_head, ratio, n_blocks), 1, 0, 2, 3));
            s = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, s, n_embd_head, ratio, n_blocks), 1, 0, 2, 3));

            comp = ggml_sum_rows(ctx0, ggml_mul(ctx0, v, ggml_soft_max(ctx0, s)));
            comp = ggml_cont(ctx0, ggml_permute(ctx0, comp, 1, 0, 2, 3));
        } else {
            // ratio 1: every token is its own block (softmax over one score is exactly 1)
            GGML_ASSERT(n_blocks == nt);
            comp = ggml_reshape_3d(ctx0, state_kv, n_embd_head, 1, nt);
        }
        comp = build_norm(comp, layer.attn_comp_norm, nullptr, LLM_NORM_RMS, il);
        cb(comp, "comp_latent", il);

        // index K from the RoPE-free latent; a latent stands for the first token of its group
        ggml_tensor * ik = build_norm(build_lora_mm(layer.indexer_attn_k, comp), layer.indexer_k_norm, nullptr, LLM_NORM_RMS, il);
        ik   = build_rope(ik,   ci.write_pos, true, false);
        comp = build_rope(comp, ci.write_pos, true, false);

        ggml_tensor * row = ggml_concat(ctx0, comp, ik, 0);
        ggml_build_forward_expand(gf, mctx41->cpy_comp(ctx0, row, ci.write_idxs, il));

        if (ratio > 1) {
            ggml_build_forward_expand(gf, mctx41->cpy_state_kv   (ctx0, ggml_get_rows(ctx0, state_kv,    ci.persist_src), ci.persist_dst, il));
            ggml_build_forward_expand(gf, mctx41->cpy_state_score(ctx0, ggml_get_rows(ctx0, state_score, ci.persist_src), ci.persist_dst, il));
        }
    }

    // raw window: store this ubatch's K, then read the window
    ggml_build_forward_expand(gf, kv);
    ggml_build_forward_expand(gf, mctx41->cpy_k(ctx0, kv, inp->k_idxs, il));

    ggml_tensor * raw_k = mctx41->get_k(ctx0, il);
    if (raw_k->ne[3] > 1) {
        // graph reservation with several sequences: one stream is the worst case of a (single sequence) ubatch
        raw_k = ggml_view_4d(ctx0, raw_k, raw_k->ne[0], raw_k->ne[1], raw_k->ne[2], 1, raw_k->nb[1], raw_k->nb[2], raw_k->nb[3], 0);
    }
    ggml_tensor * raw_mask = inp->raw_mask;

    const auto mask_cnv = [&](ggml_tensor * m) {
        return cparams.flash_attn ? ggml_cast(ctx0, m, GGML_TYPE_F16) : m;
    };

    // ALPHA_DSV41_GATHER (default on for batches of at least ALPHA_DSV41_GATHER_MIN = 256 tokens): flash attention of
    // each query over its own rows only - its window rows and its top-k compressed rows - gathered into a K of
    // [head_dim, W, 1, queries] (one sequence per query, ALPHA_DSV41_GATHER_CHUNK = 1024 queries per call), instead of
    // dense attention over every raw and compressed row with the rest masked. The same rows with the same mask values,
    // so the same result up to summation order. W is a multiple of 256 (the D=512 kernel needs it): the window part
    // takes the w rows of highest mask value, n_swa visible ones first, the rest masked.
    static const int64_t gather_min = [] {
        const char * e = getenv("ALPHA_DSV41_GATHER");
        if (e && atoi(e) == 0) {
            return INT64_MAX;
        }
        const char * m = getenv("ALPHA_DSV41_GATHER_MIN");
        return m ? (int64_t) atoll(m) : (int64_t) 256;
    }();
    static const int64_t gather_chunk = [] {
        const char * e = getenv("ALPHA_DSV41_GATHER_CHUNK");
        return e ? std::max<int64_t>(1, atoll(e)) : (int64_t) 1024;
    }();
    const int64_t n_raw  = raw_k->ne[2];
    // ALPHA_DSV41_GATHER_IL0 (debug): first layer that gathers (the earlier ones stay dense)
    static const int gather_il0 = [] { const char * e = getenv("ALPHA_DSV41_GATHER_IL0"); return e ? atoi(e) : 0; }();
    const bool    gather = cparams.flash_attn && nt >= gather_min && il >= gather_il0 && raw_k->ne[1] == 1 && raw_k->ne[3] == 1;
    const auto raw_idx = [&](int64_t w) {
        for (const auto & p : g_raw_idx) {
            if (p.first == w) {
                return p.second;
            }
        }
        GGML_ASSERT(w <= n_raw);
        ggml_tensor * t = ggml_top_k(ctx0, raw_mask, w);
        if (!ggml_is_contiguous(t)) {
            t = ggml_cont(ctx0, t);
        }
        cb(t, "gather_raw_idx", -1);
        g_raw_idx.push_back({ w, t });
        return t;
    };
    // attention output [n_head*head_dim, n] of the tokens at positions pos -> de-rope -> grouped low-rank wo_a
    // projection [o_lora_rank*n_groups, n] (wo_a holds one [group input -> o_lora_rank] matrix per group)
    const auto wo_a_of = [&](ggml_tensor * o, ggml_tensor * pos) {
        const int64_t n = pos->ne[0];
        o = ggml_reshape_3d(ctx0, o, n_embd_head, n_head, n);
        o = build_rope(o, pos, comp_rope, true);
        cb(o, "attn_derope", il);
        const int64_t group_dim = n_head/n_groups*n_embd_head;
        o = ggml_permute(ctx0, ggml_reshape_3d(ctx0, ggml_cont(ctx0, o), group_dim, n_groups, n), 0, 2, 1, 3);
        ggml_tensor * wo_a = ggml_reshape_3d(ctx0, layer.wo_a, group_dim, o_lora_rank, n_groups);
        ggml_tensor * oa = ggml_mul_mat(ctx0, wo_a, o);
        cb(oa, "attn_wo_a", il);
        return ggml_cont_2d(ctx0, ggml_permute(ctx0, oa, 0, 2, 1, 3), o_lora_rank*n_groups, n);
    };
    // ALPHA_DSV41_CHUNK_WO (default on): the gather path applies wo_a_of per query chunk, so the chunks are joined after
    // the projection (o_lora_rank*n_groups rows) instead of before it (n_head*head_dim rows)
    static const bool chunk_wo = [] { const char * e = getenv("ALPHA_DSV41_CHUNK_WO"); return !e || atoi(e) != 0; }();
    bool out_is_oa = false;
    const bool q_full_needed = !(gather && chunk_wo);
    if (q_full_needed) {
        ggml_build_forward_expand(gf, q);
    }
    // k2d [head_dim, n_kv] rows, m2d [n_kv, nt] F32 mask, idx [W, nt] I32 rows of each query
    const auto attn_gather = [&](ggml_tensor * k2d, ggml_tensor * m2d, ggml_tensor * idx) {
        const int64_t W    = idx->ne[0];
        const int64_t n_kv = k2d->ne[1];
        ggml_tensor * m3 = ggml_reshape_3d(ctx0, m2d, 1, n_kv, nt);
        std::vector<ggml_tensor *> parts;
        for (int64_t t0 = 0; t0 < nt; t0 += gather_chunk) {
            const int64_t c = std::min(gather_chunk, nt - t0);
            ggml_tensor * ic = ggml_view_2d(ctx0, idx, W, c, idx->nb[1], t0*idx->nb[1]);
            ggml_tensor * kg = ggml_get_rows_t(ctx0, k2d, ggml_reshape_1d(ctx0, ic, W*c), GGML_TYPE_F16);
            kg = ggml_reshape_4d(ctx0, kg, k2d->ne[0], W, 1, c);
            cb(kg, "gather_k", il);
            ggml_tensor * mc = ggml_view_3d(ctx0, m3, 1, n_kv, c, m3->nb[1], m3->nb[2], t0*m3->nb[2]);
            ggml_tensor * mg = ggml_reshape_4d(ctx0, ggml_get_rows_t(ctx0, mc, ic, GGML_TYPE_F16), W, 1, 1, c);
            cb(mg, "gather_mask", il);
            ggml_tensor * qc;
            if (chunk_wo) {   // this chunk's q from its qr rows
                ggml_tensor * qr_c = ggml_view_2d(ctx0, qr, qr->ne[0], c, qr->nb[1], t0*qr->nb[1]);
                ggml_tensor * qq   = ggml_reshape_3d(ctx0, build_lora_mm(layer.wq_b, qr_c), n_embd_head, n_head, c);
                qq = build_rope(qq, ggml_view_1d(ctx0, inp_pos, c, t0*inp_pos->nb[0]), comp_rope, false);
                qc = ggml_view_4d(ctx0, qq, qq->ne[0], qq->ne[1], 1, c, qq->nb[1], qq->nb[2], qq->nb[2], 0);
            } else {
                qc = ggml_view_4d(ctx0, q, q->ne[0], q->ne[1], 1, c, q->nb[1], q->nb[2], q->nb[2], t0*q->nb[2]);
            }
            qc = ggml_permute(ctx0, qc, 0, 2, 1, 3);
            ggml_tensor * fa = ggml_flash_attn_ext(ctx0, qc, kg, kg, mg, kq_scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_add_sinks(fa, layer.attn_sinks);
            ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
            cb(fa, "gather_fattn", il);
            ggml_tensor * o2 = ggml_reshape_2d(ctx0, fa, fa->ne[0]*fa->ne[1], c);
            if (chunk_wo) {
                o2 = wo_a_of(o2, ggml_view_1d(ctx0, inp_pos, c, t0*inp_pos->nb[0]));
            }
            parts.push_back(o2);
        }
        out_is_oa = chunk_wo;
        while (parts.size() > 1) {
            std::vector<ggml_tensor *> w2;
            for (size_t i = 0; i < parts.size(); i += 2) {
                w2.push_back(i + 1 < parts.size() ? ggml_concat(ctx0, parts[i], parts[i + 1], 1) : parts[i]);
            }
            parts.swap(w2);
        }
        return parts[0];
    };

    ggml_tensor * out = nullptr;
    if (ratio == 0 && gather) {
        // K is gathered from a contiguous copy of the window rows: gathering straight from the cache view gave wrong
        // answers on long prompts (f38/f39, 8230 tokens; right on short ones and on the tiny model). Root cause not
        // found; the copy is n_raw rows (4-9 MB) per window-only layer.
        // ALPHA_DSV41_GATHER_RAWFIX (bits, default 2): 1 = mask from a device copy instead of host-input views, 2 = K copy
        static const int rawfix = [] { const char * e = getenv("ALPHA_DSV41_GATHER_RAWFIX"); return e ? atoi(e) : 2; }();
        ggml_tensor * k_src = (rawfix & 2) ? ggml_cont(ctx0, raw_k) : raw_k;
        ggml_tensor * m_src = (rawfix & 1) ? ggml_scale(ctx0, raw_mask, 1.0f) : raw_mask;
        ggml_tensor * k2d = ggml_view_2d(ctx0, k_src, k_src->ne[0], n_raw, k_src->nb[2], 0);
        out = attn_gather(k2d, m_src, raw_idx(GGML_PAD((int64_t) hparams.n_swa, 256)));
        cb(out, "attn_raw", il);
    } else if (ratio == 0) {
        out = build_attn_mha(q, raw_k, raw_k, nullptr, mask_cnv(raw_mask), layer.attn_sinks, nullptr, kq_scale, il);
        cb(out, "attn_raw", il);
    } else {
        const auto & ci = inp->comp[ratio - 1];
        const int64_t n_comp = ci.kq_mask->ne[0];

        ggml_tensor * comp_all = mctx41->get_comp(ctx0, kv_src);
        GGML_ASSERT(comp_all->ne[0] == n_embd_head + n_idx_embd && n_comp <= comp_all->ne[2]);

        ggml_tensor * top_k = topk_of[topk_src];
        if (topk_src == il) {
            ggml_tensor * ik = ggml_view_4d(ctx0, comp_all, n_idx_embd, 1, n_comp, 1,
                    comp_all->nb[1], comp_all->nb[2], comp_all->nb[3], ggml_row_size(comp_all->type, n_embd_head));
            ik = ggml_cont(ctx0, ik);

            ggml_tensor * iq = ggml_reshape_3d(ctx0, build_lora_mm(layer.indexer_attn_q_b, qr), n_idx_embd, n_idx_head, nt);
            iq = build_rope(iq, inp_pos, comp_rope, false);
            iq = ggml_view_4d(ctx0, iq, iq->ne[0], iq->ne[1], nt, 1, iq->nb[1], iq->nb[2], iq->nb[2]*nt, 0);

            ggml_tensor * iw = ggml_scale(ctx0, build_lora_mm(layer.indexer_proj, cur), 1.0f/sqrtf(float(n_idx_embd*n_idx_head)));
            iw = ggml_view_4d(ctx0, iw, iw->ne[0], nt, 1, 1, iw->nb[1], iw->nb[1]*nt, iw->nb[1]*nt, 0);

            // chunk queries so the [n_ih, tokens, n_comp] scores stay near 2^26 elements (at most 64 chunks)
            const int64_t n_tok = nt;
            const int64_t chunk = std::max({ (n_tok + 63)/64, (INT64_C(1) << 26)/(n_idx_head*n_comp), INT64_C(1) });

            const int32_t cand_src = hparams.dsv41_cand_src;
            const bool    pick_blk = cand_src == il && inp->cand_pin;      // level one: pick the candidate blocks
            const bool    use_blk  = cand_src >= 0 && cand_src < il && cand_sel; // level two: score inside them
            const int64_t bs       = hparams.dsv41_cand_block_size;
            const int64_t n_blk    = pick_blk || use_blk ? (n_comp + bs - 1)/bs : 0;
            GGML_ASSERT(!pick_blk || inp->cand_pin->ne[0] == n_blk);
            GGML_ASSERT(!use_blk  || cand_n_comp == n_comp);

            ggml_tensor * ik_p = ggml_permute(ctx0, ik, 0, 2, 1, 3);
            // ALPHA_DSV41_IDX_SWAP (default on): see the score below
            static const bool idx_swap = [] { const char * e = getenv("ALPHA_DSV41_IDX_SWAP"); return !e || atoi(e) != 0; }();
            ggml_tensor * ik_f = idx_swap ? ggml_reshape_2d(ctx0, ik->type == GGML_TYPE_F32 ? ik : ggml_cast(ctx0, ik, GGML_TYPE_F32),
                                                            n_idx_embd, n_comp) : nullptr;
            std::vector<ggml_tensor *> parts_top_k;
            std::vector<ggml_tensor *> parts_sel;
            for (int64_t t0 = 0; t0 < n_tok; t0 += chunk) {
                const int64_t nc = std::min(chunk, n_tok - t0);
                const auto rows = [&](ggml_tensor * t) {
                    return ggml_view_4d(ctx0, t, t->ne[0], nc, 1, 1, t->nb[1], t->nb[2], t->nb[3], t0*t->nb[1]);
                };

                ggml_tensor * iqc = ggml_view_4d(ctx0, iq, iq->ne[0], iq->ne[1], nc, 1, iq->nb[1], iq->nb[2], iq->nb[3], t0*iq->nb[2]);

                ggml_tensor * score = nullptr;
                if (idx_swap) {
                    // [heads, tokens, n_comp] straight from the matmul: queries [dim, heads*tokens] x keys [dim, n_comp]
                    ggml_tensor * iq2 = ggml_reshape_2d(ctx0, ggml_is_contiguous(iqc) ? iqc : ggml_cont(ctx0, iqc), iqc->ne[0], iqc->ne[1]*nc);
                    ggml_tensor * kq  = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, iq2, ik_f), iqc->ne[1], nc, n_comp);
                    ggml_tensor * w3  = ggml_reshape_3d(ctx0, ggml_cont(ctx0, rows(iw)), iqc->ne[1], nc, 1);
                    // ALPHA_DSV41_IDX_FUSED (default on): the head reduction as one op (bit-identical, reads kq once)
                    static const bool idx_fused = [] { const char * e = getenv("ALPHA_DSV41_IDX_FUSED"); return !e || atoi(e) != 0; }();
                    score = idx_fused ? ggml_dsv4_idx_score(ctx0, kq, w3)
                                      : ggml_sum_rows(ctx0, ggml_mul(ctx0, ggml_relu(ctx0, kq), w3));   // [1, tokens, n_comp]
                    score = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_2d(ctx0, score, nc, n_comp)));
                    score = ggml_add(ctx0, ggml_reshape_4d(ctx0, score, n_comp, nc, 1, 1), rows(ci.kq_mask));
                } else {
                    ggml_tensor * kq = ggml_mul_mat(ctx0, ik_p, ggml_permute(ctx0, iqc, 0, 2, 1, 3));
                    kq = ggml_cont(ctx0, ggml_permute(ctx0, kq, 2, 1, 0, 3));
                    score = ggml_sum_rows(ctx0, ggml_mul(ctx0, ggml_relu(ctx0, kq), rows(iw)));
                    score = ggml_add(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, score, 2, 1, 0, 3)), rows(ci.kq_mask));
                }
                cb(score, "idx_score", il);

                if (pick_blk) {
                    // best position per block, pin the newest block, keep the top blocks
                    ggml_tensor * s2 = ggml_reshape_2d(ctx0, score, n_comp, nc);
                    if (n_blk*bs != n_comp) {
                        s2 = ggml_concat(ctx0, s2, ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_blk*bs - n_comp, nc), -INFINITY), 0);
                    }
                    ggml_tensor * bmax = ggml_pool_1d(ctx0, s2, GGML_OP_POOL_MAX, bs, bs, 0);
                    bmax = ggml_add(ctx0, ggml_reshape_4d(ctx0, bmax, n_blk, nc, 1, 1), rows(inp->cand_pin));
                    cb(bmax, "cand_block_score", il);
                    parts_sel.push_back(ggml_top_k(ctx0, bmax, std::min<int64_t>(hparams.dsv41_cand_topk_blocks, n_blk)));
                } else if (use_blk) {
                    // own scores, restricted to the source's candidate blocks: 0 inside them, -inf elsewhere
                    ggml_tensor * bsel = ggml_cont(ctx0, rows(cand_sel));
                    const int64_t k_b = bsel->ne[0];
                    ggml_tensor * bm    = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, n_blk, nc, 1), -INFINITY);
                    ggml_tensor * zeros = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, k_b, nc, 1), 0.0f);
                    bm = ggml_set_rows(ctx0, bm, zeros, ggml_reshape_3d(ctx0, bsel, k_b, nc, 1));
                    bm = ggml_reshape_2d(ctx0, ggml_repeat_4d(ctx0, bm, bs, n_blk, nc, 1), bs*n_blk, nc);
                    bm = ggml_reshape_4d(ctx0, ggml_cont(ctx0, ggml_view_2d(ctx0, bm, n_comp, nc, bm->nb[1], 0)), n_comp, nc, 1, 1);
                    cb(bm, "cand_mask", il);
                    score = ggml_add(ctx0, score, bm);
                }

                parts_top_k.push_back(ggml_top_k(ctx0, score, std::min<int64_t>(n_comp, hparams.indexer_top_k)));
            }

            const auto join = [&](std::vector<ggml_tensor *> v) {
                while (v.size() > 1) {
                    std::vector<ggml_tensor *> w;
                    for (size_t i = 0; i < v.size(); i += 2) {
                        w.push_back(i + 1 < v.size() ? ggml_concat(ctx0, v[i], v[i + 1], 1) : v[i]);
                    }
                    v.swap(w);
                }
                return v[0];
            };
            if (pick_blk) {
                cand_sel    = join(parts_sel);
                cand_n_comp = n_comp;
            }
            top_k = ggml_cont(ctx0, join(parts_top_k));
            cb(top_k, "idx_top_k", il);
            topk_of[il] = top_k;
        }
        GGML_ASSERT(top_k && "deepseek41: top-k source layer produced no top-k");

        // dense attention over [window | all compressed rows], selection applied as a mask
        ggml_tensor * comp_k = ggml_view_4d(ctx0, comp_all, n_embd_head, 1, n_comp, 1,
                comp_all->nb[1], comp_all->nb[2], comp_all->nb[3], 0);
        comp_k = comp_k->type != raw_k->type ? ggml_cast(ctx0, comp_k, raw_k->type) : ggml_cont(ctx0, comp_k);
        cb(comp_k, "comp_k", il);
        ggml_tensor * k_all = ggml_concat(ctx0, raw_k, comp_k, 2);

        if (gather) {
            // rows of k_all: the window rows, then the top-k compressed rows offset by n_raw
            const int64_t k_sel = top_k->ne[0];
            const int64_t w_raw = GGML_PAD((int64_t) hparams.n_swa + k_sel, 256) - k_sel;
            if (g_idx_of.size() < topk_of.size()) {
                g_idx_of.resize(topk_of.size(), nullptr);
            }
            ggml_tensor *& idx = g_idx_of[topk_src];
            if (!idx) {
                ggml_tensor * ic = ggml_scale_bias(ctx0, ggml_cast(ctx0, top_k, GGML_TYPE_F32), 1.0f, (float) n_raw);
                idx = ggml_concat(ctx0, raw_idx(w_raw), ggml_cast(ctx0, ic, GGML_TYPE_I32), 0);
                cb(idx, "gather_idx", il);
            }
            GGML_ASSERT(ratio >= 1 && ratio <= 2);
            ggml_tensor *& m2d = g_mask_of[ratio];
            if (!m2d) {
                m2d = ggml_concat(ctx0, raw_mask, ci.kq_mask, 0);
                cb(m2d, "gather_kq_mask", il);
            }
            ggml_tensor * k2d = ggml_view_2d(ctx0, k_all, k_all->ne[0], k_all->ne[2], k_all->nb[2], 0);
            out = attn_gather(k2d, m2d, idx);
            cb(out, "attn_comp", il);
        } else {
            ggml_tensor * sel_mask = build_top_k_mask(ci.kq_mask, top_k, "comp_top_k_mask", il);
            ggml_tensor * kq_mask  = ggml_concat(ctx0, raw_mask, sel_mask, 0);
            cb(kq_mask, "comp_kq_mask", il);

            out = build_attn_mha(q, k_all, k_all, nullptr, mask_cnv(kq_mask), layer.attn_sinks, nullptr, kq_scale, il);
            cb(out, "attn_comp", il);
        }
    }

    ggml_tensor * oa = out_is_oa ? out : wo_a_of(out, inp_pos);

    out = build_lora_mm(layer.wo_b, oa);
    cb(out, "attn_out", il);

    return out;
}

//
// graph
//

llm_build_deepseek41::llm_build_deepseek41(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params), model(model) {
    const auto * mctx41 = static_cast<const llama_kv_cache_dsv41_context *>(mctx);

    topk_of.assign(n_layer, nullptr);

    ggml_tensor * cur;

    ggml_tensor * inp = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    auto inp41_owned = std::make_unique<llm_graph_input_dsv41>(model, cparams, mctx41);
    llm_graph_input_dsv41 * inp41 = inp41_owned.get();

    inp41->stream = mctx41->get_stream();

    // raw window
    inp41->k_idxs   = mctx41->build_input_k_idxs(ctx0, ubatch);
    inp41->raw_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, mctx41->get_n_kv(), n_tokens, 1, 1);
    ggml_set_input(inp41->raw_mask);
    ggml_set_name(inp41->raw_mask, "attn_inp_kq_mask");

    // compressed rows: one set of inputs per ratio in use
    const auto new_input = [&](ggml_type type, int64_t n, const std::string & name) {
        ggml_tensor * t = ggml_new_tensor_1d(ctx0, type, n);
        ggml_set_input(t);
        ggml_set_name(t, name.c_str());
        return t;
    };
    for (uint32_t r = 1; r <= 2; ++r) {
        bool used = false;
        for (int il = 0; il < n_layer; ++il) {
            used = used || hparams.compress_ratios[il] == r;
        }
        if (!used) {
            continue;
        }
        const auto & plan = mctx41->get_plan(r, ubatch);
        auto & ci = inp41->comp[r - 1];
        const std::string pfx = "dsv41_r" + std::to_string(r);

        ci.kq_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, plan.n_comp, n_tokens, 1, 1);
        ggml_set_input(ci.kq_mask);
        ggml_set_name(ci.kq_mask, (pfx + "_kq_mask").c_str());

        ci.write_idxs = new_input(GGML_TYPE_I64, plan.n_blocks, pfx + "_write_idxs");
        ci.write_pos  = new_input(GGML_TYPE_I32, plan.n_blocks, pfx + "_write_pos");
        if (r > 1) {
            ci.read_idxs   = new_input(GGML_TYPE_I32, (int64_t) plan.read_idxs.size(),   pfx + "_read_idxs");
            ci.persist_src = new_input(GGML_TYPE_I32, (int64_t) plan.persist_src.size(), pfx + "_persist_src");
            ci.persist_dst = new_input(GGML_TYPE_I64, (int64_t) plan.persist_dst.size(), pfx + "_persist_dst");
        }
    }

    // Engram rows are read on the host from the on-disk tables; embedding input has no token ids and skips Engram
    std::vector<ggml_tensor *> engram_rows;
    for (int il = 0; ubatch.token && hparams.ple_n_heads > 0 && il < n_layer; ++il) {
        if (!hparams.is_ple(il)) {
            continue;
        }
        ggml_tensor * rows = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, (int64_t) hparams.ple_head_dim*hparams.ple_n_heads, n_tokens);
        ggml_set_input(rows);
        ggml_set_name(rows, ("engram_rows-" + std::to_string(il)).c_str());
        inp41->engram_rows.push_back(rows);
        engram_rows.push_back(rows);
    }

    if (hparams.dsv41_cand_src >= 0) {
        inp41->cand_ratio = hparams.compress_ratios[hparams.dsv41_cand_src];
        const auto & ci = inp41->comp[inp41->cand_ratio - 1];
        const int64_t bs = hparams.dsv41_cand_block_size;
        inp41->cand_pin = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, (ci.kq_mask->ne[0] + bs - 1)/bs, n_tokens, 1, 1);
        ggml_set_input(inp41->cand_pin);
        ggml_set_name(inp41->cand_pin, "cand_pin");
    }

    group_ids = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, hparams.dsv4_o_group_count, n_tokens);
    ggml_set_input(group_ids);
    ggml_set_name(group_ids, "dsv41_group_ids");
    inp41->group_ids = group_ids;

    res->add_input(std::move(inp41_owned));

    ggml_build_forward_expand(gf, inp41->raw_mask);

    const int64_t hc = hparams.hc_count;
    ggml_tensor * inpL = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens), n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    // single-pass mHC: the pre-mix used to collapse the stream lags one sublayer
    ggml_tensor * pre_prev = nullptr;

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        const int e = model.engram_index(il);
        if (e >= 0 && !engram_rows.empty()) {
            inpL = build_engram(inpL, engram_rows[e], il);
            cb(inpL, "engram_out", il);
        }

        ggml_tensor * residual = inpL;
        const hc_mixes attn_mix = build_hc_mixes(inpL, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, il);

        // the first layer's attention collapses with the identity pre-mix: copy 0
        cur = pre_prev ? build_hc_pre(inpL, pre_prev, il) : ggml_cont(ctx0, ggml_view_2d(ctx0, inpL, n_embd, n_tokens, inpL->nb[2], 0));
        cb(cur, "hc_attn_pre", il);

        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        cur = build_attention(inp41, cur, inp_pos, il);

        inpL = build_hc_post(cur, residual, attn_mix.post, attn_mix.comb, il);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;
        const hc_mixes ffn_mix = build_hc_mixes(inpL, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, il);
        pre_prev = ffn_mix.pre;

        cur = build_hc_pre(inpL, attn_mix.pre, il);
        cb(cur, "hc_ffn_pre", il);

        ggml_build_forward_expand(gf, residual);
        ggml_build_forward_expand(gf, ffn_mix.post);
        ggml_build_forward_expand(gf, ffn_mix.comb);

        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        // the reference normalizes only when it routes to more than one expert
        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm && n_expert_used > 1,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il);
        cb(moe_out, "ffn_moe_out", il);

        ggml_tensor * ffn_shexp = build_ffn(cur,
                layer.ffn_up_shexp, nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);

        inpL = build_hc_post(cur, residual, ffn_mix.post, ffn_mix.comb, il);
        cb(inpL, "l_out", il);
    }

    // no hc_head: collapse with the last FFN's pre-mix
    if (inp_out_ids) {
        inpL     = ggml_reshape_3d(ctx0, ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens), inp_out_ids), n_embd, hc, inp_out_ids->ne[0]);
        pre_prev = ggml_get_rows(ctx0, pre_prev, inp_out_ids);
    }
    cur = build_hc_pre(inpL, pre_prev, -1);
    cb(cur, "hc_out", -1);

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
