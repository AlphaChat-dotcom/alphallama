#include "llama-kv-cache-dsv41.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-model.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <stdexcept>

//
// llama_kv_cache_dsv41
//

static uint32_t dsv41_raw_size(const llama_model & model, uint32_t kv_size, uint32_t n_ubatch) {
    // as the SWA half of llama_kv_cache_iswa: the window plus one ubatch, padded to 256
    return GGML_PAD(std::min(kv_size, model.hparams.n_swa + n_ubatch), 256u);
}

llama_kv_cache_dsv41::llama_kv_cache_dsv41(
        const llama_model & model,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_ubatch) :
    llama_kv_cache(model, type_k, type_v, v_trans, offload, /* unified */ false,
                   dsv41_raw_size(model, GGML_PAD(kv_size, 256u), n_ubatch), n_seq_max, /* n_pad */ 1,
                   model.hparams.n_swa, LLAMA_SWA_TYPE_STANDARD, nullptr, nullptr),
    n_ctx_seq(kv_size),
    n_seq_max_41(n_seq_max),
    pos_next(n_seq_max, 0) {

    if (ggml_is_quantized(type_k)) {
        throw std::runtime_error("deepseek41: quantised KV cache types are not supported by this port (use f16 or f32)");
    }

    const uint32_t n_layer     = hparams.n_layer;
    const uint32_t n_embd_head = hparams.n_embd_head_k();
    const uint32_t n_row       = n_embd_head + hparams.indexer_head_size;
    const uint32_t n_stream    = n_seq_max;

    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ size_t(4u*n_layer*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map.emplace(buft, ctx);

            return ctx;
        }

        return it->second.get();
    };

    for (uint32_t il = 0; il < n_layer; ++il) {
        if (hparams.dsv41_kv_src[il] != (int32_t) il) {
            continue;
        }

        const uint32_t ratio = hparams.compress_ratios[il];
        GGML_ASSERT(ratio == 1 || ratio == 2);

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
        if (offload) {
            buft = ggml_backend_dev_buffer_type(model.dev_layer(il));
        }

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for the deepseek41 compressed rows");
        }

        comp_layer cl;
        cl.il    = (int32_t) il;
        cl.ratio = ratio;
        cl.rows  = ggml_new_tensor_3d(ctx, type_k, n_row, comp_capacity(ratio) + 1, n_stream);
        ggml_format_name(cl.rows, "dsv41_comp_l%d", il);
        if (ratio > 1) {
            cl.st_kv    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd_head, ratio, n_stream);
            cl.st_score = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd_head, ratio, n_stream);
            ggml_format_name(cl.st_kv,    "dsv41_state_kv_l%d",    il);
            ggml_format_name(cl.st_score, "dsv41_state_score_l%d", il);
        }

        comp_map[(int32_t) il] = (int32_t) comp_layers.size();
        comp_layers.push_back(cl);
    }

    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf;
        if (model.hparams.no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                t->buffer = buf;
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
        }
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for the deepseek41 compressed rows");
        }

        LLAMA_LOG_INFO("%s: %10s compressed rows buffer size = %8.2f MiB (%zu source layers, %u seqs, %u positions)\n", __func__,
                ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0, comp_layers.size(), n_stream, kv_size);

        // rows that were never written are read (masked): keep them deterministic
        ggml_backend_buffer_clear(buf, 0);
        comp_bufs.emplace_back(std::move(ctx), buf);
    }
}

uint32_t llama_kv_cache_dsv41::comp_capacity(uint32_t ratio) const {
    return GGML_PAD(std::max<uint32_t>(1, (n_ctx_seq + ratio - 1)/ratio), 256u);
}

const llama_kv_cache_dsv41::comp_layer & llama_kv_cache_dsv41::get_comp_layer(int32_t il_src) const {
    return comp_layers[comp_map.at(il_src)];
}

uint32_t llama_kv_cache_dsv41::comp_stream(llama_seq_id seq_id) const {
    GGML_ASSERT(seq_id >= 0 && (uint32_t) seq_id < n_seq_max_41);
    return get_stream(seq_id);
}

llama_pos llama_kv_cache_dsv41::get_pos_next(llama_seq_id seq_id) const {
    GGML_ASSERT(seq_id >= 0 && (uint32_t) seq_id < n_seq_max_41);
    return pos_next[seq_id];
}

void llama_kv_cache_dsv41::set_pos_next(llama_seq_id seq_id, llama_pos pos) {
    GGML_ASSERT(seq_id >= 0 && (uint32_t) seq_id < n_seq_max_41);
    pos_next[seq_id] = pos;
}

llama_dsv41_plan llama_kv_cache_dsv41::build_plan(const llama_ubatch & ubatch, uint32_t ratio) const {
    // the ubatch holds consecutive positions of one sequence (checked in init_batch)
    llama_dsv41_plan plan;
    plan.ratio = ratio;

    const int64_t   n_tokens = ubatch.n_tokens;
    const llama_pos p0       = ubatch.pos[0];
    const int64_t   cap      = comp_capacity(ratio);
    const int64_t   r        = ratio;

    plan.n_visible.resize(n_tokens);

    int64_t n_vis_max = 0;
    for (int64_t i = 0; i < n_tokens; ++i) {
        const llama_pos pos = p0 + (llama_pos) i;

        const int64_t n_visible = ((int64_t) pos + 1)/r;
        plan.n_visible[i] = (int32_t) n_visible;
        n_vis_max = std::max(n_vis_max, n_visible);

        if ((pos + 1) % r != 0) {
            continue;
        }

        const llama_pos start = pos + 1 - (llama_pos) r;

        plan.write_idxs.push_back(pos/r);
        plan.write_pos .push_back((int32_t) start);

        for (int64_t j = 0; j < r; ++j) {
            const llama_pos sp = start + (llama_pos) j;
            // tokens of this ubatch follow the ring rows in the gather source
            plan.read_idxs.push_back(sp >= p0 ? (int32_t) (r + (sp - p0)) : (int32_t) (sp % r));
        }
    }

    if (plan.write_idxs.empty()) {
        // keep the compress and write ops in the graph: a block made of the first token goes to the scratch row
        plan.write_idxs.push_back(cap);
        plan.write_pos .push_back(0);
        for (int64_t j = 0; j < r; ++j) {
            plan.read_idxs.push_back((int32_t) r);
        }
    }

    plan.n_blocks = (int64_t) plan.write_idxs.size();
    plan.n_comp   = std::max<int64_t>(GGML_PAD(n_vis_max, 256), 256);
    GGML_ASSERT(plan.n_comp <= cap);

    if (r > 1) {
        // the newest token of each ring row
        for (int64_t d = 0; d < r; ++d) {
            for (int64_t i = n_tokens - 1; i >= 0; --i) {
                if ((p0 + i) % r == d) {
                    plan.persist_src.push_back((int32_t) i);
                    plan.persist_dst.push_back((int32_t) d);
                    break;
                }
            }
        }
    }

    return plan;
}

uint32_t alpha_get_kv_reserve_cap(void);   // llama-kv-cache.cpp

llama_dsv41_plan llama_kv_cache_dsv41::build_plan_reserve(const llama_ubatch & ubatch, uint32_t ratio) const {
    llama_dsv41_plan plan;
    plan.ratio = ratio;

    const int64_t n_tokens = ubatch.n_tokens;
    const int64_t r        = ratio;

    plan.n_comp   = comp_capacity(ratio);
    {
        // ai4: reserve against the capped context (see alpha_set_kv_reserve_cap), not the whole cache
        static const int64_t env_cap = [] { const char * e = getenv("ALPHA_KV_RESERVE_CAP"); return e ? atoll(e) : 0LL; }();
        int64_t cap = (int64_t) alpha_get_kv_reserve_cap();
        if (cap <= 0) cap = env_cap;
        if (cap > 0) plan.n_comp = std::min<int64_t>(plan.n_comp, std::max<int64_t>(1, (cap + r - 1)/r));
    }
    plan.n_blocks = std::max<int64_t>(1, (n_tokens + r - 1)/r);

    plan.n_visible  .assign(n_tokens, 0);
    plan.read_idxs  .assign(r*plan.n_blocks, 0);
    plan.write_idxs .assign(plan.n_blocks, 0);
    plan.write_pos  .assign(plan.n_blocks, 0);
    if (r > 1) {
        plan.persist_src.assign(std::min<int64_t>(r, n_tokens), 0);
        plan.persist_dst.assign(std::min<int64_t>(r, n_tokens), 0);
    }

    return plan;
}

llama_memory_context_ptr llama_kv_cache_dsv41::init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) {
    GGML_UNUSED(embd_all);

    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = balloc.split_seq(n_ubatch);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            break;
        }

        // one sequence per ubatch, consecutive positions that continue the sequence
        std::vector<llama_pos> next = pos_next;
        bool ok = true;
        for (const auto & ubatch : ubatches) {
            if (ubatch.n_seqs_unq != 1) {
                LLAMA_LOG_ERROR("%s: deepseek41 does not support tokens shared between sequences\n", __func__);
                ok = false;
                break;
            }
            const llama_seq_id seq_id = ubatch.seq_id_unq[0];
            if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max_41) {
                ok = false;
                break;
            }
            for (uint32_t i = 0; i < ubatch.n_tokens && ok; ++i) {
                if (ubatch.n_seq_id[i] != 1 || ubatch.pos[i] != next[seq_id]) {
                    LLAMA_LOG_ERROR("%s: deepseek41 needs consecutive positions: seq %d expects position %d, got %d\n",
                            __func__, seq_id, next[seq_id], ubatch.pos[i]);
                    ok = false;
                    break;
                }
                next[seq_id]++;
            }
            if (!ok) {
                break;
            }
            if ((uint32_t) next[seq_id] > n_ctx_seq) {
                LLAMA_LOG_ERROR("%s: sequence %d exceeds the context size (%u)\n", __func__, seq_id, n_ctx_seq);
                ok = false;
                break;
            }
        }
        if (!ok) {
            break;
        }

        auto sinfos = prepare(ubatches);
        if (sinfos.empty()) {
            break;
        }

        return std::make_unique<llama_kv_cache_dsv41_context>(
                this, std::move(sinfos), std::move(ubatches));
    } while (false);

    return std::make_unique<llama_kv_cache_dsv41_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_kv_cache_dsv41::init_full() {
    return std::make_unique<llama_kv_cache_dsv41_context>(this);
}

bool llama_kv_cache_dsv41::get_can_shift() const {
    return false;
}

static void dsv41_clear_stream(ggml_tensor * t, uint32_t s) {
    if (t && t->buffer && ggml_backend_buffer_get_base(t->buffer) != nullptr) {
        ggml_backend_tensor_memset(t, 0, s*t->nb[2], t->nb[2]);
    }
}

void llama_kv_cache_dsv41::clear_comp(llama_seq_id seq_id) {
    for (uint32_t s = 0; s < n_seq_max_41; ++s) {
        if (seq_id >= 0 && comp_stream(seq_id) != s) {
            continue;
        }
        for (auto & cl : comp_layers) {
            dsv41_clear_stream(cl.rows,     s);
            dsv41_clear_stream(cl.st_kv,    s);
            dsv41_clear_stream(cl.st_score, s);
        }
    }
    for (uint32_t id = 0; id < n_seq_max_41; ++id) {
        if (seq_id < 0 || (llama_seq_id) id == seq_id) {
            pos_next[id] = 0;
        }
    }
}

void llama_kv_cache_dsv41::clear(bool data) {
    llama_kv_cache::clear(data);

    if (data) {
        clear_comp(-1);
    } else {
        std::fill(pos_next.begin(), pos_next.end(), 0);
    }
}

bool llama_kv_cache_dsv41::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (p1 >= 0) {
        // only a tail can be removed: the compressed rows cannot have holes
        if (seq_id >= 0 && (uint32_t) seq_id < n_seq_max_41 && p1 < pos_next[seq_id]) {
            return false;
        }
    }

    if (seq_id < 0) {
        if (p0 > 0) {
            return false;
        }
        const bool res = llama_kv_cache::seq_rm(seq_id, p0, p1);
        if (res) {
            clear_comp(-1);
        }
        return res;
    }

    if ((uint32_t) seq_id >= n_seq_max_41) {
        return false;
    }

    const llama_pos n_cur = pos_next[seq_id];

    if (p0 >= n_cur) {
        return true; // nothing to remove
    }

    if (p0 <= 0) {
        const bool res = llama_kv_cache::seq_rm(seq_id, p0, -1);
        if (res) {
            clear_comp(seq_id);
        }
        return res;
    }

    // truncation to p0 tokens. The compressed rows of the removed positions are overwritten when they are decoded
    // again and stay invisible until then. What must still be there:
    //   - the compressor state of the open block: its row is intact when the block starts at p0, or when only the
    //     last token is removed (that token did not overwrite the row of p0 - 1)
    //   - the raw window of p0
    for (const auto & cl : comp_layers) {
        if (p0 % (llama_pos) cl.ratio != 0 && p0 != n_cur - 1) {
            return false;
        }
    }
    const llama_pos p_need = std::max<llama_pos>(0, p0 - (llama_pos) hparams.n_swa + 1);
    if (llama_kv_cache::seq_pos_min(seq_id) > p_need) {
        return false;
    }

    if (!llama_kv_cache::seq_rm(seq_id, p0, -1)) {
        return false;
    }
    pos_next[seq_id] = p0;

    return true;
}

void llama_kv_cache_dsv41::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    GGML_ASSERT(p0 <= 0 && p1 < 0 && "deepseek41 only supports full sequence copies");

    llama_kv_cache::seq_cp(seq_id_src, seq_id_dst, p0, p1);

    if (seq_id_src == seq_id_dst) {
        return;
    }

    const uint32_t s_src = comp_stream(seq_id_src);
    const uint32_t s_dst = comp_stream(seq_id_dst);

    std::vector<uint8_t> tmp;
    auto cp = [&](ggml_tensor * t) {
        if (!t || !t->buffer || ggml_backend_buffer_get_base(t->buffer) == nullptr) {
            return;
        }
        tmp.resize(t->nb[2]);
        ggml_backend_tensor_get(t, tmp.data(), s_src*t->nb[2], t->nb[2]);
        ggml_backend_tensor_set(t, tmp.data(), s_dst*t->nb[2], t->nb[2]);
    };
    for (auto & cl : comp_layers) {
        cp(cl.rows);
        cp(cl.st_kv);
        cp(cl.st_score);
    }

    pos_next[seq_id_dst] = pos_next[seq_id_src];
}

void llama_kv_cache_dsv41::seq_keep(llama_seq_id seq_id) {
    llama_kv_cache::seq_keep(seq_id);

    for (uint32_t id = 0; id < n_seq_max_41; ++id) {
        if ((llama_seq_id) id != seq_id) {
            clear_comp((llama_seq_id) id);
        }
    }
}

void llama_kv_cache_dsv41::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    GGML_UNUSED(seq_id); GGML_UNUSED(p0); GGML_UNUSED(p1); GGML_UNUSED(shift);
    LLAMA_LOG_ERROR("%s: deepseek41 does not support position shifts (context shift)\n", __func__);
}

void llama_kv_cache_dsv41::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    GGML_UNUSED(seq_id); GGML_UNUSED(p0); GGML_UNUSED(p1); GGML_UNUSED(d);
    LLAMA_LOG_ERROR("%s: deepseek41 does not support position division\n", __func__);
}

llama_pos llama_kv_cache_dsv41::seq_pos_min(llama_seq_id seq_id) const {
    if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max_41) {
        return -1;
    }

    // as upstream: the compressed state cannot be rolled back inside the window, report only the current boundary
    return llama_kv_cache::seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache_dsv41::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret = llama_kv_cache::memory_breakdown();
    for (const auto & [ctx, buf] : comp_bufs) {
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf.get());

        if (hparams.no_alloc) {
            ret[buft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
        } else {
            ret[buft] += ggml_backend_buffer_get_size(buf.get());
        }
    }

    return ret;
}

void llama_kv_cache_dsv41::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    GGML_UNUSED(io); GGML_UNUSED(seq_id); GGML_UNUSED(flags);
    throw std::runtime_error("deepseek41: saving the memory state is not supported by this port");
}

void llama_kv_cache_dsv41::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(io); GGML_UNUSED(seq_id); GGML_UNUSED(flags);
    throw std::runtime_error("deepseek41: restoring the memory state is not supported by this port");
}

//
// llama_kv_cache_dsv41_context
//

llama_kv_cache_dsv41_context::llama_kv_cache_dsv41_context(llama_memory_status status) :
    llama_kv_cache_context(status) {}

llama_kv_cache_dsv41_context::llama_kv_cache_dsv41_context(llama_kv_cache_dsv41 * kv) :
    llama_kv_cache_context(kv), kv41(kv), full(true) {}

llama_kv_cache_dsv41_context::llama_kv_cache_dsv41_context(
        llama_kv_cache_dsv41 * kv,
        slot_info_vec_t sinfos,
        std::vector<llama_ubatch> ubatches) :
    llama_kv_cache_context(kv, std::move(sinfos), ubatches),
    kv41(kv),
    ubatches41(std::move(ubatches)) {
    plans.resize(ubatches41.size());
    for (size_t i = 0; i < ubatches41.size(); ++i) {
        plans[i][0] = kv->build_plan(ubatches41[i], 1);
        plans[i][1] = kv->build_plan(ubatches41[i], 2);
    }
}

bool llama_kv_cache_dsv41_context::next() {
    const bool res = llama_kv_cache_context::next();
    ++i_cur;
    return res;
}

bool llama_kv_cache_dsv41_context::apply() {
    const bool res = llama_kv_cache_context::apply();

    if (res && !full && i_cur < ubatches41.size()) {
        const auto & ubatch = ubatches41[i_cur];
        kv41->set_pos_next(ubatch.seq_id_unq[0], ubatch.pos[ubatch.n_tokens - 1] + 1);
    }

    return res;
}

uint32_t llama_kv_cache_dsv41_context::get_stream() const {
    if (full) {
        return 0;
    }
    GGML_ASSERT(i_cur < ubatches41.size());
    return kv41->comp_stream(ubatches41[i_cur].seq_id_unq[0]);
}

const llama_dsv41_plan & llama_kv_cache_dsv41_context::get_plan(uint32_t ratio, const llama_ubatch & ubatch) const {
    GGML_ASSERT(ratio == 1 || ratio == 2);
    if (full) {
        plans_reserve[ratio - 1] = kv41->build_plan_reserve(ubatch, ratio);
        return plans_reserve[ratio - 1];
    }
    GGML_ASSERT(i_cur < plans.size());
    return plans[i_cur][ratio - 1];
}

ggml_tensor * llama_kv_cache_dsv41_context::get_comp(ggml_context * ctx, int32_t il_src) const {
    const auto & cl = kv41->get_comp_layer(il_src);
    ggml_tensor * t = cl.rows;
    return ggml_view_4d(ctx, t, t->ne[0], 1, t->ne[1], 1, t->nb[1], t->nb[1], t->nb[2], get_stream()*t->nb[2]);
}

static ggml_tensor * dsv41_stream_2d(ggml_context * ctx, ggml_tensor * t, uint32_t s) {
    return ggml_view_2d(ctx, t, t->ne[0], t->ne[1], t->nb[1], s*t->nb[2]);
}

ggml_tensor * llama_kv_cache_dsv41_context::cpy_comp(ggml_context * ctx, ggml_tensor * rows, ggml_tensor * idxs, int32_t il_src) const {
    const auto & cl = kv41->get_comp_layer(il_src);
    GGML_ASSERT(rows->ne[0] == cl.rows->ne[0]);
    rows = ggml_reshape_2d(ctx, rows, rows->ne[0], ggml_nelements(rows)/rows->ne[0]);
    return ggml_set_rows(ctx, dsv41_stream_2d(ctx, cl.rows, get_stream()), rows, idxs);
}

ggml_tensor * llama_kv_cache_dsv41_context::get_state_kv(ggml_context * ctx, int32_t il_src) const {
    return dsv41_stream_2d(ctx, kv41->get_comp_layer(il_src).st_kv, get_stream());
}

ggml_tensor * llama_kv_cache_dsv41_context::get_state_score(ggml_context * ctx, int32_t il_src) const {
    return dsv41_stream_2d(ctx, kv41->get_comp_layer(il_src).st_score, get_stream());
}

ggml_tensor * llama_kv_cache_dsv41_context::cpy_state_kv(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * idxs, int32_t il_src) const {
    return ggml_set_rows(ctx, get_state_kv(ctx, il_src), cur, idxs);
}

ggml_tensor * llama_kv_cache_dsv41_context::cpy_state_score(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * idxs, int32_t il_src) const {
    return ggml_set_rows(ctx, get_state_score(ctx, il_src), cur, idxs);
}
