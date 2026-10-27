#include "llama-memory-hybrid-idx.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-context.h"

#include <algorithm>
#include <cmath>      // INFINITY
#include <cstdint>
#include <thread>

//
// llama_memory_hybrid_idx
//

llama_memory_hybrid_idx::llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
    const layer_filter_cb & filter_idx,
                ggml_type   type_idx) :
    llama_memory_hybrid(
        model,
        type_k, type_v, v_trans, kv_size, n_pad, n_swa, swa_type,
        type_r, type_s, rs_size,
        n_seq_max, offload, unified,
        filter_attn, filter_recr),
    hparams_idx(model.hparams),
    mem_idx(filter_idx == nullptr ? nullptr : [&] {
        // MQA with a single key head of indexer_head_size. The indexer stores one key per token
        // per indexer layer, not a GQA-shaped K, so the cache is given a modified copy of hparams
        // rather than the model's own.
        std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
        hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size;
        // With ALPHA_QSA_PKC, V is never read as values: it holds the pooled-key cache, idx_dim f32 per row
        hparams_idx.n_embd_head_v_full = model.hparams.indexer_head_size;
        hparams_idx.n_embd_head_v_swa  = model.hparams.indexer_head_size;

        // The cached indexer keys are raw: pooling happens first and rotation only at read time,
        // so a K-shift must not rotate them while the stream copies in the same update still apply.
        hparams_idx.rope_type = LLAMA_ROPE_TYPE_NONE;

        LLAMA_LOG_INFO("%s: creating indexer KV cache, size = %u cells, key dim = %u\n",
                       __func__, kv_size, model.hparams.indexer_head_size);

        return new llama_kv_cache(
            model, hparams_idx, type_idx, GGML_TYPE_F32, /* v_trans */ false, offload, unified,
            kv_size, n_seq_max, n_pad, n_swa, swa_type,
            filter_idx, nullptr);
    }()) {
    pkc_ok = mem_idx != nullptr;   // one stream is checked per ubatch (kv_unified is false at -np 1 yet has one stream)
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_batch(
        llama_batch_allocr & balloc,
        uint32_t n_ubatch,
        bool embd_all) {
    // note: repeats llama_memory_hybrid::init_batch, because the indexer needs the attention slot
    //       infos that the base context keeps to itself
    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                const bool unified = (get_mem_attn()->get_n_stream() == 1);
                ubatch = balloc.split_equal(n_ubatch, !unified);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!get_mem_recr()->prepare(ubatches)) {
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = get_mem_attn()->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // Invariant: the indexer reuses the attention cache's slot layout verbatim. Letting it
        // run its own prepare() would let the two drift, and build_qsa_top_k assumes cell j is the
        // same token in both.
        llama_kv_cache::slot_info_vec_t heads_idx;
        if (mem_idx) {
            heads_idx = heads_attn;
        }

        return std::make_unique<llama_memory_hybrid_idx_context>(
                this, std::move(heads_attn), std::move(heads_idx), std::move(ubatches));
    } while (false);

    return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_full() {
    return std::make_unique<llama_memory_hybrid_idx_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_idx_context>(this, lctx, optimize);
}

void llama_memory_hybrid_idx::clear(bool data) {
    qsa_pkc_invalidate();
    llama_memory_hybrid::clear(data);
    if (mem_idx) {
        mem_idx->clear(data);
    }
}

bool llama_memory_hybrid_idx::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    const bool res = llama_memory_hybrid::seq_rm(seq_id, p0, p1);
    if (mem_idx) {
        // the attention cache decides; the indexer must follow it cell for cell
        mem_idx->seq_rm(seq_id, p0, p1);
    }
    return res;
}

void llama_memory_hybrid_idx::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    llama_memory_hybrid::seq_cp(seq_id_src, seq_id_dst, p0, p1);
    if (mem_idx) {
        mem_idx->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    }
}

void llama_memory_hybrid_idx::seq_keep(llama_seq_id seq_id) {
    llama_memory_hybrid::seq_keep(seq_id);
    if (mem_idx) {
        mem_idx->seq_keep(seq_id);
    }
}

void llama_memory_hybrid_idx::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    qsa_pkc_invalidate();
    llama_memory_hybrid::seq_add(seq_id, p0, p1, shift);
    if (mem_idx) {
        mem_idx->seq_add(seq_id, p0, p1, shift);
    }
}

void llama_memory_hybrid_idx::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    qsa_pkc_invalidate();
    llama_memory_hybrid::seq_div(seq_id, p0, p1, d);
    if (mem_idx) {
        mem_idx->seq_div(seq_id, p0, p1, d);
    }
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_idx::memory_breakdown() const {
    auto ret = llama_memory_hybrid::memory_breakdown();
    if (mem_idx) {
        for (const auto & [buft, size] : mem_idx->memory_breakdown()) {
            ret[buft] += size;
        }
    }
    return ret;
}

void llama_memory_hybrid_idx::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    llama_memory_hybrid::state_write(io, seq_id, flags);
    if (mem_idx) {
        mem_idx->state_write(io, seq_id, flags);
    }
}

void llama_memory_hybrid_idx::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    qsa_pkc_invalidate();
    llama_memory_hybrid::state_read(io, seq_id, flags);
    if (mem_idx) {
        mem_idx->state_read(io, seq_id, flags);
    }
}

//
// llama_memory_hybrid_idx_context
//

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_status status) :
    llama_memory_hybrid_context(status) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem) :
    llama_memory_hybrid_context(mem),
    ctx_idx(mem->get_mem_idx() ? mem->get_mem_idx()->init_full() : nullptr),
    mem(mem) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                  llama_context * lctx,
                           bool   optimize) :
    llama_memory_hybrid_context(mem, lctx, optimize),
    ctx_idx(mem->get_mem_idx() ? mem->get_mem_idx()->init_update(lctx, optimize) : nullptr),
    mem(mem), is_update(true) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                  slot_info_vec_t   sinfos_attn,
                  slot_info_vec_t   sinfos_idx,
        std::vector<llama_ubatch>   ubatches) :
    llama_memory_hybrid_context(mem, std::move(sinfos_attn), ubatches),
    ctx_idx(mem->get_mem_idx()
            ? llama_memory_context_ptr(new llama_kv_cache_context(
                  mem->get_mem_idx(), std::move(sinfos_idx), this->ubatches))
            : nullptr),
    mem(mem) {}

bool llama_memory_hybrid_idx_context::next() {
    // advance the indexer in lockstep with the base's two children
    if (ctx_idx) {
        ctx_idx->next();
    }
    return llama_memory_hybrid_context::next();
}

bool llama_memory_hybrid_idx_context::apply() {
    // an update (stream copy, cell moves) may move V rows, which hold the pooled-key cache
    if (is_update && mem) {
        mem->qsa_pkc_invalidate();
    }
    bool res = llama_memory_hybrid_context::apply();
    if (ctx_idx) {
        res = res & ctx_idx->apply();
    }
    return res;
}

const llama_kv_cache_context * llama_memory_hybrid_idx_context::get_idx() const {
    return ctx_idx ? static_cast<const llama_kv_cache_context *>(ctx_idx.get()) : nullptr;
}

// Group the attention cells into blocks of `ratio` consecutive positions and describe them for
// build_qsa_top_k:
//   cell_blk  [n_kv, n_stream]            cell -> block id (unpooled cells point at a spare block)
//   blk_cells [ratio*n_blocks, n_stream]  block slot -> cell, so the pooler can gather members
//   blk_pos   [4*n_blocks*n_stream]       per-block mrope section positions, for rotating the
//                                         pooled key at the block's own position
//   bias      [n_blocks or n_kv, n_tps, n_stream]  additive score bias:
//                                         -inf invisible, 1e9 always-visible tail, 0 otherwise
//
// A block is keyed on (sequence set, position bucket): in a unified cache every sequence counts
// from zero, so the bucket alone would pool cells of two different sequences into one block.
// Only full blocks (all `ratio` slots present) are poolable; the rest fall into the spare block,
// which carries the tail bias so it is always selected.
void llama_memory_hybrid_idx::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias) const {
    GGML_ASSERT(ratio > 0);
    GGML_ASSERT(get_mem_idx() != nullptr);
    GGML_ASSERT(ggml_backend_buffer_is_host(cell_blk->buffer));

    const int64_t n_kv     = cell_blk->ne[0];
    const int64_t n_ns     = cell_blk->ne[1];        // streams in this ubatch
    const int64_t n_blocks = blk_pos->ne[0]/(4*n_ns);
    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t r        = ratio;

    GGML_ASSERT(n_tokens % n_ns == 0);
    const int64_t n_tps = n_tokens/n_ns;             // tokens per stream

    int32_t * dst_cell_blk  = (int32_t *) cell_blk->data;
    // Under ALPHA_QSA_PKC mode 2 the graph never reads blk_cells/blk_pos, so they get no buffer
    int32_t * dst_blk_cells = blk_cells->buffer ? (int32_t *) blk_cells->data : nullptr;
    int32_t * dst_blk_pos   = blk_pos->buffer   ? (int32_t *) blk_pos->data   : nullptr;
    float   * dst_bias      = (float   *) bias->data;

    GGML_ASSERT(r <= 64);
    const uint64_t slots_full = r == 64 ? ~uint64_t(0) : ((uint64_t(1) << r) - 1);

    std::vector<int32_t>  blk_of(n_kv);
    std::vector<int32_t>  cell_grp(n_kv);
    std::vector<int32_t>  grp_head(n_blocks);
    std::vector<int32_t>  grp_next;
    std::vector<int32_t>  grp_first;
    std::vector<int32_t>  grp_slot0;
    std::vector<uint64_t> grp_slots;
    std::vector<int32_t>  grp_bid;
    std::vector<int32_t>  bid_idx;
    std::vector<int32_t>  bid_cell;
    std::vector<int32_t>  bid_slot0;

    std::vector<int32_t> order;
    std::vector<int32_t> rank;

    if (dst_blk_pos) {
        std::fill(dst_blk_pos, dst_blk_pos + 4*n_blocks*n_ns, 0);
    }

    for (int64_t s = 0; s < n_ns; ++s) {
        // ubatch index s*n_tps belongs to this stream; ask which cells array it uses
        const llama_seq_id seq_of_stream = ubatch->seq_id[s*n_tps][0];
        const auto & cells = get_mem_idx()->get_cells(seq_of_stream);

        int32_t * cur_cell_blk  = dst_cell_blk  + s*n_kv;
        int32_t * cur_blk_cells = dst_blk_cells ? dst_blk_cells + s*(r*n_blocks) : nullptr;

        if (cur_blk_cells) {
            std::fill(cur_blk_cells, cur_blk_cells + r*n_blocks, 0);
        }

        bid_idx  .clear();
        bid_cell .clear();
        bid_slot0.clear();

        int n_seq_present = 0;

        for (int sq = 0; sq < LLAMA_MAX_SEQ && n_seq_present < 2; ++sq) {
            if (cells.seq_pos_min(sq) >= 0) {
                n_seq_present++;
            }
        }

        const bool one_seq = n_seq_present <= 1;

        // ALPHA_QSA_FAST (default on): one sequence whose cells are laid out in order (cell j holds position j, the
        // used cells a prefix): blocks are j/r, the incomplete tail goes to the spare block, and the per-cell grouping
        // below (hashing every cell) reduces to index arithmetic. Same outputs.
        // ALPHA_QSA_FAST=2 computes both and checks them.
        static const int qsa_fast = [] { const char * e = getenv("ALPHA_QSA_FAST"); return e ? atoi(e) : 1; }();
        bool fast_done = false;
        std::vector<int32_t> f_cell_blk, f_blk_cells, f_blk_pos; std::vector<float> f_bias;
        if (qsa_fast && one_seq && blk_bias) {   // distinct positions (checked below) => the mrope ranking never applies
            int64_t n_used = 0;
            bool ok = true;
            for (int64_t j = 0; j < n_kv; ++j) {
                if (cells.is_empty(j)) { if (n_used == 0 && j == 0) { } break; }
                if (cells.pos_get(j) != j) { ok = false; break; }
                n_used = j + 1;
            }
            for (int64_t j = n_used; ok && j < n_kv; ++j) if (!cells.is_empty(j)) ok = false;
            for (int64_t ii = 0; ok && ii < n_tps; ++ii) if (n_used > 0 && !cells.seq_has(0, ubatch->seq_id[s*n_tps + ii][0])) ok = false;
            if (ok && n_used / r <= n_blocks) {
                const int64_t n_full  = n_used / r;
                const int32_t n_bidf  = (int32_t) n_full;
                const bool    dead    = n_bidf < n_blocks;
                const int32_t dead_b  = dead ? n_bidf : (int32_t) (n_blocks - 1);
                f_cell_blk.resize(n_kv);
                for (int64_t j = 0; j < n_kv; ++j) f_cell_blk[j] = j < n_full*r ? (int32_t) (j/r) : dead_b;
                if (cur_blk_cells) {
                    f_blk_cells.assign(r*n_blocks, 0);
                    for (int64_t j = 0; j < n_full*r; ++j) f_blk_cells[j] = (int32_t) j;   // block b, slot j%r
                }
                if (dst_blk_pos) {
                    f_blk_pos.assign(4*n_blocks, 0);
                    for (int64_t sec = 0; sec < 4; ++sec) for (int64_t b = 0; b < n_full; ++b) f_blk_pos[sec*n_blocks + b] = (int32_t) (b*r);
                }
                // ALPHA_QSA_BIAS_MT: the bias is n_tokens x n_blocks floats (tens of millions for a long prompt chunk,
                // too slow on one thread): write it straight into the input, rows split over threads (rows are
                // independent). The check mode (ALPHA_QSA_FAST=2) keeps the serial copy.
                float * bias_out = nullptr;
                if (qsa_fast == 1) {
                    bias_out = dst_bias + (s*n_tps)*n_blocks;
                } else {
                    f_bias.resize(n_tps*n_blocks);
                    bias_out = f_bias.data();
                }
                auto bias_rows = [&](int64_t i0, int64_t i1) {
                    for (int64_t ii = i0; ii < i1; ++ii) {
                        const int64_t tail_start = ((int64_t) ubatch->pos[s*n_tps + ii] + 1)/r*r;
                        float * cb = bias_out + ii*n_blocks;
                        for (int64_t b = 0; b < n_blocks; ++b) cb[b] = b >= n_full ? -INFINITY : (b*r >= tail_start ? 1e9f : 0.0f);
                        if (dead) cb[dead_b] = 1e9f;
                    }
                };
                const int nth = (qsa_fast == 1 && n_tps*n_blocks >= (1 << 20)) ? 8 : 1;
                if (nth == 1) {
                    bias_rows(0, n_tps);
                } else {
                    std::vector<std::thread> th;
                    for (int t = 0; t < nth; ++t) th.emplace_back(bias_rows, n_tps*t/nth, n_tps*(t + 1)/nth);
                    for (auto & x : th) x.join();
                }
                fast_done = true;
            }
        }
        if (fast_done && qsa_fast == 1) {
            std::copy(f_cell_blk.begin(), f_cell_blk.end(), cur_cell_blk);
            if (cur_blk_cells) std::copy(f_blk_cells.begin(), f_blk_cells.end(), cur_blk_cells);
            if (dst_blk_pos) for (int64_t sec = 0; sec < 4; ++sec)
                std::copy(f_blk_pos.begin() + sec*n_blocks, f_blk_pos.begin() + (sec + 1)*n_blocks, dst_blk_pos + sec*(n_blocks*n_ns) + s*n_blocks);
            continue;   // the bias was written in place
        }

        // a cell no block covers needs its own -inf, which a per-block bias cannot carry
        bool oor    = false;
        bool dup    = false;
        bool ranked = false;

        auto group_cells = [&]() {
            // -1 means no usable block: an incomplete or short group cannot be pooled
            std::fill(blk_of.begin(),   blk_of.end(),   -1);
            std::fill(cell_grp.begin(), cell_grp.end(), -1);
            std::fill(grp_head.begin(), grp_head.end(), -1);

            grp_next .clear();
            grp_first.clear();
            grp_slot0.clear();
            grp_slots.clear();
            grp_bid  .clear();

            oor = false;
            dup = false;

            for (int64_t j = 0; j < n_kv; ++j) {
                if (cells.is_empty(j)) {
                    continue;
                }

                const int64_t idx = ranked ? rank[j] : cells.pos_get(j);
                const int64_t pb  = idx/r;

                if (pb >= n_blocks) {
                    oor = true;
                    continue;
                }

                int32_t g = -1;

                for (int32_t c = grp_head[pb]; c >= 0; c = grp_next[c]) {
                    if (one_seq || cells.seq_get_all((uint32_t) grp_first[c]) == cells.seq_get_all((uint32_t) j)) {
                        g = c;
                        break;
                    }
                }

                if (g < 0) {
                    g = (int32_t) grp_first.size();

                    grp_next .push_back(grp_head[pb]);
                    grp_first.push_back((int32_t) j);
                    grp_slot0.push_back(-1);
                    grp_slots.push_back(0);
                    grp_bid  .push_back(-1);

                    grp_head[pb] = g;
                }

                const uint64_t bit = uint64_t(1) << (idx%r);

                dup |= (grp_slots[g] & bit) != 0;

                cell_grp[j]   = g;
                grp_slots[g] |= bit;

                if (idx%r == 0) {
                    grp_slot0[g] = (int32_t) j;
                }
            }
        };

        group_cells();

        // mrope repeats one position across an image, so rank cells instead of using the position
        if (dup && ubatch->is_pos_2d() && one_seq) {
            order.clear();
            order.reserve(n_kv);

            for (int64_t j = 0; j < n_kv; ++j) {
                if (!cells.is_empty(j)) {
                    order.push_back((int32_t) j);
                }
            }

            // same total order the mrope causal mask uses: pos, then ext.y, then ext.x
            std::sort(order.begin(), order.end(), [&cells](int32_t a, int32_t b) {
                const llama_pos pa = cells.pos_get(a);
                const llama_pos pb = cells.pos_get(b);

                if (pa != pb) {
                    return pa < pb;
                }

                const auto & ea = cells.ext_get(a);

                return cells.ext_get(b).is_2d_gt(ea.x, ea.y);
            });

            rank.assign(n_kv, -1);

            for (int64_t k = 0; k < (int64_t) order.size(); ++k) {
                rank[order[k]] = (int32_t) k;
            }

            ranked = true;

            group_cells();
        }

        GGML_ASSERT((!blk_bias || !oor) && "qsa: cell position runs past the cell window");

        int32_t n_bid = 0;

        for (int64_t pb = 0; pb < n_blocks; ++pb) {
            for (int32_t g = grp_head[pb]; g >= 0; g = grp_next[g]) {
                if (grp_slots[g] != slots_full) {
                    continue;
                }

                grp_bid[g] = n_bid++;

                bid_idx  .push_back((int32_t) (pb*r));
                bid_cell .push_back(grp_first[g]);
                bid_slot0.push_back(grp_slot0[g]);
            }
        }

        GGML_ASSERT(n_bid <= n_blocks);

        for (int32_t b = 0; b < n_bid; ++b) {
            int32_t sec_pos[4] = { bid_idx[b], bid_idx[b], bid_idx[b], bid_idx[b] };

            if (ranked) {
                const int32_t   c = bid_slot0[b];
                const llama_pos p = cells.pos_get(c);
                const auto &    e = cells.ext_get(c);

                sec_pos[0] = p;
                sec_pos[1] = e.y;
                sec_pos[2] = e.x;
                sec_pos[3] = p;
            }

            for (int64_t sec = 0; sec < 4; ++sec) {
                if (dst_blk_pos) {
                    dst_blk_pos[sec*(n_blocks*n_ns) + s*n_blocks + b] = sec_pos[sec];
                }
            }
        }

        // unpooled cells all point at one spare block. a spare block exists only when some cell
        // is unpooled: n_bid == n_blocks means every cell sits in a full block.
        const bool    have_dead = n_bid < n_blocks;
        const int32_t dead_bid  = have_dead ? n_bid : (int32_t) (n_blocks - 1);

        for (int64_t j = 0; j < n_kv; ++j) {
            const int32_t g = cell_grp[j];

            blk_of[j] = g < 0 ? -1 : grp_bid[g];

            if (blk_of[j] >= 0) {
                const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

                if (cur_blk_cells) {
                    cur_blk_cells[blk_of[j]*r + (idx%r)] = (int32_t) j;
                }
            }

            cur_cell_blk[j] = blk_of[j] < 0 ? dead_bid : blk_of[j];
        }

        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t      i      = s*n_tps + ii;
            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            int64_t q = ubatch->pos[i];

            if (ranked) {
                const llama_pos qt = ubatch->pos[i];
                const llama_pos qy = ubatch->pos[i + n_tokens];
                const llama_pos qx = ubatch->pos[i + n_tokens*2];

                int64_t lo = 0;
                int64_t hi = (int64_t) order.size();

                while (lo < hi) {
                    const int64_t   mid = (lo + hi)/2;
                    const int32_t   c   = order[mid];
                    const llama_pos pc  = cells.pos_get(c);

                    if (pc < qt || (pc == qt && !cells.ext_get(c).is_2d_gt(qx, qy))) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }

                q = lo - 1;
            }

            // the tail is an incomplete block and is always visible, as in the reference
            const int64_t tail_start = (q + 1)/r*r;

            if (blk_bias) {
                // a block sits wholly inside or outside the tail, so one value covers it.
                // the caller adds the attention mask, which drops empty, foreign and future cells
                float * cur_blk_bias = dst_bias + i*n_blocks;

                for (int64_t b = 0; b < n_blocks; ++b) {
                    if (b >= n_bid || !cells.seq_has((uint32_t) bid_cell[b], seq_id)) {
                        cur_blk_bias[b] = -INFINITY;
                        continue;
                    }

                    // finite, so it can never meet a -inf and produce a nan
                    cur_blk_bias[b] = bid_idx[b] >= tail_start ? 1e9f : 0.0f;
                }

                // the spare block holds the unpooled cells, which are the incomplete tail, so it
                // gets the tail value. it must stay finite: a sequence with fewer than `ratio`
                // cells owns no full block, and a row of -inf only gives a nan.
                if (have_dead) {
                    cur_blk_bias[dead_bid] = 1e9f;
                }

                continue;
            }

            float * cur_bias = dst_bias + i*n_kv;

            for (int64_t j = 0; j < n_kv; ++j) {
                float v = -INFINITY;

                if (!cells.is_empty(j) && cells.seq_has(j, seq_id)) {
                    const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

                    if (idx <= q) {
                        // finite, so it can never meet a -inf and produce a nan
                        v = idx >= tail_start ? 1e9f : (blk_of[j] < 0 ? -INFINITY : 0.0f);
                    }
                }

                cur_bias[j] = v;
            }
        }
        if (fast_done) {   // ALPHA_QSA_FAST=2: the general path just ran; compare
            bool same = std::equal(f_cell_blk.begin(), f_cell_blk.end(), cur_cell_blk);
            if (cur_blk_cells) same = same && std::equal(f_blk_cells.begin(), f_blk_cells.end(), cur_blk_cells);
            if (dst_blk_pos) for (int64_t sec = 0; sec < 4; ++sec)
                same = same && std::equal(f_blk_pos.begin() + sec*n_blocks, f_blk_pos.begin() + (sec + 1)*n_blocks, dst_blk_pos + sec*(n_blocks*n_ns) + s*n_blocks);
            for (int64_t k = 0; k < n_tps*n_blocks; ++k) {
                const float x = f_bias[k], y = dst_bias[(s*n_tps)*n_blocks + k];
                if (!(x == y || (std::isinf(x) && std::isinf(y) && (x < 0) == (y < 0)))) { same = false; break; }
            }
            static long long n_chk = 0, n_bad = 0;
            n_chk++; n_bad += !same;
            if (!same || (n_chk & 255) == 0) fprintf(stderr, "[qsa-fast] check %lld: %lld mismatches\n", n_chk, n_bad);
        }
    }
}

void llama_memory_hybrid_idx_context::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias) const {
    GGML_ASSERT(mem != nullptr);

    mem->set_input_qsa(cell_blk, blk_cells, blk_pos, bias, ubatch, ratio, blk_bias);
}

int32_t llama_memory_hybrid_idx::qsa_pkc_plan(const llama_ubatch * ubatch, uint32_t ratio, int64_t n_kv,
        std::vector<int32_t> * dirty, std::vector<int32_t> * members) const {
    // mrope ubatches always report is_pos_2d; only a repeated position (an image) needs set_input_qsa's ranking,
    // and the duplicate check below turns that case away
    if (!pkc_ok || !mem_idx || get_mem_attn()->get_n_stream() != 1 || ubatch->n_tokens == 0 ||
        ratio < 2 || ratio > 64) {
        return -1;
    }
    const int64_t r        = ratio;
    const int64_t n_blocks = (n_kv + r - 1)/r;
    // row n_blocks is the scratch row that padded writes land in; it must exist inside the n_kv-row view
    if (n_blocks >= n_kv) {
        return -1;
    }
    const llama_seq_id s0 = ubatch->seq_id[0][0];
    llama_pos pmin = ubatch->pos[0], pmax = ubatch->pos[0];
    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        if (ubatch->n_seq_id[i] != 1 || ubatch->seq_id[i][0] != s0) {
            return -1;
        }
        pmin = std::min(pmin, ubatch->pos[i]);
        pmax = std::max(pmax, ubatch->pos[i]);
    }
    const auto & cells = mem_idx->get_cells(s0);
    int n_seq_present = 0;
    for (int sq = 0; sq < LLAMA_MAX_SEQ; ++sq) {
        if (cells.seq_pos_min(sq) >= 0 && ++n_seq_present > 1) {
            return -1;
        }
    }
    std::vector<int32_t> & m = pkc_scratch;
    m.assign(r*n_blocks, -1);
    for (int64_t j = 0; j < n_kv; ++j) {
        if (cells.is_empty(j)) {
            continue;
        }
        const int64_t p  = cells.pos_get(j);
        const int64_t pb = p/r;
        if (p < 0 || pb >= n_blocks || m[pb*r + p%r] >= 0) {
            return -1;
        }
        m[pb*r + p%r] = (int32_t) j;
    }
    std::vector<int32_t> & sig = pkc_sig[ratio];
    if ((int64_t) sig.size() < r*n_blocks) {
        sig.resize(r*n_blocks, -1);
    }
    if (dirty)   dirty->clear();
    if (members) members->clear();
    int32_t n_dirty = 0;
    for (int64_t pb = 0; pb < n_blocks; ++pb) {
        const int32_t * mb = m.data() + pb*r;
        bool full = true;
        for (int64_t i = 0; i < r; ++i) full &= mb[i] >= 0;
        if (!full) {
            continue;
        }
        const bool touched = pb*r <= pmax && pb*r + r - 1 >= pmin;
        bool stale = touched;
        for (int64_t i = 0; i < r && !stale; ++i) stale = sig[pb*r + i] != mb[i];
        if (!stale) {
            continue;
        }
        n_dirty++;
        if (dirty)   dirty->push_back((int32_t) pb);
        if (members) members->insert(members->end(), mb, mb + r);
    }
    return n_dirty;
}

void llama_memory_hybrid_idx::qsa_pkc_commit(uint32_t ratio, const std::vector<int32_t> & pbs,
        const std::vector<int32_t> & members) const {
    std::vector<int32_t> & sig = pkc_sig[ratio];
    for (size_t k = 0; k < pbs.size(); ++k) {
        if ((size_t) (pbs[k] + 1)*ratio > sig.size()) {
            sig.resize((pbs[k] + 1)*ratio, -1);
        }
        std::copy(members.begin() + k*ratio, members.begin() + (k + 1)*ratio, sig.begin() + (size_t) pbs[k]*ratio);
    }
}

int32_t llama_memory_hybrid_idx_context::qsa_pkc_plan(const llama_ubatch * ubatch, uint32_t ratio, int64_t n_kv) const {
    GGML_ASSERT(mem != nullptr);
    return mem->qsa_pkc_plan(ubatch, ratio, n_kv, nullptr, nullptr);
}

// mode 1: every full block is pooled in-graph; pk_map [n_blocks] sends block id b to its position bucket (row) and
//         the graph writes all of them through to the cache. mode 2: only the stale blocks are pooled: pk_upd [NU]
//         names their rows (padding goes to the scratch row n_blocks), pk_upd_cells [r*NU] their members,
//         pk_upd_pos [4*NU] their rope positions, and pk_map gathers the cached rows back in block-id order.
// Block ids are the full blocks in position order (llama_memory_hybrid_idx::set_input_qsa, one sequence).
// Any other mode: the cache is not maintained, so forget it.
void llama_memory_hybrid_idx_context::set_input_qsa_pkc(int mode, ggml_tensor * pk_map, ggml_tensor * pk_upd,
        ggml_tensor * pk_upd_cells, ggml_tensor * pk_upd_pos, const llama_ubatch * ubatch, uint32_t ratio,
        int64_t n_kv) const {
    GGML_ASSERT(mem != nullptr);
    if (mode != 1 && mode != 2) {
        mem->qsa_pkc_invalidate();
        return;
    }
    const int64_t r        = ratio;
    const int64_t n_blocks = (n_kv + r - 1)/r;
    std::vector<int32_t> dirty, members;
    // mode 1 recomputes every full block: plan against an empty table so all of them come back
    if (mode == 1) {
        mem->qsa_pkc_invalidate(ratio);
    }
    const int32_t n_dirty = mem->qsa_pkc_plan(ubatch, ratio, n_kv, &dirty, &members);
    GGML_ASSERT(n_dirty >= 0 && "qsa pkc: the cache became unusable between graph build and input");

    // block id -> row, from the cells as they are now (the full blocks in position order)
    const auto & cells = mem->get_mem_idx()->get_cells(ubatch->seq_id[0][0]);
    std::vector<uint8_t> have(r*n_blocks, 0);
    for (int64_t j = 0; j < n_kv; ++j) {
        if (!cells.is_empty(j)) {
            const int64_t p = cells.pos_get(j);
            have[(p/r)*r + p%r] = 1;
        }
    }
    int32_t * map = (int32_t *) pk_map->data;
    int64_t   b   = 0;
    for (int64_t pb = 0; pb < n_blocks; ++pb) {
        bool full = true;
        for (int64_t i = 0; i < r; ++i) full &= have[pb*r + i] != 0;
        if (full) {
            map[b++] = (int32_t) pb;
        }
    }
    for (; b < n_blocks; ++b) {
        map[b] = (int32_t) n_blocks;   // scratch: finite, and its block carries a -inf or tail bias anyway
    }

    if (mode == 2) {
        const int64_t nu = pk_upd->ne[0];
        GGML_ASSERT(n_dirty <= nu && "qsa pkc: more stale blocks than the graph recomputes");
        int32_t * upd   = (int32_t *) pk_upd->data;
        int32_t * ucell = (int32_t *) pk_upd_cells->data;
        int32_t * upos  = (int32_t *) pk_upd_pos->data;
        for (int64_t u = 0; u < nu; ++u) {
            const bool real = u < n_dirty;
            upd[u] = real ? dirty[u] : (int32_t) n_blocks;
            for (int64_t i = 0; i < r; ++i) {
                ucell[u*r + i] = real ? members[u*r + i] : 0;
            }
            for (int64_t sec = 0; sec < 4; ++sec) {
                upos[sec*nu + u] = real ? (int32_t) (dirty[u]*r) : 0;
            }
        }
    }
    mem->qsa_pkc_commit(ratio, dirty, members);
}

uint32_t llama_memory_hybrid_idx_context::get_n_stream() const {
    GGML_ASSERT(mem != nullptr);

    return mem->get_mem_attn()->get_n_stream();
}
