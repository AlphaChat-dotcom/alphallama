// SPDX-License-Identifier: Apache-2.0
#include "models/qwen35_hybrid.h"
#include "kernels/kernels.h"

#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <regex>
#include <stdexcept>
#include <vector>

namespace ai {

static constexpr int kRopeImrope = 40;   // interleaved M-RoPE, NEOX ordering

static Tensor carve_stage(Device & dev, int buf, size_t & off, int type, int64_t n0, int64_t n1) {
    Tensor t;
    t.type  = type;
    t.ne[0] = n0;
    t.ne[1] = n1;
    set_contiguous_strides(t);
    const size_t a = (dev.alloc_size(t) + dev.alignment() - 1) / dev.alignment() * dev.alignment();
    t.buf  = buf;
    t.data = (char *) dev.buffer_base(buf) + off;
    off += a;
    return t;
}

// AI_PARITY=1 asks for numerics that match ggml's graph. It is read in two places, so the predicate is shared.
static bool parity_on() {
    static const bool p = getenv("AI_PARITY") != nullptr && atoi(getenv("AI_PARITY")) != 0;
    return p;
}

// Debug: AI_DUMP_DIR=<dir> writes per-layer tensors under ggml's tensor names (tools/layerdiff.py).
static const char * dump_dir() {
    static const char * d = getenv("AI_DUMP_DIR");
    return d;
}

static void dump(Device & dev, const char * name, int il, const Tensor & t) {
    if (!dump_dir() || t.type != F32 || op::recording()) {
        return;
    }
    const int64_t cols = t.ne[0], rows = t.numel() / t.ne[0];
    std::vector<float> host((size_t) t.numel());
    dev.download(t, host.data(), 0, host.size() * 4);
    std::string path = std::string(dump_dir()) + "/" + name + "-" + std::to_string(il) + ".bin";
    FILE * f = fopen(path.c_str(), "wb");
    int64_t hdr[2] = {cols, rows};
    fwrite(hdr, 8, 2, f);
    fwrite(host.data(), 4, host.size(), f);
    fclose(f);
}

static std::string blk(int il, const char * name) {
    return "blk." + std::to_string(il) + "." + name;
}

bool Qwen35Hybrid::load(Device & dev, const std::string & path, bool verbose, bool offload_experts) {
    if (!w_.open(path)) {
        return false;
    }
    Qwen35Spec & s = spec_;
    s.arch = w_.arch();
    if (!supports(s.arch)) {
        fprintf(stderr, "[ai] Program<Qwen35Hybrid> does not support arch '%s'\n", s.arch.c_str());
        return false;
    }
    const std::string a = s.arch + ".";
    const int n_layer_all = (int) w_.get_int(a + "block_count");
    const int nextn       = (int) w_.get_int(a + "nextn_predict_layers", 0);
    s.n_layer     = n_layer_all - nextn;
    s.nextn       = nextn;
    s.n_embd      = w_.get_int(a + "embedding_length");
    s.n_head      = (int) w_.get_int(a + "attention.head_count");
    s.n_head_kv   = (int) w_.get_int(a + "attention.head_count_kv");
    s.head_dim    = (int) w_.get_int(a + "attention.key_length");
    s.n_rot       = (int) w_.get_int(a + "rope.dimension_count");
    s.rope_base   = (float) w_.get_float(a + "rope.freq_base");
    s.rms_eps     = (float) w_.get_float(a + "attention.layer_norm_rms_epsilon");
    s.n_ctx_train = (int) w_.get_int(a + "context_length");
    // A dense artifact carries no expert_count, so this is not a hard read: a missing-key error here
    // would say nothing about the real limitation.
    s.n_expert      = (int) w_.get_int(a + "expert_count", 0);
    s.n_expert_used = (int) w_.get_int(a + "expert_used_count", 0);
    s.dense         = !w_.has_key(a + "expert_count");
    s.ffn_length    = w_.get_int(a + "feed_forward_length", 0);
    s.conv_k      = (int) w_.get_int(a + "ssm.conv_kernel");
    s.ssm_state   = (int) w_.get_int(a + "ssm.state_size");
    s.ssm_groups  = (int) w_.get_int(a + "ssm.group_count");
    s.ssm_dt_rank = (int) w_.get_int(a + "ssm.time_step_rank");
    s.ssm_inner   = (int) w_.get_int(a + "ssm.inner_size");
    s.full_attn_interval = (int) w_.get_int(a + "full_attention_interval", 4);
    if (s.dense && s.ffn_length <= 0) {
        // The spec above is correct for a dense artifact, but every block then needs a plain
        // gate/up/down FFN and only the MoE path (plus its shared expert) is implemented. The
        // shared-expert op sequence is the dense FFN at a different width, so this is a binding job.
        // Refuse here, where the shape is known, rather than failing at an unbound tensor later.
        char msg[320];
        snprintf(msg, sizeof msg,
                 "Qwen35Hybrid: '%s' is a DENSE artifact (no expert_count; feed_forward_length=%lld, "
                 "n_embd=%lld, %d layers) and the dense FFN path is not implemented -- only the MoE "
                 "path is. Use a qwen35moe artifact, or implement the dense FFN branch.",
                 s.arch.c_str(), (long long) s.ffn_length, (long long) s.n_embd, s.n_layer);
        throw std::runtime_error(msg);
    }
    const auto sec = w_.get_int_array(a + "rope.dimension_sections");
    for (size_t i = 0; i < 4 && i < sec.size(); ++i) {
        s.sections[i] = (int) sec[i];
    }
    s.is_gdn.resize((size_t) s.n_layer);
    for (int il = 0; il < s.n_layer; ++il) {
        s.is_gdn[(size_t) il] = ((il + 1) % s.full_attn_interval) != 0;
    }

    // AI_HOST_EMBD=1 keeps token_embd.weight in host memory. The table is only read by get_rows, one
    // row per token, so VRAM residency buys no bandwidth. Unless the table is mapped below, graph build
    // refuses it at Call::in (buf < 0), which confirms the filter took effect.
    const bool host_embd = getenv("AI_HOST_EMBD") != nullptr;
    // AI_FFN_CPU=N | l0,l1,...: dense model, keep N dense FFN layers' weights in host memory (CpuFfn computes them on
    // the CPU in decode, staged through VRAM in prefill). N picks the last layers whose gate and up share a type.
    ffn_cpu_layer_.assign((size_t) s.n_layer, -1);
    ffn_cpu_list_.clear();
#if defined(AI_DOORBELL)
    if (const char * fc = getenv("AI_FFN_CPU"); fc && s.dense) {
        std::map<std::string, int> types;
        for (const auto & m : w_.meta()) types[m.name] = m.t.type;
        auto ok = [&](int il) {
            const auto g = types.find(blk(il, "ffn_gate.weight")), u = types.find(blk(il, "ffn_up.weight"));
            return g != types.end() && u != types.end() && g->second == u->second && types.count(blk(il, "ffn_down.weight"));
        };
        const std::string v(fc);
        if (v.find(',') != std::string::npos || v.find('-') != std::string::npos) {
            for (size_t p = 0; p < v.size();) {
                const size_t c = v.find(',', p);
                const int    il = atoi(v.substr(p, c == std::string::npos ? std::string::npos : c - p).c_str());
                if (il >= 0 && il < s.n_layer && ok(il)) ffn_cpu_list_.push_back(il);
                p = c == std::string::npos ? v.size() : c + 1;
            }
        } else {
            const int want = atoi(fc);
            for (int il = s.n_layer - 1; il >= 0 && (int) ffn_cpu_list_.size() < want; --il) {
                if (ok(il)) ffn_cpu_list_.push_back(il);
            }
        }
        std::sort(ffn_cpu_list_.begin(), ffn_cpu_list_.end());
        for (size_t i = 0; i < ffn_cpu_list_.size(); ++i) ffn_cpu_layer_[(size_t) ffn_cpu_list_[i]] = (int) i;
        if (!ffn_cpu_list_.empty()) setenv("GGML_CUDA_REGISTER_HOST", "1", 0);   // registered host block: fast staged copies
    }
#endif
    const std::vector<int> ffn_cpu_layer = ffn_cpu_layer_;
    WeightSet::HostFilter host;
    if (offload_experts || host_embd || !ffn_cpu_list_.empty()) {
        host = [offload_experts, host_embd, ffn_cpu_layer](const std::string & name) {
            if (offload_experts && name.find("_exps.weight") != std::string::npos) {
                return true;
            }
            if (host_embd && name == "token_embd.weight") {
                return true;
            }
            if (name.compare(0, 4, "blk.") == 0 && name.find(".ffn_") != std::string::npos) {
                const int il = atoi(name.c_str() + 4);
                if (il >= 0 && il < (int) ffn_cpu_layer.size() && ffn_cpu_layer[(size_t) il] >= 0) return true;
            }
            return false;
        };
        if (host_embd) {
            fprintf(stderr, "[ai] AI_HOST_EMBD=1: token_embd.weight host-kept (expect a graph-build "
                            "refusal at Call::in)\n");
        }
    }
    // adjacent placement for the fused projections (zero-copy spans)
    WeightSet::Groups groups;
    for (int il = 0; il < s.n_layer; ++il) {
        if (s.is_gdn[(size_t) il]) {
            groups.push_back({ blk(il, "attn_qkv.weight"), blk(il, "attn_gate.weight") });
            groups.push_back({ blk(il, "ssm_beta.weight"), blk(il, "ssm_alpha.weight") });
        } else {
            groups.push_back({ blk(il, "attn_q.weight"), blk(il, "attn_k.weight"), blk(il, "attn_v.weight") });
        }
    }
    for (int il = s.n_layer; il < s.n_layer + s.nextn; ++il) {
        groups.push_back({ blk(il, "attn_q.weight"), blk(il, "attn_k.weight"), blk(il, "attn_v.weight") });
    }
    for (int il = 0; il < s.n_layer + s.nextn; ++il) {
        if (s.dense) {
            groups.push_back({ blk(il, "ffn_gate.weight"), blk(il, "ffn_up.weight") });
            continue;
        }
        groups.push_back({ blk(il, "ffn_gate_inp.weight"), blk(il, "ffn_gate_inp_shexp.weight") });
        groups.push_back({ blk(il, "ffn_gate_shexp.weight"), blk(il, "ffn_up_shexp.weight") });
    }
    if (parity_on()) {
        // Report the effective padding: AI_KV_PAD overrides the default in both directions.
        const char * kvp = getenv("AI_KV_PAD");
        fprintf(stderr, "[ai] AI_PARITY=1: bit-changing fusions off, visible cells padded to %s%s "
                        "(ggml's graph pads the same way, so the attention numerics match it)\n",
                kvp ? kvp : "256", kvp ? " by AI_KV_PAD override -- parity is not expected" : "");
    }
    // AI_REQUANT: load-time requantisation for fit, e.g. "q5_K>q4_K,q6_K>q5_K:^output\.weight$": a rule SRC>DST[:regex]
    // converts every device tensor stored as SRC (whose name matches the regex, when given) to DST before upload.
    // DST needs no importance matrix (q4_0, q8_0, q4_K, q5_K, q6_K). It changes the model's numbers: use it only when
    // an artifact misses the card by a little, and check quality (paired NLL).
    WeightSet::ConvertFilter conv;
    if (const char * rq = getenv("AI_REQUANT")) {
        struct Rule { int from, to; std::regex re; bool any; };
        auto parse_type = [](std::string s) {
            for (auto & c : s) c = (char) tolower((unsigned char) c);
            return s == "q4_0" ? (int) Q4_0 : s == "q8_0" ? (int) Q8_0 : s == "q4_k" ? (int) Q4_K : s == "q5_k" ? (int) Q5_K
                 : s == "q6_k" ? (int) Q6_K : s == "iq4_xs" ? (int) IQ4_XS : s == "bf16" ? (int) BF16 : s == "f16" ? (int) F16 : -1;
        };
        auto rules = std::make_shared<std::vector<Rule>>();
        std::string spec(rq);
        size_t p = 0;
        while (p < spec.size()) {
            const size_t c = spec.find(',', p);
            const std::string item = spec.substr(p, c == std::string::npos ? std::string::npos : c - p);
            p = c == std::string::npos ? spec.size() : c + 1;
            const size_t gt = item.find('>'), co = item.find(':');
            if (gt == std::string::npos) throw std::runtime_error("AI_REQUANT: rule without '>': " + item);
            Rule r;
            r.from = parse_type(item.substr(0, gt));
            r.to   = parse_type(item.substr(gt + 1, co == std::string::npos ? std::string::npos : co - gt - 1));
            r.any  = co == std::string::npos;
            if (!r.any) r.re = std::regex(item.substr(co + 1));
            if (r.from < 0 || r.to < 0) throw std::runtime_error("AI_REQUANT: unknown type in " + item);
            rules->push_back(r);
        }
        conv = [rules](const std::string & name, const Tensor & t) -> int {
            if (t.ne[1] <= 1) return -1;   // matrices only (norms, biases and 1-D scales keep their type)
            for (const auto & r : *rules) {
                if (r.from == t.type && (r.any || std::regex_search(name, r.re))) return r.to;
            }
            return -1;
        };
        fprintf(stderr, "[ai] AI_REQUANT=%s: %zu rule(s) -- load-time requantisation changes the model's numbers\n", rq,
                rules->size());
    }
    if (!w_.load(dev, 4ULL << 30, verbose, host, groups, conv)) {
        return false;
    }
    // AI_HOST_EMBD with AI_HOST_EMBD_MAPPED (default 1): copy the host-kept table into pinned memory the kernels can read
    // (unified addressing): a gather reads one row per token over PCIe, so the device-side gathers -- including the
    // MTP head's, whose ids are produced inside the round -- need no host step. A tied table is not mapped (it would
    // put the LM head GEMV on PCIe).
    const bool emap = !getenv("AI_HOST_EMBD_MAPPED") || atoi(getenv("AI_HOST_EMBD_MAPPED")) != 0;
    if (host_embd && emap && w_.has("output.weight") && w_.is_host("token_embd.weight")) {
        Tensor     t   = w_.get("token_embd.weight");
        const int  b   = dev.alloc_host_mapped_buffer(t.nbytes() + 4096, "token_embd.mapped");
        void *     dst = dev.buffer_base(b);
        memcpy(dst, t.data, t.nbytes());
        t.buf  = b;
        t.data = dst;
        embd_mapped_ = t;
        fprintf(stderr, "[ai] AI_HOST_EMBD: token_embd mapped for device reads (%.0f MiB pinned host memory, no VRAM)\n",
                t.nbytes() / 1048576.0);
    }
    bind_layers();
    bind_mtp();
    fuse_beta_alpha(dev);
    // AI_BOUND_SUB roles that need real requantisation (head, expert_down). After bind_layers so
    // the tensors exist, and after fuse_beta_alpha so spans are already built from the originals.
    bound_apply_roles(dev);
    build_draft_head(dev);
    init_taps(dev);
    router_to_f16(dev);
    return true;
}

// F16 copies of the F32 router|shared-score spans and the GDN beta|alpha spans: half the bytes for GEMVs that
// are bandwidth-bound. This changes numerics (the router feeds top-k expert selection).
static Tensor f32_to_f16_copy(Device & dev, const Tensor & src, const char * name) {
    const int64_t      n = src.numel();
    std::vector<float> h((size_t) n);
    dev.download(src, h.data(), 0, (size_t) n * 4);
    std::vector<ggml_fp16_t> o((size_t) n);
    ggml_fp32_to_fp16_row(h.data(), o.data(), n);
    Tensor t;
    t.type = F16;
    for (int i = 0; i < 4; ++i) {
        t.ne[i] = src.ne[i];
    }
    set_contiguous_strides(t);
    t.buf  = dev.alloc_buffer((size_t) n * 2 + dev.alignment(), name);
    t.data = dev.buffer_base(t.buf);
    dev.upload(t, o.data(), 0, (size_t) n * 2);
    return t;
}

void Qwen35Hybrid::router_to_f16(Device & dev) {
    // On by default; paired NLL is unchanged within noise. AI_ROUTER_F16=0 keeps the F32 spans.
    const char * e = getenv("AI_ROUTER_F16");
    if (e && atoi(e) == 0) {
        return;
    }
    int n_r = 0, n_b = 0;
    for (int il = 0; il < spec_.n_layer; ++il) {
        op::MoeWeights & m = moe_[(size_t) il];
        if (m.route_score.valid() && m.route_score.type == F32 && m.route_score.contiguous() && m.route_score.buf >= 0) {
            m.route_score = f32_to_f16_copy(dev, m.route_score, "router_f16");
            ++n_r;
        }
        if (spec_.is_gdn[(size_t) il]) {
            op::GdnWeights & g = gdn_[(size_t) il];
            if (g.ba.valid() && g.ba.type == F32 && g.ba.contiguous() && g.ba.buf >= 0) {
                g.ba = f32_to_f16_copy(dev, g.ba, "ba_f16");
                ++n_b;
            }
        }
    }
    fprintf(stderr, "[router-f16] converted %d router|score spans and %d beta|alpha spans to F16\n", n_r, n_b);
}

static int64_t bound_row_bytes(int type, int64_t ncols);

// AI_MTP_DRAFT_Q=q4_0|q5_0|q8_0|q3_K|q2_K|iq4_xs: a separate lower-bit copy of the draft head's rows (the
// first AI_MTP_HEAD_ROWS of the output head), used only to propose tokens. The verify keeps the original
// head, so generated text cannot change, only acceptance can.
void Qwen35Hybrid::build_draft_head(Device & dev) {
    // Default q4_0; AI_MTP_DRAFT_Q=off restores the shared head rows. Lower-bit types than q4_0 are not
    // faster: the draft head is unpack-bound at these widths, not byte-bound.
    const char * e = getenv("AI_MTP_DRAFT_Q") ? getenv("AI_MTP_DRAFT_Q") : "q4_0";
    if (!mtp_.valid || *e == 0 || strcmp(e, "off") == 0) {
        return;
    }
    const int want = strcmp(e, "q4_0") == 0 ? Q4_0 : strcmp(e, "q5_0") == 0 ? Q5_0 : strcmp(e, "q8_0") == 0 ? 8
                   : strcmp(e, "q3_K") == 0 ? Q3_K : strcmp(e, "q2_K") == 0 ? Q2_K : strcmp(e, "iq4_xs") == 0 ? IQ4_XS
                   : -1;
    const Tensor & src = output_;
    if (want < 0 || !src.valid()) {
        fprintf(stderr, "[mtp] AI_MTP_DRAFT_Q=%s not understood; draft head unchanged\n", e);
        return;
    }
    const int64_t hr       = getenv("AI_MTP_HEAD_ROWS") ? atoll(getenv("AI_MTP_HEAD_ROWS")) : 131072;
    const int64_t ncols    = src.ne[0];
    const int64_t rows     = (hr > 0 && hr < src.ne[1]) ? hr : src.ne[1];
    const int64_t src_rowb = bound_row_bytes(src.type, ncols);
    const int64_t dst_rowb = bound_row_bytes(want, ncols);
    const ggml_type_traits * tr = ggml_get_type_traits((ggml_type) src.type);
    if (src_rowb == 0 || dst_rowb == 0 || tr == nullptr || tr->to_float == nullptr) {
        fprintf(stderr, "[mtp] draft head DECLINED: type %d -> %d not supported\n", src.type, want);
        return;
    }
    // oracle: the consumer's own row size, not our transcribed table
    if ((size_t) dst_rowb != ggml_row_size((ggml_type) want, ncols) ||
        (size_t) src_rowb != ggml_row_size((ggml_type) src.type, ncols)) {
        fprintf(stderr, "[mtp] draft head DECLINED: row bytes %lld/%lld disagree with ggml_row_size %zu/%zu\n",
                (long long) dst_rowb, (long long) src_rowb, ggml_row_size((ggml_type) want, ncols),
                ggml_row_size((ggml_type) src.type, ncols));
        return;
    }
    const int64_t      chunk = 4096;
    std::vector<char>  sbuf((size_t) chunk * (size_t) src_rowb);
    std::vector<float> mid((size_t) chunk * (size_t) ncols);
    // AI_DRAFT_VOCAB (default on): quantise every row on the host, so a request can pick its rows by script
    static const bool  k_dv  = getenv("AI_DRAFT_VOCAB") ? atoi(getenv("AI_DRAFT_VOCAB")) != 0 : true;
    const int64_t      qrows = (k_dv && rows < src.ne[1]) ? src.ne[1] : rows;
    std::vector<char>  dst((size_t) qrows * (size_t) dst_rowb);
    for (int64_t r0 = 0; r0 < qrows; r0 += chunk) {
        const int64_t n = std::min<int64_t>(chunk, qrows - r0);
        if (src.buf < 0) {
            memcpy(sbuf.data(), (const char *) src.data + (size_t) r0 * (size_t) src_rowb, (size_t) n * (size_t) src_rowb);
        } else {
            dev.download(src, sbuf.data(), (size_t) r0 * (size_t) src_rowb, (size_t) n * (size_t) src_rowb);
        }
        tr->to_float(sbuf.data(), mid.data(), n * ncols);
        ggml_quantize_chunk((ggml_type) want, mid.data(), dst.data() + (size_t) r0 * (size_t) dst_rowb, 0, n, ncols,
                            nullptr);
    }
    Tensor t;
    t.type  = want;
    t.ne[0] = ncols;
    t.ne[1] = rows;
    set_contiguous_strides(t);
    const size_t head_bytes = (size_t) rows * (size_t) dst_rowb;
    t.buf  = dev.alloc_buffer(head_bytes + dev.alignment(), "draft_head");
    t.data = dev.buffer_base(t.buf);
    dev.upload(t, dst.data(), 0, head_bytes);
    draft_head_ = t;
    draft_rowb_ = dst_rowb;
    if (qrows > rows) {
        std::vector<int32_t> ident((size_t) rows);
        for (int64_t i = 0; i < rows; ++i) {
            ident[(size_t) i] = (int32_t) i;
        }
        Tensor m;
        m.type  = I32;
        m.ne[0] = rows;
        set_contiguous_strides(m);
        m.buf  = dev.alloc_buffer((size_t) rows * 4 + dev.alignment(), "draft_map");
        m.data = dev.buffer_base(m.buf);
        dev.upload(m, ident.data(), 0, (size_t) rows * 4);
        draft_map_       = m;
        draft_full_host_ = std::move(dst);
        fprintf(stderr, "[draft-vocab] host copy of all %lld rows (%.0f MiB) for script-adaptive draft rows\n",
                (long long) qrows, draft_full_host_.size() / 1048576.0);
    }
    fprintf(stderr, "[mtp] draft head: %lld rows, type %d -> %d, %.0f -> %.0f MiB (verify keeps the original head)\n",
            (long long) rows, src.type, want, rows * src_rowb / 1048576.0, rows * dst_rowb / 1048576.0);
}

bool Qwen35Hybrid::set_draft_rows(Device & dev, const std::vector<int32_t> & ids) {
    if (!draft_head_.valid() || !draft_map_.valid() || draft_full_host_.empty()) {
        return false;
    }
    const int64_t rows = draft_head_.ne[1];
    const int64_t nv   = (int64_t) (draft_full_host_.size() / (size_t) draft_rowb_);
    if (ids.empty()) {
        if (draft_identity_) {
            return true;
        }
        std::vector<int32_t> ident((size_t) rows);
        for (int64_t i = 0; i < rows; ++i) {
            ident[(size_t) i] = (int32_t) i;
        }
        dev.upload(draft_head_, draft_full_host_.data(), 0, (size_t) rows * (size_t) draft_rowb_);
        dev.upload(draft_map_, ident.data(), 0, (size_t) rows * 4);
        draft_identity_ = true;
        draft_ids_.clear();
        return true;
    }
    if ((int64_t) ids.size() != rows) {
        fprintf(stderr, "[draft-vocab] %zu ids for %lld rows: refused\n", ids.size(), (long long) rows);
        return false;
    }
    for (int64_t i = 0; i < rows; ++i) {
        const int32_t id = ids[(size_t) i];
        if (id < 0 || id >= nv) {
            fprintf(stderr, "[draft-vocab] id %d out of range: refused\n", id);
            return false;
        }
    }
    // The C API sets the rows on every request (ai_engine_prefill / _chunk), and rebuilding the whole head each time
    // adds a fixed cost of hundreds of milliseconds per prompt that every other stream waits behind. Same-script prompts
    // pick the same rows, so do nothing when they match what is resident, and otherwise upload only the changed runs.
    // The device buffer ends byte-identical to a full rebuild.
    if (!draft_identity_ && ids == draft_ids_) {
        return true;
    }
    const size_t rb = (size_t) draft_rowb_;
    auto src_id = [&](int64_t i) -> int32_t { return draft_identity_ ? (int32_t) i : draft_ids_[(size_t) i]; };
    int64_t changed = 0;
    for (int64_t i = 0; i < rows;) {
        if (ids[(size_t) i] == src_id(i)) { ++i; continue; }
        int64_t j = i;
        while (j < rows && ids[(size_t) j] != src_id(j)) ++j;
        const size_t n = (size_t) (j - i);
        if (draft_stage_.size() < n * rb) draft_stage_.resize(n * rb);
        for (int64_t k = i; k < j; ++k) {
            memcpy(draft_stage_.data() + (size_t) (k - i) * rb, draft_full_host_.data() + (size_t) ids[(size_t) k] * rb, rb);
        }
        dev.upload(draft_head_, draft_stage_.data(), (size_t) i * rb, n * rb);   // synchronous (ggml_backend_tensor_set)
        dev.upload(draft_map_, ids.data() + i, (size_t) i * 4, n * 4);
        changed += j - i;
        i = j;
    }
    static const bool verbose = getenv("AI_DRAFT_VOCAB_VERBOSE") && atoi(getenv("AI_DRAFT_VOCAB_VERBOSE")) != 0;
    if (verbose) fprintf(stderr, "[draft-vocab] %lld of %lld rows changed\n", (long long) changed, (long long) rows);
    draft_ids_      = ids;
    draft_identity_ = false;
    return true;
}

void Qwen35Hybrid::init_taps(Device & dev) {
    const char * e = getenv("AI_TAPS");
    if (e == nullptr || *e == 0) {
        return;
    }
    tap_slot_.assign((size_t) spec_.n_layer, -1);
    int n = 0;
    for (const char * p = e; *p;) {
        const int il = atoi(p);
        if (il < 0 || il >= spec_.n_layer) {
            fprintf(stderr, "[taps] AI_TAPS layer %d out of range [0, %d): taps OFF\n", il, spec_.n_layer);
            tap_slot_.clear();
            return;
        }
        tap_slot_[(size_t) il] = n++;
        while (*p && *p != ',') ++p;
        if (*p == ',') ++p;
    }
    static const int64_t cap = getenv("AI_TAPS_CAP") ? atoll(getenv("AI_TAPS_CAP")) : 2048;
    taps_cap_ = cap;
    Tensor t;
    t.type  = F32;
    t.ne[0] = (int64_t) n * spec_.n_embd;
    t.ne[1] = taps_cap_;
    set_contiguous_strides(t);
    t.buf  = dev.alloc_buffer((size_t) t.nb[2] + dev.alignment(), "taps");
    t.data = dev.buffer_base(t.buf);
    taps_  = t;
    fprintf(stderr, "[taps] %d layers (%s) -> F32 [%lld, %lld] (%.0f MiB)\n", n, e, (long long) t.ne[0],
            (long long) t.ne[1], (double) t.nb[2] / 1048576.0);
}

void Qwen35Hybrid::bind_layers() {
    const Qwen35Spec & s = spec_;
    tok_embd_ = embd_mapped_.valid() ? embd_mapped_ : w_.get("token_embd.weight");
    out_norm_ = w_.get("output_norm.weight");
    output_   = w_.has("output.weight") ? w_.get("output.weight") : tok_embd_;
    spec_.n_vocab = tok_embd_.ne[1];

    int n_attn = 0, n_gdn = 0;
    for (int il = 0; il < s.n_layer; ++il) {
        attn_norm_.push_back(w_.get(blk(il, "attn_norm.weight")));
        post_norm_.push_back(w_.get(blk(il, "post_attention_norm.weight")));
        if (s.is_gdn[(size_t) il]) {
            op::GdnWeights g;
            g.qkv     = w_.get(blk(il, "attn_qkv.weight"));
            g.z       = w_.get(blk(il, "attn_gate.weight"));
            g.beta    = w_.get(blk(il, "ssm_beta.weight"));
            g.alpha   = w_.get(blk(il, "ssm_alpha.weight"));
            g.conv    = w_.get(blk(il, "ssm_conv1d.weight"));
            g.dt_bias = w_.get(blk(il, "ssm_dt.bias"));
            g.a       = w_.get(blk(il, "ssm_a"));
            g.norm    = w_.get(blk(il, "ssm_norm.weight"));
            g.out     = w_.get(blk(il, "ssm_out.weight"));
            gdn_.push_back(g);
            attn_.push_back({});
            gdn_layer_.push_back(n_gdn++);
            kv_layer_.push_back(-1);
        } else {
            op::AttnWeights at;
            at.q      = w_.get(blk(il, "attn_q.weight"));
            at.k      = w_.get(blk(il, "attn_k.weight"));
            at.v      = w_.get(blk(il, "attn_v.weight"));
            at.o      = w_.get(blk(il, "attn_output.weight"));
            at.q_norm = w_.get(blk(il, "attn_q_norm.weight"));
            at.k_norm = w_.get(blk(il, "attn_k_norm.weight"));
            attn_.push_back(at);
            gdn_.push_back({});
            kv_layer_.push_back(n_attn++);
            gdn_layer_.push_back(-1);
        }
        op::MoeWeights m;
        if (s.dense) {   // the dense FFN rides in the shared-expert slots (same op sequence)
            m.sh_up   = w_.get(blk(il, "ffn_up.weight"));
            m.sh_gate = w_.get(blk(il, "ffn_gate.weight"));
            m.sh_down = w_.get(blk(il, "ffn_down.weight"));
            moe_.push_back(m);
            continue;
        }
        m.router    = w_.get(blk(il, "ffn_gate_inp.weight"));
        m.up_exps   = w_.get(blk(il, "ffn_up_exps.weight"));
        m.gate_exps = w_.get(blk(il, "ffn_gate_exps.weight"));
        m.down_exps = w_.get(blk(il, "ffn_down_exps.weight"));
        m.sh_up     = w_.get(blk(il, "ffn_up_shexp.weight"));
        m.sh_gate   = w_.get(blk(il, "ffn_gate_shexp.weight"));
        m.sh_down   = w_.get(blk(il, "ffn_down_shexp.weight"));
        m.sh_score  = w_.get(blk(il, "ffn_gate_inp_shexp.weight"));
        moe_.push_back(m);
    }

    attn_p_.n_head     = s.n_head;
    attn_p_.n_head_kv  = s.n_head_kv;
    attn_p_.head_dim   = s.head_dim;
    attn_p_.n_rot      = s.n_rot;
    for (int i = 0; i < 4; ++i) {
        attn_p_.sections[i] = s.sections[i];
    }
    attn_p_.rope_mode  = kRopeImrope;
    attn_p_.n_ctx_orig = s.n_ctx_train;
    attn_p_.rope_base  = s.rope_base;
    attn_p_.rope_scale = 1.0f;
    attn_p_.eps        = s.rms_eps;
    attn_p_.kq_scale   = 1.0f / std::sqrt((float) s.head_dim);

    gdn_p_.conv_k   = s.conv_k;
    gdn_p_.channels = (int) s.conv_channels();
    gdn_p_.k_heads  = s.ssm_groups;
    gdn_p_.v_heads  = s.ssm_dt_rank;
    gdn_p_.head_k   = s.ssm_state;
    gdn_p_.head_v   = s.head_v();
    gdn_p_.eps      = s.rms_eps;

    moe_p_.n_expert = s.n_expert;
    moe_p_.n_used   = s.n_expert_used;
}

void Qwen35Hybrid::bind_mtp() {
    if (spec_.nextn < 1) {
        return;
    }
    const int il = spec_.n_layer;   // the first next-n block
    const auto t = [&](const char * n) { return w_.find(blk(il, n)); };
    const bool dense = spec_.dense;
    if (!t("nextn.eh_proj.weight") || !t("nextn.enorm.weight") || !t("nextn.hnorm.weight") ||
        !t("attn_q.weight") || !t(dense ? "ffn_gate.weight" : "ffn_gate_inp.weight")) {
        fprintf(stderr, "[ai] next-n block %d incomplete: MTP disabled\n", il);
        return;
    }
    MtpHead & h = mtp_;
    h.enorm       = *t("nextn.enorm.weight");
    h.hnorm       = *t("nextn.hnorm.weight");
    h.eh_proj     = *t("nextn.eh_proj.weight");
    h.shared_norm = t("nextn.shared_head_norm.weight") ? *t("nextn.shared_head_norm.weight") : out_norm_;
    h.attn_norm   = *t("attn_norm.weight");
    h.post_norm   = *t("post_attention_norm.weight");
    h.attn.q      = *t("attn_q.weight");
    h.attn.k      = *t("attn_k.weight");
    h.attn.v      = *t("attn_v.weight");
    h.attn.o      = *t("attn_output.weight");
    h.attn.q_norm = *t("attn_q_norm.weight");
    h.attn.k_norm = *t("attn_k_norm.weight");
    if (dense) {
        h.moe.sh_up   = *t("ffn_up.weight");
        h.moe.sh_gate = *t("ffn_gate.weight");
        h.moe.sh_down = *t("ffn_down.weight");
        h.valid       = true;
        return;
    }
    h.moe.router    = *t("ffn_gate_inp.weight");
    h.moe.up_exps   = *t("ffn_up_exps.weight");
    h.moe.gate_exps = *t("ffn_gate_exps.weight");
    h.moe.down_exps = *t("ffn_down_exps.weight");
    h.moe.sh_up     = *t("ffn_up_shexp.weight");
    h.moe.sh_gate   = *t("ffn_gate_shexp.weight");
    h.moe.sh_down   = *t("ffn_down_shexp.weight");
    h.moe.sh_score  = *t("ffn_gate_inp_shexp.weight");
    h.valid = true;
}

void Qwen35Hybrid::fuse_beta_alpha(Device & dev) {
    // Fused projections as zero-copy spans over adjacently placed weights.
    //   AI_FUSE_BA=0   keeps the beta and alpha projections split
    //   AI_FUSE_PROJ=0 keeps qkv|z (DeltaNet) and q|k|v (attention) split
    (void) dev;
    const char * eba = getenv("AI_FUSE_BA");
    const char * epr = getenv("AI_FUSE_PROJ");
    const bool fuse_ba   = !eba || atoi(eba) != 0;
    // fused q|k|v and qkv|z projections change bits; AI_PARITY=1 keeps them split (and also sets the KV padding)
    const bool parity    = parity_on();
    const bool fuse_proj = epr ? atoi(epr) != 0 : !parity;
    int n_ba = 0, n_qkvz = 0, n_qkv = 0;
    for (int il = 0; il < spec_.n_layer; ++il) {
        if (spec_.is_gdn[(size_t) il]) {
            op::GdnWeights & g = gdn_[(size_t) il];
            // any shared type (F32 or Q8_0 in practice): the fused GDN path only needs ba as a span
            if (fuse_ba && g.beta.valid() && g.alpha.valid() && g.beta.type == g.alpha.type &&
                g.beta.ne[0] == g.alpha.ne[0]) {
                g.ba = w_.span({ blk(il, "ssm_beta.weight"), blk(il, "ssm_alpha.weight") });
                n_ba += g.ba.valid() ? 1 : 0;
            }
            if (fuse_proj) {
                g.qkvz = w_.span({ blk(il, "attn_qkv.weight"), blk(il, "attn_gate.weight") });
                n_qkvz += g.qkvz.valid() ? 1 : 0;
                bound_halve_qkvz(g, il);
            }
        } else if (fuse_proj) {
            op::AttnWeights & a = attn_[(size_t) il];
            a.qkv = w_.span({ blk(il, "attn_q.weight"), blk(il, "attn_k.weight"), blk(il, "attn_v.weight") });
            n_qkv += a.qkv.valid() ? 1 : 0;
        }
    }
    if (fuse_proj && mtp_.valid) {
        const int il = spec_.n_layer;
        mtp_.attn.qkv = w_.span({ blk(il, "attn_q.weight"), blk(il, "attn_k.weight"), blk(il, "attn_v.weight") });
    }
    // MoE: router|shared-score and shared gate|up (decode launch count)
    int n_moe = 0;
    const auto fuse_moe = [&](op::MoeWeights & m, int il) {
        m.route_score = w_.span({ blk(il, "ffn_gate_inp.weight"), blk(il, "ffn_gate_inp_shexp.weight") });
        m.sh_gate_up  = w_.span({ blk(il, "ffn_gate_shexp.weight"), blk(il, "ffn_up_shexp.weight") });
        n_moe += m.route_score.valid() && m.sh_gate_up.valid() ? 1 : 0;
    };
    for (int il = 0; il < spec_.n_layer; ++il) {
        fuse_moe(moe_[(size_t) il], il);
    }
    if (mtp_.valid) {
        fuse_moe(mtp_.moe, spec_.n_layer);
    }
    fprintf(stderr, "[ai] fused projections: beta|alpha %d, qkv|z %d, q|k|v %d, router|score + shexp gate|up %d layers%s\n",
            n_ba, n_qkvz, n_qkv, n_moe, mtp_.valid ? " (+ MTP head)" : "");
}

Tensor Qwen35Hybrid::ffn_cpu(Device & dev, int il, const op::MoeWeights & w, const Tensor & p) {
#if defined(AI_DOORBELL)
    return cpu_ffn_->ffn(ffn_cpu_layer_[(size_t) il], w, p);
#else
    (void) dev; (void) il; (void) w; (void) p;
    throw std::runtime_error("AI_FFN_CPU needs an AI_DOORBELL build");
#endif
}

void Qwen35Hybrid::alloc_state(Device & dev, int64_t max_ctx, int n_streams, int kv_type) {
#if defined(AI_DOORBELL)
    if (!ffn_cpu_list_.empty() && !cpu_ffn_) {
        std::vector<op::MoeWeights> ls;
        for (int il : ffn_cpu_list_) ls.push_back(moe_[(size_t) il]);
        ffn_cpu_tmax_ = getenv("AI_FFN_CPU_TMAX") ? atoll(getenv("AI_FFN_CPU_TMAX")) : 16;
        cpu_ffn_ = std::make_unique<offload::CpuFfn>(dev, ls, spec_.n_embd, ffn_cpu_tmax_,
                                                     getenv("AI_FFN_CPU_THREADS") ? atoi(getenv("AI_FFN_CPU_THREADS")) : 0);
        std::string l;
        for (int il : ffn_cpu_list_) l += (l.empty() ? "" : ",") + std::to_string(il);
        fprintf(stderr, "[ai] AI_FFN_CPU: layers %s\n", l.c_str());
    }
#endif
    int n_attn = 0, n_gdn = 0;
    for (bool g : spec_.is_gdn) {
        (g ? n_gdn : n_attn)++;
    }
    if (mtp_.valid) {
        mtp_.kv = n_attn;   // one more attention store for the head
    }
    kv_.alloc(dev, n_attn + (mtp_.valid ? 1 : 0), (int64_t) spec_.head_dim * spec_.n_head_kv, max_ctx, n_streams,
              kv_type);
    const int64_t conv_n = (int64_t) (spec_.conv_k - 1) * spec_.conv_channels();
    const int64_t ssm_n  = (int64_t) spec_.head_v() * spec_.head_v() * spec_.ssm_dt_rank;
    slots_.alloc(dev, n_gdn, conv_n, ssm_n, n_streams);
    alloc_stage(dev, n_streams);
}

// One f32 activation and one q8_1 copy per layer, so no two layers ever share an address.

// AI_BOUND_QKVZ_HALF: timing only, the output is deliberately wrong.
//
// Bounds what any qkvz improvement can buy: if halving qkvz's bytes moves throughput by X, any change
// to that projection (kernel tuning, rewrite, requant) is bounded by roughly 2X.
//
// The weights are substituted, not zeroed. Zeroing qkvz zeroes q/k/v/z, which feed the recurrence and
// (through the residual) the router; degenerate activations collapse expert selection, so the MoE would
// do different work, and ggml's MUL_MAT_ID can fault on repeated ids in a row.
//
// qkvz is repointed at the model's own Q4_K expert weights: ffn_gate_exps holds enough rows of real,
// correctly scaled Q4_K data that a [n_embd, C+Sv*Hv] view of its first rows is structurally valid. Bytes
// per row fall from Q8_0's 34/32 to Q4_K's 144/256 (53%) while values stay in a sane range, so routing
// stays diverse. No allocation or host quantisation is needed.
//
// The delta is an upper bound on any change that halves qkvz's bytes with a kernel of comparable cost;
// it changes both the bytes and the codec kernel, so it is not the pure byte effect. Kernel count per
// round and token counts must match between arms, otherwise the engine did different work.

// Parse AI_BOUND_SUB="<role>:<TYPE>" once. Returns the requested ggml type for `role`, or -1.
// Roles: qkvz (attn_qkv+attn_gate span), head (output.weight, the largest tensor read once per token),
// experts (ffn_*_exps). Each type carries its own codec, as a shipped requant would, so a sweep over
// types prices the recipe directly; departure from byte-linearity across the sweep is the codec cost.
static int bound_sub_type(const char * role) {
    static const char * spec = getenv("AI_BOUND_SUB");
    if (!spec) {
        return -1;
    }
    const char * colon = strchr(spec, ':');
    if (!colon) {
        return -1;
    }
    const size_t n = (size_t) (colon - spec);
    if (strlen(role) != n || strncmp(spec, role, n) != 0) {
        return -1;
    }
    const char * t = colon + 1;
    // The legacy _0 types are here on purpose: Q4_0 has the same bytes as Q4_K (144 B per 256
    // weights) and Q5_0 as Q5_K (176 B), so a K-vs-_0 pair changes only the dequantisation work.
    // If bytes alone drove decode time, those pairs would measure the same.
    if (strcmp(t, "Q4_0") == 0) { return Q4_0; }
    if (strcmp(t, "Q5_0") == 0) { return Q5_0; }
    if (strcmp(t, "Q4_K") == 0) { return Q4_K; }
    if (strcmp(t, "Q5_K") == 0) { return Q5_K; }
    if (strcmp(t, "Q6_K") == 0) { return Q6_K; }
    if (strcmp(t, "Q8_0") == 0) { return Q8_0; }
    fprintf(stderr, "[bound] AI_BOUND_SUB: unsupported type '%s' (Q4_0|Q5_0|Q4_K|Q5_K|Q6_K|Q8_0)\n", t);
    return -1;
}

// Bytes per row for the quant types this model uses, so the log line can state the achieved ratio.
static int64_t bound_row_bytes(int type, int64_t ncols) {
    switch (type) {
        case Q4_0: return ncols / 32 * 18;    // == ncols/256 * 144, i.e. exactly Q4_K
        case Q5_0: return ncols / 32 * 22;    // == ncols/256 * 176, i.e. exactly Q5_K
        case Q2_K: return ncols / 256 * 84;
        case Q3_K: return ncols / 256 * 110;
        case IQ4_XS: return ncols / 256 * 136;
        case Q4_K: return ncols / 256 * 144;
        case Q5_K: return ncols / 256 * 176;
        case Q6_K: return ncols / 256 * 210;
        case Q8_0: return ncols / 32 * 34;
        default:   return 0;
    }
}

// Find a donor tensor of the requested type with enough rows, among the model's own weights. Using
// real model data keeps values correctly scaled, which keeps routing diverse: a degenerate top-8
// would measure collapsed routing rather than the cost of the bytes, and ggml's MUL_MAT_ID can
// fault on repeated ids in a row.
const Tensor * Qwen35Hybrid::bound_donor(int type, int64_t ncols, int64_t rows_needed) const {
    for (const auto & m : moe_) {
        const Tensor * cand[3] = { &m.gate_exps, &m.up_exps, &m.down_exps };
        for (const Tensor * c : cand) {
            if (c->valid() && c->type == type && c->ne[0] == ncols &&
                c->ne[1] * (c->ne[2] ? c->ne[2] : 1) >= rows_needed) {
                return c;
            }
        }
    }
    if (output_.valid() && output_.type == type && output_.ne[0] == ncols &&
        output_.ne[1] >= rows_needed) {
        return &output_;
    }
    return nullptr;
}

void Qwen35Hybrid::bound_halve_qkvz(op::GdnWeights & g, int il) {
    if (!g.qkvz.valid()) {
        return;
    }
    // AI_BOUND_SUB=qkvz:<TYPE> selects the type; AI_BOUND_QKVZ_HALF is an alias for qkvz:Q4_K.
    // The donor path is opt-in (AI_BOUND_QKVZ_DONOR=1). It aliases one tensor's bytes across all GDN
    // layers, so every layer reads the same data; that changes the cache footprint, not just the byte
    // count, and is not comparable with a real requant. By default bound_apply_roles requantises in place.
    int want = bound_sub_type("qkvz");
    if (want < 0 && getenv("AI_BOUND_QKVZ_HALF") != nullptr) {
        want = Q4_K;
    }
    if (getenv("AI_BOUND_QKVZ_DONOR") == nullptr) {
        return;   // leave it to bound_apply_roles, which requants for real
    }
    if (want < 0 || want == g.qkvz.type) {
        return;
    }
    const int64_t rows = g.qkvz.ne[1];
    const Tensor * donor = bound_donor(want, g.qkvz.ne[0], rows);
    if (donor == nullptr) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr, "[bound] AI_BOUND_SUB qkvz DECLINED: no donor of the requested type with "
                            "ne0=%lld and >=%lld rows. Nothing substituted; this arm is a NO-OP and "
                            "its result is NOT a measurement.\n",
                    (long long) g.qkvz.ne[0], (long long) rows);
        }
        return;
    }
    Tensor sub = *donor;
    sub.ne[1] = rows;
    sub.ne[2] = 1;
    sub.ne[3] = 1;
    set_contiguous_strides(sub);
    static bool announced = false;
    if (!announced) {
        announced = true;
        const int64_t was = bound_row_bytes(g.qkvz.type, g.qkvz.ne[0]);
        const int64_t now = bound_row_bytes(want, g.qkvz.ne[0]);
        fprintf(stderr,
                "[bound] AI_BOUND_SUB LIVE role=qkvz site=bind_layers layer=%d: type %d -> %d, "
                "[%lld,%lld], %lld -> %lld bytes/row (%.1f%%). OUTPUT WRONG BY DESIGN, timing only.\n",
                il, g.qkvz.type, want, (long long) sub.ne[0], (long long) rows,
                (long long) was, (long long) now,
                was ? 100.0 * (double) now / (double) was : 0.0);
    }
    g.qkvz = sub;
}


// Single reader for the null-arm switch, shared by the three call sites that skip same-type requants
// (here, expert_down, qkvz), so none of them can silently ignore it.
static bool bound_selfrequant() {
    static const bool on = getenv("AI_BOUND_SELFREQUANT") != nullptr;
    return on;
}

// AI_BOUND_REQUANT_DISCARD=1: run the whole dequant+requant host pass, then discard the result (no
// upload, no type change). The host CPU cost matches a real requant while weights and output stay
// bit-identical to stock, which separates the cost of the load phase from the effect of changed
// weights. (A same-type requant is not idempotent for every type, so it changes weights too.)
static bool bound_requant_discard() {
    static const bool on = getenv("AI_BOUND_REQUANT_DISCARD") != nullptr;
    return on;
}

// Requantise a tensor in place to a smaller type, for timing bounds only. Returns true if done.
// Output is deliberately degraded.
//
// AI_BOUND_SELFREQUANT=1 also permits want == t.type (a null arm): same bytes, codec and kernel, so
// the expected speed delta is zero, but it reproduces the load phase (download -> to_float ->
// quantize -> upload) that every real requant pays. That phase can leave the GPU at a lower clock
// for the first measured rounds; comparing a requant arm against this null arm, rather than stock,
// cancels that effect when clocks cannot be locked.
bool Qwen35Hybrid::bound_requant_inplace(Device & dev, Tensor & t, int want, const char * role,
                                         int idx, bool announce) {
    if (!t.valid()) {
        return false;
    }
    const bool self = (want == t.type);
    if (self && !bound_selfrequant()) {
        return false;
    }
    const int64_t ncols    = t.ne[0];
    const int64_t rows     = t.numel() / ncols;
    const int64_t src_rowb = bound_row_bytes(t.type, ncols);
    const int64_t dst_rowb = bound_row_bytes(want, ncols);
    if (src_rowb == 0 || dst_rowb == 0 || dst_rowb > src_rowb) {
        fprintf(stderr, "[bound] %s[%d] DECLINED: %lld -> %lld bytes/row does not shrink in place\n",
                role, idx, (long long) src_rowb, (long long) dst_rowb);
        return false;
    }
    const ggml_type_traits * tr = ggml_get_type_traits((ggml_type) t.type);
    if (tr == nullptr || tr->to_float == nullptr) {
        fprintf(stderr, "[bound] %s[%d] DECLINED: no dequantiser for type %d\n", role, idx, t.type);
        return false;
    }

    // Chunked so host memory stays bounded regardless of tensor size (the head is 248320 rows).
    const int64_t chunk = 4096;
    std::vector<char>  src((size_t) chunk * (size_t) src_rowb);
    std::vector<float> mid((size_t) chunk * (size_t) ncols);
    std::vector<char>  dst((size_t) rows * (size_t) dst_rowb);
    // A null arm is only null if the round trip leaves the bytes unchanged. K-quant quantisers use an
    // iterative search, so idempotence is not guaranteed; count the rows that change.
    int64_t self_diff = 0;
    // AI_BOUND_HOP2=1 checks idempotence of the target type on the real weights: hop 1 is
    // t.type -> want, hop 2 is want -> float -> want on hop 1's output, and 0/N rows should change.
    // The _0 quantisers are closed-form absmax, so this holds on the no-imatrix branch
    // (quant_weights == nullptr, as passed below); with an imatrix ggml runs a weighted search that
    // is not idempotent.
    const bool hop2 = getenv("AI_BOUND_HOP2") != nullptr;
    const ggml_type_traits * trd = hop2 ? ggml_get_type_traits((ggml_type) want) : nullptr;
    std::vector<float> mid2(hop2 ? (size_t) chunk * (size_t) ncols : 0);
    std::vector<char>  hop (hop2 ? (size_t) chunk * (size_t) dst_rowb : 0);
    int64_t hop2_diff = 0;
    for (int64_t r0 = 0; r0 < rows; r0 += chunk) {
        const int64_t n = std::min<int64_t>(chunk, rows - r0);
        // A host-resident tensor (expert offload) has buf == -1; its data pointer is already host
        // memory, and dev.download would index bufs_.at(-1) and throw. Copy directly instead.
        if (t.buf < 0) {
            memcpy(src.data(), (const char *) t.data + (size_t) r0 * (size_t) src_rowb,
                   (size_t) n * (size_t) src_rowb);
        } else {
            dev.download(t, src.data(), (size_t) r0 * (size_t) src_rowb,
                         (size_t) n * (size_t) src_rowb);
        }
        tr->to_float(src.data(), mid.data(), n * ncols);
        ggml_quantize_chunk((ggml_type) want, mid.data(), dst.data() + (size_t) r0 * (size_t) dst_rowb,
                            0, n, ncols, nullptr);
        if (self) {
            for (int64_t r = 0; r < n; ++r) {
                if (memcmp(src.data() + (size_t) r * (size_t) src_rowb,
                           dst.data() + (size_t) (r0 + r) * (size_t) dst_rowb,
                           (size_t) dst_rowb) != 0) {
                    ++self_diff;
                }
            }
        }
        if (hop2 && trd != nullptr && trd->to_float != nullptr) {
            trd->to_float(dst.data() + (size_t) r0 * (size_t) dst_rowb, mid2.data(), n * ncols);
            ggml_quantize_chunk((ggml_type) want, mid2.data(), hop.data(), 0, n, ncols, nullptr);
            for (int64_t r = 0; r < n; ++r) {
                if (memcmp(hop.data() + (size_t) r * (size_t) dst_rowb,
                           dst.data() + (size_t) (r0 + r) * (size_t) dst_rowb,
                           (size_t) dst_rowb) != 0) {
                    ++hop2_diff;
                }
            }
        }
    }
    if (hop2) {
        fprintf(stderr, "[bound] HOP-2 IDEMPOTENCE role=%s[%d] type %d: %lld/%lld rows differ on "
                        "%s -> float -> %s%s\n",
                role, idx, want, (long long) hop2_diff, (long long) rows,
                trd && trd->to_float ? "requant" : "NO DEQUANTISER",
                trd && trd->to_float ? "requant" : "",
                hop2_diff == 0 ? "  (idempotent: clean baseline)"
                               : "  <-- NOT IDEMPOTENT, this type is not a clean baseline");
    }
    if (bound_requant_discard()) {
        // The full host cost has been paid above; leave the tensor untouched.
        if (announce) {
            fprintf(stderr,
                "[bound] AI_BOUND_SUB DISCARD role=%s[%d]: type %d -> %d computed then THROWN AWAY, "
                "[%lld,%lld]. Full requant CPU cost paid; weights and output BIT-IDENTICAL to stock.\n",
                role, idx, t.type, want, (long long) ncols, (long long) rows);
        }
        return true;
    }
    if (t.buf < 0) {
        memcpy(t.data, dst.data(), dst.size());
    } else {
        dev.upload(t, dst.data(), 0, dst.size());
    }
    const int old_type = t.type;
    t.type = want;
    set_contiguous_strides(t);
    if (announce && self) {
        fprintf(stderr,
            "[bound] AI_BOUND_SUB NULL ARM role=%s[%d]: type %d -> %d (SELF), [%lld,%lld], "
            "%lld bytes/row UNCHANGED, %lld/%lld rows differ after round trip. Pays the full "
            "requant load cost; expected speed delta 0.00%%.\n",
            role, idx, old_type, want, (long long) ncols, (long long) rows,
            (long long) dst_rowb, (long long) self_diff, (long long) rows);
    } else if (announce) {
        fprintf(stderr,
            "[bound] AI_BOUND_SUB LIVE role=%s[%d]: type %d -> %d, [%lld,%lld], %lld -> %lld "
            "bytes/row (%.1f%%), requantised IN PLACE. OUTPUT DEGRADED BY DESIGN, timing only.\n",
            role, idx, old_type, want, (long long) ncols, (long long) rows,
            (long long) src_rowb, (long long) dst_rowb,
            100.0 * (double) dst_rowb / (double) src_rowb);
    }
    return true;
}

// Apply AI_BOUND_SUB for the roles that need real requantisation. Called once, after bind_layers.
void Qwen35Hybrid::bound_apply_roles(Device & dev) {
    const int head_t = bound_sub_type("head");
    if (head_t >= 0) {
        bound_requant_inplace(dev, output_, head_t, "head", 0, true);
    }
    const int down_t = bound_sub_type("expert_down");
    if (down_t >= 0) {
        int done = 0;
        for (size_t i = 0; i < moe_.size(); ++i) {
            // Only the first layer announces; the rest would be identical lines.
            if (moe_[i].down_exps.valid() &&
                (moe_[i].down_exps.type != down_t || bound_selfrequant())) {
                bound_requant_inplace(dev, moe_[i].down_exps, down_t, "expert_down", (int) i,
                                      done == 0);
                ++done;
            }
        }
        fprintf(stderr, "[bound] expert_down: %d tensors requantised in place\n", done);
    }

    // AI_BOUND_SUB=qkvz: real in-place requant. The donor path (bound_halve_qkvz) only works for a type
    // the model already contains, so types such as Q4_0 and Q5_0 need this. In-place is safe because
    // qkvz is Q8_0 and every target type is smaller.
    const int qkvz_t = bound_sub_type("qkvz");
    if (qkvz_t >= 0) {
        int done = 0, skipped_donor = 0;
        for (size_t i = 0; i < gdn_.size(); ++i) {
            Tensor & q = gdn_[i].qkvz;
            if (!q.valid() || (q.type == qkvz_t && !bound_selfrequant())) {
                continue;
            }
            // If bind_layers already swapped a donor in, q.type is the target and was skipped above.
            // In donor mode, leave remaining donor-eligible tensors to that path and count them.
            if (getenv("AI_BOUND_QKVZ_DONOR") != nullptr &&
                bound_donor(qkvz_t, q.ne[0], q.ne[1]) != nullptr) {
                ++skipped_donor;   // legacy donor mode handled it in bind_layers
                continue;
            }
            bound_requant_inplace(dev, q, qkvz_t, "qkvz", (int) i, done == 0);
            ++done;
        }
        fprintf(stderr, "[bound] qkvz: %d tensors requantised in place (%d left to the donor path)\n",
                done, skipped_donor);
    }

    // Data-identity check, on every run including stock: count distinct qkvz pointers. Matching kernel
    // counts per round cannot detect aliasing (as the donor path does), because aliasing changes
    // which bytes are read, not how much work runs.
    {
        std::vector<const void *> ptr;
        int64_t rowb = 0, rows = 0;
        for (size_t i = 0; i < gdn_.size(); ++i) {
            const Tensor & q = gdn_[i].qkvz;
            if (!q.valid()) {
                continue;
            }
            ptr.push_back((const void *) q.data);
            rowb = bound_row_bytes(q.type, q.ne[0]);
            rows = q.numel() / q.ne[0];
        }
        std::sort(ptr.begin(), ptr.end());
        const size_t uniq = (size_t) (std::unique(ptr.begin(), ptr.end()) - ptr.begin());
        fprintf(stderr,
                "[bound] qkvz DATA IDENTITY: %zu distinct pointers / %zu layers, %.1f MB distinct "
                "per round%s\n",
                uniq, ptr.size(), (double) uniq * (double) rowb * (double) rows / 1e6,
                uniq == ptr.size() ? "" :
                    "  <-- ALIASED: byte accounting is INVALID, this is not a byte arm");
    }
}

void Qwen35Hybrid::alloc_stage(Device & dev, int n_streams) {
    const int64_t E    = spec_.n_embd;
    const int64_t rows = n_streams;
    // The q8 buffers are sized for the widest round (an MTP round has T = k+1 rows per stream);
    // otherwise add_rmsnorm's `q8_buf->numel() >= bytes` check fails and the fold silently does nothing.
    //
    // stage_out_ must not be widened: it is handed downstream as a typed tensor (`o.out = *out_buf`
    // carries its ne), so a wider buffer would give the wrong shape and trip shape asserts. stage_q8_
    // is a 1-D I8 byte array whose numel is only compared, never used as a shape, so it can be oversized.
    const int64_t q8_rows = (int64_t) n_streams * unscoped_rows_;
    if (E % 512 != 0) {
        return;   // the staged copy is bit-exact only when the row length is a padding multiple
    }
    const size_t per = (size_t) E * 4 * (size_t) rows + (size_t) E * 4 * (size_t) q8_rows
                       + 2 * ((size_t) (E / 32) * 36 * (size_t) q8_rows + 4096) + 4 * dev.alignment();
    const int    buf = dev.alloc_buffer(per * (size_t) spec_.n_layer + dev.alignment(), "stage");
    size_t       off = 0;
    for (int l = 0; l < spec_.n_layer; ++l) {
        stage_out_.push_back(carve_stage(dev, buf, off, F32, E, rows));
        // +4 KiB of slack: MMVQ is written to read a q8_1 buffer that is padded for out-of-bounds
        // safety (MATRIX_ROW_PADDING exists for exactly that), and a tightly-sized staging buffer
        // would have it reading into the next layer's f32 activation.
        stage_q8_.push_back(carve_stage(dev, buf, off, I8, (E / 32) * 36 * q8_rows + 4096, 1));
        // Sized for the widest round, like stage_q8_. This is safe, unlike stage_out_, because its one
        // consumer (the merge's norm path) narrows it to a [E, T] view before anything sees its shape;
        // at n_streams rows the AI_FUSE_MERGENORM path would decline every MTP round.
        norm_out_.push_back(carve_stage(dev, buf, off, F32, E, q8_rows));
        // its own buffer, never stage_q8_: that one is overwritten by the next layer's add_rmsnorm while
        // this layer's registry entry would still point at it
        stage_q8_in_.push_back(carve_stage(dev, buf, off, I8, (E / 32) * 36 * q8_rows + 4096, 1));
    }
    // AI_GDN_GATE: per-layer output-gate buffers (f32 [NH, q8_rows] + q8_1 copy), GDN layers only
    const int64_t NH = gdn_p_.head_v * gdn_p_.v_heads;
    if (NH > 0 && NH % 512 == 0) {
        const size_t pg = (size_t) NH * 4 * (size_t) q8_rows + (size_t) (NH / 32) * 36 * (size_t) q8_rows + 4096 +
                          2 * dev.alignment();
        int n_g = 0;
        for (int l = 0; l < spec_.n_layer; ++l) {
            n_g += spec_.is_gdn[(size_t) l] ? 1 : 0;
        }
        const int gbuf = dev.alloc_buffer(pg * (size_t) n_g + dev.alignment(), "gdn_gate");
        size_t    goff = 0;
        stage_gdn_out_.assign((size_t) spec_.n_layer, Tensor{});
        stage_gdn_q8_.assign((size_t) spec_.n_layer, Tensor{});
        for (int l = 0; l < spec_.n_layer; ++l) {
            if (spec_.is_gdn[(size_t) l]) {
                stage_gdn_out_[(size_t) l] = carve_stage(dev, gbuf, goff, F32, NH, q8_rows);
                stage_gdn_q8_[(size_t) l]  = carve_stage(dev, gbuf, goff, I8, (NH / 32) * 36 * q8_rows + 4096, 1);
            }
        }
    }
}

int Qwen35Hybrid::doorbell_slots_for(size_t budget, bool with_head) const {
#if defined(AI_DOORBELL)
    const char * om = getenv("AI_OFFLOAD");
    if (!om || std::string(om) != "doorbell") return -1;
    // DoorbellMoe's device carve: per bank (n_slots + k) experts + a table; per staging group pf experts. The
    // groups are not known before the banks are compared, so the estimate takes two groups of the largest expert.
    const int k  = spec_.n_expert_used, E = spec_.n_expert;
    const int pf = getenv("AI_DOOR_PF_SLOTS") ? std::min(atoi(getenv("AI_DOOR_PF_SLOTS")), E) : std::min(E, 64);
    size_t per_slot = 0, fixed = 0, biggest = 0;
    auto add = [&](const op::MoeWeights & w) {
        const size_t b = offload::DoorbellMoe::expert_bytes(w);
        per_slot += b;
        fixed    += (size_t) k * b + (size_t) E * 4 + 4 * 256;
        biggest   = std::max(biggest, b);
    };
    for (int il = 0; il < spec_.n_layer; ++il) add(moe_[(size_t) il]);
    if (mtp_.valid && with_head) add(mtp_.moe);
    fixed += 2 * (size_t) pf * biggest + (1u << 20);
    if (per_slot == 0 || budget <= fixed) return 0;
    return (int) std::min<size_t>((size_t) E, (budget - fixed) / per_slot);
#else
    (void) budget;
    (void) with_head;
    return -1;
#endif
}

void Qwen35Hybrid::alloc_expert_pool(Device & dev, int n_slots, bool with_head, int64_t max_rows) {
#if defined(AI_DOORBELL)
    // AI_OFFLOAD=doorbell: every MoE layer (and the MTP head's) keeps n_slots resident experts in its own slot bank;
    // misses are computed by the CPU pool through doorbells, with no host synchronisation in a decode round
    if (const char * om = getenv("AI_OFFLOAD"); om && std::string(om) == "doorbell") {
        std::vector<op::MoeWeights> banks(moe_.begin(), moe_.begin() + spec_.n_layer);
        if (mtp_.valid && with_head) banks.push_back(mtp_.moe);   // bank n_layer = the MTP head
        std::vector<int> slots(banks.size(), n_slots);
        offload::DoorbellConfig dc;
        dc.t_max    = getenv("AI_DOOR_TMAX") ? atoi(getenv("AI_DOOR_TMAX"))
                                           : std::max<int64_t>({ 8, unscoped_rows_, max_rows });
        dc.admit    = getenv("AI_DOOR_ADMIT") ? atoi(getenv("AI_DOOR_ADMIT")) : 32;
        dc.threads  = getenv("AI_DOOR_THREADS") ? atoi(getenv("AI_DOOR_THREADS")) : 0;
        dc.pf_slots = getenv("AI_DOOR_PF_SLOTS") ? atoi(getenv("AI_DOOR_PF_SLOTS")) : 0;
        if (const char * pr = getenv("AI_PROFILE")) dc.profile = pr;
        door_ = std::make_unique<offload::DoorbellMoe>(dev, banks, slots, spec_.n_expert, spec_.n_expert_used,
                                                       spec_.n_embd, dc);
        return;
    }
#endif
    std::vector<std::vector<Tensor>> banks;
    for (int il = 0; il < spec_.n_layer; ++il) {
        const op::MoeWeights & m = moe_[(size_t) il];
        banks.push_back({ m.up_exps, m.gate_exps, m.down_exps });
    }
    if (mtp_.valid && with_head) {   // pool layer n_layer = the MTP head
        banks.push_back({ mtp_.moe.up_exps, mtp_.moe.gate_exps, mtp_.moe.down_exps });
    }
    pool_.alloc(dev, banks, spec_.n_expert, n_slots);
    h_route_cap_ = unscoped_rows_;
    const size_t bytes = (size_t) (spec_.n_expert_used * h_route_cap_) * 4;
    h_sel_   = (int32_t *) dev.alloc_pinned(bytes);
    h_slots_ = (int32_t *) dev.alloc_pinned(bytes);
    h_pred_ = (int32_t *) dev.alloc_pinned((size_t) (spec_.n_expert * h_route_cap_) * 4);
    pred_last_.resize((size_t) pool_.n_layers());
    h_pf_.resize((size_t) pool_.n_layers());
    for (auto & p : h_pf_) {
        p = (int32_t *) dev.alloc_pinned(bytes);
        std::fill(p, p + spec_.n_expert_used * h_route_cap_, -1);
    }
}

Tensor Qwen35Hybrid::moe_streamed_fixed(Device & dev, int il, const op::MoeWeights & mw, const Tensor & x,
                                        const Tensor & resid) {
    const int     k = moe_p_.n_used;
    const int64_t T = x.ne[1];
    if (!streamed_fixed_ok(T)) {
        throw std::runtime_error("moe_streamed_fixed: round too large for one slot group");
    }
    op::MoeRoute route = op::moe_route(dev, x, mw.router, moe_p_);
    Tensor       st    = dev.ws_alloc(I32, k, T);
    const Tensor sel   = route.sel;
    static const int pf_mode = getenv("AI_L2_PREFETCH") ? atoi(getenv("AI_L2_PREFETCH")) : 0;
    const int    nxt   = il + 1 < pool_.n_layers() && il + 1 < spec_.n_layer ? il + 1 : -1;
    Tensor       pf    = pf_mode == 1 && nxt >= 0 ? dev.ws_alloc(I32, k * T) : Tensor{};
    static const int pred_mode = getenv("AI_PF_PRED") ? atoi(getenv("AI_PF_PRED")) : 0;
    static const int pred_top  = getenv("AI_PF_TOP") ? atoi(getenv("AI_PF_TOP")) : 10;
    static const int pred_max  = getenv("AI_PF_MAX") ? atoi(getenv("AI_PF_MAX")) : 2;
    const bool   pred3 = (pred_mode == 3 || pred_mode == 4) && resid.valid() && nxt >= 0;
    const Tensor pids  = pred3 ? op::router_topk(dev, resid, post_norm_[(size_t) nxt], moe_[(size_t) nxt].router,
                                                 spec_.rms_eps, pred_top)
                               : Tensor{};
    op::host_step(dev, [this, &dev, il, k, T, sel, st, nxt, pf, pids]() {
        const size_t n  = (size_t) (k * T);
        const auto   t0 = std::chrono::steady_clock::now();
        dev.download_async(sel, h_sel_, 0, n * 4);
        if (pids.valid()) {
            dev.download_async(pids, h_pred_, 0, (size_t) (pred_top * T) * 4);
        }
        dev.synchronize();
        const auto   t1 = std::chrono::steady_clock::now();
        static const int mode = getenv("AI_ROUTE_MODE") ? atoi(getenv("AI_ROUTE_MODE")) : 0;
        if (route_prev_.size() < (size_t) pool_.n_layers()) {
            route_prev_.resize((size_t) pool_.n_layers());
            route_frozen_.resize((size_t) pool_.n_layers());
        }
        std::vector<int32_t> & prev = route_prev_[(size_t) il];
        if (mode == 1) {
            std::vector<int32_t> & fz = route_frozen_[(size_t) il];
            if (fz.empty()) {
                fz.assign(h_sel_, h_sel_ + k);
            }
            for (size_t i = 0; i < n; ++i) {
                h_sel_[i] = fz[i % (size_t) k];
            }
        } else if (mode == 2 && !prev.empty()) {
            const ExpertPool::Layer & L0 = pool_.layer(il);
            std::vector<int32_t> cand;
            for (int e = 0; e < moe_p_.n_expert; ++e) {
                if (L0.slot_of[(size_t) e] >= 0 && std::find(prev.begin(), prev.end(), e) == prev.end()) {
                    cand.push_back(e);
                }
            }
            if ((int) cand.size() >= k) {
                for (int j = 0; j < k; ++j) {   // partial Fisher-Yates
                    route_rng_ ^= route_rng_ << 13; route_rng_ ^= route_rng_ >> 7; route_rng_ ^= route_rng_ << 17;
                    const size_t r = (size_t) j + (size_t) (route_rng_ % (cand.size() - (size_t) j));
                    std::swap(cand[(size_t) j], cand[r]);
                }
                for (size_t i = 0; i < n; ++i) {
                    h_sel_[i] = cand[i % (size_t) k];
                }
            }
        }
        for (size_t i = 0; i < n; ++i) {
            prof_.overlap_all++;
            prof_.overlap_hit += std::find(prev.begin(), prev.end(), h_sel_[i]) != prev.end() ? 1 : 0;
        }
        prev.assign(h_sel_, h_sel_ + n);
        std::vector<int32_t> uniq;
        uniq.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            if (std::find(uniq.begin(), uniq.end(), h_sel_[i]) == uniq.end()) {
                uniq.push_back(h_sel_[i]);
            }
        }
        {
            // prediction quality: of the experts that were cold before this step, how many were predicted
            const ExpertPool::Layer & Lq = pool_.layer(il);
            const std::vector<int32_t> & pl = pred_last_[(size_t) il];
            for (int32_t e : uniq) {
                const bool pf = Lq.group >= 0 && Lq.slot_of[(size_t) e] >= 0 &&
                                pool_.group(Lq.group).prefetched[(size_t) Lq.slot_of[(size_t) e]];
                if (Lq.slot_of[(size_t) e] < 0 || pf) {
                    pred_.cold_true++;
                    pred_.cold_pred += std::find(pl.begin(), pl.end(), e) != pl.end() ? 1 : 0;
                }
            }
            pred_last_[(size_t) il].clear();
        }
        pool_.ensure(dev, il, uniq);   // missing row spans: async copies from registered host memory
        const ExpertPool::Layer & L = pool_.layer(il);
        for (size_t i = 0; i < n; ++i) {
            h_slots_[i] = L.slot_of[(size_t) h_sel_[i]];
        }
        dev.upload_async(st, h_slots_, 0, n * 4);
        if (pids.valid()) {
            // two-stage lookahead: layer l+1's router on the residual entering layer l's MoE; the
            // copies overlap layer l's experts and layer l+1's mixer (mode 4 = read only)
            const ExpertPool::Layer & Ln = pool_.layer(nxt);
            std::vector<int32_t> cold;
            for (int r = 0; r < pred_top; ++r) {
                for (int64_t t = 0; t < T; ++t) {
                    const int32_t e = h_pred_[(size_t) (t * pred_top + r)];
                    if (Ln.slot_of[(size_t) e] < 0 && std::find(cold.begin(), cold.end(), e) == cold.end()) {
                        cold.push_back(e);
                    }
                }
            }
            pred_.steps++;
            pred_.cand += cold.size();
            pred_last_[(size_t) nxt].assign(cold.begin(), cold.begin() + std::min<size_t>(cold.size(), (size_t) pred_max));
            if (pred_mode == 3) {
                pool_.prefetch(dev, nxt, cold, pred_max);
            }
        }
        if (pf.valid()) {
            // guess for layer l+1: the slots its previous step used (still resident ones)
            const std::vector<int32_t> & pv = route_prev_[(size_t) nxt];
            const ExpertPool::Layer &    Ln = pool_.layer(nxt);
            int32_t * g = h_pf_[(size_t) nxt];
            for (size_t i = 0; i < n; ++i) {
                g[i] = i < pv.size() ? Ln.slot_of[(size_t) pv[i]] : -1;
            }
            dev.upload_async(pf, g, 0, n * 4);
        }
        const auto t2 = std::chrono::steady_clock::now();
        prof_.wait_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        prof_.cpu_ms  += std::chrono::duration<double, std::milli>(t2 - t1).count();
        prof_.steps++;
    });
    const ExpertPool::Layer & L = pool_.layer(il);
    if (pf_mode == 2) {
        op::l2_prefetch(dev, L.up.pool, L.gate.pool, L.down.pool, st);
    }
    if (pf.valid()) {
        const ExpertPool::Layer & Ln = pool_.layer(nxt);
        op::l2_prefetch(dev, Ln.up.pool, Ln.gate.pool, Ln.down.pool, pf);
    }
    Tensor part = op::moe_experts(dev, x, L.up.pool, L.gate.pool, L.down.pool, st, route.weights, k);
    return op::add(dev, part, op::moe_shared(dev, x, mw));
}

void Qwen35Hybrid::predict_next(Device & dev, int il, const Tensor & y) {
    static const int mode = getenv("AI_PF_PRED") ? atoi(getenv("AI_PF_PRED")) : 0;
    if (mode != 1 && mode != 2) {
        return;
    }
    static const int top = getenv("AI_PF_TOP") ? atoi(getenv("AI_PF_TOP")) : 10;
    static const int mx  = getenv("AI_PF_MAX") ? atoi(getenv("AI_PF_MAX")) : 2;
    const int     nxt = il + 1;
    const int64_t T   = y.ne[1];
    Tensor ids = op::router_topk(dev, y, post_norm_[(size_t) nxt], moe_[(size_t) nxt].router, spec_.rms_eps, top);
    op::host_step(dev, [this, &dev, nxt, T, ids]() {
        const size_t n = (size_t) (top * T);
        dev.download_async(ids, h_pred_, 0, n * 4);
        dev.synchronize();
        // rank-interleaved over rows: rank 0 of every row, then rank 1, ...
        const ExpertPool::Layer & L = pool_.layer(nxt);
        std::vector<int32_t> cold;
        for (int r = 0; r < top; ++r) {
            for (int64_t t = 0; t < T; ++t) {
                const int32_t e = h_pred_[(size_t) (t * top + r)];
                if (L.slot_of[(size_t) e] < 0 && std::find(cold.begin(), cold.end(), e) == cold.end()) {
                    cold.push_back(e);
                }
            }
        }
        pred_.steps++;
        pred_.cand += cold.size();
        // what would be prefetched (for the accuracy counter): the first mx cold candidates
        std::vector<int32_t> & pl = pred_last_[(size_t) nxt];
        pl.assign(cold.begin(), cold.begin() + std::min<size_t>(cold.size(), (size_t) mx));
        if (mode == 1) {
            pool_.prefetch(dev, nxt, cold, mx);
        }
    });
}

Tensor Qwen35Hybrid::moe_streamed(Device & dev, int il, const op::MoeWeights & mw, const Tensor & x) {
    const int     k = moe_p_.n_used;
    const int64_t T = x.ne[1];
    op::MoeRoute route = op::moe_route(dev, x, mw.router, moe_p_);

    // the one host round trip of the streamed assembly: which experts does this layer need?
    std::vector<int32_t> sel((size_t) (k * T));
    dev.download(route.sel, sel.data(), 0, sel.size() * 4);

    const ExpertPool::Layer & L = pool_.layer(il);
    std::vector<char>    want((size_t) moe_p_.n_expert, 0);
    std::vector<int32_t> resident, missing;
    for (int32_t e : sel) {
        if (want[(size_t) e]) {
            continue;
        }
        want[(size_t) e] = 1;
        (L.slot_of[(size_t) e] >= 0 ? resident : missing).push_back(e);
    }
    std::vector<int32_t> order = resident;
    order.insert(order.end(), missing.begin(), missing.end());

    const int     cap    = pool_.n_slots();
    const size_t  groups = (order.size() + (size_t) cap - 1) / (size_t) cap;
    std::vector<int32_t> slots((size_t) (k * T));
    std::vector<float>   mask((size_t) (k * T));
    Tensor moe;
    for (size_t gi = 0; gi < groups; ++gi) {
        const size_t b = gi * (size_t) cap, e = std::min(order.size(), b + (size_t) cap);
        std::vector<int32_t> group(order.begin() + (long) b, order.begin() + (long) e);
        pool_.ensure(dev, il, group);
        std::vector<char> in_group((size_t) moe_p_.n_expert, 0);
        for (int32_t x_e : group) {
            in_group[(size_t) x_e] = 1;
        }
        // Rows keep k distinct slot ids: ggml's grouped matmul faults on a slot repeated within
        // a row. Experts outside this group take unused occupied slots (weight 0, so
        // their output is dropped).
        const int         bank = pool_.bank_slots(il);   // slot ids span the (shared) bank
        std::vector<char> used((size_t) bank, 0);
        for (int64_t row = 0; row < T; ++row) {
            const size_t r0 = (size_t) (row * k);
            std::fill(used.begin(), used.end(), 0);
            for (int j = 0; j < k; ++j) {
                const int32_t e = sel[r0 + (size_t) j];
                if (in_group[(size_t) e]) {
                    used[(size_t) L.slot_of[(size_t) e]] = 1;
                }
            }
            int next = 0;
            for (int j = 0; j < k; ++j) {
                const int32_t e   = sel[r0 + (size_t) j];
                const bool    hit = in_group[(size_t) e] != 0;
                if (hit) {
                    slots[r0 + (size_t) j] = L.slot_of[(size_t) e];
                } else {
                    // an occupied slot: an empty one holds uninitialised bytes, and 0 * inf is NaN
                    while (next < bank && (used[(size_t) next] || !pool_.occupied(il, next))) {
                        ++next;
                    }
                    if (next == bank) {
                        throw std::runtime_error("moe_streamed: no distinct filler slot");
                    }
                    used[(size_t) next]    = 1;
                    slots[r0 + (size_t) j] = next;
                }
                mask[r0 + (size_t) j] = hit ? 1.0f : 0.0f;
            }
        }
        Tensor st = dev.ws_alloc(I32, k, T);
        dev.upload(st, slots.data(), 0, slots.size() * 4);
        Tensor w = route.weights;
        if (groups > 1) {
            Tensor mk = dev.ws_alloc(F32, 1, k, T);
            dev.upload(mk, mask.data(), 0, mask.size() * 4);
            w = op::mul(dev, route.weights, mk);
        }
        Tensor part = op::moe_experts(dev, x, L.up.pool, L.gate.pool, L.down.pool, st, w, k);
        moe = moe.valid() ? op::add(dev, moe, part) : part;
    }
    return op::add(dev, moe, op::moe_shared(dev, x, mw));
}

Tensor Qwen35Hybrid::ffn_residual(Device & dev, int pool_layer, const op::MoeWeights & w, const Tensor & p,
                                  const Tensor & r, int dump_layer) {
    if (spec_.dense) {
        if (pool_layer < (int) ffn_cpu_layer_.size() && ffn_cpu_layer_[(size_t) pool_layer] >= 0) {
            return op::add(dev, ffn_cpu(dev, pool_layer, w, p), r);   // AI_FFN_CPU
        }
        return op::dense_ffn_residual_norm(dev, p, r, w, nullptr, nullptr, spec_.rms_eps).y;
    }
    static const bool native_merge = !getenv("AI_NATIVE_MERGE") || atoi(getenv("AI_NATIVE_MERGE")) != 0;
    if (!offloaded() && native_merge) {
        return op::moe_residual(dev, p, r, w, moe_p_);
    }
    Tensor f;
#if defined(AI_DOORBELL)
    if (door_) {
        if (p.ne[1] <= door_->t_max()) {
            f = door_->moe(pool_layer, w, moe_p_, p);
        } else if (op::recording()) {
            throw std::runtime_error("ffn: a recorded doorbell MoE round of " + std::to_string(p.ne[1]) +
                                     " rows exceeds the doorbell's t_max " + std::to_string(door_->t_max()) +
                                     " (set AI_DOOR_TMAX >= " + std::to_string(p.ne[1]) + ")");
        } else {
            f = door_->moe_bulk(pool_layer, w, moe_p_, p);
        }
        if (dump_layer >= 0) {
            dump(dev, "ffn_out", dump_layer, f);
        }
        return op::add(dev, f, r);
    }
#endif
    if (!pool_.n_slots()) {
        f = op::moe_block(dev, p, w, moe_p_);
    } else if (streamed_fixed_ok(p.ne[1])) {
        f = moe_streamed_fixed(dev, pool_layer, w, p, r);
    } else if (op::recording()) {
        throw std::runtime_error("ffn: streamed MoE round too large to record (raise --expert-slots)");
    } else {
        f = moe_streamed(dev, pool_layer, w, p);
    }
    if (dump_layer >= 0) {
        dump(dev, "ffn_out", dump_layer, f);
    }
    return op::add(dev, f, r);
}

Tensor Qwen35Hybrid::ffn_residual_norm(Device & dev, int pool_layer, const op::MoeWeights & w, const Tensor & p,
                                       const Tensor & r, int dump_layer, const Tensor * next_norm,
                                       const Tensor * norm_buf, float eps, Tensor & pre_normed,
                                       const Tensor * q8_in, const Tensor * router_pre, bool router_cmp) {
    static const bool native_merge = !getenv("AI_NATIVE_MERGE") || atoi(getenv("AI_NATIVE_MERGE")) != 0;
    if (spec_.dense) {
        if (pool_layer < (int) ffn_cpu_layer_.size() && ffn_cpu_layer_[(size_t) pool_layer] >= 0) {   // AI_FFN_CPU
            const Tensor f = ffn_cpu(dev, pool_layer, w, p);
            if (next_norm != nullptr && next_norm->valid()) {   // the residual add also yields the next layer's norm
                const op::AddNorm an = op::add_rmsnorm(dev, f, r, *next_norm, eps, norm_buf, q8_in);
                pre_normed = an.out;
                return an.sum;
            }
            pre_normed = Tensor{};
            return op::add(dev, f, r);
        }
        auto o     = op::dense_ffn_residual_norm(dev, p, r, w, next_norm, norm_buf, eps, q8_in);
        pre_normed = o.normed;
        return o.y;
    }
    if (!offloaded() && native_merge && next_norm != nullptr && norm_buf != nullptr) {
        auto o     = op::moe_residual_norm(dev, p, r, w, moe_p_, *next_norm, *norm_buf, eps, q8_in, router_pre,
                                           router_cmp);
        pre_normed = o.normed;
        return o.y;
    }
    return ffn_residual(dev, pool_layer, w, p, r, dump_layer);
}

op::AttnInputs Qwen35Hybrid::attn_inputs(const Inputs & in) const {
    op::AttnInputs a;
    a.pos4         = in.pos4;
    a.write_cells  = in.cells;
    a.mask         = in.mask;
    a.n_kv         = in.n_kv;
    a.B            = in.B;
    a.region_cells = kv_.region_cells();
    a.first_cell   = kv_.first_cell(in.first_region);
    return a;
}

size_t Qwen35Hybrid::workspace_bytes(int64_t t_max, int64_t n_logits) const {
    return workspace_bytes_for(t_max, n_logits, kv_.capacity_cells(), kv_.n_regions());
}

int64_t Qwen35Hybrid::kv_cache_bytes(int64_t max_ctx, int n_streams, int kv_type, size_t alignment) const {
    int n_attn = 0;
    for (bool g : spec_.is_gdn) n_attn += g ? 0 : 1;
    const int     layers = n_attn + (mtp_.valid ? 1 : 0);
    const int64_t region = (max_ctx + PagedKv::kPage - 1) / PagedKv::kPage * PagedKv::kPage;   // as PagedKv::alloc
    const size_t  row    = ggml_row_size((ggml_type) kv_type, (int64_t) spec_.head_dim * spec_.n_head_kv);
    const size_t  per    = (row * (size_t) (region * n_streams) + alignment - 1) / alignment * alignment;
    return (int64_t) (per * 2 * (size_t) layers);
}

int64_t Qwen35Hybrid::state_bytes(int n_streams, int64_t q8_rows_per_stream, size_t alignment) const {
    int n_gdn = 0;
    for (bool g : spec_.is_gdn) n_gdn += g ? 1 : 0;
    const int64_t conv_n = (int64_t) (spec_.conv_k - 1) * spec_.conv_channels();
    const int64_t ssm_n  = (int64_t) spec_.head_v() * spec_.head_v() * spec_.ssm_dt_rank;
    const size_t  slots  = ((size_t) (conv_n + ssm_n) * 4 * (size_t) n_streams + 4 * alignment) * (size_t) n_gdn;
    const int64_t E      = spec_.n_embd;
    const size_t  q8     = (size_t) n_streams * (size_t) q8_rows_per_stream;
    const size_t  stage  = E % 512 != 0 ? 0
                           : ((size_t) E * 4 * (size_t) n_streams + (size_t) E * 4 * q8 +
                              2 * ((size_t) (E / 32) * 36 * q8 + 4096) + 4 * alignment) * (size_t) spec_.n_layer + alignment;
    return (int64_t) (slots + stage);
}

bool Qwen35Hybrid::chunk_scope_on() {   // AI_CHUNK_SCOPE, default on; 0 lets a short prefill chunk run unscoped
    static const bool v = getenv("AI_CHUNK_SCOPE") ? atoi(getenv("AI_CHUNK_SCOPE")) != 0 : true;
    return v;
}

bool Qwen35Hybrid::mix_scope_on() {   // AI_MIX_SCOPE, default on; 0 restores the whole-layer scope
    static const bool v = getenv("AI_MIX_SCOPE") ? atoi(getenv("AI_MIX_SCOPE")) != 0 : true;
    return v;
}

size_t Qwen35Hybrid::workspace_bytes_for(int64_t t_max, int64_t n_logits, int64_t capacity_cells, int n_regions) const {
    // Per-row high-water of one layer scope (all f32): the MoE block dominates
    // (8 routed rows of up/gate/act/down plus weighting and the running sum).
    const int64_t E   = spec_.n_embd;
    const int64_t C   = spec_.conv_channels();
    // dense: one "expert" of the FFN width (gate, up, act and down rows in scope)
    const int64_t F   = spec_.dense ? spec_.ffn_length : 512;
    const int64_t k   = spec_.dense ? 1 : spec_.n_expert_used;
    // AI_GDN16: native chunked GDN scratch per row: normalised q and k (2 Hk Sk), gate and beta (2 Hv), and the control
    // packets (2176 B per head per 16 tokens = 34 floats per head). Its output replaces the ggml result, already counted.
    const int64_t gdn16_row = kern::gdn16_enabled() ? 2 * (int64_t) spec_.ssm_groups * spec_.ssm_state + 36 * (int64_t) spec_.ssm_dt_rank : 0;
    const int64_t per_row =
        4 * (6 * E                                   // norms, residuals, carries in scope
             + 3 * C                                 // qkv, conv input, conv output
             + 4 * (int64_t) spec_.ssm_inner         // z, gdn out, norm, gate
             + 2 * (int64_t) spec_.head_dim * spec_.n_head * 2   // q+gate, rope
             + 2 * spec_.n_expert                    // router logits, probs
             + k * (4 * F + 3 * E)                   // up, gate, act, mul; down, weighted, sums
             + 8 * E                                 // shared expert and final adds
             // add_rmsnorm's sum + normed output for the dense FFN residual, and the attention prologue's
             // q / gate outputs (+ compare shadows); MoE terms can hide their absence, a dense model cannot.
             + 4 * E
             + 4 * (int64_t) spec_.head_dim * spec_.n_head
             + gdn16_row);
    // AI_MIX_SCOPE: a scoped layer rolls its mixer block back before the FFN, so its peak is the larger block, not the
    // sum: common + max(mixer, ffn) + the mixer's output slot.
    const int64_t mixer_row = 4 * (3 * C + 4 * (int64_t) spec_.ssm_inner + 2 * (int64_t) spec_.head_dim * spec_.n_head * 2 +
                                   4 * (int64_t) spec_.head_dim * spec_.n_head + gdn16_row);
    const int64_t ffn_row   = 4 * (2 * spec_.n_expert + k * (4 * F + 3 * E) + 8 * E + 4 * E);
    const int64_t scoped_row = mix_scope_on() ? 4 * 6 * E + std::max(mixer_row, ffn_row) + 4 * E : per_row;
    const int64_t fixed = 96LL << 20;   // 64 MiB is too tight for a dense model at --chunk 1
    // AI_FA_Q (kernels/fa_decode_q.cu): its per-call partials. Added only when the path can run, and sized for every
    // stream at <= 8 rows.
    const int64_t fa_q = kern::fa_decode_q_enabled() && spec_.n_head_kv > 0
                             ? kern::fa_decode_q_ws_bound(spec_.n_head, spec_.n_head_kv,
                                                          std::max<int64_t>(1, n_regions))
                             : 0;
    // The f16 [t_max, capacity] mask is not in the workspace: round masks live in the engine's ingress buffers (sized
    // there and counted by plan_kv); counting it here would double-count it. Only the MTP chain's per-step [n_kv, B]
    // masks are here.
    (void) capacity_cells;
    const int64_t carry = 2 * 4 * E * t_max + 4 * t_max * 8;
    const int64_t head  = n_logits * (spec_.n_vocab * 4 + E * 8);
    // a record-mode round wider than the unscoped budget keeps its GDN records in an arena
    const int64_t rec_arena = n_logits > unscoped_rows_ ? (int64_t) rec_arena_bytes(n_logits) : 0;
    // Rounds of <= 16 rows keep every layer's activations (no per-layer rollback), so size for
    // the larger of one scoped layer at t_max and all layers unscoped at min(t_max, 16).
    const int64_t scoped   = scoped_row * t_max * 3 / 2;
    const int64_t unscoped = per_row * std::min<int64_t>(t_max, unscoped_rows_) * (spec_.n_layer + spec_.nextn) * 5 / 4;
    return (size_t) (std::max(scoped, unscoped) + fixed + fa_q + carry + head + rec_arena);
}

size_t Qwen35Hybrid::rec_arena_bytes(int64_t T) const {
    // per GDN layer, upper bounds of op::GdnRecord at T rows (f32): conv input (history ++ T columns) and conv output
    // over the conv channels; packed q / k (<= the conv channels); beta|alpha, beta, alpha, gate, sigmoid(beta)
    // (Hv each); plus 256-B alignment for its 9 tensors. persist() throws if a real record exceeds it.
    const int64_t C  = spec_.conv_channels();
    const int64_t Hv = spec_.ssm_dt_rank;
    const int64_t K  = std::max(1, spec_.conv_k);
    const int64_t per_layer = 4 * ((K - 1 + T) * C + C * T + 2 * C * T + 8 * Hv * T) + 9 * 256;
    return (size_t) (per_layer * (int64_t) gdn_.size());
}

int64_t Qwen35Hybrid::visible_cells(int64_t pos0, int64_t T) const {
    // ggml's graph pads the visible-cell count to 256. Computing it exactly changes numerics noticeably
    // (the padded cells change the softmax reduction), and the padding accounts for nearly all of the
    // difference from ggml's forward, so AI_PARITY=1 implies it. AI_KV_PAD overrides, in both directions.
    static const int64_t kv_pad = getenv("AI_KV_PAD") ? atoll(getenv("AI_KV_PAD"))
                                                      : (parity_on() ? 256 : 1);
    return std::min((pos0 + T + kv_pad - 1) / kv_pad * kv_pad, kv_.region_cells());
}

void Qwen35Hybrid::set_pos_override(int seq, const int32_t * pos4, int64_t n) {
    if (seq < 0) return;
    if ((size_t) seq >= pos_ov_.size()) pos_ov_.resize((size_t) seq + 1);
    pos_ov_[(size_t) seq] = { pos4, pos4 ? n : 0 };
}

void Qwen35Hybrid::set_rope_delta(int seq, int64_t delta) {
    if (seq < 0) return;
    if ((size_t) seq >= rope_delta_.size()) rope_delta_.resize((size_t) seq + 1, 0);
    rope_delta_[(size_t) seq] = delta;
}

void Qwen35Hybrid::pos_at(int seq, int64_t cell, int32_t out[4]) const {
    if (seq >= 0 && (size_t) seq < pos_ov_.size() && cell < pos_ov_[(size_t) seq].n) {
        const PosOverride & o = pos_ov_[(size_t) seq];
        for (int k = 0; k < 4; ++k) out[k] = o.pos4[(size_t) k * (size_t) o.n + (size_t) cell];
        return;
    }
    const int32_t p = (int32_t) (cell + rope_delta(seq));
    out[0] = out[1] = out[2] = p;
    out[3] = 0;
}

void Qwen35Hybrid::fill_inputs(int seq, const int32_t * tokens, int64_t L, int64_t pos0, int64_t n_kv,
                               int64_t row0, int64_t T, int32_t * ids, int32_t * pos4, int64_t * cells,
                               uint16_t * mask) {
    kv_.reserve(seq, std::max(n_kv, pos0 + L));
    for (int64_t j = 0; j < L; ++j) {
        const int64_t i = row0 + j;
        int32_t       rp[4];
        pos_at(seq, pos0 + j, rp);   // == (cell, cell, cell, 0) for text
        ids[i]         = tokens[j];
        pos4[i]        = rp[0];
        pos4[T + i]    = rp[1];
        pos4[2*T + i]  = rp[2];
        pos4[3*T + i]  = rp[3];
        cells[i]       = kv_.cell(seq, pos0 + j);
        if (!mask) {
            continue;   // AI_DECODE_MASK_DEV: the round writes the mask on device (kern::decode_causal_mask)
        }
        const int64_t visible = pos0 + j;   // region cells 0..visible are attendable
        uint16_t * row = mask + i * n_kv;
        const int64_t open = std::min(visible + 1, n_kv);
        std::fill(row, row + open, (uint16_t) 0x0000);         // f16 0
        std::fill(row + open, row + n_kv, (uint16_t) 0xFC00);  // f16 -inf
    }
}

Tensor Qwen35Hybrid::forward(Device & dev, const Inputs & in, bool all_logits, GdnRecords * rec) {
    const Qwen35Spec & s = spec_;
    const int64_t E      = s.n_embd;
    const int64_t T      = in.T;
    const int64_t B      = in.B;
    const int64_t L      = T / B;
    const int     slot   = in.first_region;
    if (rec) {
        // A record-mode round wider than the unscoped budget (AI_LOOKUP) runs scoped (per-layer rollback,
        // like a prefill chunk); its GDN records are copied into an arena reserved below, before the layer loop, so
        // they outlive each layer's rollback. Batch 1 only (scoped rounds are).
        if (T > unscoped_rows_ && B > 1) {
            throw std::runtime_error("forward: record mode wider than the unscoped budget is batch-1 only");
        }
        rec->layers.assign(gdn_.size(), op::GdnRecord{});
    }
    const op::AttnInputs ain = attn_inputs(in);

    // Large chunks roll the workspace back per layer, so the residual stream lives in two
    // carries. Small rounds (decode, verify) keep every layer's activations: no carry copies.
    // AI_CHUNK_SCOPE (default 1): a prefill chunk is scoped at any width. An unscoped chunk keeps all n_layer layers'
    // activations, so a short last chunk (prompt length mod chunk <= unscoped_rows_) could overflow a workspace that
    // every full chunk fits in. Scoping only adds the carry copies; the kernels and inputs are the same, so the result is too.
    const bool scoped = T > unscoped_rows_ || (in.chunk && B == 1 && T > 1 && chunk_scope_on());
    if (getenv("AI_SHAPE_TRACE")) {
        fprintf(stderr, "[shape] forward T=%lld B=%lld L=%lld unscoped=%lld scoped=%d ids=[%lld,%lld]\n",
                (long long) T, (long long) B, (long long) L, (long long) unscoped_rows_, (int) scoped,
                (long long) in.ids.ne[0], (long long) in.ids.ne[1]);
    }
    if (scoped && B > 1) {
        throw std::runtime_error("forward: batched rounds must fit the unscoped row budget");
    }
    if (in.chunk) {
        kern::q8_stage_clear();   // AI_CHUNK_SCOPE: a prefill chunk starts at ws 0: drop the last round's entries
    }
    // AI_CHUNK_SCOPE: a scoped rollback recycles [mark, high water): drop the staging entries keyed inside it
    const auto drop_staged = [&dev](size_t mark) {
        if (dev.ws_base() != nullptr && dev.ws_capacity() > mark) {
            kern::q8_stage_invalidate((const char *) dev.ws_base() + mark, dev.ws_capacity() - mark);
        }
    };
    Tensor  carry[2];
    int     which = 0;
    int64_t rows  = T;
    Tensor  cur;
    // AI_HOST_EMBD: with a host-kept table the round's rows are already staged as a small q8_0
    // block (Engine::Ingress::alloc_embd), so this is the same get_rows over a gathered prefix with
    // an iota id list -- same kernel, same bytes, bit-identical result.
    const Tensor & etbl = in.embd.valid() ? in.embd : tok_embd_;
    const Tensor & eids = in.embd.valid() ? in.embd_iota : in.ids;
    if (scoped) {
        carry[0] = dev.ws_alloc(F32, E, T);
        carry[1] = dev.ws_alloc(F32, E, T);
        Device::Scope sc(dev);
        op::copy_into(dev, op::embed(dev, etbl, eids), carry[0], "embed->carry");
        cur = carry[0];
    } else {
        cur = op::embed(dev, etbl, eids);
    }
    // image rows: the projector's embeddings replace the placeholder tokens' rows (contiguous runs)
    for (const auto & r : in.img_runs) {
        const size_t rb  = cur.nb[1];
        const Tensor dst = cur.view(E, r[1], 1, 1, rb, rb * (size_t) r[1], rb * (size_t) r[1], rb * (size_t) r[0]);
        const size_t ib  = in.img.nb[1];
        const Tensor src = in.img.view(E, r[1], 1, 1, ib, ib * (size_t) r[1], ib * (size_t) r[1], ib * (size_t) r[2]);
        op::copy_into(dev, src, dst, "mm image rows");
    }

    // AI_MIX_SCOPE: in a scoped (prefill) layer the mixer's output goes to a slot allocated first and its intermediates
    // are rolled back before the FFN, so the MoE block does not run on top of dead GDN intermediates. Copy only.
    const bool k_mix_scope = mix_scope_on();
    // the record arena (see above): one bound per GDN layer, carved by persist() after each gdn_block
    Tensor rec_arena;
    size_t rec_off = 0;
    if (rec && scoped) {
        const size_t bytes = rec_arena_bytes(T);
        rec_arena = dev.ws_alloc(I8, (int64_t) bytes);
    }
    const auto persist = [&](Tensor & t) {
        if (!t.valid()) return;
        Tensor d;
        d.type = t.type;
        for (int i = 0; i < 4; ++i) d.ne[i] = t.ne[i];
        set_contiguous_strides(d);
        const size_t nb = (d.nbytes() + 255) / 256 * 256;
        if (rec_off + nb > (size_t) rec_arena.ne[0]) {
            throw std::runtime_error("forward: wide-verify record arena too small (rec_arena_bytes bound)");
        }
        d.data = (char *) rec_arena.data + rec_off;
        d.buf  = rec_arena.buf;
        rec_off += nb;
        op::copy_into(dev, t, d, "gdn record -> arena");
        t = d;
    };
    Tensor pre_normed;   // the previous layer's merge kernel may have produced this layer's norm
    for (int il = 0; il < s.n_layer; ++il) {
        const size_t mark = dev.ws_mark();
        const bool   mix_scoped = scoped && k_mix_scope;
        Tensor       mix_slot   = mix_scoped ? dev.ws_alloc(F32, E, T) : Tensor{};
        const size_t mix_mark   = dev.ws_mark();
        Tensor h;
        if (pre_normed.valid() && pre_normed.ne[0] == cur.ne[0] && pre_normed.ne[1] == cur.ne[1]) {
            h = pre_normed;   // produced by the previous layer's merge kernel
        } else {
            h = op::rmsnorm(dev, cur, attn_norm_[(size_t) il], s.rms_eps);
        }
        pre_normed = Tensor{};
        op::set_dump_layer(dump_dir() ? il : -1);   // labels the MoE op's router dumps
        dump(dev, "attn_norm", il, h);
        Tensor mix;
        if (s.is_gdn[(size_t) il]) {
            const int g = gdn_layer_[(size_t) il];
            mix = op::gdn_block(dev, h, gdn_[(size_t) il], gdn_p_,
                                gdn_state(g, il, slot, (int) B),
                                rec ? &rec->layers[(size_t) g] : nullptr);
            if (rec && scoped) {   // before the mixer rollback below frees them
                op::GdnRecord & R = rec->layers[(size_t) g];
                for (Tensor * t : { &R.conv_input, &R.conv_out, &R.ba, &R.beta, &R.alpha, &R.qh, &R.kh, &R.gh, &R.bh }) {
                    persist(*t);
                }
            }
            dump(dev, "linear_attn_out", il, mix);
        } else {
            const int a = kv_layer_[(size_t) il];
            mix = op::attn_block(dev, h, attn_[(size_t) il], attn_p_, { kv_.k(a), kv_.v(a) }, ain);
            dump(dev, "attn_output", il, mix);
        }
        if (mix_scoped && mix.type == F32 && mix.ne[0] == E && mix.ne[1] == T) {
            op::copy_into(dev, mix, mix_slot, "mix->slot");
            dev.ws_reset(mix_mark);   // the block's intermediates are dead; h is not used past this point
            drop_staged(mix_mark);   // AI_CHUNK_SCOPE: entries keyed on the recycled range would alias later tensors
            mix = mix_slot;
        }
        Tensor res = cur;
        if (il == s.n_layer - 1 && !all_logits && L > 1) {
            mix  = op::rows(dev, mix, in.last_id);
            res  = op::rows(dev, cur, in.last_id);
            rows = B;
        }
        if (getenv("AI_SHAPE_TRACE") && il == s.n_layer - 1) {
            fprintf(stderr, "[shape] last il=%d mix=[%lld,%lld] res=[%lld,%lld] cur=[%lld,%lld] "
                            "last_id=[%lld] rows=%lld\n",
                    il, (long long) mix.ne[0], (long long) mix.ne[1],
                    (long long) res.ne[0], (long long) res.ne[1],
                    (long long) cur.ne[0], (long long) cur.ne[1],
                    (long long) in.last_id.ne[0], (long long) rows);
        }
        static const bool k_fuse_addnorm = []() {
            const char * e = getenv("AI_FUSE_ADDNORM");
            return e ? atoi(e) != 0 : true;
        }();
        Tensor r, p, router;
        bool   router_cmp = false;
        if (k_fuse_addnorm && mix.type == F32 && res.type == F32 && mix.contiguous() && res.contiguous()) {
            // one launch instead of ADD + (RMS_NORM|MUL); bit-exact, so it is on by default
            const Tensor * sout = il < (int) stage_out_.size() ? &stage_out_[(size_t) il] : nullptr;
            const Tensor * sq8  = il < (int) stage_q8_.size() ? &stage_q8_[(size_t) il] : nullptr;
            // AI_ADDNORM_ROUTER: the MoE router GEMV rides in the same launch (only with the native merge path)
            const op::MoeWeights & mw = moe_[(size_t) il];
            const Tensor * rw = (!offloaded() && mw.route_score.valid()) ? &mw.route_score : nullptr;
            auto an = op::add_rmsnorm(dev, mix, res, post_norm_[(size_t) il], s.rms_eps, sout, sq8, rw);
            r = an.sum;
            p = an.out;
            router     = an.router;
            router_cmp = an.router_cmp;
            dump(dev, "attn_residual", il, r);
        } else {
            r = op::add(dev, mix, res);
            dump(dev, "attn_residual", il, r);
            p = op::rmsnorm(dev, r, post_norm_[(size_t) il], s.rms_eps);
        }
        const Tensor * nn  = (il + 1 < s.n_layer) ? &attn_norm_[(size_t) (il + 1)] : nullptr;
        const Tensor * nob = il < (int) norm_out_.size() ? &norm_out_[(size_t) il] : nullptr;
        const Tensor * q8i = il < (int) stage_q8_in_.size() ? &stage_q8_in_[(size_t) il] : nullptr;
        Tensor y = ffn_residual_norm(dev, il, moe_[(size_t) il], p, r, il, nn, nob, s.rms_eps, pre_normed, q8i,
                                     router.valid() ? &router : nullptr, router_cmp);
        if (pool_.n_slots() && il + 1 < s.n_layer && streamed_fixed_ok(y.ne[1])) {
            predict_next(dev, il, y);
        }
        dump(dev, "l_out", il, y);
        if (!tap_slot_.empty() && tap_slot_[(size_t) il] >= 0) {
            // AI_TAPS: this layer's output into its row band of the persistent tap buffer (all rows of the round)
            if (y.ne[1] > taps_cap_) {
                throw std::runtime_error("AI_TAPS: round of " + std::to_string(y.ne[1]) + " rows exceeds the tap buffer (" +
                                         std::to_string(taps_cap_) + "; raise AI_TAPS_CAP)");
            }
            const int64_t slot_i = tap_slot_[(size_t) il];
            Tensor dst = taps_.view(s.n_embd, y.ne[1], 1, 1, taps_.nb[1], taps_.nb[1] * (size_t) y.ne[1],
                                    taps_.nb[1] * (size_t) y.ne[1], (size_t) slot_i * (size_t) s.n_embd * 4);
            op::copy_into(dev, y, dst, "layer->taps");
            taps_rows_ = y.ne[1];
        }

        if (!scoped) {
            cur = y;
            continue;
        }
        which ^= 1;
        const Tensor & c = carry[which];
        Tensor dst = c.view(E, rows, 1, 1, c.nb[1], c.nb[1] * (size_t) rows, c.nb[1] * (size_t) rows, 0);
        op::copy_into(dev, y, dst, "layer->carry");
        dev.ws_reset(mark);
        // AI_CHUNK_SCOPE: the q8_1 staging registry is keyed on the f32 activation's address (and geometry). A rollback
        // hands this layer's addresses to the next one, so an entry left here could match a different tensor there and
        // the consuming GEMV would read a stale q8 copy (short scoped chunks take the GEMV path, which does the lookup).
        // Only the recycled range is dropped: entries keyed on persistent per-layer buffers (norm_out_, the next
        // layer's input copy) stay, as in an unscoped pass; a blanket clear would make those consumers quantise for
        // themselves, which is not bit-identical.
        drop_staged(mark);
        cur = dst;
    }
    if (taps_.valid() && taps_rows_ > 0) {   // AI_TAPS + AI_DUMP_DIR: the tap buffer, for tools that check it
        dump(dev, "taps", 0, taps_.view(taps_.ne[0], taps_rows_, 1, 1, taps_.nb[1], taps_.nb[1] * (size_t) taps_rows_,
                                         taps_.nb[1] * (size_t) taps_rows_, 0));
    }
    return cur;   // final residual [E, rows]; lives until the workspace is reset
}

void Qwen35Hybrid::fold_masked(Device & dev, const Inputs & in, const GdnRecords & rec, const Tensor & colmask,
                               const Tensor & m) {
    const int slot = in.first_region;
    const int B    = (int) in.B;
    for (int il = 0; il < spec_.n_layer; ++il) {
        const int g = gdn_layer_[(size_t) il];
        if (g < 0) {
            continue;
        }
        op::gdn_fold_masked(dev, gdn_[(size_t) il], gdn_p_, { slots_.conv(g, slot, B), slots_.ssm(g, slot, B) },
                            rec.layers.at((size_t) g), colmask, m);
    }
}

void Qwen35Hybrid::fill_head_inputs(int seq, int64_t L, int64_t pos0, int64_t n_kv, int64_t row0, int64_t T,
                                    int32_t * pos4, int64_t * cells, uint16_t * mask) {
    kv_.reserve(seq, std::max(n_kv, pos0 + L));
    for (int64_t j = 0; j < L; ++j) {
        const int64_t i  = row0 + j;
        int32_t       rp[4];
        pos_at(seq, pos0 + j, rp);   // == (cell, cell, cell, 0) for text
        pos4[i]       = rp[0];
        pos4[T + i]   = rp[1];
        pos4[2*T + i] = rp[2];
        pos4[3*T + i] = rp[3];
        cells[i]      = kv_.cell(seq, pos0 + j);
        if (!mask) {
            continue;   // AI_HEAD_MASK_DEV: the round writes the mask on device (kern::head_causal_mask)
        }
        uint16_t * row = mask + i * n_kv;
        const int64_t open = std::min<int64_t>(pos0 + j + 1, n_kv);
        std::fill(row, row + n_kv, (uint16_t) 0xFC00);
        if (open > 1) {
            std::fill(row + 1, row + open, (uint16_t) 0x0000);   // cells 1..p
        }
    }
}

Qwen35Hybrid::MtpOut Qwen35Hybrid::mtp(Device & dev, const Tensor & ids, const Tensor & H, const Inputs & head_in,
                                       HeadRows rows, const Tensor & m) {
    if (!mtp_.valid) {
        throw std::runtime_error("mtp: model has no next-n block");
    }
    const float eps = spec_.rms_eps;
    const op::AttnInputs ain = attn_inputs(head_in);

    Tensor x = op::mtp_input(dev, tok_embd_, ids, H, mtp_.enorm, mtp_.hnorm, mtp_.eh_proj, eps);
    Tensor h = op::rmsnorm(dev, x, mtp_.attn_norm, eps);
    Tensor a = op::attn_block(dev, h, mtp_.attn, attn_p_, { kv_.k(mtp_.kv), kv_.v(mtp_.kv) }, ain);
    Tensor r = op::add(dev, a, x);
    Tensor p = op::rmsnorm(dev, r, mtp_.post_norm, eps);
    Tensor y = ffn_residual(dev, spec_.n_layer, mtp_.moe, p, r, -1);
    MtpOut out;
    const int64_t T = head_in.T, B = head_in.B, L = T / B;
    switch (rows) {
        case HeadRows::None:
            out.hidden = y;
            return out;
        case HeadRows::All:
            break;
        case HeadRows::Last:
            if (B != 1) {
                throw std::runtime_error("mtp: HeadRows::Last is single-stream");
            }
            y = y.view(y.ne[0], 1, 1, 1, y.nb[1], y.nb[1], y.nb[1], y.nb[1] * (size_t) (T - 1));
            break;
        case HeadRows::Accepted:
            y = op::select_rows(dev, y, L, B, m);
            break;
    }
    out.hidden = y;
    // Draft-head shortlist: the first rows of Qwen's merge-ordered vocab.
    // Only proposals are limited; the verify always uses the full head. AI_MTP_HEAD_ROWS=0 = full.
    static const int64_t hrows = getenv("AI_MTP_HEAD_ROWS") ? atoll(getenv("AI_MTP_HEAD_ROWS")) : 131072;
    Tensor ow = draft_head_.valid() ? draft_head_ : output_;
    if (!draft_head_.valid() && hrows > 0 && hrows < ow.ne[1]) {
        ow = ow.view(ow.ne[0], hrows, 1, 1, ow.nb[1], ow.nb[1] * (size_t) hrows, ow.nb[1] * (size_t) hrows, 0);
    }
    out.logits = op::head(dev, y, mtp_.shared_norm, ow, eps);
    out.token  = op::argmax(dev, out.logits, (draft_head_.valid() && draft_map_.valid()) ? &draft_map_ : nullptr);
    static const bool k_gate = getenv("AI_MTP_GATE") != nullptr && atof(getenv("AI_MTP_GATE")) > 0.0;
    if (k_gate) {
        out.conf = op::max_softmax(dev, out.logits);
    }
    return out;
}

Tensor Qwen35Hybrid::logits(Device & dev, const Tensor & hidden) {
    return op::head(dev, hidden, out_norm_, output_, spec_.rms_eps);
}

}  // namespace ai
