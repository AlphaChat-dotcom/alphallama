// ReplaySSM record side. See alpha_gdn_record.h for the design and why this needs no kernel change.
// Default off (ALPHA_REPLAYSSM=1 to enable); when off every hook is a no-op and the arch builders take
// their normal publish path, so the graph is unchanged.
#include "alpha_gdn_record.h"

#include "ggml.h"

#include <cstdio>
#include <cstdlib>

static alpha_gdn_record_t g_rec;

bool alpha_gdn_record_enabled() {
    static const bool on = []() {
        const char * e = getenv("ALPHA_REPLAYSSM");
        return e && atoi(e) != 0;
    }();
    return on;
}

static bool g_dbg = getenv("ALPHA_REPLAYSSM_DBG") != nullptr;

// ggml-alloc reuses a tensor's slot as soon as its last consumer has run. Marking it as an output
// pins it for the whole graph. A view does not own storage, so pin its source too - otherwise the
// parent can be recycled underneath a pinned view.
static void pin(ggml_tensor * t) {
    if (!t) return;
    ggml_set_output(t);
    // Pin the storage owner too: a view owns nothing, so pinning only the view lets ggml-alloc
    // recycle the parent underneath it. But pin the parent only when it is graph-allocated: at
    // build time a graph tensor still has data == NULL, whereas a tensor belonging to a persistent
    // buffer (the memory module's recurrent state) already has data. Permanently flagging one of
    // those as an output would follow it into every future graph.
    if (t->view_src && t->view_src->data == nullptr) ggml_set_output(t->view_src);
}

void alpha_gdn_record_arm(int64_t n_tokens) {
    if (!alpha_gdn_record_enabled()) return;
    // Do not clear `layers`. The builder hooks only run when the graph is rebuilt, and llama.cpp
    // reuses a cached graph whenever the next ubatch has the same shape (e.g. chained verify windows
    // with ALPHA_MTP_LOOP>1). A reused graph recomputes the same tensors at the same addresses, so
    // the previous pass's pointers are still right and their data has just been refreshed. Clearing
    // them would make the fold refuse; that is safe (the caller relays) but costs a full forward.
    g_rec.n_gdn = g_rec.n_conv = g_rec.n_state = 0;
    g_rec.n_tokens = n_tokens;
    g_rec.armed    = true;
    g_rec.valid    = false;
}

bool alpha_gdn_record_armed() {
    return g_rec.armed;
}

static alpha_gdn_rec_layer & slot(int il) {
    if ((int) g_rec.layers.size() <= il) {
        g_rec.layers.resize(il + 1);
    }
    return g_rec.layers[il];
}

void alpha_gdn_record_gdn(int il, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                          ggml_tensor * g, ggml_tensor * b, ggml_tensor * s) {
    if (!g_rec.armed || il < 0) return;
    auto & L = slot(il);
    L.q = q; L.k = k; L.v = v; L.g = g; L.b = b; L.s = s;
    L.dt = nullptr; L.a = nullptr; L.eps = 0.0f; L.flags = 0;
    L.n_tokens = q ? q->ne[2] : 0;
    L.have_gdn = q && k && v && g && b && s;
    if (!L.have_gdn) return;
    pin(q); pin(k); pin(v); pin(g); pin(b);
    // s comes from build_rs, which yields either a view of the committed state buffer (nothing for
    // ggml-alloc to recycle) or a graph-allocated gather. In the second case pinning only the view
    // would let ggml-alloc recycle the gather once the op consumed it, and the fold would read a
    // destroyed S_0; pin() also covers the parent when it is graph-allocated.
    pin(s);
    g_rec.n_gdn++;
}

void alpha_gdn_record_gdn_ext(int il, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                              ggml_tensor * g, ggml_tensor * b, ggml_tensor * s,
                              ggml_tensor * dt, ggml_tensor * a, float eps, int32_t flags) {
    alpha_gdn_record_gdn(il, q, k, v, g, b, s);
    if (!g_rec.armed || il < 0) return;
    auto & L = slot(il);
    if (!L.have_gdn) return;
    L.dt = dt; L.a = a; L.eps = eps; L.flags = flags;   // model weights: nothing for ggml-alloc to recycle
}

bool alpha_gdn_record_state(int il, ggml_tensor * state_dst, ggml_tensor * new_state) {
    if (!g_rec.armed || il < 0) return true;   // not recording -> caller publishes as usual
    auto & L = slot(il);
    L.state_dst = state_dst;
    L.new_state = new_state;
    pin(new_state);
    g_rec.n_state++;
    return false;                              // recording -> the fold owns this destination
}

bool alpha_gdn_record_conv(int il, ggml_tensor * conv_input, ggml_tensor * conv_src,
                           ggml_tensor * conv_dst) {
    if (!g_rec.armed || il < 0) return true;
    auto & L = slot(il);
    L.conv_input = conv_input;
    L.conv_src   = conv_src;
    L.conv_dst   = conv_dst;
    L.have_conv  = conv_input && conv_src && conv_dst;
    if (!L.have_conv) return true;             // could not record -> fall back to publishing
    pin(conv_input);
    g_rec.n_conv++;
    return false;
}

bool alpha_gdn_record_conv2(int il, ggml_tensor * conv_input, ggml_tensor * conv_src,
                            ggml_tensor * conv_dst) {
    if (!g_rec.armed || il < 0) return true;
    auto & L = slot(il);
    L.conv2_input = conv_input;
    L.conv2_src   = conv_src;
    L.conv2_dst   = conv_dst;
    L.have_conv2  = conv_input && conv_src && conv_dst;
    if (!L.have_conv2) return true;            // could not record -> the caller publishes
    pin(conv_input);
    return false;
}

// Caller contract: suppressing the publish leaves the committed recurrent state at S_0, the
// pre-window value. That is the state the relay path rolls back to before re-laying the accepted
// prefix, so "fold refused -> run the relay" is always correct and needs no snapshot. But it is
// correct only if the caller honours it on every armed decode, including one where the whole
// window was accepted and no relay would otherwise run. A refused fold that is not followed by a
// relay leaves the state a full window behind, silently.
void alpha_gdn_record_end(int n_recurrent_expected) {
    if (!g_rec.armed) return;
    g_rec.armed = false;
    // A recording is usable only if every recurrent layer was captured on all three axes. A
    // partial recording means some layer's publish was suppressed with no fold to replace it,
    // which would leave that layer's state stale and silently corrupt generation.
    const bool complete = n_recurrent_expected > 0 &&
                          g_rec.n_gdn   == n_recurrent_expected &&
                          g_rec.n_state == n_recurrent_expected &&
                          g_rec.n_conv  == n_recurrent_expected;
    if (complete) {
        g_rec.valid          = true;
        g_rec.prev_complete  = true;
        g_rec.prev_n_tokens  = g_rec.n_tokens;
        g_rec.prev_expected  = n_recurrent_expected;
    } else if (g_rec.n_gdn == 0 && g_rec.n_state == 0 && g_rec.n_conv == 0 &&
               g_rec.prev_complete && g_rec.prev_n_tokens == g_rec.n_tokens &&
               g_rec.prev_expected == n_recurrent_expected) {
        // Nothing recorded because the graph was reused, not because anything went wrong. Same
        // graph => same tensors => same addresses, and graph_compute just refreshed their contents,
        // so the retained pointers describe this pass exactly. Requiring all three counters to be
        // zero keeps this narrow: a genuinely partial build (some layers hooked, some not) still
        // refuses rather than folding from a mix of two passes.
        g_rec.valid = true;
        g_rec.reused++;
    } else {
        g_rec.valid         = false;
        g_rec.prev_complete = false;
    }
    // One-shot warning. A refused fold is safe (the caller relays), so output stays correct while
    // every window pays an extra full forward, and the detailed line below is behind
    // ALPHA_REPLAYSSM_DBG. g_rec.valid is the exact discriminator: `complete` and the graph-reuse
    // branch set it, and only a genuine refusal leaves it false, so this cannot fire on graph reuse.
    // It names the missing axis rather than an arch, so a recurrent model with an MTP head that
    // lacks the hooks reports itself on the first window.
    if (!g_rec.valid) {
        static bool warned = false;
        if (!warned) { warned = true;
            const bool hooks_missing = g_rec.n_gdn > 0 && (g_rec.n_state == 0 || g_rec.n_conv == 0);
            fprintf(stderr,
                "[replayssm] WARNING: fold REFUSED (gdn=%d state=%d conv=%d expected=%d). "
                "Falling back to the relay -- output stays correct, but every window pays an extra "
                "forward.%s Set ALPHA_REPLAYSSM_DBG=1 for per-window detail. (warned once)\n",
                g_rec.n_gdn, g_rec.n_state, g_rec.n_conv, n_recurrent_expected,
                hooks_missing
                    ? "  CAUSE: the gdn axis recorded but state/conv did not => this arch's model"
                      " file is missing its alpha_gdn_record_state/_conv call sites (they must wrap"
                      " the recurrent-state and conv-state publishes, as in models/qwen35.cpp)."
                    : "");
        }
    }
    if (!complete && g_dbg) {
        fprintf(stderr, "[replayssm] INCOMPLETE recording: gdn=%d state=%d conv=%d expected=%d%s\n",
                g_rec.n_gdn, g_rec.n_state, g_rec.n_conv, n_recurrent_expected,
                // The most likely cause: the record hook sits at the fused call site, but
                // llm_build_delta_net_base::build_delta_net only routes a multi-token batch there when
                // cparams.fused_gdn_ch is set, and the capability probe clears that flag whenever a
                // layer's device differs from its GDN tensor's, i.e. on any partially-offloaded model.
                // Look for "fused Gated Delta Net (chunked) not supported, set to disabled" at startup.
                (g_rec.n_gdn == 0 && g_rec.n_state > 0)
                    ? "  [gdn=0 with state>0 => the verify took the CHUNKED path;"
                      " check for 'fused Gated Delta Net (chunked) ... disabled' at startup"
                    : "");
    }
}

const alpha_gdn_record_t & alpha_gdn_record_get() {
    return g_rec;
}

void alpha_gdn_record_invalidate() {
    g_rec.valid = false;
    g_rec.armed = false;
}
