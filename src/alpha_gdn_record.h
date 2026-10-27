#pragma once
// ReplaySSM: record/replay of recurrent state for the speculative reject path (ALPHA_MTP).
//
// On a reject the window rewinds and would otherwise re-run the accepted prefix through a full
// llama_decode that produces no new tokens. Attention KV for the accepted prefix is already correct
// after the verify, so seq_rm(pos0+n_acc+1, -1) handles it. The two recurrent states cannot be
// partially undone:
//     conv (r_l): tail_{K-1}( old_state || qkv_mixed[0:n] )
//     gdn  (s_l): the gated-delta recurrence over k/v/g/beta[0:n]
// Both depend on the pre-window state and the per-token inputs, so "state after m tokens" has to be
// re-derived. Three properties of the builder make that cheap:
//
// (1) No new kernel and no second implementation of the recurrence.
//     ggml_gated_delta_net(ctx, q, k, v, g, beta, state) only reads its inputs, so recording means
//     retaining tensors that already exist, and the fold re-invokes the same op on records sliced
//     to m. The per-token transition does not depend on n_tokens (the reduce is over S_v), so the
//     result is bit-exact by construction.
//
// (2) S_0 needs no storage. The only writer of the committed state is the explicit publish in the
//     arch builder; with that publish suppressed, ssm_states_all still holds S_0 when the fold runs.
//
// (3) The conv half is a view offset, not a recurrence.
//     conv_input       = concat(conv_states /* = H_0, K-1 cols */, qkv_mixed /* n cols */)
//     published state  = view(conv_input, K-1 cols @ offset ne[0]-(K-1))
//     "conv state after m tokens" is tail_{K-1}(H_0 || p_1..p_m), the same tensor at offset m:
//         fold_conv(m) = view_3d(conv_input, K-1, ch, n_seqs, nb1, nb2, m*esz)
//     Only conv_input has to survive the verify.
//
// Retention: records are marked ggml_set_output so ggml-alloc does not reuse their slots, and are
// consumed before any other graph runs on the main scheduler. The fold runs on sched_alt, which has
// its own compute buffer, so it does not clobber what it reads. This is not a state snapshot.
//
// Safety: with the publish suppressed, the committed state is stale until the fold runs. Therefore:
//   - arming is per-decode and cleared unconditionally when that decode ends;
//   - the fold refuses unless every recurrent layer that ran was recorded and had its publish
//     suppressed (`expected_layers` vs `recorded`), and reports the refusal;
//   - on refusal the caller falls back to the relay decode.
// A partial recording never publishes a partial state.
#include <cstdint>
#include <vector>

struct ggml_context;
struct ggml_tensor;

// Per recurrent layer, the tensors the fold needs. All of these are graph tensors owned by the
// verify graph; we only hold pointers plus the data/buffer they resolve to.
struct alpha_gdn_rec_layer {
    // ggml_gated_delta_net inputs, exactly as build_delta_net passed them.
    ggml_tensor * q = nullptr;
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
    ggml_tensor * g = nullptr;
    ggml_tensor * b = nullptr;
    ggml_tensor * dt = nullptr;  // fused-input form (ggml_gated_delta_net_ext): dt bias, A, eps, flags
    ggml_tensor * a  = nullptr;
    float         eps   = 0.0f;
    int32_t       flags = 0;
    ggml_tensor * s = nullptr;   // S_0: a view of ssm_states_all, still valid because the
                                 // publish that would have overwritten it was suppressed.
    // Destinations the suppressed publishes would have written.
    ggml_tensor * state_dst = nullptr;
    ggml_tensor * new_state = nullptr;   // the verify's OWN result -- echo mode copies this instead
                                         // of re-invoking the op, to separate plumbing from math
    ggml_tensor * conv_input = nullptr;
    ggml_tensor * conv_src   = nullptr;   // the K-1 column view the publish would have copied;
                                          // the fold rebuilds it at column offset m instead
    ggml_tensor * conv_dst   = nullptr;
    // A second conv-shaped history on the same recurrent row (qwen4exp's PLE layer keeps its dilated
    // conv history in the row's tail): same layout as conv_* -- tokens on ne[0], history = the last
    // ne[0] columns -- so the fold rebuilds it the same way, a copy at column offset m.
    ggml_tensor * conv2_input = nullptr;
    ggml_tensor * conv2_src   = nullptr;
    ggml_tensor * conv2_dst   = nullptr;
    int64_t n_tokens = 0;        // T covered by this verify
    bool    have_gdn  = false;
    bool    have_conv = false;
    bool    have_conv2 = false;
};

// Global record state for the in-flight graph build. Single-threaded graph building, matching the
// rest of the builder.
struct alpha_gdn_record_t {
    std::vector<alpha_gdn_rec_layer> layers;   // indexed by layer id
    int  n_gdn        = 0;   // layers whose gdn inputs were recorded
    int  n_conv       = 0;   // layers whose conv publish was suppressed and recorded
    int  n_state      = 0;   // layers whose state publish was suppressed and recorded
    int64_t n_tokens  = 0;
    bool armed        = false;   // record this graph build
    bool valid        = false;   // a complete recording is available to fold
    // Graph reuse support: the builder hooks only fire on a rebuild, so a reused graph records
    // nothing even though last pass's pointers remain exactly valid. See alpha_gdn_record_arm.
    bool    prev_complete = false;
    int64_t prev_n_tokens = -1;
    int     prev_expected = -1;
    int64_t reused        = 0;   // folds served by a retained (reused-graph) recording
};

// Is the feature compiled-in and enabled? (ALPHA_REPLAYSSM=1; default off.)
bool alpha_gdn_record_enabled();

// Arm capture for the next graph build only. Cleared by alpha_gdn_record_end().
void alpha_gdn_record_arm(int64_t n_tokens);
bool alpha_gdn_record_armed();

// Called by the arch builder at each publish site. Returns true when the caller should perform
// its normal ggml_build_forward_expand(ggml_cpy(...)) publish, false when record mode has taken
// responsibility for writing that destination via the fold.
bool alpha_gdn_record_state(int il, ggml_tensor * state_dst, ggml_tensor * new_state);
bool alpha_gdn_record_conv (int il, ggml_tensor * conv_input, ggml_tensor * conv_src,
                            ggml_tensor * conv_dst);
// the optional second conv-shaped history of a layer (see conv2_* above); same return contract
bool alpha_gdn_record_conv2(int il, ggml_tensor * conv_input, ggml_tensor * conv_src,
                            ggml_tensor * conv_dst);

// Called generically from llm_build_delta_net_base::build_delta_net for every recurrent layer.
void alpha_gdn_record_gdn(int il, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                          ggml_tensor * g, ggml_tensor * b, ggml_tensor * s);
// fused-input form: g/b are the raw alpha/beta, dt/a the per-head bias/decay weights (see ggml_gated_delta_net_ext)
void alpha_gdn_record_gdn_ext(int il, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                              ggml_tensor * g, ggml_tensor * b, ggml_tensor * s,
                              ggml_tensor * dt, ggml_tensor * a, float eps, int32_t flags);

// End of a decode. `n_recurrent_expected` is how many recurrent layers the model has; the
// recording is marked valid only if every one of them was fully captured.
void alpha_gdn_record_end(int n_recurrent_expected);

// Access for the fold (llama-context.cpp).
const alpha_gdn_record_t & alpha_gdn_record_get();
void alpha_gdn_record_invalidate();
