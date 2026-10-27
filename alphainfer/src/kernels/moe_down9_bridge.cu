// Bridge from AlphaInfer's MoE op to ggml's nine-way down kernel (ALPHA_MOE_DOWN9).
//
// The kernel uses one block per output row and one warp per expert path, with the codec chosen by
// warp index. The expert axis is intra-block, so folding the paths together costs no parallelism.
//
// It replaces, per layer: quantise(act) + MUL_MAT_ID(down) + quantise(shared act) +
// MUL_MAT(shared down) + our native merge, and the [n_embd, k, T] per-slot intermediate no longer
// exists, which saves its HBM round trip.
//
// Its public entry point takes ggml_tensors, and our producers are native kernels with no ggml nodes,
// so this builds the descriptors by hand from our own Tensors. Our DType values are ggml's (F32 = 0,
// I32 = 26), so the type field carries across unchanged.
#include "kernels/moe_down9.h"

#include "ggml-backend-impl.h"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/moe-down9.cuh"

namespace {

ggml_tensor as_ggml(const ai::Tensor & t) {
    ggml_tensor g{};
    g.type = (ggml_type) t.type;
    for (int i = 0; i < 4; ++i) {
        g.ne[i] = t.ne[i];
        g.nb[i] = t.nb[i];
    }
    g.data = const_cast<void *>((const void *) t.data);
    g.op   = GGML_OP_NONE;
    return g;
}

}   // namespace

namespace ai::kern {

bool moe_down9_width_ok(int64_t T) {
    return ggml_cuda_moe_down9_width_ok(T);
}

bool moe_down9(ggml_backend * backend, const Tensor & down_exps, const Tensor & act_routed, const Tensor & ids,
               const Tensor & weights, const Tensor & down_shexp, const Tensor & act_shexp, const Tensor & score,
               const Tensor & residual, const Tensor & dst) {
    if (backend == nullptr || backend->context == nullptr) {
        return false;
    }
    ggml_tensor t_down = as_ggml(down_exps);
    ggml_tensor t_act  = as_ggml(act_routed);
    ggml_tensor t_ids  = as_ggml(ids);
    ggml_tensor t_wts  = as_ggml(weights);
    ggml_tensor t_sdw  = as_ggml(down_shexp);
    ggml_tensor t_sact = as_ggml(act_shexp);
    ggml_tensor t_sc   = as_ggml(score);
    ggml_tensor t_res  = as_ggml(residual);
    ggml_tensor t_dst  = as_ggml(dst);

    const ggml_tensor * addends[1] = { &t_res };
    if (!ggml_cuda_moe_down9_supported(&t_down, &t_act, &t_ids, &t_wts, &t_sdw, &t_sact, &t_sc, addends, 1,
                                       &t_dst)) {
        // A decline here is fatal upstream (the graph was already built without the downs), so log the
        // geometry once. The clause most likely to fire is n_ff: the kernel requires the shared
        // expert's FF width to equal the routed one, because it
        // quantises both activations with the routed n_ff and gives both a single blocks-per-row.
        static bool logged = false;
        if (!logged) {
            logged = true;
            const long long nff = (long long) t_down.ne[0];
            fprintf(stderr, "[moe-down9] DECLINED -- geometry follows; compare against the contract\n");
            fprintf(stderr, "  down_exps  type=%d ne=[%lld,%lld,%lld]  (n_ff=%lld, n_embd=%lld)\n",
                    (int) t_down.type, (long long) t_down.ne[0], (long long) t_down.ne[1],
                    (long long) t_down.ne[2], nff, (long long) t_down.ne[1]);
            fprintf(stderr, "  act_routed type=%d ne=[%lld,%lld,%lld,%lld] contig=%d  (want ne0==n_ff)\n",
                    (int) t_act.type, (long long) t_act.ne[0], (long long) t_act.ne[1],
                    (long long) t_act.ne[2], (long long) t_act.ne[3], (int) ggml_is_contiguous(&t_act));
            fprintf(stderr, "  ids        type=%d ne=[%lld,%lld]   weights type=%d ne=[%lld,%lld,%lld]\n",
                    (int) t_ids.type, (long long) t_ids.ne[0], (long long) t_ids.ne[1],
                    (int) t_wts.type, (long long) t_wts.ne[0], (long long) t_wts.ne[1],
                    (long long) t_wts.ne[2]);
            fprintf(stderr, "  down_shexp type=%d ne=[%lld,%lld]  <-- ne0 MUST equal n_ff=%lld %s\n",
                    (int) t_sdw.type, (long long) t_sdw.ne[0], (long long) t_sdw.ne[1], nff,
                    t_sdw.ne[0] == nff ? "(ok)" : "(MISMATCH: shared FF width differs)");
            fprintf(stderr, "  act_shexp  type=%d ne=[%lld,%lld] contig=%d  score ne=[%lld,%lld] nb1=%lld\n",
                    (int) t_sact.type, (long long) t_sact.ne[0], (long long) t_sact.ne[1],
                    (int) ggml_is_contiguous(&t_sact), (long long) t_sc.ne[0], (long long) t_sc.ne[1],
                    (long long) t_sc.nb[1]);
            fprintf(stderr, "  residual   ne=[%lld,%lld]   dst ne=[%lld,%lld] contig=%d\n",
                    (long long) t_res.ne[0], (long long) t_res.ne[1], (long long) t_dst.ne[0],
                    (long long) t_dst.ne[1], (int) ggml_is_contiguous(&t_dst));
        }
        return false;
    }
    ggml_backend_cuda_context & ctx = *(ggml_backend_cuda_context *) backend->context;
    ggml_cuda_op_moe_down9(ctx, &t_down, &t_act, &t_ids, &t_wts, &t_sdw, &t_sact, &t_sc, addends, 1, &t_dst);
    static bool logged = false;
    if (!logged) {
        logged = true;
        fprintf(stderr, "[moe-down9] LAUNCHED (one block per output row, one warp per path)\n");
    }
    return true;
}

}   // namespace ai::kern
