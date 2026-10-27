#pragma once

// Combined MeZO + sparse backprop training
//
// Single-phase architecture extracts two gradient signals per step:
//   1. MeZO (zeroth-order): noisy gradient for all LoRA params
//   2. Exact backprop: precise gradient for the most important params (top 15% by default)
//
// A block-Fisher mask identifies the important params from a few batches.
// A convergence monitor stops training automatically, with quality gates.
//
// Architecture:
//   Step 0:  Block-Fisher mask (5 batches)
//   Step 1+: Combined step: 2 forwards (MeZO) + sparse backprop on the masked params
//            Activations reused from +eps forward (O(eps) error, negligible)

#include "llama.h"
#include "ggml.h"
#include "alpha_train_novel.h"

#include <vector>

// Convergence Monitor

struct alpha_convergence_monitor {
    // Windowed loss statistics
    float loss_window[512];
    int   window_pos;
    int   window_count;
    int   window_size;         // W (default 200)

    // Phase gate values
    float eval_loss_baseline;  // M0 (pretrained)
    float eval_loss_best;      // best ever seen
    int   best_step;

    // Convergence detection
    float loss_velocity;       // slope of loss over window (should be negative)
    float loss_variance;       // noise floor
    int   plateau_count;       // consecutive flat windows
    bool  converged;

    // Diminishing returns
    float decay_tau;           // estimated time constant
    float remaining_improvement;

    // Adaptive tuning state
    float recent_avg_loss;     // EMA of recent losses (for spike detection)
    float last_eval_loss;      // eval loss at last check
    int   spike_count;         // number of spikes detected
    int   overfit_count;       // consecutive eval-checks where train improves but eval doesn't
    int   stall_count;         // consecutive checks with no train improvement
    int   n_adjustments;       // total adjustments made
};

// Fisher Mask

struct alpha_fisher_mask {
    // Per-LoRA-layer importance scores and mask bits
    struct layer_mask {
        std::vector<float> importance;  // per-element importance (|delta_loss / eps|)
        std::vector<bool>  mask;        // true = important (in top sparsity_ratio)
        int n_masked;                   // count of masked elements
    };

    std::vector<layer_mask> layers;     // one per LoRA layer
    float sparsity_ratio;               // fraction of params masked (default 0.15)
    int   n_batches;                    // batches used for stability (default 5)
    int   total_masked;
    int   total_params;
};

// Combined Trainer Config

struct alpha_combined_train_config {
    // LoRA
    int    rank;              // LoRA rank (default: 32)

    // Optimization
    float  lr;                // learning rate for exact backprop channel (default: 2e-5)
    float  lr_mezo;           // learning rate for MeZO channel (default: 1e-4)
    float  beta1, beta2, eps; // Adam
    float  weight_decay;
    int    warmup_steps;

    // MeZO
    float  mezo_eps;          // perturbation magnitude (default: 1e-3)

    // Fisher mask
    float  sparsity_ratio;    // fraction of params for exact backprop (default: 0.15)
    int    fisher_batches;    // batches for mask stability (default: 5)

    // Convergence
    int    max_steps;         // hard stop (default: 10000)
    int    convergence_window;// window size for convergence test (default: 200)
    int    plateau_patience;  // consecutive flat windows before stop (default: 3)
    float  gate_threshold;    // eval improvement required (default: 0.95 = 5% better)

    // Elastic anchoring (prevents basin escape during backprop)
    float  elastic_lambda;    // anchor strength (default: 0.1)

    // Logging
    int    log_every;
    int    eval_every;        // evaluate on held-out data every N steps
    int    save_every;
    bool   json_progress;     // output JSON lines for programmatic progress tracking
};

struct alpha_combined_trainer;

// Default config
static inline alpha_combined_train_config alpha_combined_default_config() {
    alpha_combined_train_config cfg = {};
    cfg.rank = 32;
    cfg.lr = 1e-5f;            // backprop channel - lower to prevent spikes with 15% mask
    cfg.lr_mezo = 5e-5f;       // MeZO channel - moderate for broad exploration
    cfg.beta1 = 0.9f;
    cfg.beta2 = 0.999f;
    cfg.eps = 1e-8f;
    cfg.weight_decay = 0.01f;     // regularization to prevent overfitting
    cfg.warmup_steps = 300;       // Adam cold-start on 15% backprop needs longer warmup
    cfg.mezo_eps = 1e-3f;
    cfg.sparsity_ratio = 0.15f;   // 15% mask from step 0 - wider coverage, single phase
    cfg.fisher_batches = 5;
    cfg.max_steps = 5000;          // converges faster with wider mask
    cfg.convergence_window = 500;  // enough room for auto-tune to correct and re-converge
    cfg.plateau_patience = 3;
    cfg.gate_threshold = 0.95f;
    cfg.elastic_lambda = 0.2f;    // stronger anchoring for wider 15% mask
    cfg.log_every = 10;
    cfg.eval_every = 100;
    cfg.save_every = 500;
    return cfg;
}

// Public API

// Initialize combined trainer (creates LoRA adapter + Fisher mask)
alpha_combined_trainer * alpha_combined_trainer_init(
    struct llama_model   * model,
    struct llama_context * ctx,
    alpha_combined_train_config config);

// Compute Block-Fisher mask (call once before training loop)
// Uses eval_tokens for importance estimation (5 batches x 1 forward)
void alpha_combined_compute_mask(
    alpha_combined_trainer * trainer,
    const llama_token * eval_tokens,
    int n_eval_tokens,
    int seq_len);

// Run one combined step: MeZO forward pair + sparse backprop on masked params
// Returns loss
float alpha_combined_step(
    alpha_combined_trainer * trainer,
    const llama_token * tokens,
    int n_tokens);

// Check convergence (call after each step or batch of steps)
// Returns true if training should stop
bool alpha_combined_check_convergence(
    alpha_combined_trainer * trainer,
    float loss);

// Quality gate: compare eval loss against baseline
// Returns true if model improved (gate passes)
bool alpha_combined_quality_gate(
    alpha_combined_trainer * trainer,
    const llama_token * eval_tokens,
    int n_eval_tokens);

// Save LoRA adapter to GGUF
void alpha_combined_save(const alpha_combined_trainer * trainer, const char * path);

// Print stats
void alpha_combined_print_stats(const alpha_combined_trainer * trainer);

// Free
void alpha_combined_trainer_free(alpha_combined_trainer * trainer);

// Full training loop (convenience wrapper)
LLAMA_API int alpha_combined_train(
    struct llama_model * model,
    struct llama_context * ctx,
    const char * data_file,
    const char * output_file,
    alpha_combined_train_config config,
    int epochs,
    int seq_len);

// Get eval callback for activation capture
LLAMA_API ggml_backend_sched_eval_callback alpha_combined_get_callback(void);
