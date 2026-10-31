#include "alpha_train_optim.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <vector>

struct alpha_train_optim {
    alpha_optim_config config;
    std::vector<alpha_optim_param_group> groups;
};

alpha_train_optim * alpha_optim_init(alpha_optim_config config) {
    auto * optim = new alpha_train_optim();
    optim->config = config;

    // Set defaults
    if (optim->config.lr <= 0)    optim->config.lr = 2e-5f;
    if (optim->config.beta1 <= 0) optim->config.beta1 = 0.9f;
    if (optim->config.beta2 <= 0) optim->config.beta2 = 0.999f;
    if (optim->config.eps <= 0)   optim->config.eps = 1e-8f;
    if (optim->config.warmup_steps <= 0) optim->config.warmup_steps = 10;

    return optim;
}

void alpha_optim_add_param_group(
    alpha_train_optim * optim,
    float * params,
    float * grads,
    int n_params) {

    if (!optim || !params || !grads || n_params <= 0) return;

    alpha_optim_param_group group = {};
    group.params   = params;
    group.grads    = grads;
    group.n_params = n_params;

    // Allocate Adam moments (CPU, could be pinned memory for DMA)
    group.m = (float *)calloc(n_params, sizeof(float));
    group.v = (float *)calloc(n_params, sizeof(float));

    optim->groups.push_back(group);
}

float alpha_optim_get_lr(const alpha_train_optim * optim, int step) {
    if (!optim) return 0;
    float lr = optim->config.lr;
    int warmup = optim->config.warmup_steps;
    if (step < warmup && warmup > 0) {
        lr *= (float)(step + 1) / (float)warmup;
    }
    return lr;
}

void alpha_optim_step(alpha_train_optim * optim, int step) {
    if (!optim) return;

    float lr    = alpha_optim_get_lr(optim, step);
    float beta1 = optim->config.beta1;
    float beta2 = optim->config.beta2;
    float eps   = optim->config.eps;
    float wd    = optim->config.weight_decay;

    // Bias correction
    int t = step + 1;
    float bc1 = 1.0f - powf(beta1, (float)t);
    float bc2 = 1.0f - powf(beta2, (float)t);

    for (auto & group : optim->groups) {
        for (int i = 0; i < group.n_params; i++) {
            float g = group.grads[i];

            // Weight decay (decoupled, AdamW style)
            if (wd > 0) {
                group.params[i] -= lr * wd * group.params[i];
            }

            // Adam moments
            group.m[i] = beta1 * group.m[i] + (1.0f - beta1) * g;
            group.v[i] = beta2 * group.v[i] + (1.0f - beta2) * g * g;

            // Bias-corrected estimates
            float m_hat = group.m[i] / bc1;
            float v_hat = group.v[i] / bc2;

            // Parameter update
            group.params[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
        }
    }
}

void alpha_optim_zero_grad(alpha_train_optim * optim) {
    if (!optim) return;
    for (auto & group : optim->groups) {
        memset(group.grads, 0, group.n_params * sizeof(float));
    }
}

void alpha_optim_free(alpha_train_optim * optim) {
    if (!optim) return;
    for (auto & group : optim->groups) {
        free(group.m);
        free(group.v);
    }
    delete optim;
}
