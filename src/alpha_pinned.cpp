#include "alpha_pinned.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <vector>

// Try to find the CUDA host register function dynamically
// This avoids hard dependency on CUDA headers
typedef bool (*register_host_fn)(void *, size_t);
typedef void (*unregister_host_fn)(void *);

static register_host_fn   g_register_fn   = nullptr;
static unregister_host_fn g_unregister_fn = nullptr;
static bool               g_init_done     = false;

static std::vector<void *> g_registered_buffers;

static void init_pinned_fns() {
    if (g_init_done) return;
    g_init_done = true;

    // Check if CUDA backend is available by looking for a GPU device
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            // CUDA is available — the register functions are linked
            // We use the ggml-cuda.h functions directly
            // (they're in libggml-cuda.so which is already loaded)
            break;
        }
    }
}

int alpha_pinned_register_model(struct llama_model * model) {
    if (!model) return 0;

    init_pinned_fns();

    // Get model size info
    uint64_t model_size = llama_model_size(model);
    if (model_size == 0) return 0;

    // We can't easily access the model's internal CPU buffers from the public API.
    // The model's buffers are private (pimpl pattern).
    //
    // Alternative approach: use ggml_backend_cuda_register_host_buffer()
    // on the buffer data pointer, but we need the pointer from the model internals.
    //
    // For now, this is a stub that logs the intent. The actual registration
    // would require either:
    // 1. Adding a public API to llama.h to iterate model buffers
    // 2. Modifying llama-model.cpp to use pinned allocation at load time
    // 3. Using dlsym to find cudaHostRegister and calling it on mmap'd regions
    //
    // The most practical fix is option 2: modify make_cpu_buft_list() to
    // force CUDA_Host buffer type even for quantized tensors, by using
    // a custom buffer type that wraps cudaHostAlloc.

    fprintf(stderr, "alpha_pinned: model size %llu MB — pinned memory registration pending\n",
            (unsigned long long)(model_size / (1024 * 1024)));
    fprintf(stderr, "alpha_pinned: note: llama.cpp's CPU buffer doesn't expose data pointer for post-load registration\n");
    fprintf(stderr, "alpha_pinned: the effective fix is to modify llama-model.cpp make_cpu_buft_list()\n");

    return 0;
}

void alpha_pinned_unregister_model(struct llama_model * model) {
    (void)model;
    // Unregister any buffers we registered
    // Currently no-op since registration is pending
}
