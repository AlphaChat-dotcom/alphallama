#pragma once

#include <string>

// 0.8B error assistant, loaded on-demand, CPU-only.
// Generates helpful error explanations when users hit common problems.
// Falls back to static messages if oracle model is not found.

struct alpha_error_assistant {
    // Load oracle model on CPU. Returns true on success.
    // Searches: (1) ALPHALLAMA_ORACLE_PATH env, (2) same dir as binary,
    // (3) ~/.alphallama/models/oracle-0.8b.gguf
    bool load();

    // Generate an explanation for an error.
    // Returns AI-generated help text, or empty string if oracle unavailable.
    std::string explain(const std::string & error_msg,
                        const std::string & command_line = "",
                        const std::string & extra_context = "");

    // Release resources.
    void unload();

    // Check if oracle is available (file exists, not necessarily loaded).
    static bool available();

    // Get the oracle model path (for logging).
    static std::string find_model_path();
};

// Convenience: try to explain an error, fall back to static message.
// Usage: alpha_error_explain("CUDA OOM", "alphallama -m big.gguf -ngl 99", "GPU has 16GB")
// Returns the explanation (AI or static); always non-empty.
std::string alpha_error_explain(const std::string & error_msg,
                                const std::string & command_line = "",
                                const std::string & extra_context = "");

