// Fallback tokenizer loader for HuggingFace tokenizer.json (ALPHA_SPLIT_GGUF).
// Used when split GGUF shard 1 is too small to hold the tokenizer KV arrays.
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

struct llama_tokenizer_json_data {
    std::vector<std::string> tokens;     // indexed by token id
    std::vector<float>       scores;     // indexed by token id
    std::vector<int>         types;      // LLAMA_TOKEN_TYPE_* values
    std::vector<std::string> merges;     // BPE merge strings ("token1 token2")
    std::string              model;      // "gpt2", "llama", etc.
    std::string              pre;        // pre-tokenizer type

    // Special token IDs (-1 if not found)
    int bos_id = -1;
    int eos_id = -1;
    int unk_id = -1;
    int pad_id = -1;
};

// Load tokenizer data from tokenizer.json in the given directory.
// Returns true on success. On failure, logs a warning and returns false.
bool llama_load_tokenizer_json(const std::string & model_dir, llama_tokenizer_json_data & out);
