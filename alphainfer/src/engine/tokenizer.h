// Tokenizer shim: llama.cpp's vocabulary code, loaded vocab-only and reused as-is.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct llama_model;
struct llama_vocab;

namespace ai {

class Tokenizer {
public:
    ~Tokenizer();
    bool load(const std::string & gguf_path);

    std::vector<int32_t> encode(const std::string & text, bool parse_special = true) const;
    std::string          piece(int32_t token) const;
    bool                 is_eog(int32_t token) const;
    int32_t              n_vocab() const;

    // ChatML turn for the Qwen3.5 family; thinking disabled by an empty think block.
    static std::string chat_prompt(const std::string & user, bool think);

private:
    llama_model *       model_ = nullptr;
    const llama_vocab * vocab_ = nullptr;
};

}  // namespace ai
