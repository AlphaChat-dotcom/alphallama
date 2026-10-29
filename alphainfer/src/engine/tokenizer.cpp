// SPDX-License-Identifier: Apache-2.0
#include "engine/tokenizer.h"

#include "llama.h"

#include <stdexcept>

namespace ai {

Tokenizer::~Tokenizer() {
    if (model_) {
        llama_model_free(model_);
    }
}

bool Tokenizer::load(const std::string & gguf_path) {
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only   = true;
    mp.n_gpu_layers = 0;
    model_ = llama_model_load_from_file(gguf_path.c_str(), mp);
    if (!model_) {
        return false;
    }
    vocab_ = llama_model_get_vocab(model_);
    return vocab_ != nullptr;
}

std::vector<int32_t> Tokenizer::encode(const std::string & text, bool parse_special) const {
    std::vector<int32_t> out(text.size() + 16);
    int n = llama_tokenize(vocab_, text.c_str(), (int32_t) text.size(), out.data(), (int32_t) out.size(),
                           /*add_special=*/ false, parse_special);
    if (n < 0) {
        out.resize((size_t) -n);
        n = llama_tokenize(vocab_, text.c_str(), (int32_t) text.size(), out.data(), (int32_t) out.size(),
                           false, parse_special);
    }
    if (n < 0) {
        throw std::runtime_error("tokenize failed");
    }
    out.resize((size_t) n);
    return out;
}

std::string Tokenizer::piece(int32_t token) const {
    char buf[256];
    int n = llama_token_to_piece(vocab_, token, buf, sizeof(buf), 0, /*special=*/ false);
    return n > 0 ? std::string(buf, (size_t) n) : std::string();
}

int32_t Tokenizer::n_vocab() const {
    return llama_vocab_n_tokens(vocab_);
}

bool Tokenizer::is_eog(int32_t token) const {
    return llama_vocab_is_eog(vocab_, token);
}

std::string Tokenizer::chat_prompt(const std::string & user, bool think) {
    std::string s = "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n";
    if (!think) {
        s += "<think>\n\n</think>\n\n";
    }
    return s;
}

}  // namespace ai
