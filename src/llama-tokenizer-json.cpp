// ALPHA_SPLIT_GGUF: fallback tokenizer loader from HuggingFace tokenizer.json
// Used when split GGUF shard 1 is too small to hold tokenizer KV arrays.

#include "llama-tokenizer-json.h"
#include "llama-impl.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <string>

using json = nlohmann::json;

static std::string path_join(const std::string & dir, const std::string & file) {
    if (dir.empty()) return file;
    char last = dir.back();
    if (last == '/' || last == '\\') return dir + file;
    return dir + "/" + file;
}

static std::string path_parent(const std::string & path) {
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) return ".";
    return path.substr(0, pos);
}

bool llama_load_tokenizer_json(const std::string & model_path, llama_tokenizer_json_data & out) {
    // model_path is the GGUF file path; look for tokenizer.json in the same directory
    std::string dir = path_parent(model_path);
    std::string tokenizer_path = path_join(dir, "tokenizer.json");

    std::ifstream f(tokenizer_path);
    if (!f.is_open()) {
        LLAMA_LOG_WARN("%s: tokenizer.json not found at %s\n", __func__, tokenizer_path.c_str());
        return false;
    }

    LLAMA_LOG_INFO("%s: loading tokenizer fallback from %s\n", __func__, tokenizer_path.c_str());

    json data;
    try {
        data = json::parse(f);
    } catch (const json::parse_error & e) {
        LLAMA_LOG_ERROR("%s: failed to parse tokenizer.json: %s\n", __func__, e.what());
        return false;
    }

    // Extract model type
    if (data.contains("model") && data["model"].contains("type")) {
        std::string model_type = data["model"]["type"].get<std::string>();
        // Map HF model types to GGUF tokenizer model names
        if (model_type == "BPE") {
            out.model = "gpt2";
        } else if (model_type == "Unigram") {
            out.model = "llama";
        } else if (model_type == "WordPiece") {
            out.model = "bert";
        } else {
            out.model = "gpt2"; // default fallback
        }
    }

    // Extract vocab
    if (!data.contains("model") || !data["model"].contains("vocab")) {
        LLAMA_LOG_ERROR("%s: tokenizer.json missing model.vocab\n", __func__);
        return false;
    }

    const auto & vocab = data["model"]["vocab"];
    if (!vocab.is_object()) {
        LLAMA_LOG_ERROR("%s: model.vocab is not an object\n", __func__);
        return false;
    }

    // Find max token id to size the arrays
    int max_id = -1;
    for (auto it = vocab.begin(); it != vocab.end(); ++it) {
        int id = it.value().get<int>();
        if (id > max_id) max_id = id;
    }

    if (max_id < 0) {
        LLAMA_LOG_ERROR("%s: empty vocab in tokenizer.json\n", __func__);
        return false;
    }

    int n_tokens = max_id + 1;
    out.tokens.resize(n_tokens);
    out.scores.resize(n_tokens, 0.0f);
    out.types.resize(n_tokens, 1);  // 1 = LLAMA_TOKEN_TYPE_NORMAL

    for (auto it = vocab.begin(); it != vocab.end(); ++it) {
        int id = it.value().get<int>();
        if (id >= 0 && id < n_tokens) {
            out.tokens[id] = it.key();
        }
    }

    // Extract added_tokens for special token detection
    if (data.contains("added_tokens") && data["added_tokens"].is_array()) {
        for (const auto & at : data["added_tokens"]) {
            int id = at.value("id", -1);
            bool special = at.value("special", false);
            if (id >= 0 && id < n_tokens && special) {
                out.types[id] = 3;  // 3 = LLAMA_TOKEN_TYPE_CONTROL

                // Detect common special tokens by content
                std::string content = at.value("content", "");
                if (content == "<|endoftext|>" || content == "</s>" || content == "<|im_end|>") {
                    if (out.eos_id < 0) out.eos_id = id;
                }
                if (content == "<s>" || content == "<|begin_of_text|>" || content == "<|im_start|>") {
                    if (out.bos_id < 0) out.bos_id = id;
                }
                if (content == "<unk>") {
                    if (out.unk_id < 0) out.unk_id = id;
                }
                if (content == "<pad>" || content == "<|padding|>") {
                    if (out.pad_id < 0) out.pad_id = id;
                }
            }
        }
    }

    // Extract BPE merges
    if (data.contains("model") && data["model"].contains("merges") && data["model"]["merges"].is_array()) {
        for (const auto & m : data["model"]["merges"]) {
            out.merges.push_back(m.get<std::string>());
        }
    }

    // Try to detect pre-tokenizer type from tokenizer_config.json
    std::string config_path = path_join(dir, "tokenizer_config.json");
    std::ifstream cf(config_path);
    if (cf.is_open()) {
        try {
            json config = json::parse(cf);
            // Some models store the pre-tokenizer type here
            if (config.contains("model_type")) {
                std::string mt = config["model_type"].get<std::string>();
                // Map common model types to pre-tokenizer types
                if (mt.find("qwen") != std::string::npos) {
                    out.pre = "qwen2";
                } else if (mt.find("llama") != std::string::npos) {
                    out.pre = "llama3";
                } else if (mt.find("deepseek") != std::string::npos) {
                    out.pre = "deepseek3-llm";
                }
            }
        } catch (...) {
            // tokenizer_config.json is optional
        }
    }

    LLAMA_LOG_INFO("%s: loaded %d tokens, %zu merges, %d special tokens from tokenizer.json\n",
                   __func__, n_tokens, out.merges.size(),
                   (int)std::count(out.types.begin(), out.types.end(), 3));

    return true;
}
