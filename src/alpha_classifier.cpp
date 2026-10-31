#include "alpha_classifier.h"
#include "alpha_moe_optimizer.h"  // for alpha_moe_detect_output_domain (keyword scanner)
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

struct alpha_classifier {
    llama_model * model;
    llama_context * ctx;
    const llama_vocab * vocab;
    bool model_loaded;
};

// Domain names - must match profile file names (*_dp.json)
static const char * DOMAIN_NAMES[] = {
    "chemical", "math", "latex", "scientific", "medical",
    "legal", "multilingual", "regex", "binary_hex", "phonetic",
    "musical", "assembly", "symbolic_logic", "code", "json_output",
    "sql", "markdown", "technical", "conversational", "factual",
    "instructional", "analytical", "narrative", "general", nullptr
};

alpha_classifier * alpha_classifier_init(const char * model_path) {
    auto * cls = new alpha_classifier();
    cls->model = nullptr;
    cls->ctx = nullptr;
    cls->vocab = nullptr;
    cls->model_loaded = false;

    if (!model_path || strlen(model_path) == 0) {
        fprintf(stderr, "[CLASSIFIER] keyword-only mode (no model)\n");
        return cls;
    }

    // Load small model for classification
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;  // CPU only - don't use GPU VRAM

    cls->model = llama_model_load_from_file(model_path, mparams);
    if (!cls->model) {
        fprintf(stderr, "[CLASSIFIER] WARNING: failed to load %s, using keyword-only\n", model_path);
        return cls;
    }

    auto cparams = llama_context_default_params();
    cparams.n_ctx = 256;      // small context for classification
    cparams.n_batch = 256;
    cparams.flash_attn = true;

    cls->ctx = llama_init_from_model(cls->model, cparams);
    if (!cls->ctx) {
        fprintf(stderr, "[CLASSIFIER] WARNING: failed to create context, using keyword-only\n");
        llama_model_free(cls->model);
        cls->model = nullptr;
        return cls;
    }

    cls->vocab = llama_model_get_vocab(cls->model);
    cls->model_loaded = true;
    fprintf(stderr, "[CLASSIFIER] loaded %s (CPU, 256 ctx)\n", model_path);
    return cls;
}

// Build classification prompt
static std::string build_classify_prompt(const char * user_prompt, int max_len) {
    std::string prompt =
        "<|im_start|>system\n"
        "You are a topic classifier. Given a user prompt, output ONLY ONE word "
        "from this list: chemical, math, latex, scientific, medical, legal, "
        "multilingual, regex, binary_hex, phonetic, musical, assembly, "
        "symbolic_logic, code, json_output, sql, markdown, technical, "
        "conversational, factual, instructional, analytical, narrative, general.\n"
        "Output only the category word, nothing else.<|im_end|>\n"
        "<|im_start|>user\n";

    // Truncate user prompt if too long (classification doesn't need full text)
    int len = (int)strlen(user_prompt);
    if (len > max_len) len = max_len;
    prompt.append(user_prompt, len);

    prompt += "<|im_end|>\n<|im_start|>assistant\n";
    return prompt;
}

// Match generated text against known domain names
static const char * match_domain(const char * text, int text_len) {
    // Try exact match first
    for (int i = 0; DOMAIN_NAMES[i]; i++) {
        if (strncmp(text, DOMAIN_NAMES[i], strlen(DOMAIN_NAMES[i])) == 0) {
            return DOMAIN_NAMES[i];
        }
    }
    // Try case-insensitive substring match
    std::string lower(text, text_len);
    for (auto & c : lower) c = tolower(c);
    for (int i = 0; DOMAIN_NAMES[i]; i++) {
        if (lower.find(DOMAIN_NAMES[i]) != std::string::npos) {
            return DOMAIN_NAMES[i];
        }
    }
    return "general";
}

const char * alpha_classifier_detect(
    alpha_classifier * cls,
    const char * prompt,
    int prompt_len)
{
    if (!cls || !prompt || prompt_len <= 0) return "general";

    // Step 1: Fast keyword scan (0ms)
    const char * keyword_result = alpha_moe_detect_output_domain(prompt, prompt_len);
    if (keyword_result) {
        return keyword_result;  // keyword matched - no need for model
    }

    // Step 2: Model fallback (if loaded)
    if (!cls->model_loaded) {
        return "general";  // no model, no keyword match
    }

    // Build classification prompt (truncate user prompt to 150 chars)
    std::string cls_prompt = build_classify_prompt(prompt, 150);

    // Tokenize
    int n_toks = -llama_tokenize(cls->vocab, cls_prompt.c_str(),
        cls_prompt.size(), nullptr, 0, true, true);
    if (n_toks <= 0 || n_toks > 200) return "general";

    std::vector<llama_token> tokens(n_toks);
    llama_tokenize(cls->vocab, cls_prompt.c_str(), cls_prompt.size(),
        tokens.data(), tokens.size(), true, true);

    // Clear KV cache
    llama_memory_seq_rm(llama_get_memory(cls->ctx), 0, -1, -1);

    // Decode prompt
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(cls->ctx, batch) != 0) {
        return "general";
    }

    // Sample up to 3 tokens (domain name is usually 1-2 tokens)
    char result[64] = {};
    int result_len = 0;

    for (int i = 0; i < 3; i++) {
        // Greedy sample from logits
        const float * logits = llama_get_logits(cls->ctx);
        if (!logits) break;

        int n_vocab = llama_vocab_n_tokens(cls->vocab);
        int best_id = 0;
        float best_logit = logits[0];
        for (int v = 1; v < n_vocab; v++) {
            if (logits[v] > best_logit) {
                best_logit = logits[v];
                best_id = v;
            }
        }

        // Convert token to text
        char tbuf[32] = {};
        int tlen = llama_token_to_piece(cls->vocab, best_id, tbuf, sizeof(tbuf), 0, true);
        if (tlen <= 0) break;

        // Append to result
        if (result_len + tlen < (int)sizeof(result) - 1) {
            memcpy(result + result_len, tbuf, tlen);
            result_len += tlen;
        }

        // Check if we have a match already
        const char * domain = match_domain(result, result_len);
        if (strcmp(domain, "general") != 0) {
            return domain;
        }

        // Decode this token for next iteration
        if (i < 2) {
            llama_batch next = llama_batch_get_one(&best_id, 1);
            if (llama_decode(cls->ctx, next) != 0) break;
        }
    }

    // Final match attempt
    return match_domain(result, result_len);
}

void alpha_classifier_free(alpha_classifier * cls) {
    if (!cls) return;
    if (cls->ctx) llama_free(cls->ctx);
    if (cls->model) llama_model_free(cls->model);
    delete cls;
}
