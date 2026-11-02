// SPDX-License-Identifier: Apache-2.0
// Script-adaptive MTP draft vocabulary (AI_DRAFT_VOCAB).
//
// The draft head holds a fixed number of rows (131072). By default they are the first 131072 ids in merge
// order, which covers Latin and most CJK but leaves out large parts of other scripts (Cyrillic, Arabic, Hangul,
// kana): a large share of the tokens generated in those languages are ones the draft can never propose. A
// full-vocab draft head would fix acceptance there but slows English down.
//
// Here the same 131072 rows are chosen per request: for a prompt whose letters are mostly Latin, the default
// rows (nothing changes); otherwise every token of the prompt's scripts, every common/byte/control token, and
// low ids to fill the rest. Same draft cost, full coverage of the language. Drafts only propose: the verify
// keeps the full head, so generated text is unchanged; only acceptance moves.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ai {

class Tokenizer;

enum DraftScript : uint8_t {
    DS_COMMON = 0,   // no letters, byte fragments, control tokens: always kept
    DS_LATIN,
    DS_CJK,
    DS_KANA,
    DS_HANGUL,
    DS_CYRILLIC,
    DS_ARABIC,
    DS_GREEK,
    DS_HEBREW,
    DS_THAI,
    DS_DEVANAGARI,
    DS_OTHER,        // letters of a script not listed
    DS_MIXED,        // more than one script in one token
    DS_COUNT
};

const char * draft_script_name(int s);

// One class per token id, from the token's decoded bytes.
std::vector<uint8_t> classify_vocab(const Tokenizer & tok, int32_t n_vocab);
// The same classes from a GGUF's tokenizer.ggml.tokens (GPT-2 byte-level BPE text), without a llama vocab: the
// qwen4exp runner has no Tokenizer.
std::vector<uint8_t> classify_gguf_tokens(const std::vector<std::string> & tokens);

// For AI_DF2_ROUTE: the prompt is Latin-dominant (>= 90% of its letters), by the draft-vocab classes
bool prompt_is_latin(const std::vector<uint8_t> & cls, const std::vector<int32_t> & prompt);
// For AI_DF2_ROUTE: share of the prompt's words that are common English function words (0..1)
double prompt_english_share(const Tokenizer & tok, const std::vector<int32_t> & prompt);
// the same share from text (the qwen4exp runner has no Tokenizer: it decodes the GGUF pieces, gguf_token_texts)
double text_english_share(const std::string & text);
// tokenizer.ggml.tokens (GPT-2 byte-encoded) -> the raw bytes of every piece
std::vector<std::string> gguf_token_texts(const std::vector<std::string> & tokens);
// [AI_DF2_ROUTE_EDIT] the last user turn carries a fenced code block (>= 600 chars) and asks for it edited (rename/fix/...)
bool prompt_is_code_edit(const Tokenizer & tok, const std::vector<int32_t> & prompt);
// The draft rows for a prompt, ascending ids, exactly `rows` of them; empty when the default rows should stay
// (a Latin-dominant prompt). `why` receives a one-line description.
std::vector<int32_t> draft_rows_for(const std::vector<uint8_t> & cls, const std::vector<int32_t> & prompt,
                                    int64_t rows, std::string * why);

}  // namespace ai
