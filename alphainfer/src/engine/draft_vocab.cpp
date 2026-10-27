#include "engine/draft_vocab.h"

#include "engine/tokenizer.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ai {

const char * draft_script_name(int s) {
    static const char * n[DS_COUNT] = { "common", "latin", "cjk", "kana", "hangul", "cyrillic", "arabic",
                                        "greek", "hebrew", "thai", "devanagari", "other", "mixed" };
    return s >= 0 && s < DS_COUNT ? n[s] : "?";
}

// script of one code point; DS_COMMON for non-letters
static int cp_script(uint32_t c) {
    if (c < 0x80) {
        return ((c | 0x20) >= 'a' && (c | 0x20) <= 'z') ? DS_LATIN : DS_COMMON;
    }
    if ((c >= 0xC0 && c <= 0x24F && c != 0xD7 && c != 0xF7) || (c >= 0x1E00 && c <= 0x1EFF) ||
        (c >= 0x2C60 && c <= 0x2C7F) || (c >= 0xA720 && c <= 0xA7FF)) {
        return DS_LATIN;
    }
    if ((c >= 0x370 && c <= 0x3FF) || (c >= 0x1F00 && c <= 0x1FFF)) return DS_GREEK;
    if ((c >= 0x400 && c <= 0x52F) || (c >= 0x1C80 && c <= 0x1C8F) || (c >= 0x2DE0 && c <= 0x2DFF) ||
        (c >= 0xA640 && c <= 0xA69F)) {
        return DS_CYRILLIC;
    }
    if (c >= 0x590 && c <= 0x5FF) return DS_HEBREW;
    if ((c >= 0x600 && c <= 0x6FF) || (c >= 0x750 && c <= 0x77F) || (c >= 0x8A0 && c <= 0x8FF) ||
        (c >= 0xFB50 && c <= 0xFDFF) || (c >= 0xFE70 && c <= 0xFEFF)) {
        return DS_ARABIC;
    }
    if (c >= 0x900 && c <= 0x97F) return DS_DEVANAGARI;
    if (c >= 0xE00 && c <= 0xE7F) return DS_THAI;
    if ((c >= 0x1100 && c <= 0x11FF) || (c >= 0x3130 && c <= 0x318F) || (c >= 0xAC00 && c <= 0xD7AF)) return DS_HANGUL;
    if ((c >= 0x3040 && c <= 0x30FF) || (c >= 0x31F0 && c <= 0x31FF) || (c >= 0xFF66 && c <= 0xFF9F)) return DS_KANA;
    if ((c >= 0x3400 && c <= 0x4DBF) || (c >= 0x4E00 && c <= 0x9FFF) || (c >= 0xF900 && c <= 0xFAFF) ||
        (c >= 0x20000 && c <= 0x2FFFF)) {
        return DS_CJK;
    }
    // punctuation, symbols, CJK punctuation, full-width forms, emoji: no script
    if ((c >= 0x2000 && c <= 0x2BFF) || (c >= 0x3000 && c <= 0x303F) || (c >= 0xFF00 && c <= 0xFF65) ||
        (c >= 0x1F000 && c <= 0x1FAFF) || (c >= 0x80 && c <= 0xBF) || c == 0xD7 || c == 0xF7) {
        return DS_COMMON;
    }
    return DS_OTHER;
}

static int piece_script(const std::string & s) {
    int  found = DS_COMMON;
    size_t i   = 0;
    while (i < s.size()) {
        const unsigned char b = (unsigned char) s[i];
        uint32_t            c;
        int                 len;
        if (b < 0x80) { c = b; len = 1; }
        else if ((b & 0xE0) == 0xC0) { c = b & 0x1F; len = 2; }
        else if ((b & 0xF0) == 0xE0) { c = b & 0x0F; len = 3; }
        else if ((b & 0xF8) == 0xF0) { c = b & 0x07; len = 4; }
        else { return DS_COMMON; }   // a byte fragment: every script may need it
        if (i + (size_t) len > s.size()) {
            return DS_COMMON;   // truncated multi-byte sequence: a byte-fallback piece
        }
        for (int k = 1; k < len; ++k) {
            const unsigned char cb = (unsigned char) s[i + (size_t) k];
            if ((cb & 0xC0) != 0x80) {
                return DS_COMMON;
            }
            c = (c << 6) | (cb & 0x3F);
        }
        const int sc = cp_script(c);
        if (sc != DS_COMMON) {
            if (found == DS_COMMON) {
                found = sc;
            } else if (found != sc) {
                return DS_MIXED;
            }
        }
        i += (size_t) len;
    }
    return found;
}

std::vector<uint8_t> classify_vocab(const Tokenizer & tok, int32_t n_vocab) {
    std::vector<uint8_t> cls((size_t) n_vocab, DS_COMMON);
    for (int32_t id = 0; id < n_vocab; ++id) {
        cls[(size_t) id] = (uint8_t) piece_script(tok.piece(id));
    }
    return cls;
}

// GPT-2 byte-level BPE token text (tokenizer.ggml.tokens) -> the bytes it stands for: bytes 33-126, 161-172 and
// 174-255 are their own code points, the other 68 bytes are U+0100.. in byte order. Code points outside that map
// (none in a byte-level vocab) are kept as their UTF-8.
static std::string gpt2_token_bytes(const std::string & t) {
    static const std::vector<int> inv = [] {
        std::vector<int> v(256 + 68, -1);
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            const bool self = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            if (self) v[(size_t) b] = b; else v[(size_t) (256 + n++)] = b;
        }
        return v;
    }();
    std::string out;
    size_t i = 0;
    while (i < t.size()) {
        const unsigned char b = (unsigned char) t[i];
        uint32_t c = b;
        int len = 1;
        if ((b & 0xE0) == 0xC0) { c = b & 0x1F; len = 2; }
        else if ((b & 0xF0) == 0xE0) { c = b & 0x0F; len = 3; }
        else if ((b & 0xF8) == 0xF0) { c = b & 0x07; len = 4; }
        if (i + (size_t) len > t.size()) { out.append(t, i, std::string::npos); break; }
        for (int k = 1; k < len; ++k) c = (c << 6) | ((unsigned char) t[i + (size_t) k] & 0x3F);
        if (c < inv.size() && inv[c] >= 0) out += (char) inv[c];
        else out.append(t, i, (size_t) len);
        i += (size_t) len;
    }
    return out;
}

std::vector<uint8_t> classify_gguf_tokens(const std::vector<std::string> & tokens) {
    std::vector<uint8_t> cls(tokens.size(), DS_COMMON);
    for (size_t id = 0; id < tokens.size(); ++id) cls[id] = (uint8_t) piece_script(gpt2_token_bytes(tokens[id]));
    return cls;
}

double prompt_english_share(const Tokenizer & tok, const std::vector<int32_t> & prompt) {
    // share of the prompt's words that are common English function words (English prose typically scores 12-37%,
    // other European languages ~1-2%, mostly-code ~10%). DFlash2's drafter only beats the MTP head on English.
    static const char * const kWords[] = { "the", "of", "and", "to", "a", "in", "is", "that", "for", "it", "as", "with",
        "was", "on", "be", "by", "this", "are", "or", "at", "from", "an", "not", "have", "which", "you", "your", "we",
        "they", "he", "she", "his", "her", "its", "can", "will", "would", "there", "their", "what", "when", "how", "all",
        "if", "do", "does", "has", "but", "more", "one", "so", "been", "than", "about", "into", "only", "also", "these",
        "those", "our", "may", "should" };
    std::string text;
    for (int32_t id : prompt) text += tok.piece(id);
    long long words = 0, hits = 0;
    std::string w;
    auto flush = [&]() {
        if (w.empty()) return;
        words++;
        for (const char * k : kWords) if (w == k) { hits++; break; }
        w.clear();
    };
    for (unsigned char ch : text) {
        if (ch >= 'A' && ch <= 'Z') w += (char) (ch - 'A' + 'a');
        else if (ch >= 'a' && ch <= 'z') w += (char) ch;
        else if (ch >= 0x80) { w += '#'; }   // non-ASCII letter bytes: the word is not an English function word
        else flush();
    }
    flush();
    return words ? (double) hits / (double) words : 0.0;
}

bool prompt_is_code_edit(const Tokenizer & tok, const std::vector<int32_t> & prompt) {
    // AI_DF2_ROUTE_EDIT: the request hands over a file in a fenced code block and asks for it back edited (rename / fix / ...).
    // The answer is then mostly a verbatim copy of the block, where lookup on the MTP head beats lookup on DFlash2.
    // Copy tasks that answer in lists or prose (log triage, signature quotes, doc QA, JSON) stay on DFlash2, which wins those.
    // Read: the last user turn only (a system prompt cannot trigger it); its longest fenced block must be >= 600 chars;
    // the instruction is the text after the last block (or before the first, when nothing follows), so prose between
    // blocks and inside documents is not read.
    static const char * const kStems[] = { "renam", "refactor", "rewrit", "modif", "fix", "replac", "chang", "updat",
        "edit", "apply", "patch" };
    static const size_t min_block = getenv("AI_DF2_ROUTE_EDIT_MIN") ? (size_t) atol(getenv("AI_DF2_ROUTE_EDIT_MIN")) : 600;
    std::string text;
    for (int32_t id : prompt) text += tok.piece(id);
    size_t beg = text.rfind("<|im_start|>user");
    beg = beg == std::string::npos ? 0 : beg + 16;
    size_t end = text.find("<|im_end|>", beg);
    if (end == std::string::npos) end = text.size();
    std::vector<size_t> f;   // fence offsets in the turn
    for (size_t i = text.find("```", beg); i != std::string::npos && i < end; i = text.find("```", i + 3)) f.push_back(i);
    if (f.size() < 2) return false;
    size_t longest = 0;
    for (size_t k = 0; k + 1 < f.size(); k += 2) longest = std::max(longest, f[k + 1] - f[k]);
    if (longest < min_block) return false;
    const size_t last_close = f[(f.size() / 2) * 2 - 1] + 3;
    auto letters = [&](size_t a, size_t b) {
        size_t n = 0;
        for (size_t i = a; i < b; ++i) n += std::isalpha((unsigned char) text[i]) ? 1 : 0;
        return n;
    };
    size_t a = last_close, b = end;
    if (letters(a, b) < 10) { a = beg; b = f[0]; }
    std::string w;
    auto hit = [&]() {
        bool h = false;
        for (const char * k : kStems) h = h || w.compare(0, strlen(k), k) == 0;
        w.clear();
        return h;
    };
    for (size_t i = a; i < b; ++i) {
        const unsigned char ch = (unsigned char) text[i];
        if (ch < 0x80 && std::isalpha(ch)) w += (char) std::tolower(ch);
        else if (!w.empty() && hit()) return true;
    }
    return !w.empty() && hit();
}

bool prompt_is_latin(const std::vector<uint8_t> & cls, const std::vector<int32_t> & prompt) {
    long long latin = 0, letters = 0;   // the same rule as draft_rows_for: >= 90% of the letters are Latin
    for (int32_t id : prompt) {
        if (id < 0 || (size_t) id >= cls.size()) continue;
        const int s = cls[(size_t) id];
        if (s != DS_COMMON && s != DS_MIXED) {
            letters++;
            latin += s == DS_LATIN;
        }
    }
    return letters == 0 || latin * 10 >= letters * 9;
}

std::vector<int32_t> draft_rows_for(const std::vector<uint8_t> & cls, const std::vector<int32_t> & prompt,
                                    int64_t rows, std::string * why) {
    long long cnt[DS_COUNT] = { 0 }, letters = 0;
    for (int32_t id : prompt) {
        if (id < 0 || (size_t) id >= cls.size()) {
            continue;
        }
        const int s = cls[(size_t) id];
        if (s != DS_COMMON && s != DS_MIXED) {
            cnt[s]++;
            letters++;
        }
    }
    const int64_t n = (int64_t) cls.size();
    // AI_DRAFT_VOCAB=1 (default): Latin-dominant prompts get Latin-priority rows (common + Latin, ascending, until
    // full); =3 gives them the first rows instead. Non-Latin prompts are script-adaptive in both modes.
    static const int mode = getenv("AI_DRAFT_VOCAB") ? atoi(getenv("AI_DRAFT_VOCAB")) : 1;
    const bool latin = letters == 0 || cnt[DS_LATIN] * 10 >= letters * 9;
    if (rows >= n || (latin && mode == 3)) {
        if (why) *why = "latin-dominant (or no letters): default rows";
        return {};
    }
    bool keep[DS_COUNT] = { false };
    keep[DS_COMMON] = keep[DS_MIXED] = keep[DS_OTHER] = true;
    std::string desc = "scripts:";
    if (latin) {
        keep[DS_LATIN] = true;
        desc += " latin";
    }
    for (int s = 1; s < DS_COUNT; ++s) {
        // >= 5% of the prompt's letters, or an absolute floor (AI_DRAFT_SCRIPT_MIN, default 4 tokens): a code prompt with a
        // short Chinese instruction ("explain in Chinese") is < 5% CJK but answers in CJK, and drafting it from Latin
        // rows alone collapses MTP acceptance.
        static const long long min_tok = getenv("AI_DRAFT_SCRIPT_MIN") ? atoll(getenv("AI_DRAFT_SCRIPT_MIN")) : 4;
        if (s != DS_LATIN && (cnt[s] * 20 >= letters || (min_tok > 0 && cnt[s] >= min_tok))) {
            keep[s] = true;
            desc += std::string(" ") + draft_script_name(s);
        }
    }
    // CJK and kana travel together in Japanese
    if (keep[DS_KANA]) keep[DS_CJK] = true;
    std::vector<char> sel((size_t) n, 0);
    int64_t           taken = 0;
    for (int64_t id = 0; id < n && taken < rows; ++id) {
        if (keep[cls[(size_t) id]]) {
            sel[(size_t) id] = 1;
            ++taken;
        }
    }
    const int64_t script_rows = taken;
    for (int64_t id = 0; id < n && taken < rows; ++id) {   // fill with the lowest remaining ids
        if (!sel[(size_t) id]) {
            sel[(size_t) id] = 1;
            ++taken;
        }
    }
    std::vector<int32_t> out;
    out.reserve((size_t) rows);
    for (int64_t id = 0; id < n; ++id) {
        if (sel[(size_t) id]) {
            out.push_back((int32_t) id);
        }
    }
    if (why) {
        char b[160];
        snprintf(b, sizeof(b), " | %lld script/common rows + %lld fill rows", (long long) script_rows,
                 (long long) (taken - script_rows));
        *why = desc + b;
    }
    return out;
}

}  // namespace ai
