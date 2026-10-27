#pragma once

// DeltaNet Digest Pipeline: 5-layer lossless extraction for prescan
//
// Converts 16K chunks into ~50-token digests that keep the key content, so prescan
// processes far fewer tokens than the full text.
//
// Layers:
//   1. BM25 IDF keywords (top-15 unique terms by distinctiveness)
//   2. NER entities (FILE, FUNCTION, CLASS, PERSON, DATE, etc.)
//   3. First-sentence extraction (topic sentences per paragraph), phase 2 only
//   4. Code skeleton (imports + function signatures), code files only
//   5. Relation pairs (cross-type entity co-occurrence)

#include "server-bm25.h"
#include "server-entity.h"
#include <string>
#include <vector>
#include <unordered_set>

// Layer 3: first-sentence extraction.
// Extract the first sentence of each paragraph (topic sentences), ~60 tokens for 6 paragraphs.
// Phase 2 only, for queries the 50-token digest cannot answer (multi-hop, subtle).
static std::string extract_first_sentences(const std::string & text, int max_sentences = 6) {
    std::string result;
    int count = 0;
    size_t pos = 0;

    // First paragraph starts at pos 0
    while (pos < text.size() && count < max_sentences) {
        // Skip leading whitespace
        while (pos < text.size() && (text[pos] == '\n' || text[pos] == '\r' || text[pos] == ' ')) pos++;
        if (pos >= text.size()) break;

        // Find end of first sentence (. ! ? or newline)
        size_t end = pos;
        while (end < text.size() && end < pos + 150) {  // cap at 150 chars
            char c = text[end];
            if (c == '.' || c == '!' || c == '?') { end++; break; }
            if (c == '\n') break;
            end++;
        }
        if (end > pos) {
            std::string sentence = text.substr(pos, end - pos);
            // Trim trailing whitespace
            while (!sentence.empty() && (sentence.back() == ' ' || sentence.back() == '\n'))
                sentence.pop_back();
            if (sentence.size() >= 10) {  // skip tiny fragments
                if (!result.empty()) result += " | ";
                result += sentence;
                count++;
            }
        }

        // Advance to next paragraph (double newline)
        size_t next = text.find("\n\n", end);
        if (next == std::string::npos) break;
        pos = next + 2;
    }
    return result;
}

// Layer 5: relation extraction.
// Extract cross-type entity co-occurrence pairs.
// Enables multi-hop: "Bell->telephone" + "telephone->Meucci" -> cross-reference.
static std::string extract_relations(const std::vector<entity_t> & entities, int max_pairs = 8) {
    std::string result;
    int count = 0;
    std::unordered_set<std::string> seen;

    for (size_t i = 0; i < entities.size() && count < max_pairs; i++) {
        for (size_t j = i + 1; j < entities.size() && count < max_pairs; j++) {
            // Cross-type pairs are most informative
            if (entities[i].type == entities[j].type) continue;
            // Skip generic pairs
            if (entities[i].type == entity_type::GENERIC || entities[j].type == entity_type::GENERIC) continue;

            std::string pair = entities[i].text + "\xe2\x86\x92" + entities[j].text;  // U+2192 (right arrow) in UTF-8
            if (seen.count(pair)) continue;
            seen.insert(pair);

            if (!result.empty()) result += " ";
            result += pair;
            count++;
        }
    }
    return result;
}

// Layer 3b: boundary text.
// Borrow first paragraph from the next chunk to cover boundary content.
// Tagged [BOUNDARY] so model treats as low-confidence.
static std::string extract_boundary(const std::string & next_chunk_text, int max_chars = 200) {
    if (next_chunk_text.empty()) return "";

    // Find end of first paragraph
    size_t end = next_chunk_text.find("\n\n");
    if (end == std::string::npos) end = std::min((size_t)max_chars, next_chunk_text.size());
    else end = std::min(end, (size_t)max_chars);

    std::string boundary = next_chunk_text.substr(0, end);
    // Trim
    while (!boundary.empty() && (boundary.back() == '\n' || boundary.back() == ' '))
        boundary.pop_back();

    return boundary.empty() ? "" : "[BOUNDARY] " + boundary;
}

// Main digest builder.
// Phase 1 (50 tokens): Layers 1 + 2 + 5 only (IDF + NER + relations)
// Phase 2 (215 tokens): + Layer 3 (first sentences) + Layer 3b (boundary)
//
// `include_sentences`: false for Phase 1 (50-token test), true for Phase 2
// `next_chunk_text`: empty string if no boundary needed or last chunk
static std::string build_chunk_digest(
    const bm25_index & bm25,
    int chunk_id,
    const std::vector<entity_t> & entities,
    const std::vector<std::string> & imports,
    const std::string & raw_text,
    bool include_sentences = false,
    const std::string & next_chunk_text = "")
{
    std::string digest;

    // Layer 1: BM25 IDF keywords (reuses the existing index)
    std::string idf_kw = bm25.get_idf_keywords_str(chunk_id, 15);
    if (!idf_kw.empty()) {
        digest += "[IDF: " + idf_kw + "]\n";
    }

    // Layer 2: NER entities (already extracted during ingest)
    if (!entities.empty()) {
        digest += "[NER:";
        for (auto & e : entities) {
            switch (e.type) {
                case entity_type::FILE:     digest += " FILE:" + e.text; break;
                case entity_type::FUNCTION: digest += " FUNC:" + e.text; break;
                case entity_type::CLASS:    digest += " CLASS:" + e.text; break;
                case entity_type::ERROR:    digest += " ERR:" + e.text; break;
                case entity_type::BUG_ID:   digest += " BUG:" + e.text; break;
                case entity_type::VARIABLE: digest += " VAR:" + e.text; break;
                case entity_type::MODULE:   digest += " MOD:" + e.text; break;
                default: break;
            }
        }
        digest += "]\n";
    }

    // Layer 4: Code skeleton (imports + function signatures)
    if (!imports.empty()) {
        digest += "[CODE:";
        for (auto & imp : imports) {
            digest += " IMPORT:" + imp;
        }
        digest += "]\n";
    }

    // Layer 5: Relation pairs (cross-type entity co-occurrence)
    if (entities.size() >= 2) {
        std::string rels = extract_relations(entities, 8);
        if (!rels.empty()) {
            digest += "[REL: " + rels + "]\n";
        }
    }

    // Phase 2 additions (only if include_sentences = true)
    if (include_sentences) {
        // Layer 3: First-sentence extraction
        std::string sentences = extract_first_sentences(raw_text, 6);
        if (!sentences.empty()) {
            digest += "[STRUCT: " + sentences + "]\n";
        }

        // Layer 3b: Boundary text from next chunk
        if (!next_chunk_text.empty()) {
            std::string boundary = extract_boundary(next_chunk_text);
            if (!boundary.empty()) {
                digest += boundary + "\n";
            }
        }
    }

    return digest;
}

// end of server-digest.h
