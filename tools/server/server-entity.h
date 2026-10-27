#pragma once

// Layer 2: NER + Entity Linking for cross-reference retrieval.
// Catches renamed files, call chains, and cross-service dependencies
// that BM25 keyword matching misses.
//
// Architecture:
//   Pre-scan:  Extract entities from each chunk (regex, CPU, 0.02ms/chunk)
//   Index:     entity -> chunk_ids, chunk_id -> entities
//   Linking:   Detect renames (git log) + call chains (import parsing)
//   Query:     3-hop: BM25 -> entity expand -> follow links
//
// No external deps. Regex-based extraction (no spaCy).

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <regex>
#include <algorithm>
#include <cctype>
#include <functional>

// <windows.h> (pulled in transitively by cpp-httplib via <winsock2.h>) defines several
// single-word macros that collide with ordinary identifiers. wingdi.h has `#define ERROR 0`, which
// rewrites `entity_type::ERROR` into `entity_type::0`; MSVC then reports "unexpected type
// 'unknown-type'" and mis-attributes the follow-on errors to BUG_ID/VARIABLE/MODULE/GENERIC on the
// next lines, which is why the first diagnostic points at the wrong enumerator.
#if defined(_WIN32)
#  undef ERROR
#endif

// Entity types

enum class entity_type {
    FILE,       // "auth_handler.py", "src/utils/token.js"
    FUNCTION,   // "cleanup_tokens", "verify_jwt"
    CLASS,      // "AuthHandler", "TokenCache"
    ERROR,      // "ERROR_4091", "NullPointerException"
    BUG_ID,     // "P3-001", "JIRA-1234", "BUG-42"
    VARIABLE,   // "TOKEN_TTL", "MAX_RETRIES"
    MODULE,     // "auth_service", "payment_api" (from imports)
    GENERIC,    // catch-all for NER-detected entities
};

static const char * entity_type_str(entity_type t) {
    switch (t) {
        case entity_type::FILE:     return "FILE";
        case entity_type::FUNCTION: return "FUNCTION";
        case entity_type::CLASS:    return "CLASS";
        case entity_type::ERROR:    return "ERROR";
        case entity_type::BUG_ID:   return "BUG_ID";
        case entity_type::VARIABLE: return "VARIABLE";
        case entity_type::MODULE:   return "MODULE";
        case entity_type::GENERIC:  return "GENERIC";
    }
    return "UNKNOWN";
}

struct entity_t {
    std::string text;    // normalized (lowercase, trimmed)
    entity_type type;
};

// Code Entity Extractor
// Regex-based extraction for code entities. No spaCy dependency.
// Runs on CPU, ~0.02ms per chunk.

struct code_entity_extractor {

    // Pre-compiled regexes (compiled once, reused)
    std::regex re_file;
    std::regex re_function;
    std::regex re_class;
    std::regex re_error;
    std::regex re_bug_id;
    std::regex re_variable;
    std::regex re_py_import;
    std::regex re_js_import;
    // PHP imports (path-aware)
    std::regex re_php_use;
    std::regex re_php_require_quoted;
    std::regex re_php_require_dir;
    std::regex re_php_require_dirname;
    // C/C++ includes
    std::regex re_c_include;
    // Java/C# imports
    std::regex re_java_import;

    code_entity_extractor() {
        // File paths: word/word.ext (1-4 char extension)
        re_file = std::regex(R"([\w/\\]+\.\w{1,4})");
        // Function calls: word( - capture the function name
        re_function = std::regex(R"((\w{2,})\s*\()");
        // Class names: CamelCase (at least 2 uppercase transitions)
        re_class = std::regex(R"(\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b)");
        // Error codes: UPPER_DIGITS or XxxException
        re_error = std::regex(R"([A-Z_]{2,}_\d+|\w+Exception|\w+Error)");
        // Bug IDs: P0-001, JIRA-1234, BUG-42, ISSUE-99
        re_bug_id = std::regex(R"((?:P[0-3]|JIRA|BUG|ISSUE|TICKET)[-_]\d+)");
        // Constants/variables: ALL_UPPER_WITH_UNDERSCORES (min 2 parts)
        re_variable = std::regex(R"(\b[A-Z][A-Z0-9]*_[A-Z0-9_]+\b)");
        // Python imports: from X import / import X
        re_py_import = std::regex(R"((?:from\s+([\w.]+)\s+import|import\s+([\w.]+)))");
        // JS/TS imports: require('X') / from 'X'
        re_js_import = std::regex(R"((?:require\s*\(\s*['"]([^'"]+)['"]|from\s+['"]([^'"]+)['"]))");
        // PHP namespace use: use App\Models\User;
        re_php_use = std::regex(R"(\buse\s+([\w\\]+)\s*;)");
        // PHP require/include with quoted path: require 'file.php';
        re_php_require_quoted = std::regex(R"((?:require_once|include_once|require|include)\s*\(?\s*['"]([^'"$]+\.php)['"]\s*\)?)");
        // PHP require with __DIR__: require __DIR__ . '/path/file.php';
        re_php_require_dir = std::regex(R"((?:require_once|include_once|require|include)\s*\(?\s*__DIR__\s*\.\s*['"]([^'"$]+\.php)['"]\s*\)?)");
        // PHP require with dirname(): require dirname(__DIR__) . '/path/file.php';
        re_php_require_dirname = std::regex(R"((?:require_once|include_once|require|include)\s*\(?\s*dirname\s*\([^)]+\)\s*\.\s*['"]([^'"$]+\.php)['"]\s*\)?)");
        // C/C++ includes
        re_c_include = std::regex(R"(#include\s*[<"]([^>"]+)[>"])");
        // Java/C# imports: import X.Y.Z; / using X.Y;
        re_java_import = std::regex(R"((?:import|using)\s+([\w.]+)\s*;)");
    }

    // Extract all entities from chunk text
    std::vector<entity_t> extract(const std::string & text) const {
        std::vector<entity_t> entities;
        entities.reserve(32);

        extract_regex(text, re_file, entity_type::FILE, entities);
        extract_regex(text, re_function, entity_type::FUNCTION, entities, 1);
        extract_regex(text, re_class, entity_type::CLASS, entities);
        extract_regex(text, re_error, entity_type::ERROR, entities);
        extract_regex(text, re_bug_id, entity_type::BUG_ID, entities);
        extract_regex(text, re_variable, entity_type::VARIABLE, entities);

        // Deduplicate
        deduplicate(entities);
        return entities;
    }

    // Extract imports (for call chain building) - language-aware
    // Covers PHP path-aware imports, C/C++ includes, Java/C# imports
    std::vector<std::string> extract_imports(const std::string & text) const {
        std::vector<std::string> imports;

        // Python imports
        extract_import_regex(text, re_py_import, imports);

        // JS/TS imports
        extract_import_regex(text, re_js_import, imports);

        // Java/C# imports
        extract_import_matches(text, re_java_import, imports);

        // C/C++ includes
        extract_import_matches(text, re_c_include, imports);

        // PHP imports (4 patterns)
        // Namespace use: use App\Models\User;
        extract_import_matches(text, re_php_use, imports);
        // Quoted require: require 'file.php';
        extract_import_matches(text, re_php_require_quoted, imports);
        // __DIR__ require: require __DIR__ . '/path/file.php';
        {
            auto begin = std::sregex_iterator(text.begin(), text.end(), re_php_require_dir);
            auto end = std::sregex_iterator();
            for (auto it = begin; it != end; ++it) {
                std::string path = (*it)[1].str();
                // Strip leading /. chars
                while (!path.empty() && (path[0] == '/' || path[0] == '.')) path.erase(0, 1);
                if (!path.empty()) imports.push_back(normalize(path));
            }
        }
        // dirname() require: require dirname(__DIR__) . '/path/file.php';
        {
            auto begin = std::sregex_iterator(text.begin(), text.end(), re_php_require_dirname);
            auto end = std::sregex_iterator();
            for (auto it = begin; it != end; ++it) {
                std::string path = (*it)[1].str();
                while (!path.empty() && (path[0] == '/' || path[0] == '.')) path.erase(0, 1);
                if (!path.empty()) imports.push_back(normalize(path));
            }
        }

        return imports;
    }

    // File-reference scan - find basename references in text
    // Skip generic names that would cause false positives
    static bool is_generic_basename(const std::string & name) {
        static const std::unordered_set<std::string> skip = {
            "__init__.py", "index.js", "index.ts", "index.tsx", "index.jsx",
            "main.py", "main.java", "main.go", "main.c", "main.cpp",
            "setup.py", "config.py", "config.js", "config.ts",
            "utils.py", "utils.js", "utils.ts", "helpers.py", "helpers.js",
            "test.py", "tests.py", "app.py", "app.js", "app.tsx",
            "mod.rs", "lib.rs", "Makefile", "CMakeLists.txt",
        };
        return skip.count(name) > 0;
    }

    // Function call ambiguity - track which functions are defined in multiple files
    // Returns only functions unique to a single file (safe to use as cross-file call evidence)
    static std::unordered_map<std::string, std::string> unique_function_owners(
        const std::unordered_map<std::string, std::vector<std::string>> & func_to_files)
    {
        std::unordered_map<std::string, std::string> unique;
        for (auto it = func_to_files.begin(); it != func_to_files.end(); ++it) {
            if (it->second.size() == 1) {
                unique[it->first] = it->second[0];
            }
        }
        return unique;
    }

private:
    static std::string normalize(const std::string & s) {
        std::string out;
        out.reserve(s.size());
        for (char c : s) {
            out += (char)std::tolower((unsigned char)c);
        }
        // trim whitespace
        size_t start = out.find_first_not_of(" \t\n\r");
        size_t end = out.find_last_not_of(" \t\n\r");
        if (start == std::string::npos) return "";
        return out.substr(start, end - start + 1);
    }

    void extract_regex(const std::string & text, const std::regex & re,
                       entity_type type, std::vector<entity_t> & out,
                       int capture_group = 0) const {
        auto begin = std::sregex_iterator(text.begin(), text.end(), re);
        auto end = std::sregex_iterator();
        for (auto it = begin; it != end; ++it) {
            std::string match = (*it)[capture_group].str();
            if (match.size() >= 2) {
                out.push_back({normalize(match), type});
            }
        }
    }

    // Helper for import extraction with two capture groups (Python/JS pattern)
    void extract_import_regex(const std::string & text, const std::regex & re,
                              std::vector<std::string> & imports) const {
        auto begin = std::sregex_iterator(text.begin(), text.end(), re);
        auto end = std::sregex_iterator();
        for (auto it = begin; it != end; ++it) {
            std::string mod = (*it)[1].matched ? (*it)[1].str() : (*it)[2].str();
            if (!mod.empty()) imports.push_back(normalize(mod));
        }
    }

    // Helper for import extraction with single capture group
    void extract_import_matches(const std::string & text, const std::regex & re,
                                std::vector<std::string> & imports) const {
        auto begin = std::sregex_iterator(text.begin(), text.end(), re);
        auto end = std::sregex_iterator();
        for (auto it = begin; it != end; ++it) {
            std::string mod = (*it)[1].str();
            if (!mod.empty()) imports.push_back(normalize(mod));
        }
    }

    static void deduplicate(std::vector<entity_t> & entities) {
        std::unordered_set<std::string> seen;
        auto it = std::remove_if(entities.begin(), entities.end(),
            [&seen](const entity_t & e) {
                std::string key = e.text + ":" + std::to_string((int)e.type);
                return !seen.insert(key).second;
            });
        entities.erase(it, entities.end());
    }
};

// Entity Index
// Maps entity -> chunk_ids and chunk_id -> entities.
// Built during pre-scan alongside BM25 index.

struct entity_index {
    // entity_key (text:type) -> list of chunk IDs
    std::unordered_map<std::string, std::vector<int>> entity_to_chunks;
    // chunk_id -> list of entity keys
    std::unordered_map<int, std::vector<std::string>> chunk_to_entities;
    // entity_key -> entity_type
    std::unordered_map<std::string, entity_type> entity_types;

    int n_entities = 0;  // total entity entries (with duplicates across chunks)
    int n_unique = 0;    // unique entity keys

    void clear() {
        entity_to_chunks.clear();
        chunk_to_entities.clear();
        entity_types.clear();
        n_entities = 0;
        n_unique = 0;
    }

    // Add extracted entities for a chunk
    void add(int chunk_id, const std::vector<entity_t> & entities) {
        for (auto & e : entities) {
            std::string key = e.text;  // use text as key (already normalized)

            auto it = entity_to_chunks.find(key);
            if (it == entity_to_chunks.end()) {
                entity_to_chunks[key] = {chunk_id};
                entity_types[key] = e.type;
                n_unique++;
            } else {
                it->second.push_back(chunk_id);
            }

            chunk_to_entities[chunk_id].push_back(key);
            n_entities++;
        }
    }

    // Register expected files that have no content in the corpus.
    // Creates entity entries so they can be found via entity linking even
    // without parsed code (a dataset may include only part of the listed files).
    void register_file(const std::string & filename, int chunk_id = -1) {
        std::string key = normalize_key(filename);
        if (entity_to_chunks.find(key) == entity_to_chunks.end()) {
            entity_to_chunks[key] = {};
            entity_types[key] = entity_type::FILE;
            n_unique++;
        }
        if (chunk_id >= 0) {
            entity_to_chunks[key].push_back(chunk_id);
        }
        // Also register basename for module matching
        auto pos = key.find_last_of("/\\");
        std::string base = (pos != std::string::npos) ? key.substr(pos + 1) : key;
        auto dot = base.find_last_of('.');
        if (dot != std::string::npos) base = base.substr(0, dot);
        if (base.size() >= 2 && entity_to_chunks.find(base) == entity_to_chunks.end()) {
            entity_to_chunks[base] = entity_to_chunks[key];
            entity_types[base] = entity_type::MODULE;
            n_unique++;
        }
    }

    // Class-reference scan - find cross-file class usage.
    // For each class defined in chunk A, check if its name appears in chunk B's text.
    // Call after all chunks are indexed. Adds MODULE-type edges.
    void scan_class_references(
        const std::unordered_map<int, std::string> & chunk_texts,
        const std::unordered_map<int, std::vector<std::string>> & chunk_classes)
    {
        // Build class -> defining chunk map
        std::unordered_map<std::string, std::unordered_set<int>> class_to_chunks;
        for (auto it = chunk_classes.begin(); it != chunk_classes.end(); ++it) {
            for (auto & cls : it->second) {
                if (cls.size() > 2) {
                    class_to_chunks[cls].insert(it->first);
                }
            }
        }

        // For each chunk, check if it references classes from other chunks
        for (auto it = chunk_texts.begin(); it != chunk_texts.end(); ++it) {
            int cid = it->first;
            const std::string & text = it->second;
            auto & local_entities = chunk_to_entities[cid];
            for (auto jt = class_to_chunks.begin(); jt != class_to_chunks.end(); ++jt) {
                const std::string & cls = jt->first;
                const auto & owners = jt->second;
                // Skip if this chunk defines the class itself
                if (owners.count(cid)) continue;
                // Skip if already linked
                bool already = false;
                for (auto & e : local_entities) {
                    if (e == cls) { already = true; break; }
                }
                if (already) continue;
                // Check if class name appears in text (word boundary match)
                if (text.find(cls) != std::string::npos) {
                    // Add as entity reference
                    entity_t ref = {cls, entity_type::CLASS};
                    add(cid, {ref});
                }
            }
        }
    }

    // File-reference scan - find basename references in text.
    // Language-agnostic fallback: if "utils.py" appears in another file's text,
    // they're related. Skips generic names (__init__.py, index.js, etc.).
    void scan_file_references(
        const std::unordered_map<int, std::string> & chunk_texts,
        const std::unordered_map<int, std::string> & chunk_filenames)
    {
        for (auto it_a = chunk_texts.begin(); it_a != chunk_texts.end(); ++it_a) {
            int cid_a = it_a->first;
            const std::string & text_a = it_a->second;
            for (auto it_b = chunk_filenames.begin(); it_b != chunk_filenames.end(); ++it_b) {
                int cid_b = it_b->first;
                const std::string & fname_b = it_b->second;
                if (cid_a == cid_b) continue;
                // Get basename
                auto pos = fname_b.find_last_of("/\\");
                std::string basename = (pos != std::string::npos) ? fname_b.substr(pos + 1) : fname_b;
                if (basename.size() <= 4) continue;
                if (code_entity_extractor::is_generic_basename(basename)) continue;
                // Check if basename appears in text
                if (text_a.find(basename) != std::string::npos) {
                    // Check not already linked
                    std::string key_b = normalize_key(fname_b);
                    auto & ents = chunk_to_entities[cid_a];
                    bool already = false;
                    for (auto & e : ents) {
                        if (e == key_b || e == basename) { already = true; break; }
                    }
                    if (!already) {
                        entity_t ref = {key_b, entity_type::FILE};
                        add(cid_a, {ref});
                    }
                }
            }
        }
    }

    // Find all chunks containing this entity
    std::vector<int> find_chunks(const std::string & entity) const {
        std::string key = normalize_key(entity);
        auto it = entity_to_chunks.find(key);
        if (it != entity_to_chunks.end()) return it->second;
        return {};
    }

    // Find all entities in a chunk (co-occurrence = related)
    std::vector<std::string> find_entities(int chunk_id) const {
        auto it = chunk_to_entities.find(chunk_id);
        if (it != chunk_to_entities.end()) return it->second;
        return {};
    }

    // Get entity type
    entity_type get_type(const std::string & entity) const {
        auto it = entity_types.find(entity);
        if (it != entity_types.end()) return it->second;
        return entity_type::GENERIC;
    }

private:
    static std::string normalize_key(const std::string & s) {
        std::string out;
        out.reserve(s.size());
        for (char c : s) out += (char)std::tolower((unsigned char)c);
        size_t start = out.find_first_not_of(" \t\n\r'\"");
        size_t end = out.find_last_not_of(" \t\n\r'\"");
        if (start == std::string::npos) return "";
        return out.substr(start, end - start + 1);
    }
};

// Entity Linker
// Detects renames, aliases, and call chain dependencies.
// Built after pre-scan from git history + import statements.

struct entity_linker {
    // Bidirectional rename links: old <-> new
    std::unordered_map<std::string, std::string> rename_links;

    // Call chain: file/module -> list of dependencies
    std::unordered_map<std::string, std::vector<std::string>> call_chain;

    // Reverse call chain: dependency -> list of callers
    std::unordered_map<std::string, std::vector<std::string>> reverse_chain;

    void clear() {
        rename_links.clear();
        call_chain.clear();
        reverse_chain.clear();
    }

    // Parse git log output for file renames.
    // Input: lines from `git log --diff-filter=R --summary --format=""`
    // Format: " rename old_name => new_name (100%)"
    //    or:  " rename {old => new}/path (100%)"
    void detect_renames(const std::string & git_log) {
        // Pattern: rename X => Y  or  rename {X => Y}/rest
        std::regex re_rename(R"(rename\s+(?:\{([^}]+)\s+=>\s+([^}]+)\}(/\S+)?|(\S+)\s+=>\s+(\S+)))");

        auto begin = std::sregex_iterator(git_log.begin(), git_log.end(), re_rename);
        auto end = std::sregex_iterator();

        for (auto it = begin; it != end; ++it) {
            std::string old_name, new_name;
            if ((*it)[1].matched) {
                // {old => new}/rest format
                std::string suffix = (*it)[3].matched ? (*it)[3].str() : "";
                old_name = (*it)[1].str() + suffix;
                new_name = (*it)[2].str() + suffix;
            } else {
                // old => new format
                old_name = (*it)[4].str();
                new_name = (*it)[5].str();
            }

            old_name = normalize(old_name);
            new_name = normalize(new_name);

            if (!old_name.empty() && !new_name.empty()) {
                rename_links[old_name] = new_name;
                rename_links[new_name] = old_name;  // bidirectional
            }
        }
    }

    // Build call chain from import statements found during NER extraction.
    // Called with: source file/module name + its imports
    void add_imports(const std::string & source, const std::vector<std::string> & imports) {
        std::string src = normalize(source);
        for (auto & imp : imports) {
            call_chain[src].push_back(imp);
            reverse_chain[imp].push_back(src);
        }
    }

    // Resolve: given an entity, return all linked entities
    std::unordered_set<std::string> resolve(const std::string & entity) const {
        std::unordered_set<std::string> linked;
        std::string key = normalize(entity);

        // Check renames
        auto it_rename = rename_links.find(key);
        if (it_rename != rename_links.end()) {
            linked.insert(it_rename->second);
        }

        // Forward call chain (what does this depend on?)
        auto it_fwd = call_chain.find(key);
        if (it_fwd != call_chain.end()) {
            for (auto & dep : it_fwd->second) {
                linked.insert(dep);
            }
        }

        // Reverse call chain (what depends on this?)
        auto it_rev = reverse_chain.find(key);
        if (it_rev != reverse_chain.end()) {
            for (auto & caller : it_rev->second) {
                linked.insert(caller);
            }
        }

        return linked;
    }

private:
    static std::string normalize(const std::string & s) {
        std::string out;
        out.reserve(s.size());
        for (char c : s) out += (char)std::tolower((unsigned char)c);
        // trim
        size_t start = out.find_first_not_of(" \t\n\r");
        size_t end = out.find_last_not_of(" \t\n\r");
        if (start == std::string::npos) return "";
        return out.substr(start, end - start + 1);
    }
};

// Multi-Hop Query
// 3-hop pipeline: BM25 -> entity co-occurrence -> entity links

struct multihop_result {
    std::vector<int> chunk_ids;     // all evidence chunk IDs (deduplicated)
    int hop1_count = 0;             // from BM25
    int hop2_count = 0;             // from entity co-occurrence
    int hop3_count = 0;             // from entity links
};

static multihop_result multihop_query(
    const std::string & query,
    const std::vector<int> & bm25_hits,     // pre-computed BM25 top-k
    const entity_index & eidx,
    const entity_linker & elink,
    const code_entity_extractor & extractor,
    int max_evidence = 50)
{
    multihop_result result;
    std::unordered_set<int> seen;

    // Hop 1: BM25 keyword results (already computed)
    for (int cid : bm25_hits) {
        if (seen.insert(cid).second) {
            result.chunk_ids.push_back(cid);
            result.hop1_count++;
        }
    }

    // Extract entities from the query itself
    auto query_entities = extractor.extract(query);

    // Hop 2: Entity co-occurrence expansion
    // For top-10 BM25 results + query entities, find related chunks
    int hop2_budget = std::max(0, max_evidence - (int)result.chunk_ids.size());

    // From query entities
    for (auto & qe : query_entities) {
        auto related = eidx.find_chunks(qe.text);
        for (int cid : related) {
            if (hop2_budget <= 0) break;
            if (seen.insert(cid).second) {
                result.chunk_ids.push_back(cid);
                result.hop2_count++;
                hop2_budget--;
            }
        }
    }

    // From top-10 BM25 result entities
    int top_n = std::min(10, (int)bm25_hits.size());
    for (int i = 0; i < top_n && hop2_budget > 0; i++) {
        auto entities = eidx.find_entities(bm25_hits[i]);
        for (auto & ent : entities) {
            auto related = eidx.find_chunks(ent);
            for (int cid : related) {
                if (hop2_budget <= 0) break;
                if (seen.insert(cid).second) {
                    result.chunk_ids.push_back(cid);
                    result.hop2_count++;
                    hop2_budget--;
                }
            }
        }
    }

    // Hop 3: Follow entity links (renames, call chains)
    int hop3_budget = std::max(0, max_evidence - (int)result.chunk_ids.size());

    // From top-5 BM25 results
    int link_n = std::min(5, (int)bm25_hits.size());
    for (int i = 0; i < link_n && hop3_budget > 0; i++) {
        auto entities = eidx.find_entities(bm25_hits[i]);
        for (auto & ent : entities) {
            auto linked = elink.resolve(ent);
            for (auto & linked_ent : linked) {
                auto related = eidx.find_chunks(linked_ent);
                for (int cid : related) {
                    if (hop3_budget <= 0) break;
                    if (seen.insert(cid).second) {
                        result.chunk_ids.push_back(cid);
                        result.hop3_count++;
                        hop3_budget--;
                    }
                }
            }
        }
    }

    return result;
}

// Tagged Edge + Bidirectional Filtering + NER Summary Builder

enum class edge_source { IMPORT, CLASS_REF, FILE_REF, FUNC_CALL };

struct tagged_edge {
    std::string from;
    std::string to;
    edge_source source;
    std::string func_name;  // only for FUNC_CALL edges
};

// Remove bidirectional inferred edges (ambiguous direction).
// Keep bidirectional import edges (real circular deps in JS/TS).
static std::vector<tagged_edge> filter_bidirectional(const std::vector<tagged_edge> & edges) {
    // Find bidirectional inferred edges
    std::unordered_set<std::string> inferred_set;
    for (auto & e : edges) {
        if (e.source != edge_source::IMPORT) {
            inferred_set.insert(e.from + "\t" + e.to);
        }
    }
    std::unordered_set<std::string> bidirectional;
    for (auto & e : edges) {
        if (e.source != edge_source::IMPORT) {
            std::string reverse = e.to + "\t" + e.from;
            if (inferred_set.count(reverse)) {
                bidirectional.insert(e.from + "\t" + e.to);
                bidirectional.insert(reverse);
            }
        }
    }

    std::vector<tagged_edge> clean;
    for (auto & e : edges) {
        std::string key = e.from + "\t" + e.to;
        if (!bidirectional.count(key)) {
            clean.push_back(e);
        }
    }
    return clean;
}

// Build NER summary text for prompt prepending.
// Includes bidirectional filtering and always-list-all-files.
static std::string build_ner_summary(
    const std::vector<tagged_edge> & raw_edges,
    const std::vector<std::string> & all_expected_files,
    const std::unordered_map<std::string, std::vector<std::string>> & file_classes)
{
    auto edges = filter_bidirectional(raw_edges);

    std::string summary = "[NER Analysis]\n";

    // Import edges
    bool has_imports = false;
    std::unordered_set<std::string> seen;
    for (auto & e : edges) {
        if (e.source == edge_source::IMPORT || e.source == edge_source::CLASS_REF ||
            e.source == edge_source::FILE_REF) {
            std::string key = e.from + "->" + e.to;
            if (seen.insert(key).second) {
                if (!has_imports) {
                    summary += "Import dependencies detected:\n";
                    has_imports = true;
                }
                summary += "  " + e.from + " imports from " + e.to + "\n";
            }
        }
    }

    // Function call edges
    bool has_funcs = false;
    seen.clear();
    for (auto & e : edges) {
        if (e.source == edge_source::FUNC_CALL) {
            std::string key = e.from + "->" + e.to + ":" + e.func_name;
            if (seen.insert(key).second) {
                if (!has_funcs) {
                    summary += "Cross-file function calls detected:\n";
                    has_funcs = true;
                }
                summary += "  " + e.from + " calls " + e.func_name + "() defined in " + e.to + "\n";
            }
        }
    }

    // Class listings
    for (auto it = file_classes.begin(); it != file_classes.end(); ++it) {
        if (!it->second.empty()) {
            summary += "Classes in " + it->first + ": ";
            for (size_t i = 0; i < it->second.size(); i++) {
                if (i > 0) summary += ", ";
                summary += it->second[i];
            }
            summary += "\n";
        }
    }

    // Always list all expected files so model doesn't drop them
    std::unordered_set<std::string> listed;
    for (auto & e : edges) {
        listed.insert(e.from);
        listed.insert(e.to);
    }
    for (auto & f : all_expected_files) {
        if (!listed.count(f)) {
            summary += "  " + f + ": (no dependencies detected, include in chain)\n";
        }
    }

    return summary;
}

// Build NER summary from retrieved chunk IDs.
// Server-side wrapper: extracts entities from chunk index, builds tagged edges,
// calls build_ner_summary() with the right data structures.
static std::string build_ner_summary_for_chunks(
    const std::vector<int> & chunk_ids,
    const entity_index & eidx,
    const entity_linker & elink,
    const code_entity_extractor & extractor,
    const std::function<std::string(int)> & get_source,
    const std::function<std::string(int)> & get_text)
{
    std::vector<std::string> all_files;
    std::unordered_map<std::string, std::vector<std::string>> file_classes;
    std::unordered_map<std::string, std::string> file_texts;

    for (int cid : chunk_ids) {
        std::string source = get_source(cid);
        if (source.empty()) continue;
        all_files.push_back(source);
        std::string text = get_text(cid);
        file_texts[source] = text;
        auto entities = extractor.extract(text);
        for (auto & e : entities) {
            if (e.type == entity_type::CLASS) {
                file_classes[source].push_back(e.text);
            }
        }
    }
    if (all_files.empty()) return "";

    std::vector<tagged_edge> edges;
    std::unordered_map<std::string, std::string> base_to_file;
    for (auto & f : all_files) {
        std::string base = f;
        auto pos = base.find_last_of("/\\");
        if (pos != std::string::npos) base = base.substr(pos + 1);
        auto dot = base.find_last_of('.');
        if (dot != std::string::npos) base = base.substr(0, dot);
        for (auto & c : base) c = (char)std::tolower((unsigned char)c);
        if (base.size() >= 2) base_to_file[base] = f;
    }

    for (auto & f : all_files) {
        auto it = file_texts.find(f);
        if (it == file_texts.end()) continue;
        const std::string & text = it->second;

        auto imports = extractor.extract_imports(text);
        for (auto & imp : imports) {
            std::string imp_lower = imp;
            for (auto & c : imp_lower) c = (char)std::tolower((unsigned char)c);
            std::string segment;
            for (char c : imp_lower) {
                if (c == '/' || c == '\\' || c == '.') {
                    if (segment.size() >= 2) {
                        auto mt = base_to_file.find(segment);
                        if (mt != base_to_file.end() && mt->second != f) {
                            edges.push_back({f, mt->second, edge_source::IMPORT, ""});
                        }
                    }
                    segment.clear();
                } else {
                    segment += c;
                }
            }
            if (segment.size() >= 2) {
                auto mt = base_to_file.find(segment);
                if (mt != base_to_file.end() && mt->second != f) {
                    edges.push_back({f, mt->second, edge_source::IMPORT, ""});
                }
            }
        }

        auto entities = extractor.extract(text);
        for (auto & e : entities) {
            if (e.type == entity_type::CLASS) continue;
            if (e.type == entity_type::FUNCTION) {
                for (auto & other_f : all_files) {
                    if (other_f == f) continue;
                    auto oit = file_texts.find(other_f);
                    if (oit == file_texts.end()) continue;
                    if (oit->second.find(e.text) != std::string::npos) {
                        edges.push_back({f, other_f, edge_source::FUNC_CALL, e.text});
                        break;
                    }
                }
            }
        }
    }

    return build_ner_summary(edges, all_files, file_classes);
}
