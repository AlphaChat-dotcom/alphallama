// SPDX-License-Identifier: Apache-2.0
#include "serve-intelligence.h"
#ifdef ALPHA_HAS_REDIS
#include "server-redis.h"
#endif

#include <cstdio>
#include <unordered_set>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <chrono>

namespace fs = std::filesystem;

namespace alphaserve {

// Budget Computation

context_budget compute_budget(int ctx_size, bool has_tools, int n_history_msgs) {
    int tool_overhead  = has_tools ? 1500 : 0;
    int gen_reserve    = 1024;
    int sys_overhead   = 200;
    int history_tokens = n_history_msgs * 150;

    int available_tokens = ctx_size - tool_overhead - gen_reserve - sys_overhead - history_tokens;
    if (available_tokens < 512) available_tokens = 512;

    int available_chars = (int)(available_tokens * 3.5f);

    context_budget b;
    b.max_experience_count = 3;
    b.max_experience_chars = std::min(5000, available_chars / 10);
    b.max_ner_chars        = available_chars * 15 / 100;
    b.max_code_chars       = available_chars - b.max_ner_chars - b.max_experience_chars;
    if (b.max_code_chars < 500) b.max_code_chars = 500;
    return b;
}

// Pre-Inference Augmentation
// Extracted from server-context.cpp corpus_augment_messages()
// with llama.cpp dependencies removed.

void intelligence_augment(intelligence_ctx & ctx, json & body) {
    bool has_bm25 = ctx.bm25 && ctx.bm25->N > 0;
    bool has_redis = false;
#ifdef ALPHA_HAS_REDIS
    has_redis = ctx.use_redis && ctx.redis && ctx.redis->is_connected();
#endif
    if (!has_bm25 && !has_redis) return;
    if (!body.contains("messages") || !body["messages"].is_array()) return;

    // 1. Extract query from last user message
    std::string query;
    auto & msgs = body["messages"];
    for (int i = (int)msgs.size() - 1; i >= 0; i--) {
        if (msgs[i].value("role", "") == "user" &&
            msgs[i].contains("content") && msgs[i]["content"].is_string()) {
            query = msgs[i]["content"].get<std::string>();
            break;
        }
    }
    if (query.empty()) return;

    auto t_start = std::chrono::high_resolution_clock::now();

    // 2. Compute budget
    bool has_tools = body.contains("tools");
    auto budget = compute_budget(ctx.ctx_size, has_tools, (int)msgs.size());

    std::string code_context;
    std::string ner_summary;
    std::string experience_context;
    int code_chars = 0, exp_chars = 0, exp_count = 0;

    // 3. Try Redis first (if enabled)
#ifdef ALPHA_HAS_REDIS
    if (ctx.use_redis && ctx.redis) {
        auto redis_result = ctx.redis->query(query);
        if (redis_result.found) {
            std::string augmented;

            if (redis_result.method == "graph") {
                augmented += "[Knowledge Graph]\n";
                for (auto & sent : redis_result.sentences) {
                    augmented += "  " + sent + "\n";
                }
                for (auto & ent : redis_result.entities) {
                    auto ent_result = ctx.redis->try_entity_lookup(ent);
                    if (ent_result.found) {
                        for (auto & s : ent_result.sentences) {
                            if ((int)augmented.size() + (int)s.size() > budget.max_code_chars) break;
                            augmented += s + "\n";
                        }
                    }
                    auto imports = ctx.redis->get_imports(ent);
                    if (!imports.empty()) {
                        augmented += ent + " imports:";
                        for (auto & imp : imports) augmented += " " + imp;
                        augmented += "\n";
                    }
                }
                auto ft = ctx.redis->ft_search(query, 10);
                if (!ft.sentences.empty()) {
                    augmented += "\n[Related Text]\n";
                    for (auto & s : ft.sentences) {
                        if ((int)augmented.size() + (int)s.size() > budget.max_code_chars) break;
                        augmented += s + "\n";
                    }
                }
            } else if (redis_result.method == "aggregation") {
                augmented += "[Records Found]\n";
                for (auto & ent : redis_result.entities) {
                    augmented += "  * " + ent + "\n";
                    std::string src = ctx.redis->get_code_source(ent);
                    if (!src.empty()) augmented += "    Source: " + src + "\n";
                }
            } else if (redis_result.method == "entity") {
                augmented += "[Entity Locations]\n";
                for (auto & loc : redis_result.sentences) {
                    augmented += "  " + loc + "\n";
                }
                auto ft = ctx.redis->ft_search(query, 20);
                if (!ft.sentences.empty()) {
                    augmented += "\n[Relevant Text]\n";
                    for (auto & s : ft.sentences) {
                        if ((int)augmented.size() + (int)s.size() > budget.max_code_chars) break;
                        augmented += s + "\n";
                    }
                }
            } else {
                for (auto & sent : redis_result.sentences) {
                    if ((int)augmented.size() + (int)sent.size() > budget.max_code_chars) break;
                    augmented += sent + "\n";
                }
            }

            if (!augmented.empty()) {
                json sys_msg = {{"role", "system"}, {"content", augmented}};
                msgs.insert(msgs.begin(), sys_msg);
                fprintf(stderr, "[intel] redis [%s] %d chars, %.1fms\n",
                        redis_result.method.c_str(), (int)augmented.size(), redis_result.latency_ms);
                return;
            }
        }
    }
#endif

    // 4. BM25 fallback + optional NER multihop
    if (!has_bm25) return;
    auto bm25_ranked = ctx.bm25->top_k(query, 50);
    std::vector<int> ranked;

    if (ctx.use_ner && ctx.entities && ctx.entities->n_unique > 0 &&
        ctx.linker && ctx.extractor) {
        auto mh = multihop_query(query, bm25_ranked,
                                  *ctx.entities, *ctx.linker,
                                  *ctx.extractor, 50);
        ranked = mh.chunk_ids;
    } else {
        ranked = bm25_ranked;
    }
    if (ranked.empty()) return;

    // 5. Partition code vs experience chunks
    std::vector<int> code_ranked, exp_ranked;
    for (int cid : ranked) {
        std::string source = ctx.bm25->get_source(cid);
        if (source.find("experience") == 0) {
            exp_ranked.push_back(cid);
        } else {
            code_ranked.push_back(cid);
        }
    }

    // 6. NER summary
    if (ctx.use_ner && ctx.entities && ctx.extractor && !code_ranked.empty()) {
        auto get_source = [&](int cid) { return ctx.bm25->get_source(cid); };
        auto get_text = [&](int cid) { return ctx.bm25->get_text(cid); };
        ner_summary = build_ner_summary_for_chunks(
            code_ranked, *ctx.entities, *ctx.linker,
            *ctx.extractor, get_source, get_text);
        if ((int)ner_summary.size() > budget.max_ner_chars) {
            ner_summary.resize(budget.max_ner_chars);
            auto last_nl = ner_summary.rfind('\n');
            if (last_nl != std::string::npos && last_nl > ner_summary.size() / 2)
                ner_summary.resize(last_nl + 1);
        }
    }

    // 7. Fill code context (deduplicate by parent file)
    std::unordered_set<int> seen_parents;
    for (int cid : code_ranked) {
        int parent_id = ctx.bm25->get_parent_id(cid);
        if (seen_parents.count(parent_id)) continue;
        seen_parents.insert(parent_id);

        std::string source = ctx.bm25->get_parent_source(cid);
        std::string text = ctx.bm25->get_text(cid);
        if (text.empty()) continue;
        if (code_chars + (int)text.size() > budget.max_code_chars) break;
        code_context += "// File: " + source + "\n" + text + "\n\n";
        code_chars += (int)text.size();
    }

    // 8. Fill experience context
    for (int cid : exp_ranked) {
        if (exp_count >= budget.max_experience_count) break;
        std::string text = ctx.bm25->get_text(cid);
        if (text.empty()) continue;
        if (exp_chars + (int)text.size() > budget.max_experience_chars) continue;
        std::string source = ctx.bm25->get_source(cid);
        experience_context += "// Experience: " + source + "\n" + text + "\n\n";
        exp_chars += (int)text.size();
        exp_count++;
    }

    // 9. Assemble
    std::string augmented;
    if (!ner_summary.empty()) augmented += ner_summary + "\n\n";
    augmented += code_context;
    if (!experience_context.empty()) {
        augmented += "\n--- Prior Experiences ---\n" + experience_context;
    }
    if (augmented.empty()) return;

    // 10. OOM fallback
    int est_chars = (int)augmented.size();
    for (auto & msg : msgs) {
        if (msg.contains("content") && msg["content"].is_string())
            est_chars += (int)msg["content"].get<std::string>().size();
    }
    int est_tokens = (int)(est_chars / 3.5f) + (has_tools ? 1500 : 0) + 1024;
    if (est_tokens > ctx.ctx_size) {
        fprintf(stderr, "[intel] est %d tokens > ctx %d — truncating\n", est_tokens, ctx.ctx_size);
        augmented.clear();
        if (!ner_summary.empty()) augmented += ner_summary + "\n\n";
        if (code_context.size() > 1000) {
            code_context.resize(code_context.size() / 2);
            auto nl = code_context.rfind('\n');
            if (nl != std::string::npos) code_context.resize(nl + 1);
        }
        augmented += code_context;
    }

    json sys_msg = {{"role", "system"}, {"content", augmented}};
    msgs.insert(msgs.begin(), sys_msg);

    auto t_end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    fprintf(stderr, "[intel] augment: %d code + %d ner + %d exp (%d) in %.1fms\n",
            code_chars, (int)ner_summary.size(), exp_chars, exp_count, ms);
}

// Post-Inference

void intelligence_post_process(intelligence_ctx & ctx,
                                const json & request,
                                const json & response) {
    // Stub: experience recording will be added here
    // For now, just log
    if (response.contains("choices") && !response["choices"].empty()) {
        auto & c = response["choices"][0];
        std::string finish = c.value("finish_reason", "unknown");
        int n_tok = 0;
        if (response.contains("usage") && response["usage"].contains("completion_tokens")) {
            n_tok = response["usage"]["completion_tokens"].get<int>();
        }
        fprintf(stderr, "[intel] post: finish=%s tokens=%d\n", finish.c_str(), n_tok);
    }
}

// Corpus Ingestion

static const std::vector<std::string> INGEST_EXTENSIONS = {
    ".txt", ".md", ".py", ".cpp", ".c", ".h", ".hpp",
    ".js", ".ts", ".java", ".go", ".rs", ".php",
    ".json", ".csv", ".log", ".xml", ".yaml", ".yml",
    ".rb", ".sh", ".sql", ".css", ".html",
};

static bool has_ingest_extension(const std::string & path) {
    for (auto & ext : INGEST_EXTENSIONS) {
        if (path.size() >= ext.size() &&
            path.compare(path.size() - ext.size(), ext.size(), ext) == 0) {
            return true;
        }
    }
    return false;
}

bool intelligence_init_corpus(intelligence_ctx & ctx, const std::string & corpus_dir) {
    if (corpus_dir.empty()) return false;

    ctx.corpus_dir = corpus_dir;
    if (!ctx.bm25) {
        static bm25_index bm25_instance;
        bm25_instance.warn_unfinalized = false;   // small resident corpus, never finalized: the forward scan
        ctx.bm25 = &bm25_instance;
    }
    if (!ctx.entities) {
        static entity_index entity_instance;
        ctx.entities = &entity_instance;
    }
    if (!ctx.extractor) {
        static code_entity_extractor extractor_instance;
        ctx.extractor = &extractor_instance;
    }
    if (!ctx.linker) {
        static entity_linker linker_instance;
        ctx.linker = &linker_instance;
    }

    auto t_start = std::chrono::high_resolution_clock::now();
    int n_files = 0;
    int n_chunks = 0;

    try {
        for (auto & entry : fs::recursive_directory_iterator(corpus_dir)) {
            if (!entry.is_regular_file()) continue;
            std::string path = entry.path().string();
            if (!has_ingest_extension(path)) continue;

            std::ifstream f(path);
            if (!f.is_open()) continue;

            std::string text((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
            if (text.empty()) continue;

            // Relative path for source
            std::string rel = fs::relative(entry.path(), corpus_dir).string();

            // Add to BM25 (always - serves as fallback)
            ctx.bm25->add_document(n_chunks, text, rel);

            // NER extraction
            auto entities = ctx.extractor->extract(text);
            ctx.entities->add(n_chunks, entities);

            // Extract imports for entity linker
            auto imports = ctx.extractor->extract_imports(text);
            if (!imports.empty()) {
                ctx.linker->add_imports(rel, imports);
            }

            // Ingest into Redis (primary index)
#ifdef ALPHA_HAS_REDIS
            if (ctx.use_redis && ctx.redis && ctx.redis->is_connected()) {
                std::vector<redis_entity> r_ents;
                for (auto & e : entities) {
                    r_ents.push_back({e.text, entity_type_str(e.type), rel, 0});
                }

                // Build relationships: import edges + cross-type entity pairs
                std::vector<redis_relationship> r_rels;
                for (auto & imp : imports) {
                    r_rels.push_back({rel, "imports", imp});
                }
                // Cross-type entity pairs (e.g., PERSON->FILE, CLASS->FUNC)
                for (size_t ei = 0; ei < entities.size() && ei < 50; ei++) {
                    for (size_t ej = ei + 1; ej < entities.size() && ej < 50; ej++) {
                        if (entities[ei].type == entities[ej].type) continue;
                        if (entities[ei].type == entity_type::GENERIC ||
                            entities[ej].type == entity_type::GENERIC) continue;
                        r_rels.push_back({entities[ei].text, "related_to", entities[ej].text});
                    }
                }

                ctx.redis->ingest_file(rel, text, r_ents, r_rels, imports);

                // Per-file function/class indexes: funcs:{file}, classes:{file}
                for (auto & e : entities) {
                    if (e.type == entity_type::FUNCTION) {
                        ctx.redis->add_func(rel, e.text);
                    } else if (e.type == entity_type::CLASS) {
                        ctx.redis->add_class(rel, e.text);
                    }
                }
            }
#endif

            n_chunks++;
            n_files++;
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "[intel] corpus scan error: %s\n", e.what());
    }

    // Flush Redis pipeline after batch ingest
#ifdef ALPHA_HAS_REDIS
    if (ctx.use_redis && ctx.redis && ctx.redis->is_connected()) {
        ctx.redis->flush();
    }
#endif

    auto t_end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    bool redis_active = false;
#ifdef ALPHA_HAS_REDIS
    redis_active = ctx.use_redis && ctx.redis && ctx.redis->is_connected();
#endif
    fprintf(stderr, "[intel] corpus: %d files, %d chunks, %d entities in %.0fms (primary: %s)\n",
            n_files, n_chunks, ctx.entities->n_unique, ms,
            redis_active ? "redis" : "bm25");

    return n_chunks > 0;
}

} // namespace alphaserve
