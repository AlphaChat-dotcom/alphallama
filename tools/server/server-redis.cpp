// Redis integration implementation
// Requires: libhiredis-dev (apt install libhiredis-dev)

#include "server-redis.h"
#include <cstdarg>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <regex>
#include <set>
#include <unordered_set>
#include <sstream>

// hiredis headers (dynamically linked)
#include <hiredis/hiredis.h>

// Connection

redis_index::~redis_index() {
    if (ctx) {
        redisFree(ctx);
        ctx = nullptr;
    }
}

bool redis_index::connect(const std::string & host, int port) {
    struct timeval timeout = {5, 0};  // 5 second timeout
    ctx = redisConnectWithTimeout(host.c_str(), port, timeout);
    if (!ctx || ctx->err) {
        if (ctx) {
            fprintf(stderr, "redis: connection failed: %s\n", ctx->errstr);
            redisFree(ctx);
            ctx = nullptr;
        }
        return false;
    }
    fprintf(stderr, "redis: connected to %s:%d\n", host.c_str(), port);
    return true;
}

redisReply * redis_index::cmd(const char * fmt, ...) {
    if (!ctx) return nullptr;
    va_list ap;
    va_start(ap, fmt);
    auto * reply = (redisReply *)redisvCommand(ctx, fmt, ap);
    va_end(ap);
    return reply;
}

// Indexes

bool redis_index::create_indexes() {
    if (!ctx) return false;

    // Drop existing indexes (ignore errors if they don't exist)
    auto * r = cmd("FT.DROPINDEX sentences DD");
    if (r) freeReplyObject(r);
    r = cmd("FT.DROPINDEX code_idx DD");
    if (r) freeReplyObject(r);

    // Create sentence full-text index (replaces BM25)
    r = cmd("FT.CREATE sentences ON HASH PREFIX 1 sent: "
            "SCHEMA text TEXT WEIGHT 1.0 source TAG pos NUMERIC SORTABLE");
    if (!r || r->type == REDIS_REPLY_ERROR) {
        fprintf(stderr, "redis: failed to create sentences index: %s\n",
                r ? r->str : "null reply");
        if (r) freeReplyObject(r);
        return false;
    }
    freeReplyObject(r);

    fprintf(stderr, "redis: created RediSearch indexes\n");
    return true;
}

// Pipeline helpers

void redis_index::pipeline_start() {
    pipeline_active = true;
    pipeline_count = 0;
}

void redis_index::pipeline_append(const char * fmt, ...) {
    if (!ctx) return;
    va_list ap;
    va_start(ap, fmt);
    redisvAppendCommand(ctx, fmt, ap);
    va_end(ap);
    pipeline_count++;

    // Flush every 1000 commands to avoid OOM
    if (pipeline_count >= 1000) {
        pipeline_flush();
        pipeline_start();
    }
}

void redis_index::pipeline_flush() {
    if (!ctx || pipeline_count == 0) return;
    redisReply * reply;
    for (int i = 0; i < pipeline_count; i++) {
        if (redisGetReply(ctx, (void**)&reply) == REDIS_OK && reply) {
            freeReplyObject(reply);
        }
    }
    pipeline_active = false;
    pipeline_count = 0;
}

void redis_index::flush() {
    if (pipeline_active) {
        pipeline_flush();
    }
}

// Ingest

void redis_index::add_sentence(const std::string & source, int pos, const std::string & text,
                               const std::string & record) {
    if (!ctx) return;
    // Key: sent:{source}:{pos}
    // Prepend source to text so RediSearch can match filenames + content
    std::string indexed = "[" + source + "] " + text;
    char key[512];
    snprintf(key, sizeof(key), "sent:%s:%d", source.c_str(), pos);
    // `record` is the parent record/block this sentence belongs to. FT matches at sentence
    // granularity, but a hit injects the whole record so the answer field is present even when
    // it lives in a different sentence. Non-schema field: stored on the hash, returned by FT.SEARCH.
    const std::string & rec = record.empty() ? text : record;
    pipeline_append("HSET %s text %b source %s pos %d record %b",
                    key, indexed.c_str(), indexed.size(), source.c_str(), pos,
                    rec.c_str(), rec.size());
    n_sentences++;
}

void redis_index::add_entity(const std::string & name, const std::string & type,
                              const std::string & source, int line) {
    if (!ctx) return;
    // Entity set: e:{name} -> {source:line, source:line, ...}
    char member[512];
    snprintf(member, sizeof(member), "%s:%d", source.c_str(), line);
    pipeline_append("SADD e:%s %s", name.c_str(), member);

    // Type index: etype:{type} -> {name, name, ...}
    pipeline_append("SADD etype:%s %s", type.c_str(), name.c_str());
    n_entities++;
}

void redis_index::add_relationship(const std::string & subject,
                                    const std::string & predicate,
                                    const std::string & object) {
    if (!ctx) return;
    // Relationship: r:{subject} -> {predicate: object}
    pipeline_append("HSET r:%s %s %s", subject.c_str(), predicate.c_str(), object.c_str());
    // Reverse index for graph traversal
    pipeline_append("HSET rev:%s %s %s", object.c_str(), predicate.c_str(), subject.c_str());
    n_relationships++;
}

void redis_index::add_code_pattern(const std::string & code, const std::string & source) {
    if (!ctx) return;
    // Extract prefix: COBALT-742 -> COBALT
    auto dash = code.find('-');
    std::string prefix = (dash != std::string::npos) ? code.substr(0, dash) : code;

    // Set: codes:{prefix} -> {code1, code2, ...}
    pipeline_append("SADD codes:%s %s", prefix.c_str(), code.c_str());
    // Code source: c:{code} -> source
    pipeline_append("HSET c:%s source %s", code.c_str(), source.c_str());
    n_codes++;
}

void redis_index::add_import(const std::string & file, const std::string & imported) {
    if (!ctx) return;
    pipeline_append("SADD imports:%s %s", file.c_str(), imported.c_str());
    pipeline_append("SADD imported_by:%s %s", imported.c_str(), file.c_str());
}

void redis_index::add_func(const std::string & file, const std::string & func_name) {
    if (!ctx) return;
    pipeline_append("SADD funcs:%s %s", file.c_str(), func_name.c_str());
}

void redis_index::add_class(const std::string & file, const std::string & class_name) {
    if (!ctx) return;
    pipeline_append("SADD classes:%s %s", file.c_str(), class_name.c_str());
}

int redis_index::ingest_file(const std::string & filepath, const std::string & text,
                              const std::vector<redis_entity> & entities,
                              const std::vector<redis_relationship> & relationships,
                              const std::vector<std::string> & imports) {
    pipeline_start();

    // Split text into records (blank-line-separated blocks), then into sentences within each
    // record. FT indexes sentences; each sentence carries its parent record so a hit injects the
    // whole record, since the answer field is often in a different sentence of the same record.
    std::vector<std::string> sentences;
    std::vector<std::string> sent_records;   // parallel: parent record text for each sentence
    auto split_sentences = [](const std::string & block, std::vector<std::string> & out) {
        std::string current;
        for (size_t i = 0; i < block.size(); i++) {
            current += block[i];
            char c = block[i];
            bool is_sent_end = (c == '.' || c == '!' || c == '?') &&
                               (i + 1 >= block.size() || block[i + 1] == ' ' || block[i + 1] == '\n');
            bool is_code_end = (c == '{' || c == '}' || c == ';');
            bool is_newline = (c == '\n');
            if ((is_sent_end || is_code_end || is_newline) && current.size() >= 15) {
                while (!current.empty() && (current.back() == ' ' || current.back() == '\n'
                                             || current.back() == '\r'))
                    current.pop_back();
                if (current.size() >= 15) out.push_back(current);
                current.clear();
            }
        }
        if (current.size() >= 15) out.push_back(current);
    };
    size_t rstart = 0;
    while (rstart < text.size()) {
        size_t rend = text.find("\n\n", rstart);
        std::string record = text.substr(rstart, (rend == std::string::npos ? text.size() : rend) - rstart);
        std::vector<std::string> rsents;
        split_sentences(record, rsents);
        for (auto & s : rsents) { sentences.push_back(s); sent_records.push_back(record); }
        rstart = (rend == std::string::npos) ? text.size() : rend + 2;
    }

    // Store sentences (each tagged with its parent record for record-granularity injection)
    for (int i = 0; i < (int)sentences.size(); i++) {
        add_sentence(filepath, i, sentences[i], sent_records[i]);
    }

    // Store entities
    for (auto & e : entities) {
        add_entity(e.name, e.type, filepath, e.line);
    }

    // Store relationships
    for (auto & rel : relationships) {
        add_relationship(rel.subject, rel.predicate, rel.object);
    }

    // Store imports
    for (auto & imp : imports) {
        add_import(filepath, imp);
    }

    // Extract and store code patterns (WORD-DIGITS)
    static const std::regex code_re("[A-Z][A-Z0-9]+-[A-Z0-9]+");
    std::sregex_iterator it(text.begin(), text.end(), code_re);
    std::sregex_iterator end;
    for (; it != end; ++it) {
        add_code_pattern(it->str(), filepath);
    }

    pipeline_flush();
    return (int)sentences.size();
}

// Query

std::vector<std::string> redis_index::get_imports(const std::string & file) {
    std::vector<std::string> result;
    auto * r = cmd("SMEMBERS imports:%s", file.c_str());
    if (r && r->type == REDIS_REPLY_ARRAY) {
        for (size_t i = 0; i < r->elements; i++) {
            if (r->element[i]->type == REDIS_REPLY_STRING) {
                result.push_back(r->element[i]->str);
            }
        }
    }
    if (r) freeReplyObject(r);
    return result;
}

std::vector<std::string> redis_index::get_imported_by(const std::string & file) {
    std::vector<std::string> result;
    auto * r = cmd("SMEMBERS imported_by:%s", file.c_str());
    if (r && r->type == REDIS_REPLY_ARRAY) {
        for (size_t i = 0; i < r->elements; i++) {
            if (r->element[i]->type == REDIS_REPLY_STRING) {
                result.push_back(r->element[i]->str);
            }
        }
    }
    if (r) freeReplyObject(r);
    return result;
}

std::vector<std::string> redis_index::get_codes(const std::string & prefix) {
    std::vector<std::string> result;
    auto * r = cmd("SMEMBERS codes:%s", prefix.c_str());
    if (r && r->type == REDIS_REPLY_ARRAY) {
        for (size_t i = 0; i < r->elements; i++) {
            if (r->element[i]->type == REDIS_REPLY_STRING) {
                result.push_back(r->element[i]->str);
            }
        }
    }
    if (r) freeReplyObject(r);
    return result;
}

// Structured-record analytics (FT.AGGREGATE)

redisReply * redis_index::cmd_argv(const std::vector<std::string> & args) {
    if (!ctx || args.empty()) return nullptr;
    std::vector<const char *> argv(args.size());
    std::vector<size_t>       argvlen(args.size());
    for (size_t i = 0; i < args.size(); i++) {
        argv[i]    = args[i].data();
        argvlen[i] = args[i].size();
    }
    return (redisReply *) redisCommandArgv(ctx, (int) args.size(), argv.data(), argvlen.data());
}

bool redis_index::create_record_index(const std::string & index,
                                      const std::vector<std::string> & tag_fields,
                                      const std::vector<std::string> & numeric_fields) {
    if (!ctx) return false;

    // Drop an existing index of this name (keep the docs: no DD) so re-create is idempotent.
    if (auto * d = cmd("FT.DROPINDEX %s", index.c_str())) freeReplyObject(d);

    std::vector<std::string> args = {"FT.CREATE", index, "ON", "HASH", "PREFIX", "1",
                                     index + ":", "SCHEMA"};
    for (const auto & t : tag_fields)     { args.push_back(t); args.push_back("TAG"); }
    for (const auto & n : numeric_fields) { args.push_back(n); args.push_back("NUMERIC"); args.push_back("SORTABLE"); }

    auto * r = cmd_argv(args);
    bool ok = r && r->type != REDIS_REPLY_ERROR;
    if (!ok) {
        fprintf(stderr, "redis: FT.CREATE %s failed: %s\n", index.c_str(), r ? r->str : "null reply");
    }
    if (r) freeReplyObject(r);
    if (!ok) return false;

    // Remember the schema so aggregate() can whitelist field names.
    record_schema sch;
    for (const auto & t : tag_fields)     sch.tags.insert(t);
    for (const auto & n : numeric_fields) sch.numerics.insert(n);
    record_schemas[index] = std::move(sch);
    fprintf(stderr, "redis: created record index `%s` (%zu tag + %zu numeric fields)\n",
            index.c_str(), tag_fields.size(), numeric_fields.size());
    return true;
}

void redis_index::add_record(const std::string & index, const std::string & id,
                             const std::vector<std::pair<std::string, std::string>> & fields) {
    if (!ctx) return;
    std::vector<std::string> args = {"HSET", index + ":" + id};
    for (const auto & kv : fields) { args.push_back(kv.first); args.push_back(kv.second); }
    if (args.size() <= 2) return;  // nothing to set
    // Route through the pipeline for batch-ingest throughput.
    std::vector<const char *> argv(args.size());
    std::vector<size_t>       argvlen(args.size());
    for (size_t i = 0; i < args.size(); i++) { argv[i] = args[i].data(); argvlen[i] = args[i].size(); }
    redisAppendCommandArgv(ctx, (int) args.size(), argv.data(), argvlen.data());
    pipeline_count++;
    pipeline_active = true;
    if (pipeline_count >= 1000) { pipeline_flush(); pipeline_start(); }
}

redis_agg_result redis_index::aggregate(const std::string & index,
                                        const std::string & metric,
                                        const std::string & field,
                                        const std::string & group_by,
                                        const std::vector<redis_agg_filter> & filters) {
    redis_agg_result out;
    out.metric = metric; out.field = field; out.group_by = group_by;

    // metric -> RediSearch reducer + arity (0 = COUNT takes no field).
    std::string reducer; int nargs = 1;
    if      (metric == "sum")   reducer = "SUM";
    else if (metric == "avg")   reducer = "AVG";
    else if (metric == "min")   reducer = "MIN";
    else if (metric == "max")   reducer = "MAX";
    else if (metric == "count") { reducer = "COUNT"; nargs = 0; }
    else { out.error = "unknown metric '" + metric + "' (use sum|avg|count|min|max)"; return out; }

    // Every field name must be a declared field of this index.
    auto sit = record_schemas.find(index);
    if (sit == record_schemas.end()) {
        out.error = "unknown record index '" + index + "' (call create_record_index first)";
        return out;
    }
    const record_schema & sch = sit->second;
    auto is_tag     = [&](const std::string & f){ return sch.tags.count(f)     > 0; };
    auto is_numeric = [&](const std::string & f){ return sch.numerics.count(f) > 0; };

    if (!is_tag(group_by) && !is_numeric(group_by)) {
        out.error = "unknown group_by field '" + group_by + "'"; return out;
    }
    if (nargs == 1 && !is_numeric(field)) {
        out.error = "unknown or non-numeric field '" + field + "' for metric '" + metric + "'"; return out;
    }

    // Build the RediSearch query string from validated filters:
    //   numeric field -> @f:[lo hi] ;  tag field -> @f:{value}
    std::string q;
    for (const auto & f : filters) {
        std::string clause;
        if (is_numeric(f.field) && !f.hi.empty()) {
            clause = "@" + f.field + ":[" + f.lo + " " + f.hi + "]";
        } else if (is_tag(f.field)) {
            clause = "@" + f.field + ":{" + f.lo + "}";
        } else if (is_numeric(f.field)) {
            // range with only a low bound given -> treat as exact point [lo lo]
            clause = "@" + f.field + ":[" + f.lo + " " + f.lo + "]";
        } else {
            out.error = "unknown filter field '" + f.field + "'"; return out;
        }
        if (!q.empty()) q += " ";
        q += clause;
    }
    if (q.empty()) q = "*";

    std::vector<std::string> args = {"FT.AGGREGATE", index, q,
                                     "GROUPBY", "1", "@" + group_by, "REDUCE", reducer};
    if (nargs == 1) { args.push_back("1"); args.push_back("@" + field); }
    else            { args.push_back("0"); }
    args.push_back("AS"); args.push_back("value");
    args.push_back("LIMIT"); args.push_back("0"); args.push_back("100000");

    auto t0 = std::chrono::steady_clock::now();
    auto * r = cmd_argv(args);
    auto t1 = std::chrono::steady_clock::now();
    out.latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (!r || r->type == REDIS_REPLY_ERROR) {
        out.error = std::string("FT.AGGREGATE failed: ") + (r ? r->str : "null reply");
        if (r) freeReplyObject(r);
        return out;
    }
    // reply[0] = result count; reply[1..] = rows, each a flat [field,val,...] array.
    for (size_t i = 1; i < r->elements; i++) {
        if (r->element[i]->type != REDIS_REPLY_ARRAY) continue;
        auto * row = r->element[i];
        redis_agg_row ar; bool have_g = false, have_v = false;
        for (size_t j = 0; j + 1 < row->elements; j += 2) {
            const char * k = row->element[j]->str;
            const char * v = row->element[j + 1]->str;
            if (!k || !v) continue;
            if (group_by == k) { ar.group = v; have_g = true; }
            else if (strcmp(k, "value") == 0) { ar.value = atof(v); have_v = true; }
        }
        if (have_g && have_v) out.rows.push_back(std::move(ar));
    }
    freeReplyObject(r);
    out.ok = true;
    return out;
}

std::string redis_index::get_code_source(const std::string & code) {
    auto * r = cmd("HGET c:%s source", code.c_str());
    std::string result;
    if (r && r->type == REDIS_REPLY_STRING) {
        result = r->str;
    }
    if (r) freeReplyObject(r);
    return result;
}

redis_query_result redis_index::try_entity_lookup(const std::string & q) {
    redis_query_result result;
    // Extract entities: names, files, codes
    std::vector<std::string> entities;
    static const std::regex name_re("[A-Z][a-z]+(?:\\s+[A-Z][a-z]+)+");
    for (std::sregex_iterator it(q.begin(), q.end(), name_re); it != std::sregex_iterator(); ++it)
        entities.push_back(it->str());
    static const std::regex file_re("[a-zA-Z0-9_/]+\\.[a-z]{1,4}\\b");
    for (std::sregex_iterator it(q.begin(), q.end(), file_re); it != std::sregex_iterator(); ++it)
        entities.push_back(it->str());

    for (auto & entity : entities) {
        auto * r = cmd("SMEMBERS e:%s", entity.c_str());
        if (r && r->type == REDIS_REPLY_ARRAY && r->elements > 0) {
            result.found = true;
            result.method = "entity";
            result.entities.push_back(entity);
            // Fetch actual sentence text for each location
            for (size_t i = 0; i < r->elements; i++) {
                if (r->element[i]->type == REDIS_REPLY_STRING) {
                    std::string loc = r->element[i]->str;  // "source:pos"
                    // Fetch sentence text: HGET sent:{source}:{pos} text
                    std::string sent_key = "sent:" + loc;
                    auto * sr = cmd("HGET %s text", sent_key.c_str());
                    if (sr && sr->type == REDIS_REPLY_STRING) {
                        result.sentences.push_back(sr->str);
                    } else {
                        result.sentences.push_back(loc);  // fallback to location
                    }
                    if (sr) freeReplyObject(sr);
                }
            }
        }
        if (r) freeReplyObject(r);
    }
    return result;
}

redis_query_result redis_index::try_graph_traversal(const std::string & q) {
    redis_query_result result;
    // Extract entities: capitalized names, file paths, compound codes
    std::vector<std::string> entities;
    // Capitalized multi-word names
    static const std::regex name_re("[A-Z][a-z]+(?:\\s+[A-Z][a-z]+)+");
    for (std::sregex_iterator it(q.begin(), q.end(), name_re); it != std::sregex_iterator(); ++it)
        entities.push_back(it->str());
    // File paths (word.ext)
    static const std::regex file_re("[a-zA-Z0-9_/]+\\.[a-z]{1,4}\\b");
    for (std::sregex_iterator it(q.begin(), q.end(), file_re); it != std::sregex_iterator(); ++it)
        entities.push_back(it->str());
    // Compound codes (WORD-DIGITS)
    static const std::regex code_re("[A-Z][A-Z0-9]+-[A-Z0-9]+");
    for (std::sregex_iterator it(q.begin(), q.end(), code_re); it != std::sregex_iterator(); ++it)
        entities.push_back(it->str());

    for (auto & entity : entities) {
        // Check relationships: r:{entity} -> all predicates
        auto * r = cmd("HGETALL r:%s", entity.c_str());
        if (r && r->type == REDIS_REPLY_ARRAY && r->elements >= 2) {
            result.found = true;
            result.method = "graph";
            for (size_t i = 0; i + 1 < r->elements; i += 2) {
                std::string pred = r->element[i]->str;
                std::string obj = r->element[i + 1]->str;
                result.sentences.push_back(entity + " " + pred + " " + obj);
                result.entities.push_back(obj);

                // Follow one hop: r:{object}
                auto * r2 = cmd("HGETALL r:%s", obj.c_str());
                if (r2 && r2->type == REDIS_REPLY_ARRAY) {
                    for (size_t j = 0; j + 1 < r2->elements; j += 2) {
                        result.sentences.push_back(
                            obj + " " + r2->element[j]->str + " " + r2->element[j + 1]->str);
                    }
                }
                if (r2) freeReplyObject(r2);
            }
        }
        if (r) freeReplyObject(r);
    }
    return result;
}

redis_query_result redis_index::try_aggregation(const std::string & q) {
    redis_query_result result;
    // Look for code prefixes in query (COBALT, OPAL, etc.)
    static const std::regex prefix_re("\\b([A-Z][A-Z0-9]{2,})\\b");
    std::sregex_iterator it(q.begin(), q.end(), prefix_re);
    for (; it != std::sregex_iterator(); ++it) {
        auto codes = get_codes(it->str());
        if (!codes.empty()) {
            result.found = true;
            result.method = "aggregation";
            for (auto & c : codes) {
                result.sentences.push_back("Code: " + c);
                result.entities.push_back(c);
            }
        }
    }
    return result;
}

// Build a RediSearch OR-union query from a natural-language question.
// Passed verbatim, the question is AND-intersected by RediSearch (all content terms must
// co-occur in one sentence), which matches almost nothing for natural-language questions.
// Union semantics `(t1|t2|...)` match any term, and the TFIDF scorer still ranks sentences
// that match more terms higher.
static std::string build_ft_or_query(const std::string & q) {
    // Minimal stopword set; RediSearch drops its own indexed stopwords, but excluding
    // these from the OR keeps high-frequency function words from flooding the union.
    static const std::set<std::string> stop = {
        "the","a","an","of","to","in","on","for","and","or","is","are","was","were","be",
        "this","that","which","who","whom","whose","what","how","when","where","why","does",
        "do","did","with","at","by","from","as","it","its","you","your","i","we","they",
        "he","she","his","her","their","our","me","my","us","can","could","would","should",
        "will","shall","may","might","has","have","had","not","no","if","then","than","so",
        "there","here","about","into","over","under","between"
    };
    std::string term, out;
    auto flush = [&]() {
        if (term.size() >= 2 && !stop.count(term)) {
            if (!out.empty()) out += '|';
            out += term;
        }
        term.clear();
    };
    for (char c : q) {
        if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) {
            term += c;
        } else if (c >= 'A' && c <= 'Z') {
            term += (char)(c - 'A' + 'a');   // lowercase; also strips RediSearch special chars
        } else {
            flush();
        }
    }
    flush();
    if (out.empty()) return "";
    return "(" + out + ")";
}

redis_query_result redis_index::ft_search(const std::string & q, int limit) {
    redis_query_result result;
    auto t0 = std::chrono::steady_clock::now();

    // Rewrite the NL question into a sanitized OR-union query (see build_ft_or_query).
    std::string ftq = build_ft_or_query(q);
    if (ftq.empty()) {
        result.latency_ms = 0.0;
        return result;   // nothing searchable -> let caller fall back
    }

    // RediSearch full-text search on sentences (BM25/TFIDF-ranked union). Match at sentence
    // granularity but inject the parent `record` (deduplicated) so the answer field is present
    // even when it is in a different sentence of the same record.
    std::unordered_set<std::string> seen_records;
    auto * r = cmd("FT.SEARCH sentences %s LIMIT 0 %d", ftq.c_str(), limit);
    if (r && r->type == REDIS_REPLY_ARRAY && r->elements > 1) {
        result.found = true;
        result.method = "ftsearch_or";
        // FT.SEARCH returns: [total, key1, [field1, val1, ...], key2, ...]
        for (size_t i = 1; i < r->elements; i += 2) {
            if (i + 1 < r->elements && r->element[i + 1]->type == REDIS_REPLY_ARRAY) {
                auto * fields = r->element[i + 1];
                const char * txt = nullptr;
                const char * rec = nullptr;
                for (size_t j = 0; j + 1 < fields->elements; j += 2) {
                    if (strcmp(fields->element[j]->str, "text") == 0)   txt = fields->element[j + 1]->str;
                    if (strcmp(fields->element[j]->str, "record") == 0) rec = fields->element[j + 1]->str;
                }
                // Prefer the whole parent record; dedup so one record injects once.
                const char * inject = rec ? rec : txt;
                if (inject && seen_records.insert(inject).second) {
                    result.sentences.push_back(inject);
                }
            }
        }
    }
    if (r) freeReplyObject(r);

    auto t1 = std::chrono::steady_clock::now();
    result.latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return result;
}

redis_query_result redis_index::query(const std::string & q) {
    // Run all structured queries and merge the results rather than short-circuiting:
    // multi-hop questions need both entity lookup and graph traversal.
    redis_query_result combined;

    auto entity_r = try_entity_lookup(q);
    auto graph_r = try_graph_traversal(q);
    auto agg_r = try_aggregation(q);

    // Merge: graph > aggregation > entity > ftsearch
    if (graph_r.found && !graph_r.sentences.empty()) {
        combined = graph_r;
        // Also add entity sentences for richer context
        for (auto & s : entity_r.sentences) combined.sentences.push_back(s);
        for (auto & e : entity_r.entities) combined.entities.push_back(e);
    } else if (agg_r.found) {
        combined = agg_r;
    } else if (entity_r.found) {
        combined = entity_r;
    }

    if (combined.found) return combined;

    return ft_search(q);
}
