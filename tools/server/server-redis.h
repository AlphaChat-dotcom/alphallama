#pragma once

// Redis integration for AlphaLlama retrieval.
// Replaces BM25 with RediSearch full-text, entity graph, and sentence index.
//
// Architecture:
//   Ingest: NER extraction -> entities/relations/sentences -> Redis pipeline
//   Query:  entity lookup -> graph traversal -> aggregation -> FT.SEARCH
//
// Requires: Redis Stack (RediSearch module) on --redis host:port
// Fallback: in-process BM25 when Redis not available

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <utility>

// Forward declare hiredis types (linked at runtime)
struct redisContext;
struct redisReply;

// Entity/relationship types for Redis graph storage
struct redis_entity {
    std::string name;
    std::string type;       // FILE, FUNC, CLASS, PERSON, DATE, etc.
    std::string source;     // file path
    int line = 0;
};

struct redis_relationship {
    std::string subject;
    std::string predicate;  // imports, extends, born_in, employed_at, etc.
    std::string object;
};

struct redis_sentence {
    std::string text;
    std::string source;     // file path
    int position = 0;       // sentence index within file
};

// Query result (unified across all retrieval methods)
struct redis_query_result {
    bool found = false;
    std::string method;     // "entity", "graph", "aggregation", "ftsearch"
    std::vector<std::string> sentences;  // retrieved sentences
    std::vector<std::string> entities;   // matched entities
    double latency_ms = 0;
};

// Structured-record analytics (FT.AGGREGATE).
// Retrieve-then-inject only sees the top-k chunks, so it cannot sum or group-by a whole
// corpus. These run the reduction in Redis over all records and return a small table that
// fits the injection budget. Works for any tabular dataset; fields are declared per index
// at ingest.

// One numeric range or tag-equality filter. When `hi` is empty the field is treated
// as a TAG and matched for equality against `lo`; otherwise it is a NUMERIC range [lo,hi].
struct redis_agg_filter {
    std::string field;
    std::string lo;
    std::string hi;
};

struct redis_agg_row {
    std::string group;   // group-by key value
    double      value;   // reduced metric
};

struct redis_agg_result {
    bool        ok = false;
    std::string error;         // non-empty on validation/exec failure; no query runs on a validation error
    std::string metric;        // sum|avg|count|min|max
    std::string field;         // numeric field reduced (empty for count)
    std::string group_by;
    std::vector<redis_agg_row> rows;
    double      latency_ms = 0;
};

class redis_index {
public:
    redis_index() = default;
    ~redis_index();

    // Connect to Redis server. Returns false if connection fails.
    bool connect(const std::string & host, int port = 6379);
    bool is_connected() const { return ctx != nullptr; }

    // Create RediSearch indexes (call once at startup)
    bool create_indexes();

    // Ingest a file: extract entities, split sentences, store in Redis
    int ingest_file(const std::string & filepath, const std::string & text,
                    const std::vector<redis_entity> & entities,
                    const std::vector<redis_relationship> & relationships,
                    const std::vector<std::string> & imports);

    // Ingest a single sentence
    void add_sentence(const std::string & source, int pos, const std::string & text,
                      const std::string & record = "");

    // Add entity to graph
    void add_entity(const std::string & name, const std::string & type,
                    const std::string & source, int line);

    // Add relationship
    void add_relationship(const std::string & subject, const std::string & predicate,
                          const std::string & object);

    // Add code pattern (for aggregation: COBALT-742 -> codes:COBALT set)
    void add_code_pattern(const std::string & code, const std::string & source);

    // Add import relationship (for DependEval)
    void add_import(const std::string & file, const std::string & imported);

    // Add function/class to per-file index
    void add_func(const std::string & file, const std::string & func_name);
    void add_class(const std::string & file, const std::string & class_name);

    // Flush pipeline (call after batch ingest)
    void flush();

    // Unified query: tries entity -> graph -> aggregation -> FT.SEARCH
    redis_query_result query(const std::string & q);

    // Individual query methods
    redis_query_result try_entity_lookup(const std::string & q);
    redis_query_result try_graph_traversal(const std::string & q);
    redis_query_result try_aggregation(const std::string & q);
    redis_query_result ft_search(const std::string & q, int limit = 50);

    // DependEval: get full import chain for a file
    std::vector<std::string> get_imports(const std::string & file);
    std::vector<std::string> get_imported_by(const std::string & file);

    // Aggregation: get all codes matching a prefix
    std::vector<std::string> get_codes(const std::string & prefix);

    // Create a structured-record FT index. `tag_fields` become GROUPBY/equality
    // dimensions (TAG); `numeric_fields` become reducible + range-filterable
    // (NUMERIC SORTABLE). Records the schema so aggregate() can reject unknown
    // (hallucinated) field names. Keys are `{index}:{id}`. Returns false on error.
    bool create_record_index(const std::string & index,
                             const std::vector<std::string> & tag_fields,
                             const std::vector<std::string> & numeric_fields);

    // HSET one structured record into `index` (key `{index}:{id}`).
    void add_record(const std::string & index, const std::string & id,
                    const std::vector<std::pair<std::string, std::string>> & fields);

    // FT.AGGREGATE over all records in `index`:
    //   GROUPBY @group_by REDUCE <metric>(@field)  [+ numeric/tag FILTERs]
    // metric is one of sum, avg, count, min, max. metric, field, group_by and every filter
    // field are validated against the index schema; on any mismatch returns
    // {ok=false, error=...} without running a query.
    redis_agg_result aggregate(const std::string & index,
                               const std::string & metric,
                               const std::string & field,
                               const std::string & group_by,
                               const std::vector<redis_agg_filter> & filters);

    // Get source file for a code pattern
    std::string get_code_source(const std::string & code);

    // Stats
    int n_sentences = 0;
    int n_entities = 0;
    int n_relationships = 0;
    int n_codes = 0;

private:
    redisContext * ctx = nullptr;
    bool pipeline_active = false;
    int pipeline_count = 0;

    // Per-index declared schema for structured-record analytics (field-name whitelist).
    struct record_schema { std::unordered_set<std::string> tags; std::unordered_set<std::string> numerics; };
    std::unordered_map<std::string, record_schema> record_schemas;

    // Execute a Redis command, return reply (caller must free)
    redisReply * cmd(const char * fmt, ...);

    // Execute a command built from a pre-tokenized argv (safe for arbitrary/validated
    // arg counts; used by aggregate()). Caller must free the reply.
    redisReply * cmd_argv(const std::vector<std::string> & args);

    // Pipeline helpers
    void pipeline_start();
    void pipeline_append(const char * fmt, ...);
    void pipeline_flush();
};

// end of server-redis.h
