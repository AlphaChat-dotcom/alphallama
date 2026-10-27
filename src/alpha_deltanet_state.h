#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// DeltaNet pre-scan state file format:
// Header: magic(4) + version(4) + n_layers(4) + state_size_per_layer(4) + tokens_scanned(8)
// Data:   n_layers x state_size_per_layer floats (contiguous)

#define ALPHA_DNST_MAGIC 0x54534E44  // "DNST"
#define ALPHA_DNST_VERSION 1

struct alpha_deltanet_state_header {
    uint32_t magic;
    uint32_t version;
    uint32_t n_layers;           // number of recurrent layers
    uint32_t state_size;         // floats per layer (n_embd_s)
    uint64_t tokens_scanned;     // total tokens processed into this state
};

// In-memory snapshot of recurrent state at a corpus position
struct alpha_deltanet_snapshot {
    uint64_t chunk_pos;          // corpus position (chunk index)
    uint64_t tokens_scanned;     // total tokens at this point
    std::vector<float> state;    // n_recurrent * n_embd_s floats
};

// Configuration for sharded prescan
struct alpha_deltanet_prescan_config {
    uint32_t snapshot_interval = 100000;   // chunks between in-memory snapshots
    uint32_t shard_size        = 1000000;  // chunks per shard
    float    shard_overlap     = 0.5f;     // overlap fraction between adjacent shards
    std::string output_dir;                // where to write final shard states
    std::string progress_path;             // crash-recovery file (overwritten, not numbered)
    bool     verbose           = false;    // show progress logs
};

// Simple mode (backward compat)

// Pre-scan: iterate chunk files, prefill each through model, save final state
void alpha_deltanet_prescan(
    struct llama_context * ctx,
    const std::string & chunks_dir,
    const std::string & state_path);

// Save recurrent state from model memory to file
bool alpha_deltanet_state_save(
    const std::string & path,
    struct llama_context * ctx,
    uint64_t tokens_scanned);

// Load pre-scanned state from file into model memory
// Returns tokens_scanned on success, 0 on failure
uint64_t alpha_deltanet_state_load(
    const std::string & path,
    struct llama_context * ctx);

// Sharded mode (anti-leak)

// Pre-scan with in-memory snapshot ring + randomized progress save + batch shard output.
// Snapshots every snapshot_interval chunks (memory only, never disk).
// Progress save at random intervals (single overwriting file).
// Final output: N shard states written simultaneously.
void alpha_deltanet_prescan_sharded(
    struct llama_context * ctx,
    const std::string & input_path,
    const alpha_deltanet_prescan_config & cfg);

// Load multiple shard states from a directory, blend and inject into model
// Returns total tokens across all shards on success, 0 on failure
uint64_t alpha_deltanet_shards_load(
    const std::string & shard_dir,
    struct llama_context * ctx);
