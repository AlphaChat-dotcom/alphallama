#include "alpha_deltanet_state.h"
#include "alpha_crypto.h"
#include "llama.h"
#include "llama-impl.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-memory-recurrent.h"
#include "llama-memory-hybrid.h"
#include "llama-hparams.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <filesystem>
#ifndef _WIN32
#include <unistd.h>
#endif

// Get the recurrent memory from context (handles both pure recurrent and hybrid)
static llama_memory_recurrent * get_recurrent_memory(struct llama_context * ctx) {
    llama_memory_i * memory = llama_get_memory(ctx);
    if (!memory) return nullptr;

    // Try direct cast to recurrent
    auto * recurrent = dynamic_cast<llama_memory_recurrent *>(memory);
    if (recurrent) return recurrent;

    // Try hybrid (has recurrent component)
    auto * hybrid = dynamic_cast<llama_memory_hybrid *>(memory);
    if (hybrid) {
        return hybrid->get_mem_recr();
    }

    return nullptr;
}

// Flush a token batch through the model
static void flush_batch(struct llama_context * ctx, std::vector<llama_token> & batch_tokens, uint64_t & total_tokens) {
    if (batch_tokens.empty()) return;

    llama_batch batch = llama_batch_init(batch_tokens.size(), 0, 1);
    for (size_t t = 0; t < batch_tokens.size(); ++t) {
        batch.token[t] = batch_tokens[t];
        batch.pos[t] = total_tokens + t;
        batch.n_seq_id[t] = 1;
        batch.seq_id[t][0] = 0;
        batch.logits[t] = (t == batch_tokens.size() - 1) ? 1 : 0;
    }
    batch.n_tokens = batch_tokens.size();

    int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);

    if (rc != 0) {
        LLAMA_LOG_WARN("prescan: decode failed (rc=%d)\n", rc);
    }
    total_tokens += batch_tokens.size();

    // Clear KV cache but keep recurrent state
    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
    batch_tokens.clear();
}

void alpha_deltanet_prescan(
    struct llama_context * ctx,
    const std::string & input_path,
    const std::string & state_path)
{
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    const int32_t ctx_size = llama_n_ctx(ctx);
    const int32_t max_batch = std::min(ctx_size, (int32_t)1024);
    uint64_t total_tokens = 0;
    auto t0 = ggml_time_ms();

    // Detect if input is a file or directory
    if (!std::filesystem::exists(input_path)) {
        LLAMA_LOG_ERROR("%s: path not found: %s\n", __func__, input_path.c_str());
        return;
    }
    bool is_dir = std::filesystem::is_directory(input_path);

    std::vector<llama_token> batch_tokens;
    batch_tokens.reserve(max_batch);
    size_t n_batches = 0;

    // Convolutional checkpoints: save state periodically during prescan.
    // Each checkpoint captures recurrent state at that point, the freshest
    // memory of recently scanned chunks. At query time, load the checkpoint
    // nearest to the BM25-found chunk for best recall.
    // Interval: 150K tokens (~1K digest chunks at ~150 tok/chunk).
    // Gives fine-grained checkpoints for per-chunk oracle queries.
    const uint64_t checkpoint_interval = 150000;  // tokens between checkpoints
    uint64_t tokens_since_checkpoint = 0;
    int checkpoint_id = 0;
    std::string checkpoint_dir;
    if (!state_path.empty()) {
        // Create checkpoint directory: {state_path}.ckpt/
        checkpoint_dir = state_path + ".ckpt";
        std::filesystem::create_directories(checkpoint_dir);
    }

    auto save_checkpoint = [&]() {
        if (checkpoint_dir.empty()) return;
        // Flush any pending tokens first
        if (!batch_tokens.empty()) {
            flush_batch(ctx, batch_tokens, total_tokens);
            n_batches++;
        }
        char ckpt_path[512];
        snprintf(ckpt_path, sizeof(ckpt_path), "%s/ckpt_%04d.bin",
                 checkpoint_dir.c_str(), checkpoint_id);
        alpha_deltanet_state_save(ckpt_path, ctx, total_tokens);
        LLAMA_LOG_INFO("%s: checkpoint %d saved at %llu tokens → %s\n",
                      __func__, checkpoint_id, (unsigned long long)total_tokens, ckpt_path);
        checkpoint_id++;
        tokens_since_checkpoint = 0;
    };

    if (!is_dir) {
        // Streaming mode: read a large text file in blocks
        FILE * f = fopen(input_path.c_str(), "r");
        if (!f) {
            LLAMA_LOG_ERROR("%s: failed to open %s\n", __func__, input_path.c_str());
            return;
        }

        size_t file_size = (size_t)std::filesystem::file_size(input_path);
        LLAMA_LOG_INFO("%s: streaming prescan from %s (%.1f GB), checkpoint every %llu tokens\n",
                       __func__, input_path.c_str(), file_size / 1e9,
                       (unsigned long long)checkpoint_interval);

        char buf[8192];
        size_t bytes_read = 0;

        while (!feof(f)) {
            size_t n = fread(buf, 1, sizeof(buf) - 1, f);
            if (n == 0) break;
            buf[n] = 0;
            bytes_read += n;

            // Tokenize block
            std::vector<llama_token> tokens(n + 256);
            int n_tokens = llama_tokenize(vocab, buf, n, tokens.data(), tokens.size(), false, true);
            if (n_tokens <= 0) continue;
            tokens.resize(n_tokens);

            // Add to batch, flush when full
            for (int t = 0; t < n_tokens; ++t) {
                batch_tokens.push_back(tokens[t]);
                if ((int32_t)batch_tokens.size() >= max_batch) {
                    flush_batch(ctx, batch_tokens, total_tokens);
                    n_batches++;
                    tokens_since_checkpoint += max_batch;

                    // Save checkpoint at interval
                    if (tokens_since_checkpoint >= checkpoint_interval) {
                        save_checkpoint();
                    }
                }
            }

            // Progress every 100MB
            if (bytes_read % (100 * 1024 * 1024) < sizeof(buf)) {
                auto elapsed = (ggml_time_ms() - t0) / 1000.0;
                double rate = total_tokens / std::max(elapsed, 0.001);
                double pct = 100.0 * bytes_read / std::max(file_size, (size_t)1);
                double eta = (file_size - bytes_read) / std::max((double)bytes_read / elapsed, 0.001) / 60.0;
                LLAMA_LOG_INFO("%s: %.1f%% (%zu MB), %llu tokens, %.0f tok/s, ETA %.1f min, %d checkpoints\n",
                              __func__, pct, bytes_read / (1024*1024),
                              (unsigned long long)total_tokens, rate, eta, checkpoint_id);
            }
        }
        fclose(f);

    } else if (is_dir) {
        // Directory mode: list and process .txt files
        // For directories with many files, prefer streaming mode with a concatenated file
        std::vector<std::string> chunk_files;
        for (const auto & entry : std::filesystem::directory_iterator(input_path)) {
            if (entry.is_regular_file()) {
                std::string name = entry.path().filename().string();
                if (name.size() > 4 && name.substr(name.size() - 4) == ".txt") {
                    chunk_files.push_back(name);
                }
            }
        }
        std::sort(chunk_files.begin(), chunk_files.end());

        LLAMA_LOG_INFO("%s: directory prescan: %zu chunks from %s\n",
                       __func__, chunk_files.size(), input_path.c_str());

        for (size_t i = 0; i < chunk_files.size(); ++i) {
            std::string path = input_path + "/" + chunk_files[i];
            // Read the whole chunk; a fixed-size single fread would absorb only the first bytes of
            // each file and leave the state blind to most of the corpus. Corpus chunk files are
            // bounded (~16-64KB), so read once and tokenize once, which also avoids splitting a
            // token across a fixed-size block boundary.
            std::string text;
            {
                FILE * f = fopen(path.c_str(), "rb");
                if (!f) continue;
                fseek(f, 0, SEEK_END);
                long fsz = ftell(f);
                fseek(f, 0, SEEK_SET);
                if (fsz <= 0) { fclose(f); continue; }
                text.resize((size_t)fsz);
                size_t got = fread(&text[0], 1, (size_t)fsz, f);
                fclose(f);
                text.resize(got);
                if (text.empty()) continue;
            }

            std::vector<llama_token> tokens(text.size() + 256);
            int n_tokens = llama_tokenize(vocab, text.data(), (int32_t)text.size(),
                                          tokens.data(), (int32_t)tokens.size(), false, true);
            if (n_tokens <= 0) continue;
            tokens.resize(n_tokens);

            for (int t = 0; t < n_tokens; ++t) {
                batch_tokens.push_back(tokens[t]);
                if ((int32_t)batch_tokens.size() >= max_batch) {
                    flush_batch(ctx, batch_tokens, total_tokens);
                    n_batches++;
                }
            }

            if ((i + 1) % 1000 == 0) {
                auto elapsed = (ggml_time_ms() - t0) / 1000.0;
                double rate = (i + 1) / std::max(elapsed, 0.001);
                LLAMA_LOG_INFO("%s: %zu/%zu chunks, %llu tokens, %.0f chunks/s\n",
                              __func__, i + 1, chunk_files.size(),
                              (unsigned long long)total_tokens, rate);
            }
        }
    }

    // Flush remaining
    flush_batch(ctx, batch_tokens, total_tokens);

    // Save final checkpoint
    save_checkpoint();

    auto elapsed = (ggml_time_ms() - t0) / 1000.0;
    double tok_per_s = total_tokens / std::max(elapsed, 0.001);
    LLAMA_LOG_INFO("%s: complete: %llu tokens in %.0fs (%.0f tok/s, %zu batches, %d checkpoints)\n",
                   __func__, (unsigned long long)total_tokens, elapsed, tok_per_s, n_batches, checkpoint_id);

    // Also save the final combined state
    if (!state_path.empty()) {
        alpha_deltanet_state_save(state_path, ctx, total_tokens);
    }
}

bool alpha_deltanet_state_save(
    const std::string & path,
    struct llama_context * ctx,
    uint64_t tokens_scanned)
{
    auto * recurrent = get_recurrent_memory(ctx);
    if (!recurrent) {
        LLAMA_LOG_ERROR("%s: model has no recurrent memory\n", __func__);
        return false;
    }

    const auto & hparams = llama_get_model(ctx)->hparams;
    const uint32_t n_layer = hparams.n_layer;
    const uint32_t n_embd_s = hparams.n_embd_s();

    // Count recurrent layers
    uint32_t n_recurrent = 0;
    for (uint32_t il = 0; il < n_layer; ++il) {
        if (hparams.is_recurrent(il)) {
            n_recurrent++;
        }
    }

    if (n_recurrent == 0) {
        LLAMA_LOG_ERROR("%s: no recurrent layers found\n", __func__);
        return false;
    }

    // Build DNST payload in memory
    alpha_deltanet_state_header hdr = {};
    hdr.magic = ALPHA_DNST_MAGIC;
    hdr.version = ALPHA_DNST_VERSION;
    hdr.n_layers = n_recurrent;
    hdr.state_size = n_embd_s;
    hdr.tokens_scanned = tokens_scanned;

    size_t payload_size = sizeof(hdr) + (size_t)n_recurrent * n_embd_s * sizeof(float);
    std::vector<uint8_t> payload(payload_size);
    memcpy(payload.data(), &hdr, sizeof(hdr));

    LLAMA_LOG_INFO("%s: saving state: %u layers, %u floats/layer, %llu tokens scanned\n",
                   __func__, n_recurrent, n_embd_s, (unsigned long long)tokens_scanned);

    size_t offset = sizeof(hdr);
    std::vector<float> buf(n_embd_s);

    for (uint32_t il = 0; il < n_layer; ++il) {
        if (!hparams.is_recurrent(il)) continue;

        ggml_tensor * s = recurrent->s_l[il];
        if (!s) {
            LLAMA_LOG_ERROR("%s: s_l[%d] is null\n", __func__, il);
            memset(payload.data() + offset, 0, n_embd_s * sizeof(float));
        } else {
            ggml_backend_tensor_get(s, payload.data() + offset, 0, n_embd_s * sizeof(float));
        }
        offset += n_embd_s * sizeof(float);
    }

    // Encrypt and write; no unencrypted fallback in production
    if (alpha_acx_encrypt_file(path, payload.data(), payload.size())) {
        LLAMA_LOG_INFO("%s: saved %s (%.1f MB, %u layers)\n",
                       __func__, path.c_str(), payload_size / 1e6, n_recurrent);
    } else {
#ifdef ALPHA_INTERNAL
        // Dev builds: fall back to raw write for debugging
        FILE * f = fopen(path.c_str(), "wb");
        if (!f) {
            LLAMA_LOG_ERROR("%s: failed to open %s for writing\n", __func__, path.c_str());
            return false;
        }
        fwrite(payload.data(), 1, payload.size(), f);
        fclose(f);
        LLAMA_LOG_INFO("%s: saved %s (%.1f MB, %u layers, dev-unencrypted)\n",
                       __func__, path.c_str(), payload_size / 1e6, n_recurrent);
#else
        LLAMA_LOG_ERROR("%s: encryption unavailable — cannot save state\n", __func__);
        return false;
#endif
    }
    return true;
}

// Parse a DNST payload (already decrypted or raw) and inject into recurrent memory.
// Returns tokens_scanned on success, 0 on failure.
static uint64_t inject_dnst_payload(
    const uint8_t * data, size_t data_len,
    struct llama_context * ctx, const std::string & path_for_log)
{
    auto * recurrent = get_recurrent_memory(ctx);
    if (!recurrent) {
        LLAMA_LOG_ERROR("%s: model has no recurrent memory\n", __func__);
        return 0;
    }

    const auto & hparams = llama_get_model(ctx)->hparams;
    const uint32_t n_layer = hparams.n_layer;
    const uint32_t n_embd_s = hparams.n_embd_s();

    if (data_len < sizeof(alpha_deltanet_state_header)) {
        LLAMA_LOG_ERROR("%s: data too small for header\n", __func__);
        return 0;
    }

    alpha_deltanet_state_header hdr;
    memcpy(&hdr, data, sizeof(hdr));

    if (hdr.magic != ALPHA_DNST_MAGIC) {
        LLAMA_LOG_ERROR("%s: invalid magic: 0x%08x (expected 0x%08x)\n",
                       __func__, hdr.magic, ALPHA_DNST_MAGIC);
        return 0;
    }
    if (hdr.version != ALPHA_DNST_VERSION) {
        LLAMA_LOG_ERROR("%s: unsupported version: %u\n", __func__, hdr.version);
        return 0;
    }
    if (hdr.state_size != n_embd_s) {
        LLAMA_LOG_ERROR("%s: state size mismatch: file=%u, model=%u\n",
                       __func__, hdr.state_size, n_embd_s);
        return 0;
    }

    uint32_t n_recurrent = 0;
    for (uint32_t il = 0; il < n_layer; ++il) {
        if (hparams.is_recurrent(il)) n_recurrent++;
    }
    if (hdr.n_layers != n_recurrent) {
        LLAMA_LOG_ERROR("%s: layer count mismatch: file=%u, model=%u\n",
                       __func__, hdr.n_layers, n_recurrent);
        return 0;
    }

    size_t expected = sizeof(hdr) + (size_t)n_recurrent * n_embd_s * sizeof(float);
    if (data_len < expected) {
        LLAMA_LOG_ERROR("%s: data too small: %zu < %zu\n", __func__, data_len, expected);
        return 0;
    }

    LLAMA_LOG_INFO("%s: loading state: %u layers, %u floats/layer, %llu tokens scanned\n",
                   __func__, hdr.n_layers, hdr.state_size, (unsigned long long)hdr.tokens_scanned);

    // Install the payload as the recurrent state prior, not just as a write into cell 0.
    //
    // Writing cell 0 alone does not reach inference, for two reasons:
    //   1. s_l[il] is [n_embd_s, mem_size] (one column per cell), so offset 0 primes only cell 0.
    //      With -np>1 no other slot would see the state.
    //   2. The server's slot reset calls llama_memory_seq_rm(mem, id, -1, -1), which invalidates the
    //      sequence tail; the next decode then seeds the sequence from cell rs_z (zeros), discarding
    //      whatever was in cell 0.
    // The prior fixes both: find_slot() primes rs_z, so every fresh sequence on every slot starts
    // from this state, and it survives any number of slot resets.
    // The cell-0 write is kept as well so a context used without going through find_slot still sees
    // the state.
    const uint8_t * ptr = data + sizeof(hdr);
    std::vector<std::vector<uint8_t>> prior(n_layer);
    for (uint32_t il = 0; il < n_layer; ++il) {
        if (!hparams.is_recurrent(il)) continue;

        ggml_tensor * s = recurrent->s_l[il];
        if (s) {
            ggml_backend_tensor_set(s, ptr, 0, n_embd_s * sizeof(float));
            prior[il].assign(ptr, ptr + n_embd_s * sizeof(float));
        }
        ptr += n_embd_s * sizeof(float);
    }
    recurrent->set_state_prior(prior);
    if (!recurrent->has_state_prior()) {
        // Every layer was rejected on row-size grounds -> the state does not match this model.
        // Fail loudly: silently continuing would look like "pre-scan does not help".
        LLAMA_LOG_ERROR("%s: state prior rejected (row size mismatch) — state will NOT reach inference\n",
                        __func__);
        return 0;
    }

    LLAMA_LOG_INFO("%s: loaded %s (%.1f MB, %llu tokens)\n",
                   __func__, path_for_log.c_str(), expected / 1e6,
                   (unsigned long long)hdr.tokens_scanned);
    return hdr.tokens_scanned;
}

uint64_t alpha_deltanet_state_load(
    const std::string & path,
    struct llama_context * ctx)
{
    // Try .acx (encrypted) first
    auto decrypted = alpha_acx_decrypt_file(path);
    if (!decrypted.empty()) {
        return inject_dnst_payload(decrypted.data(), decrypted.size(), ctx, path);
    }

#ifdef ALPHA_INTERNAL
    // Dev builds: fall back to raw unencrypted file
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        LLAMA_LOG_ERROR("%s: failed to open %s\n", __func__, path.c_str());
        return 0;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size <= 0) { fclose(f); return 0; }

    std::vector<uint8_t> raw(size);
    size_t n = fread(raw.data(), 1, size, f);
    fclose(f);
    if ((long)n != size) return 0;

    return inject_dnst_payload(raw.data(), raw.size(), ctx, path);
#else
    LLAMA_LOG_ERROR("%s: failed to load %s (encrypted state required)\n", __func__, path.c_str());
    return 0;
#endif
}

// Sharded prescan: in-memory snapshots + randomized progress append + batch shard output

// Snapshot current recurrent state from GPU into a CPU buffer
static alpha_deltanet_snapshot snapshot_state_to_memory(
    struct llama_context * ctx, uint64_t chunk_pos, uint64_t tokens_scanned)
{
    alpha_deltanet_snapshot snap;
    snap.chunk_pos = chunk_pos;
    snap.tokens_scanned = tokens_scanned;

    auto * recurrent = get_recurrent_memory(ctx);
    if (!recurrent) return snap;

    const auto & hparams = llama_get_model(ctx)->hparams;
    const uint32_t n_layer = hparams.n_layer;
    const uint32_t n_embd_s = hparams.n_embd_s();

    uint32_t n_recurrent = 0;
    for (uint32_t il = 0; il < n_layer; ++il) {
        if (hparams.is_recurrent(il)) n_recurrent++;
    }

    snap.state.resize((size_t)n_recurrent * n_embd_s);
    size_t offset = 0;
    for (uint32_t il = 0; il < n_layer; ++il) {
        if (!hparams.is_recurrent(il)) continue;
        ggml_tensor * s = recurrent->s_l[il];
        if (s) {
            ggml_backend_tensor_get(s, snap.state.data() + offset, 0, n_embd_s * sizeof(float));
        } else {
            memset(snap.state.data() + offset, 0, n_embd_s * sizeof(float));
        }
        offset += n_embd_s;
    }
    return snap;
}

// Append a snapshot to the progress file (binary: chunk_pos + tokens + state floats)
static void append_snapshot_to_file(FILE * f, const alpha_deltanet_snapshot & snap) {
    fwrite(&snap.chunk_pos, sizeof(uint64_t), 1, f);
    fwrite(&snap.tokens_scanned, sizeof(uint64_t), 1, f);
    uint64_t n_floats = snap.state.size();
    fwrite(&n_floats, sizeof(uint64_t), 1, f);
    fwrite(snap.state.data(), sizeof(float), snap.state.size(), f);
    fflush(f);
}

// Read all snapshots from progress file
static std::vector<alpha_deltanet_snapshot> read_snapshots_from_file(const std::string & path) {
    std::vector<alpha_deltanet_snapshot> snaps;
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return snaps;

    while (true) {
        alpha_deltanet_snapshot snap;
        if (fread(&snap.chunk_pos, sizeof(uint64_t), 1, f) != 1) break;
        if (fread(&snap.tokens_scanned, sizeof(uint64_t), 1, f) != 1) break;
        uint64_t n_floats = 0;
        if (fread(&n_floats, sizeof(uint64_t), 1, f) != 1) break;
        snap.state.resize(n_floats);
        if (fread(snap.state.data(), sizeof(float), n_floats, f) != n_floats) break;
        snaps.push_back(std::move(snap));
    }
    fclose(f);
    return snaps;
}

// Blend snapshots within a chunk range into a single state (element-wise average)
static std::vector<float> blend_snapshots(
    const std::vector<alpha_deltanet_snapshot> & snaps,
    uint64_t range_start, uint64_t range_end)
{
    std::vector<float> blended;
    int count = 0;
    for (const auto & s : snaps) {
        if (s.chunk_pos >= range_start && s.chunk_pos <= range_end) {
            if (blended.empty()) {
                blended = s.state;
            } else {
                for (size_t i = 0; i < blended.size() && i < s.state.size(); ++i) {
                    blended[i] += s.state[i];
                }
            }
            count++;
        }
    }
    if (count > 1) {
        float inv = 1.0f / count;
        for (auto & v : blended) v *= inv;
    }
    return blended;
}

// Write a single shard state file from pre-blended floats
static bool write_shard_file(
    const std::string & path, const std::vector<float> & state,
    uint32_t n_recurrent, uint32_t n_embd_s, uint64_t tokens_scanned)
{
    // Build payload: header + state data
    alpha_deltanet_state_header hdr = {};
    hdr.magic = ALPHA_DNST_MAGIC;
    hdr.version = ALPHA_DNST_VERSION;
    hdr.n_layers = n_recurrent;
    hdr.state_size = n_embd_s;
    hdr.tokens_scanned = tokens_scanned;

    std::vector<uint8_t> payload(sizeof(hdr) + state.size() * sizeof(float));
    memcpy(payload.data(), &hdr, sizeof(hdr));
    memcpy(payload.data() + sizeof(hdr), state.data(), state.size() * sizeof(float));

    // Encrypt and write
    if (alpha_acx_encrypt_file(path, payload.data(), payload.size())) {
        return true;
    }
#ifdef ALPHA_INTERNAL
    // Dev fallback: raw write
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) return false;
    fwrite(payload.data(), 1, payload.size(), f);
    fclose(f);
    return true;
#else
    LLAMA_LOG_ERROR("%s: encryption unavailable — cannot save shard\n", __func__);
    return false;
#endif
}

void alpha_deltanet_prescan_sharded(
    struct llama_context * ctx,
    const std::string & input_path,
    const alpha_deltanet_prescan_config & cfg)
{
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    const int32_t ctx_size = llama_n_ctx(ctx);
    const int32_t max_batch = std::min(ctx_size, (int32_t)1024);
    const auto & hparams = llama_get_model(ctx)->hparams;
    uint64_t total_tokens = 0;
    auto t0 = ggml_time_ms();

    // Count recurrent layers for shard output
    uint32_t n_recurrent = 0;
    for (uint32_t il = 0; il < hparams.n_layer; ++il) {
        if (hparams.is_recurrent(il)) n_recurrent++;
    }
    const uint32_t n_embd_s = hparams.n_embd_s();

    // Random interval generator for progress saves
    std::mt19937 rng(42 ^ (uint32_t)ggml_time_ms());
    std::uniform_int_distribution<uint32_t> save_dist(200000, 800000);
    uint64_t next_save_chunk = save_dist(rng);

    // In-memory snapshot buffer (holds 2-3 snapshots, ~190 MB max)
    std::vector<alpha_deltanet_snapshot> pending_snaps;

    // Open progress file for append (crash recovery + snapshot accumulation)
    std::string progress_path = cfg.progress_path.empty()
        ? cfg.output_dir + "/progress.bin" : cfg.progress_path;
    FILE * progress_f = fopen(progress_path.c_str(), "ab");
    if (!progress_f) {
        LLAMA_LOG_ERROR("%s: failed to open progress file: %s\n", __func__, progress_path.c_str());
        return;
    }

    // Check for resume: count existing snapshots
    uint64_t resume_chunk = 0;
    {
        auto existing = read_snapshots_from_file(progress_path);
        if (!existing.empty()) {
            resume_chunk = existing.back().chunk_pos;
            if (cfg.verbose) {
                LLAMA_LOG_INFO("%s: resuming from chunk %llu (%zu snapshots on disk)\n",
                               __func__, (unsigned long long)resume_chunk, existing.size());
            }
        }
    }

    // Detect input type
    if (!std::filesystem::exists(input_path)) {
        LLAMA_LOG_ERROR("%s: path not found: %s\n", __func__, input_path.c_str());
        fclose(progress_f);
        return;
    }

    std::vector<llama_token> batch_tokens;
    batch_tokens.reserve(max_batch);
    size_t n_batches = 0;
    size_t chunks_done = 0;
    size_t last_snapshot_chunk = 0;

    if (std::filesystem::is_directory(input_path)) {
        // Directory mode: process chunk files
        std::vector<std::string> chunk_files;
        for (const auto & entry : std::filesystem::directory_iterator(input_path)) {
            if (entry.is_regular_file()) {
                std::string name = entry.path().filename().string();
                if (name.size() > 4 && name.substr(name.size() - 4) == ".txt") {
                    chunk_files.push_back(name);
                }
            }
        }
        std::sort(chunk_files.begin(), chunk_files.end());

        if (cfg.verbose) {
            LLAMA_LOG_INFO("%s: sharded prescan: %zu chunks\n", __func__, chunk_files.size());
        }

        for (size_t i = 0; i < chunk_files.size(); ++i) {
            // Skip already-processed chunks (resume)
            if (i < resume_chunk) { chunks_done = i + 1; continue; }

            std::string path = input_path + "/" + chunk_files[i];
            // Read the whole chunk (see the same note in the directory prescan above).
            std::string text;
            {
                FILE * f = fopen(path.c_str(), "rb");
                if (!f) continue;
                fseek(f, 0, SEEK_END);
                long fsz = ftell(f);
                fseek(f, 0, SEEK_SET);
                if (fsz <= 0) { fclose(f); continue; }
                text.resize((size_t)fsz);
                size_t got = fread(&text[0], 1, (size_t)fsz, f);
                fclose(f);
                text.resize(got);
                if (text.empty()) continue;
            }

            std::vector<llama_token> tokens(text.size() + 256);
            int n_tokens = llama_tokenize(vocab, text.data(), (int32_t)text.size(),
                                          tokens.data(), (int32_t)tokens.size(), false, true);
            if (n_tokens <= 0) continue;
            tokens.resize(n_tokens);

            for (int t = 0; t < n_tokens; ++t) {
                batch_tokens.push_back(tokens[t]);
                if ((int32_t)batch_tokens.size() >= max_batch) {
                    flush_batch(ctx, batch_tokens, total_tokens);
                    n_batches++;
                }
            }
            chunks_done = i + 1;

            // In-memory snapshot every snapshot_interval chunks
            if (chunks_done - last_snapshot_chunk >= cfg.snapshot_interval) {
                pending_snaps.push_back(snapshot_state_to_memory(ctx, chunks_done, total_tokens));
                last_snapshot_chunk = chunks_done;
            }

            // Flush pending snapshots to disk at random intervals
            if (chunks_done >= next_save_chunk && !pending_snaps.empty()) {
                for (const auto & snap : pending_snaps) {
                    append_snapshot_to_file(progress_f, snap);
                }
                pending_snaps.clear();
                next_save_chunk = chunks_done + save_dist(rng);
            }

            if (cfg.verbose && chunks_done % 10000 == 0) {
                auto elapsed = (ggml_time_ms() - t0) / 1000.0;
                LLAMA_LOG_INFO("%s: %zu/%zu, %llu tok, %.0f tok/s\n",
                               __func__, chunks_done, chunk_files.size(),
                               (unsigned long long)total_tokens,
                               total_tokens / std::max(elapsed, 0.001));
            }
        }
    } else if (std::filesystem::is_regular_file(input_path)) {
        // Streaming mode: large text file, snapshot by token count
        FILE * f = fopen(input_path.c_str(), "r");
        if (!f) { fclose(progress_f); return; }

        size_t file_size = (size_t)std::filesystem::file_size(input_path);
        char buf[8192];
        size_t bytes_read = 0;
        // Approximate: 1 chunk ~ 400 bytes. Use token-based interval instead.
        uint64_t tokens_per_snapshot = (uint64_t)cfg.snapshot_interval * 120; // ~120 tok/chunk avg
        uint64_t last_snapshot_tok = 0;

        while (!feof(f)) {
            size_t n = fread(buf, 1, sizeof(buf) - 1, f);
            if (n == 0) break;
            buf[n] = 0;
            bytes_read += n;

            std::vector<llama_token> tokens(n + 256);
            int n_tokens = llama_tokenize(vocab, buf, n, tokens.data(), tokens.size(), false, true);
            if (n_tokens <= 0) continue;
            tokens.resize(n_tokens);

            for (int t = 0; t < n_tokens; ++t) {
                batch_tokens.push_back(tokens[t]);
                if ((int32_t)batch_tokens.size() >= max_batch) {
                    flush_batch(ctx, batch_tokens, total_tokens);
                    n_batches++;
                }
            }

            chunks_done = bytes_read / 400; // approximate chunk count

            if (total_tokens - last_snapshot_tok >= tokens_per_snapshot) {
                pending_snaps.push_back(snapshot_state_to_memory(ctx, chunks_done, total_tokens));
                last_snapshot_tok = total_tokens;
            }

            if (chunks_done >= next_save_chunk && !pending_snaps.empty()) {
                for (const auto & snap : pending_snaps) {
                    append_snapshot_to_file(progress_f, snap);
                }
                pending_snaps.clear();
                next_save_chunk = chunks_done + save_dist(rng);
            }
        }
        fclose(f);
    }

    // Flush remaining batch + final snapshot
    flush_batch(ctx, batch_tokens, total_tokens);
    pending_snaps.push_back(snapshot_state_to_memory(ctx, chunks_done, total_tokens));
    for (const auto & snap : pending_snaps) {
        append_snapshot_to_file(progress_f, snap);
    }
    pending_snaps.clear();
    fclose(progress_f);

    auto elapsed = (ggml_time_ms() - t0) / 1000.0;
    if (cfg.verbose) {
        LLAMA_LOG_INFO("%s: prescan complete: %llu tokens in %.0fs (%.0f tok/s)\n",
                       __func__, (unsigned long long)total_tokens, elapsed,
                       total_tokens / std::max(elapsed, 0.001));
    }

    // Compute and write shard states
    auto all_snaps = read_snapshots_from_file(progress_path);
    if (all_snaps.empty()) {
        LLAMA_LOG_ERROR("%s: no snapshots to build shards from\n", __func__);
        return;
    }

    uint64_t total_chunks = all_snaps.back().chunk_pos;
    uint32_t n_shards = std::max(1u, (uint32_t)std::ceil((double)total_chunks / cfg.shard_size));
    uint64_t step = total_chunks / n_shards;
    uint64_t overlap_chunks = (uint64_t)(step * cfg.shard_overlap);

    if (cfg.verbose) {
        LLAMA_LOG_INFO("%s: building %u shards from %zu snapshots (%llu total chunks)\n",
                       __func__, n_shards, all_snaps.size(), (unsigned long long)total_chunks);
    }

    // Create output directory
    std::filesystem::create_directories(cfg.output_dir);

    for (uint32_t si = 0; si < n_shards; ++si) {
        uint64_t shard_start = (si == 0) ? 0 : si * step - overlap_chunks;
        uint64_t shard_end = std::min((uint64_t)(si + 1) * step + overlap_chunks, total_chunks);

        auto blended = blend_snapshots(all_snaps, shard_start, shard_end);
        if (blended.empty()) continue;

        // tokens_scanned for this shard = tokens in the shard's range (not total)
        uint64_t shard_tokens = 0;
        for (const auto & s : all_snaps) {
            if (s.chunk_pos >= shard_start && s.chunk_pos <= shard_end) {
                shard_tokens = s.tokens_scanned; // use last matching snapshot's count
            }
        }

        char fname[64];
        snprintf(fname, sizeof(fname), "cache_%04u.acx", si);
        std::string shard_path = cfg.output_dir + "/" + fname;
        write_shard_file(shard_path, blended, n_recurrent, n_embd_s, shard_tokens);

        if (cfg.verbose) {
            LLAMA_LOG_INFO("%s: wrote %s (chunks %llu-%llu, %.1f MB)\n",
                           __func__, shard_path.c_str(),
                           (unsigned long long)shard_start, (unsigned long long)shard_end,
                           blended.size() * sizeof(float) / 1e6);
        }
    }

    // Clean up progress file
    unlink(progress_path.c_str());

    LLAMA_LOG_INFO("%s: %u shards written to %s\n", __func__, n_shards, cfg.output_dir.c_str());
}

uint64_t alpha_deltanet_shards_load(
    const std::string & shard_dir,
    struct llama_context * ctx)
{
    auto * recurrent = get_recurrent_memory(ctx);
    if (!recurrent) {
        LLAMA_LOG_ERROR("%s: model has no recurrent memory\n", __func__);
        return 0;
    }

    const auto & hparams = llama_get_model(ctx)->hparams;
    const uint32_t n_layer = hparams.n_layer;
    const uint32_t n_embd_s = hparams.n_embd_s();

    uint32_t n_recurrent = 0;
    for (uint32_t il = 0; il < n_layer; ++il) {
        if (hparams.is_recurrent(il)) n_recurrent++;
    }

    // Find shard files
    std::vector<std::string> shard_files;
    if (!std::filesystem::is_directory(shard_dir)) {
        LLAMA_LOG_ERROR("%s: failed to open shard dir: %s\n", __func__, shard_dir.c_str());
        return 0;
    }
    for (const auto & entry : std::filesystem::directory_iterator(shard_dir)) {
        if (entry.is_regular_file()) {
            std::string name = entry.path().filename().string();
            if (name.size() > 4 && name.substr(name.size() - 4) == ".acx") {
                shard_files.push_back(shard_dir + "/" + name);
            }
        }
    }
    std::sort(shard_files.begin(), shard_files.end());

    if (shard_files.empty()) {
        LLAMA_LOG_ERROR("%s: no cache files in %s\n", __func__, shard_dir.c_str());
        return 0;
    }

    // Load and average all shard states
    size_t state_floats = (size_t)n_recurrent * n_embd_s;
    std::vector<float> blended(state_floats, 0.0f);
    uint64_t total_tokens = 0;
    int n_loaded = 0;

    for (const auto & path : shard_files) {
        // Try decryption first
        auto decrypted = alpha_acx_decrypt_file(path);
        const uint8_t * data_ptr = nullptr;
        size_t data_size = 0;
        std::vector<uint8_t> raw_buf;

        if (!decrypted.empty()) {
            data_ptr = decrypted.data();
            data_size = decrypted.size();
        }
#ifdef ALPHA_INTERNAL
        else {
            // Dev fallback: try raw file
            FILE * f = fopen(path.c_str(), "rb");
            if (!f) continue;
            fseek(f, 0, SEEK_END);
            long fsize = ftell(f);
            fseek(f, 0, SEEK_SET);
            if (fsize <= 0) { fclose(f); continue; }
            raw_buf.resize(fsize);
            fread(raw_buf.data(), 1, fsize, f);
            fclose(f);
            data_ptr = raw_buf.data();
            data_size = raw_buf.size();
        }
#else
        else { continue; }
#endif

        if (data_size < sizeof(alpha_deltanet_state_header)) continue;

        alpha_deltanet_state_header hdr;
        memcpy(&hdr, data_ptr, sizeof(hdr));
        if (hdr.magic != ALPHA_DNST_MAGIC ||
            hdr.n_layers != n_recurrent ||
            hdr.state_size != n_embd_s) {
            continue;
        }

        size_t expected = sizeof(hdr) + state_floats * sizeof(float);
        if (data_size < expected) continue;

        const float * state_data = (const float *)(data_ptr + sizeof(hdr));
        for (size_t i = 0; i < state_floats; ++i) {
            blended[i] += state_data[i];
        }
        total_tokens = std::max(total_tokens, hdr.tokens_scanned);
        n_loaded++;
    }

    if (n_loaded == 0) {
        LLAMA_LOG_ERROR("%s: failed to load any shards\n", __func__);
        return 0;
    }

    // Average
    if (n_loaded > 1) {
        float inv = 1.0f / n_loaded;
        for (auto & v : blended) v *= inv;
    }

    // Inject into recurrent memory
    size_t offset = 0;
    for (uint32_t il = 0; il < n_layer; ++il) {
        if (!hparams.is_recurrent(il)) continue;
        ggml_tensor * s = recurrent->s_l[il];
        if (s) {
            ggml_backend_tensor_set(s, blended.data() + offset, 0, n_embd_s * sizeof(float));
        }
        offset += n_embd_s;
    }

    LLAMA_LOG_INFO("%s: loaded %d shards from %s (%.1f MB, %llu tokens)\n",
                   __func__, n_loaded, shard_dir.c_str(),
                   state_floats * sizeof(float) / 1e6,
                   (unsigned long long)total_tokens);
    return total_tokens;
}
