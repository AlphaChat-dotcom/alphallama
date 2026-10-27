// ALPHA_Q8_STAGE: see q8stage.cuh
#include "q8stage.cuh"

#include <cstdio>
#include <cstdlib>

#define ALPHA_Q8_STAGE_MAX 16

static alpha_q8_staged g_stage[ALPHA_Q8_STAGE_MAX];
static int             g_stage_n = 0;
static long long       g_hits = 0, g_staged = 0, g_declined = 0;

bool ggml_cuda_q8_stage_enabled() {
    static const bool v = []() {
        const char * e = getenv("ALPHA_Q8_STAGE");
        return e ? atoi(e) != 0 : false;     // default off
    }();
    return v;
}

// ALPHA_Q8_STAGE_HOST_LIFETIME: make the automatic per-pass reset a no-op and let the host own entry
// lifetime via ggml_cuda_q8_stage_clear().
//
// "One pass" means different things to different callers. For this engine a pass is a token, so
// resetting per pass is right. For a caller that issues many graph_compute calls per round (e.g. a
// native producer in one span and its consumers in the next), the automatic reset would drop entries
// between the producer and its consumer and every lookup would miss. A host that owns the lifetime can
// give entries a layer lifetime, which is shorter than per-pass, so this is strictly safer against the
// stale-address hazard.
static bool q8_stage_host_lifetime() {
    static const bool v = []() {
        const char * e = getenv("ALPHA_Q8_STAGE_HOST_LIFETIME");
        return e ? atoi(e) != 0 : false;
    }();
    return v;
}

void ggml_cuda_q8_stage_reset() {
    ggml_cuda_q8_stage_report();
    if (q8_stage_host_lifetime()) {
        return;
    }
    g_stage_n = 0;
}

void ggml_cuda_q8_stage_clear() {
    g_stage_n = 0;
}

void ggml_cuda_q8_stage_put(const alpha_q8_staged & e) {
    if (g_stage_n >= ALPHA_Q8_STAGE_MAX || e.src1_data == nullptr || e.q8 == nullptr) {
        return;
    }
    // An allocator that reuses addresses (ggml's graph allocator across layers, a bump arena reset per
    // layer) will hand a later activation the same pointer as an earlier one, with identical geometry
    // because every layer has the same shapes. A stale entry would then validate perfectly and the
    // consumer would read the previous layer's activation. So the newest registration for an address
    // always replaces the older one; never keep both.
    for (int i = 0; i < g_stage_n; ++i) {
        if (g_stage[i].src1_data == e.src1_data) {
            g_stage[i] = g_stage[--g_stage_n];
            break;
        }
    }
    g_stage[g_stage_n++] = e;
    g_staged++;
}

const char * ggml_cuda_q8_stage_get(const void * src1_data,
                                    int64_t ne10, int64_t ne11, int64_t ne12, int64_t ne13,
                                    int64_t s11, int64_t s12, int64_t s13) {
    for (int i = 0; i < g_stage_n; ++i) {
        const alpha_q8_staged & e = g_stage[i];
        if (e.src1_data != src1_data) {
            continue;
        }
        if (e.ne10 == ne10 && e.ne11 == ne11 && e.ne12 == ne12 && e.ne13 == ne13 &&
            e.s11 == s11 && e.s12 == s12 && e.s13 == s13) {
            return e.q8;
        }
        // A contiguous reshape of the staged tensor reads the identical q8_1 buffer: the layout is
        // [blocks_per_row][rows] packed, so only the row length and the total row count matter, not how
        // those rows are split across ne11/ne12/ne13. This is the common case: the expert MUL_MAT_ID
        // sees a reshape of the same normed activation the dense GEMVs see. Both sides must be
        // contiguous, or the packed layout assumption does not hold.
        const bool both_contiguous =
            e.s11 == e.ne10 && e.s12 == e.ne10 * e.ne11 && e.s13 == e.ne10 * e.ne11 * e.ne12 &&
            s11   == ne10   && s12   == ne10   * ne11   && s13   == ne10   * ne11   * ne12;
        if (e.ne10 == ne10 && both_contiguous &&
            e.ne11 * e.ne12 * e.ne13 == ne11 * ne12 * ne13) {
            return e.q8;
        }
        // An entry for this activation exists but the geometry differs. Falling back silently would be
        // indistinguishable from "no producer staged this" and hide a mis-keyed entry, so say which
        // field disagreed.
        g_declined++;
        static bool once = false;
        if (!once) {
            once = true;
            const char * f = e.ne10 != ne10 ? "ne10" : e.ne11 != ne11 ? "ne11" :
                             e.ne12 != ne12 ? "ne12" : e.ne13 != ne13 ? "ne13" :
                             e.s11  != s11  ? "s11"  : e.s12  != s12  ? "s12"  : "s13";
            const long long got = e.ne10 != ne10 ? (long long) e.ne10 : e.ne11 != ne11 ? (long long) e.ne11 :
                                  e.ne12 != ne12 ? (long long) e.ne12 : e.ne13 != ne13 ? (long long) e.ne13 :
                                  e.s11  != s11  ? (long long) e.s11  : e.s12  != s12  ? (long long) e.s12  :
                                                                        (long long) e.s13;
            const long long want = e.ne10 != ne10 ? (long long) ne10 : e.ne11 != ne11 ? (long long) ne11 :
                                   e.ne12 != ne12 ? (long long) ne12 : e.ne13 != ne13 ? (long long) ne13 :
                                   e.s11  != s11  ? (long long) s11  : e.s12  != s12  ? (long long) s12  :
                                                                       (long long) s13;
            fprintf(stderr, "[q8-stage] declined: %s %lld vs %lld\n", f, got, want);
        }
        return nullptr;
    }
    return nullptr;
}

void ggml_cuda_q8_stage_invalidate(const void * data, size_t nbytes, int node_idx) {
    if (data == nullptr || nbytes == 0) {
        return;
    }
    const char * a = (const char *) data;
    for (int i = 0; i < g_stage_n; ) {
        if (g_stage[i].produced_at == node_idx) {
            ++i;                    // the producer writing its own output
            continue;
        }
        const char * b = (const char *) g_stage[i].src1_data;
        // the staged copy corresponds to specific bytes; once those are overwritten it is stale
        const size_t bn = (size_t) g_stage[i].ne11 * g_stage[i].ne12 * g_stage[i].ne13 *
                          (size_t) g_stage[i].s11 * sizeof(float);
        if (a < b + bn && b < a + nbytes) {
            g_stage[i] = g_stage[--g_stage_n];
        } else {
            ++i;
        }
    }
}

void ggml_cuda_q8_stage_note_hit()    { g_hits++; }
void ggml_cuda_q8_stage_note_staged() { g_staged++; }

#define ALPHA_Q8_MISS_SHAPES 12
struct q8_miss_shape { int64_t ne[4]; long long n; };
static q8_miss_shape g_miss[ALPHA_Q8_MISS_SHAPES];
static int           g_miss_n = 0;

// Same census, but for producer sites reached, to tell "never fired" apart from "fired and did not help"
// (and an arena that silently filled from a plan that declined).
static q8_miss_shape g_seen[ALPHA_Q8_MISS_SHAPES];
static int           g_seen_n = 0;
static long long     g_arena_full = 0;

void ggml_cuda_q8_stage_note_arena_full() { g_arena_full++; }

struct q8_named { char name[48]; long long n; };
static q8_named g_hitters[ALPHA_Q8_MISS_SHAPES];
static int      g_hitters_n = 0;

void ggml_cuda_q8_stage_note_hit_named(const char * consumer) {
    if (consumer == nullptr) {
        return;
    }
    for (int i = 0; i < g_hitters_n; ++i) {
        bool same = true;
        for (int c = 0; c < 47; ++c) {
            if (g_hitters[i].name[c] != consumer[c]) { same = false; break; }
            if (consumer[c] == '\0') { break; }
        }
        if (same) {
            g_hitters[i].n++;
            return;
        }
    }
    if (g_hitters_n < ALPHA_Q8_MISS_SHAPES) {
        int c = 0;
        for (; c < 47 && consumer[c] != '\0'; ++c) {
            g_hitters[g_hitters_n].name[c] = consumer[c];
        }
        g_hitters[g_hitters_n].name[c] = '\0';
        g_hitters[g_hitters_n].n = 1;
        g_hitters_n++;
    }
}

void ggml_cuda_q8_stage_note_producer(int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    for (int i = 0; i < g_seen_n; ++i) {
        if (g_seen[i].ne[0] == ne0 && g_seen[i].ne[1] == ne1 &&
            g_seen[i].ne[2] == ne2 && g_seen[i].ne[3] == ne3) {
            g_seen[i].n++;
            return;
        }
    }
    if (g_seen_n < ALPHA_Q8_MISS_SHAPES) {
        g_seen[g_seen_n].ne[0] = ne0; g_seen[g_seen_n].ne[1] = ne1;
        g_seen[g_seen_n].ne[2] = ne2; g_seen[g_seen_n].ne[3] = ne3;
        g_seen[g_seen_n].n = 1;
        g_seen_n++;
    }
}

void ggml_cuda_q8_stage_note_miss(int64_t ne10, int64_t ne11, int64_t ne12, int64_t ne13) {
    for (int i = 0; i < g_miss_n; ++i) {
        if (g_miss[i].ne[0] == ne10 && g_miss[i].ne[1] == ne11 &&
            g_miss[i].ne[2] == ne12 && g_miss[i].ne[3] == ne13) {
            g_miss[i].n++;
            return;
        }
    }
    if (g_miss_n < ALPHA_Q8_MISS_SHAPES) {
        g_miss[g_miss_n].ne[0] = ne10; g_miss[g_miss_n].ne[1] = ne11;
        g_miss[g_miss_n].ne[2] = ne12; g_miss[g_miss_n].ne[3] = ne13;
        g_miss[g_miss_n].n = 1;
        g_miss_n++;
    }
}

void ggml_cuda_q8_stage_report() {
    static const long long every = []() {
        const char * e = getenv("ALPHA_Q8_STAGE_STATS");
        return e ? atoll(e) : 0;
    }();
    if (every <= 0) {
        return;
    }
    static long long calls = 0;
    if (++calls % every == 0) {
        fprintf(stderr, "[q8-stage] staged=%lld consumed=%lld declined=%lld\n", g_staged, g_hits, g_declined);
        fprintf(stderr, "[q8-stage] arena_full=%lld\n", g_arena_full);
        for (int i = 0; i < g_hitters_n; ++i) {
            fprintf(stderr, "[q8-stage] consumer: %s x%lld\n", g_hitters[i].name, g_hitters[i].n);
        }
        for (int i = 0; i < g_seen_n; ++i) {
            fprintf(stderr, "[q8-stage] producer reached: ne=[%lld,%lld,%lld,%lld] x%lld\n",
                    (long long) g_seen[i].ne[0], (long long) g_seen[i].ne[1],
                    (long long) g_seen[i].ne[2], (long long) g_seen[i].ne[3], g_seen[i].n);
        }
        for (int i = 0; i < g_miss_n; ++i) {
            fprintf(stderr, "[q8-stage] still quantising: ne=[%lld,%lld,%lld,%lld] x%lld\n",
                    (long long) g_miss[i].ne[0], (long long) g_miss[i].ne[1],
                    (long long) g_miss[i].ne[2], (long long) g_miss[i].ne[3], g_miss[i].n);
        }
    }
}
