// SPDX-License-Identifier: Apache-2.0
/* Check AlphaInfer's dtype table against ggml's own size functions.
 *
 * ggml_type_size() / ggml_blck_size() / ggml_row_size() are the values the kernels actually use, so
 * comparing against them is an independent check rather than a restatement of the table's constants,
 * and it covers every type in the table.
 *
 * A can't-fail row runs first: a deliberately wrong pairing must be reported as a mismatch, or the
 * comparison is inert.
 *
 * Host-only: no CUDA, no device, no model. Links libggml-base for the size queries.
 */
#include "core/tensor.h"

#include "ggml.h"

#include <cstdio>
#include <stdexcept>

namespace {

int checks = 0;
int fails  = 0;

struct Row { int ours; ggml_type theirs; const char * label; };

void cmp(const Row & r) {
    ++checks;
    const size_t  ts_ours = ai::type_size(r.ours);
    const int64_t bs_ours = ai::block_size(r.ours);
    const size_t  ts_them = ggml_type_size(r.theirs);
    const int64_t bs_them = ggml_blck_size(r.theirs);
    // row_size over one block and over eight, so a right type_size with a wrong block_size still fails
    const size_t  rs_ours = ai::row_size(r.ours, bs_ours * 8);
    const size_t  rs_them = (size_t) ggml_row_size(r.theirs, bs_them * 8);
    const bool ok = ts_ours == ts_them && bs_ours == bs_them && rs_ours == rs_them;
    if (!ok) { ++fails; }
    printf("  %-8s %-7s ours %4zu B / %4lld   ggml %4zu B / %4lld   row*8 %6zu vs %6zu\n",
           ok ? "[ok]" : "[FAIL]", r.label,
           ts_ours, (long long) bs_ours, ts_them, (long long) bs_them, rs_ours, rs_them);
}

}  // namespace

int main() {
    /* can't-fail row first: a deliberately wrong pairing must be caught, or the comparison is inert */
    printf("can't-fail row -- a deliberately wrong pairing must FAIL:\n");
    const int before = fails;
    cmp(Row{ ai::Q4_K, GGML_TYPE_Q6_K, "q4K~q6K" });
    if (fails == before) {
        printf("  FAIL: the oracle did not flag a known-wrong pairing; the comparison proves nothing.\n");
        return 1;
    }
    printf("  => caught it, so the comparison is live. Resetting the counter.\n\n");
    fails  = 0;
    checks = 0;

    printf("every type in AlphaInfer's table vs ggml:\n");
    cmp(Row{ ai::F32,    GGML_TYPE_F32,    "f32"    });
    cmp(Row{ ai::F16,    GGML_TYPE_F16,    "f16"    });
    cmp(Row{ ai::BF16,   GGML_TYPE_BF16,   "bf16"   });
    cmp(Row{ ai::Q4_0,   GGML_TYPE_Q4_0,   "q4_0"   });
    cmp(Row{ ai::Q4_1,   GGML_TYPE_Q4_1,   "q4_1"   });
    cmp(Row{ ai::Q5_0,   GGML_TYPE_Q5_0,   "q5_0"   });
    cmp(Row{ ai::Q8_0,   GGML_TYPE_Q8_0,   "q8_0"   });
    cmp(Row{ ai::Q4_K,   GGML_TYPE_Q4_K,   "q4_K"   });
    cmp(Row{ ai::Q5_K,   GGML_TYPE_Q5_K,   "q5_K"   });
    cmp(Row{ ai::Q6_K,   GGML_TYPE_Q6_K,   "q6_K"   });
    cmp(Row{ ai::IQ4_NL, GGML_TYPE_IQ4_NL, "iq4_nl" });
    cmp(Row{ ai::IQ4_XS, GGML_TYPE_IQ4_XS, "iq4_xs" });
    cmp(Row{ ai::Q2_K,   GGML_TYPE_Q2_K,   "q2_K"   });
    cmp(Row{ ai::Q3_K,   GGML_TYPE_Q3_K,   "q3_K"   });
    cmp(Row{ ai::IQ3_S,  GGML_TYPE_IQ3_S,  "iq3_s"  });
    cmp(Row{ ai::NVFP4,  GGML_TYPE_NVFP4,  "nvfp4"  });
    cmp(Row{ ai::I8,     GGML_TYPE_I8,     "i8"     });
    cmp(Row{ ai::I32,    GGML_TYPE_I32,    "i32"    });
    cmp(Row{ ai::I64,    GGML_TYPE_I64,    "i64"    });

    /* the enum values themselves must match ggml's, or a GGUF's type field is misread */
    printf("\nenum values match ggml:\n");
    struct { int ours; int theirs; const char * n; } ev[] = {
        { ai::IQ4_NL, (int) GGML_TYPE_IQ4_NL, "IQ4_NL" },
        { ai::IQ4_XS, (int) GGML_TYPE_IQ4_XS, "IQ4_XS" },
        { ai::IQ3_S,  (int) GGML_TYPE_IQ3_S,  "IQ3_S"  },
        { ai::NVFP4,  (int) GGML_TYPE_NVFP4,  "NVFP4"  },
        { ai::Q4_K,   (int) GGML_TYPE_Q4_K,   "Q4_K"   },
        { ai::I64,    (int) GGML_TYPE_I64,    "I64"    },
    };
    for (const auto & e : ev) {
        ++checks;
        const bool ok = e.ours == e.theirs;
        if (!ok) { ++fails; }
        printf("  %-8s %-7s ours=%d ggml=%d\n", ok ? "[ok]" : "[FAIL]", e.n, e.ours, e.theirs);
    }

    /* block_size() must throw on an unknown blocked type rather than returning 1 */
    printf("\nblock_size refuses an unknown type:\n");
    ++checks;
    bool threw = false;
    try {
        (void) ai::block_size(9999);
    } catch (const std::exception &) {
        threw = true;
    }
    if (!threw) { ++fails; }
    printf("  %-8s block_size(9999) threw instead of returning 1\n", threw ? "[ok]" : "[FAIL]");

    printf("\n%d/%d checks passed\n", checks - fails, checks);
    if (fails == 0) {
        printf("=> PASS: the dtype table agrees with ggml on every entry, including the two new IQ types.\n");
        return 0;
    }
    printf("=> FAIL: %d mismatch(es).\n", fails);
    return 1;
}
