# Host-only unit tests (no GPU, no model)

## `dtype_oracle_test.cpp`

Checks AlphaInfer's dtype table against **ggml's own** `ggml_type_size` / `ggml_blck_size` /
`ggml_row_size`, rather than against restated constants. Asserting numbers I had just read out of
`ggml-common.h` would only prove I can copy; the oracle is the values the kernels actually use, and it
covers **every** entry in our table so a pre-existing wrong one surfaces too.

⭐ Carries its own can't-fail row, run first: a deliberately wrong pairing (`Q4_K` vs `GGML_TYPE_Q6_K`)
must be reported as a mismatch. If it is not, the comparison is inert and the test refuses to report a
pass.

Also checks that the enum values equal ggml's (a mismatch there misreads a GGUF's type field — which is
exactly the bug that put wrong IQ counts in an earlier report), and that `block_size()` now **throws** on
an unknown type instead of returning 1.

Build (seconds, safe to run alongside a GPU benchmark):

```sh
cd ~/alphainfer && FORK=$PWD/third_party/build-fork/bin
g++ -std=c++17 -O0 -I src -I third_party/fork-73c397f7b/ggml/include \
    -o /tmp/dtype_oracle tools/unit-tests/dtype_oracle_test.cpp src/core/tensor.cpp \
    $FORK/libggml-base.so -Wl,-rpath,$FORK
/tmp/dtype_oracle
```

Last run: 20/20, can't-fail row caught its deliberate mismatch.
