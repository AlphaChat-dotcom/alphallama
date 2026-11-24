# alphallama-latest

`latest/` is the source of `alphallama-latest`, AlphaLlama's second engine for model families that the main engine
does not implement yet (currently Inkling 2). It is llama.cpp at a recent release plus the pull requests that model
support needs, plus our kernels, built as its own CMake project into one static executable.

`alphallama-serve` hands a model over to it by itself: when the GGUF's `general.architecture` is unknown to the main
engine and `alphallama-latest` sits next to `alphallama-serve`, the process continues as `alphallama-latest` on the
same port with the same OpenAI-compatible API (`ALPHA_LATEST=0` turns this off). Where they are not given, it applies
the measured serving profile: one slot, 32K context, q8_0 KV cache, a 512 MiB fit margin, sampling on the GPU, and
n-gram lookup speculation (`--spec-type ngram-mod`). `ALPHA_LATEST_DRAFTER=1` uses a DFlash / DSpark drafter found in
the model's directory instead. Speculation here is not token-identical to plain decoding.

Build: configure AlphaLlama with `-DALPHA_LATEST=ON` (the backend options carry over; `-DALPHA_LATEST_JOBS=N` limits
the parallel jobs). The release packages for Linux CUDA, Windows CUDA and macOS include it.

## Source

- llama.cpp release `b12631` (https://github.com/ggml-org/llama.cpp), MIT License.
- The pull requests below, merged in this order on top of it (the set Unsloth pins for its llama.cpp build
  `b12631-mix-acc0843`, https://github.com/unslothai/llama.cpp), MIT License.
- Our changes, MIT License like the rest of this directory:
  - `ggml/src/ggml-cuda/loop-ctrl.cu`: the loop control of looped layers in two kernels (exact where the unfused
    chain runs `mul_mat_vec_f`)
  - `ggml/include/ggml-depth-mix.h`, `ggml/src/ggml-cuda/depth-mix.cu`: attention over the depth as one op
  - `ggml/src/ggml-cuda/conv-res.cu`: the residual short convolution in one kernel (exact)
  - `ggml/src/ggml-cuda/concat-store.cu`: a short convolution input and its state store in one kernel (exact)
  - `ggml/include/ggml-rel-bias.h`, `ggml/src/ggml-cuda/rel-bias.cu`: the relative attention bias folded into the
    attention mask as one op
  - `ggml/src/ggml-cuda/topk-moe.cu`: the top-k routing fusion also covers a weight norm with an eps
  - `tools/server`: no `/v1/messages` endpoints, as in AlphaLlama
- Trimmed to what the server builds from: the backends other than CPU, CUDA, Metal, BLAS and RPC, the tests,
  the examples, the tools other than the server and mtmd, and the web UI sources are not included.

| Repository | Pull request | Commit |
|---|---|---|
| ggml-org/llama.cpp | #24423 | `9f55089d59b3` |
| unslothai/llama.cpp | #378 | `19360e53362a` |
| unslothai/llama.cpp | #405 | `0a9105de42d2` |
| unslothai/llama.cpp | #395 | `dd330cc0024f` |
| unslothai/llama.cpp | #396 | `9b6b8a4a0cb6` |
| unslothai/llama.cpp | #397 | `805b04c23d1f` |
| unslothai/llama.cpp | #385 | `54dcc0553d46` |
| unslothai/llama.cpp | #386 | `6348de261e82` |
| unslothai/llama.cpp | #70 | `70af2a4db8fe` |
| unslothai/llama.cpp | #61 | `b74bd087653c` |
| unslothai/llama.cpp | #95 | `496a8a646f19` |
| unslothai/llama.cpp | #158 | `fb9b9462041a` |
| unslothai/llama.cpp | #157 | `fec8345236da` |
| unslothai/llama.cpp | #149 | `efa8b0e689ac` |
| unslothai/llama.cpp | #176 | `95de917806d4` |
| unslothai/llama.cpp | #239 | `9923be53f13d` |
| unslothai/llama.cpp | #240 | `924234599ba0` |
| unslothai/llama.cpp | #241 | `ee25de406273` |
| unslothai/llama.cpp | #243 | `30a106edde47` |
| unslothai/llama.cpp | #251 | `827478f5aead` |
| unslothai/llama.cpp | #264 | `fd6fae58b5b2` |
| unslothai/llama.cpp | #266 | `af3b2bc3201e` |
| unslothai/llama.cpp | #283 | `644ca7d817b8` |
| unslothai/llama.cpp | #332 | `ce078ec1223f` |
| unslothai/llama.cpp | #304 | `e4e1eba07030` |
| unslothai/llama.cpp | #311 | `819e95b31674` |
| unslothai/llama.cpp | #320 | `579e70abf11a` |
| unslothai/llama.cpp | #328 | `02428e283cb0` |
| unslothai/llama.cpp | #348 | `56ec1e88a052` |
| unslothai/llama.cpp | #335 | `b240c56069c2` |
| unslothai/llama.cpp | #376 | `f12ce09a08c9` |
| unslothai/llama.cpp | #400 | `a5667a9c0bd5` |
| unslothai/llama.cpp | #401 | `e418bb288b59` |
| unslothai/llama.cpp | #402 | `0bfd204bd4de` |
