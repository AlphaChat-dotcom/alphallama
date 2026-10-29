# AlphaLlama

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

AlphaLlama is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) for running large language models on
consumer GPUs, with a focus on Mixture-of-Experts (MoE) models that do not fit in VRAM.

It keeps the llama.cpp model format (GGUF), tools and C API, and adds:

- **Expert offload for MoE models.** VRAM, pinned system RAM and the model file on disk act as one expert store.
  Hot experts stay on the GPU and are read in place; the rest are streamed in or computed on the CPU in parallel
  with the GPU, so a model several times larger than VRAM still decodes at interactive speed.
- **Multi-token prediction (MTP) decoding** for models that ship an MTP head (for example Qwen3.5 / Qwen3.6 MTP
  GGUFs), with greedy output identical to plain decoding, plus prompt-lookup drafting for repetitive text.
- **`alphallama-serve`**, an OpenAI-compatible HTTP server that picks a configuration for the model and GPU
  automatically (`alphallama-serve -m model.gguf` is enough), with optional multi-user batching.
- **Retrieval over large local corpora** in the `alphallama` server: a BM25 keyword index (in memory or
  memory-mapped), optional dense rescoring, and query-time context assembly.
- Extra quantization types (including the ik_llama.cpp types) and CUDA kernels for MoE decode.
- **AlphaInfer** ([alphainfer/](alphainfer)), a second CUDA engine with hand-written forward passes and captured
  decode rounds for selected model families (Qwen3.5 / Qwen3.6 / Qwen3.8), usable on its own or as a backend of
  `alphallama-serve`.

All MoE optimizations are generic: they use the model's metadata (expert count, experts per token, tensor shapes)
and are not tied to one model family.

## Build

Requirements: CMake 3.18+, a C++17 compiler, and for GPU support the CUDA toolkit (12.x).

```sh
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Binaries are written to `build/bin/`. Set `-DCMAKE_CUDA_ARCHITECTURES=86` (or your GPU's compute capability) to
shorten the build. CPU-only builds work without `-DGGML_CUDA=ON`. See [docs/build.md](docs/build.md) for other
backends and platforms; everything that builds in llama.cpp builds here.

Optional components:

| CMake option | Default | What it adds |
|---|---|---|
| `ALPHA_FLASHINFER` | ON if `vendor/flashinfer` exists | FlashInfer batched attention and the Marlin INT4 path in `alphallama-serve`. Clone [FlashInfer](https://github.com/flashinfer-ai/flashinfer) into `vendor/flashinfer` to enable it. |
| `ALPHA_ALPHAINFER` | OFF | The AlphaInfer engine and its `alphallama-serve` backend (CUDA only). See [alphainfer/README.md](alphainfer/README.md). |
| `ALPHA_DENSE_L0` | OFF | Dense (embedding) retrieval backend for the `alphallama` server. |
| `ALPHA_DELTANET` | OFF | DeltaNet corpus pre-scan and state save/load (experimental). |

## Quick start

```sh
# OpenAI-compatible server; picks offload, batch sizes and decoding mode for the model and GPU
build/bin/alphallama-serve -m Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf --port 8080

curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'

# Several concurrent users
build/bin/alphallama-serve -m model.gguf -np 4

# Interactive chat in the terminal
build/bin/alphachat -m model.gguf
```

`alphallama-serve` with no `-m` prints its options. The standard llama.cpp tools are built too (`llama-cli`,
`llama-bench`, `llama-quantize`, `llama-perplexity`, ...); the llama.cpp server is built as `alphallama`.

On Windows, some antivirus products suspend a freshly built, unsigned server binary after a few requests: the
process stops responding with no error. If `alphallama-serve` or `alphallama` freezes this way, add `build\bin`
to the antivirus exclusions.

## Tuning

Defaults are chosen automatically and should be a good starting point. Advanced settings are exposed as
`ALPHA_*` environment variables; each one is documented where it is read in the source. `alphallama-serve`
prints the settings it chose at startup. Speed depends strongly on the workload, so when comparing settings,
measure decode speed and end-to-end time separately and use the same prompts.

## Relationship to llama.cpp

AlphaLlama tracks llama.cpp and keeps its file formats and public API. Files that come from llama.cpp remain under
the MIT License; see [NOTICE](NOTICE). Documentation in [docs/](docs) is llama.cpp's and applies here unless noted.

## License

MIT License, see [LICENSE](LICENSE), except the AlphaInfer engine ([alphainfer/](alphainfer)), the AlphaServe
server ([tools/serve/](tools/serve)) and the fused-infer expert-offload code, which are Apache License 2.0. The
exact file list is in [NOTICE](NOTICE). Third-party code and licenses are listed in [NOTICE](NOTICE) and
[licenses/](licenses).
