# AlphaInfer

AlphaInfer is a second inference engine in this repository. Instead of building a ggml graph per step, it runs a
hand-written forward pass for each supported model family over a fixed set of operations, with memory allocated
once at load time, paged KV cache and recurrent-state slots. Whole decode rounds are captured as CUDA graphs.

It uses this repository's ggml for the operations it has no native kernel for, and ships its own CUDA kernels
for the hot paths (fused MoE merge and norm, router top-k, Gated DeltaNet convolution, decode attention, MTP).

Supported model architectures:

| GGUF architecture | Models | Notes |
|---|---|---|
| `qwen35moe` | Qwen3.5 / Qwen3.6 MoE (for example Qwen3.6-35B-A3B) | expert offload, MTP |
| `qwen35` | Qwen3.5 / Qwen3.8 dense (for example 27B) | MTP, DFlash2 drafter, CPU FFN offload |
| `qwen4exp` | Qwen3.8 Flash-Next | built with `AI_Q4X_FORWARD` (default on), run with `ai-q4x-run` |

## Build

AlphaInfer is built from the top level of the repository and needs CUDA:

```sh
cmake -B build -DGGML_CUDA=ON -DALPHA_ALPHAINFER=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This builds the `alphainfer` command-line tool, the support tools (`ai-*`), and the `alphainfer` backend of
`alphallama-serve`.

## Usage

```sh
# Generate 256 tokens from a chat prompt with 3 MTP drafts per step
build/bin/alphainfer -m Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf --chat -f prompt.txt -n 256 --mtp 3

# MoE model larger than VRAM: experts in host memory, 118 device slots per layer, CPU computes misses
AI_OFFLOAD=doorbell build/bin/alphainfer -m model.gguf --chat -p "Hello" --expert-slots 118 --mtp 3

# Behind the OpenAI-compatible server
build/bin/alphallama-serve --backend alphainfer -m Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf -np 2
```

`alphainfer` without arguments lists its options. MTP verification is exact by default on MoE models: the output
is identical to plain greedy decoding. Settings that are mainly useful for tuning are read from `AI_*`
environment variables and documented where they are read.

## Layout

- `src/core`: tensors, device memory, weights, KV cache
- `src/ops`: operations, either native kernels or calls into ggml
- `src/models`: one forward pass per model family
- `src/engine`: rounds, scheduling, MTP policy, tokenizer
- `src/offload`: doorbell expert offload and CPU expert computation
- `src/kernels`: CUDA kernels
- `src/api`: C API (`ai_engine.h`) used by `alphallama-serve`
- `apps`, `tools`: the command-line tool, kernel checks and benchmarks

## License

Apache License 2.0, see [LICENSE](LICENSE). This covers all of `alphainfer/`, including AlphaInfer 2 (the qwen4exp
Flash-Next engine and expert offload v2). Every source file carries an `SPDX-License-Identifier: Apache-2.0` header.
