cmake_minimum_required(VERSION 3.22)
project(alphainfer CXX CUDA)

# Built from the top-level CMakeLists.txt (ALPHA_ALPHAINFER=ON), AlphaInfer links this repository's own ggml and
# llama targets. A standalone configure of this directory links a ggml/llama build given by AI_VENDOR_BUILD.
if (TARGET ggml AND TARGET llama)
    set(AI_IN_TREE ON)
else()
    set(AI_IN_TREE OFF)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
find_package(CUDAToolkit REQUIRED)

# ggml + libllama: the engine calls ggml ops for the layers it has no native kernel for, and reads the CUDA
# backend context header in src/kernels/cudart.cu.
if (AI_IN_TREE)
    set(AI_VENDOR_SRC   "${CMAKE_CURRENT_SOURCE_DIR}/..")
    set(AI_VENDOR_BUILD "${CMAKE_BINARY_DIR}")
    set(AI_GGML ggml)
    set(AI_GGML_BASE ggml-base)
    set(AI_GGML_CUDA ggml-cuda)
    set(AI_GGML_CPU ggml-cpu)
    set(AI_LLAMA llama)
else()
    set(AI_VENDOR_SRC   "${CMAKE_CURRENT_SOURCE_DIR}/.."       CACHE PATH "ggml / llama source tree")
    set(AI_VENDOR_BUILD "${CMAKE_CURRENT_SOURCE_DIR}/../build" CACHE PATH "ggml / llama build tree")
    find_library(AI_GGML       ggml       PATHS "${AI_VENDOR_BUILD}/bin" NO_DEFAULT_PATH REQUIRED)
    find_library(AI_GGML_BASE  ggml-base  PATHS "${AI_VENDOR_BUILD}/bin" NO_DEFAULT_PATH REQUIRED)
    find_library(AI_GGML_CUDA  ggml-cuda  PATHS "${AI_VENDOR_BUILD}/bin" NO_DEFAULT_PATH REQUIRED)
    find_library(AI_GGML_CPU   ggml-cpu   PATHS "${AI_VENDOR_BUILD}/bin" NO_DEFAULT_PATH REQUIRED)
    find_library(AI_LLAMA      llama      PATHS "${AI_VENDOR_BUILD}/bin" NO_DEFAULT_PATH REQUIRED)
endif()

set(AI_VENDOR_INCLUDES
    "${AI_VENDOR_SRC}/ggml/include"
    "${AI_VENDOR_SRC}/include"
    "${AI_VENDOR_SRC}/ggml/src")

add_library(aicore STATIC
    src/core/tensor.cpp
    src/core/device.cpp
    src/core/weights.cpp
    src/core/host_mem.cpp
    src/core/kv.cpp
    src/core/expert_pool.cpp
    src/ops/adapter.cpp
    src/ops/ops.cpp
    src/ops/dflash2.cpp
    src/api/ai_engine.cpp
    src/models/qwen35_hybrid.cpp
    src/models/dflash2.cpp
    src/models/qwen4exp.cpp
    src/engine/tokenizer.cpp
    src/engine/draft_vocab.cpp
    src/engine/engine.cpp
    src/engine/mtp_policy.cpp
    src/engine/engine_mtp.cpp
    src/kernels/cudart.cu
    src/kernels/moe_merge.cu
    src/kernels/gdn_conv.cu
    src/kernels/gdn_chunk16.cu
    src/kernels/gdn_conv_pf.cu
    src/kernels/gemm_f16w_pf.cu
    src/kernels/argmax.cu
    src/kernels/add_rmsnorm.cu
    src/kernels/moe_merge_norm.cu
    src/kernels/moe_merge_norm_coop.cu
    src/kernels/moe_merge_norm_last.cu
    src/kernels/moe_merge_partials.cu
    src/kernels/fa_decode.cu
    src/kernels/fa_decode_q.cu
    src/kernels/stream_fork.cu
    src/kernels/gdn_gate.cu
    src/kernels/gemv_f16w.cu
    src/kernels/router_topk.cu
    src/kernels/moe_gu_verify.cu
    src/kernels/moe_gu_tc.cu
    src/kernels/moe_pf.cu
    src/kernels/draft_conf.cu
    src/kernels/topk_native.cu
    src/kernels/attn_prologue.cu
    src/kernels/embed_q4k.cu
    src/kernels/moe_down9_bridge.cu
    src/kernels/q8stage_bridge.cu
    src/kernels/gemv_q8.cu
    src/kernels/gemv_q8_planar.cu
    src/kernels/mtp.cu
    src/kernels/prefetch.cu
    src/kernels/dflash2.cu)
# install(EXPORT) cannot export a bare source-dir path, so the public include is a generator
# expression: BUILD_INTERFACE for targets in this project, INSTALL_INTERFACE for an external consumer.
target_include_directories(aicore
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src> $<INSTALL_INTERFACE:include>
    PRIVATE ${AI_VENDOR_INCLUDES})
target_compile_options(aicore PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter>)
# Turn on only when the linked ggml is built with -DALPHA_BATCH_INVARIANT_MMVQ=1 (MMVQ pinned to the 1-row shape at
# 1..8 columns): enables the dense exact default with K = 7 + lookup (engine/mtp_policy.cpp).
option(AI_BI_FORK "ggml is built with the MMVQ batch-invariance pin (ALPHA_BATCH_INVARIANT_MMVQ=1)" OFF)
if (AI_BI_FORK)
    target_compile_definitions(aicore PRIVATE AI_BI_FORK=1)
endif()
target_link_libraries(aicore PUBLIC ${AI_LLAMA} ${AI_GGML_CUDA} ${AI_GGML} ${AI_GGML_BASE} CUDA::cudart)
# native kernels: no FMA contraction, so a fused kernel stays bitwise equal to the ops it replaces
set_source_files_properties(src/kernels/moe_merge.cu PROPERTIES COMPILE_OPTIONS "-fmad=false")
# router_topk.cu reproduces ggml's topk_moe, which is built -use_fast_math: match its flush-to-zero
set_source_files_properties(src/kernels/router_topk.cu PROPERTIES COMPILE_OPTIONS "-ftz=true")
# moe_gu_verify.cu runs ggml's vec_dot code and op_silu: build it with the same math flags as ggml-cuda
set_source_files_properties(src/kernels/moe_gu_verify.cu PROPERTIES COMPILE_OPTIONS "-use_fast_math")
set_source_files_properties(src/kernels/moe_gu_tc.cu PROPERTIES COMPILE_OPTIONS "-use_fast_math")
set_source_files_properties(src/kernels/topk_native.cu PROPERTIES COMPILE_OPTIONS "-use_fast_math")
set_source_files_properties(src/kernels/attn_prologue.cu PROPERTIES COMPILE_OPTIONS "-use_fast_math")
# gdn_conv_pf.cu reproduces ggml's SSM_CONV + SiLU (built -use_fast_math) bit for bit
set_source_files_properties(src/kernels/gdn_conv_pf.cu PROPERTIES COMPILE_OPTIONS "-use_fast_math")

# Install aicore, the C API headers (src/api) and an exported target set, so another project
# (e.g. AlphaServe) can find_package(alphainfer).
if (NOT AI_IN_TREE)
    install(TARGETS aicore EXPORT alphainferTargets ARCHIVE DESTINATION lib)
    install(DIRECTORY src/api/ DESTINATION include/api FILES_MATCHING PATTERN "*.h")
    install(EXPORT alphainferTargets NAMESPACE alphainfer:: DESTINATION lib/cmake/alphainfer)
endif()

add_executable(alphainfer apps/alphainfer.cpp)
target_link_libraries(alphainfer PRIVATE aicore)

add_executable(ai-bench-gemv tools/bench_gemv.cpp)
target_link_libraries(ai-bench-gemv PRIVATE aicore)
add_executable(ai-bench-fixed tools/bench_fixed.cpp)
target_link_libraries(ai-bench-fixed PRIVATE aicore)
# Is the LM head weight read once per call or once per row? op::head is a single ggml_mul_mat, so
# this probes the kernel selected below it. It prints the implied bandwidth under both assumptions
# (one weight read, n reads); whichever stays flat near peak is the answer.
add_executable(ai-bench-head-ncols tools/bench_head_ncols.cpp)
target_link_libraries(ai-bench-head-ncols PRIVATE aicore)

# Per-kernel differential test: every native fused kernel against the ggml chain it replaces,
# byte for byte. Runs eager (never under capture -- a sync inside a capturing stream is illegal and
# silently invalidates the comparison) and sweeps T = 1..16, where batch-variance bugs live.
add_executable(ai-kernel-diff tools/kernel_diff.cpp)
target_include_directories(ai-kernel-diff PRIVATE ${AI_VENDOR_INCLUDES})
target_link_libraries(ai-kernel-diff PRIVATE aicore CUDA::cudart)

# DFlash2 drafter forward against the z-lab reference on identical inputs (the oracle's schedule)
add_executable(ai-dflash2-check tools/dflash2_check.cpp)
target_link_libraries(ai-dflash2-check PRIVATE aicore)

# Kernel-level check for ALPHA_SMALLM_TC: small-M mul_mat on real 27B shapes, dumped for an A/B diff
add_executable(ai-smallm-check tools/smallm_check.cpp)
target_include_directories(ai-smallm-check PRIVATE ${AI_VENDOR_INCLUDES})
target_link_libraries(ai-smallm-check PRIVATE ${AI_GGML_CUDA} ${AI_GGML} ${AI_GGML_BASE})

# Kernel check for AI_FA_Q: accuracy vs f64, MTP == plain identity, GB/s vs the native q4_0 kernel
add_executable(ai-fadq-check tools/fadq_check.cpp)
target_link_libraries(ai-fadq-check PRIVATE aicore CUDA::cudart)

# Requantise a GGUF's 2-D matrices to one type (any arch; used for the DFlash2 drafter)
add_executable(ai-requant tools/requant_gguf.cpp)
target_include_directories(ai-requant PRIVATE ${AI_VENDOR_INCLUDES})
target_link_libraries(ai-requant PRIVATE ${AI_GGML} ${AI_GGML_BASE})

add_executable(ai-ref tools/ref_oracle.cpp)
target_include_directories(ai-ref PRIVATE ${AI_VENDOR_INCLUDES})
target_link_libraries(ai-ref PRIVATE ${AI_LLAMA} ${AI_GGML_CUDA} ${AI_GGML} ${AI_GGML_BASE})

# GPU <-> CPU doorbell handshake inside a captured graph (standalone, CUDA runtime only)
add_executable(ai-doorbell-bench tools/doorbell_bench.cu)
set_target_properties(ai-doorbell-bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# The qwen4exp forward, with its Op bodies in src/ops/q4x_ops.cpp
option(AI_Q4X_FORWARD "build the qwen4exp runtime (src/models/qwen4exp_run.cpp + src/ops/q4x_ops.cpp)" ON)
if (AI_Q4X_FORWARD)
    target_sources(aicore PRIVATE src/models/qwen4exp_run.cpp src/models/qwen4exp_mtp.cpp src/models/qwen4exp_prefill.cpp src/ops/q4x_ops.cpp src/kernels/q4x_qsa.cu src/kernels/q4x_route.cu src/kernels/q4x_experts.cu src/kernels/q4x_dense.cu
                                  src/kernels/q4x_hc.cu src/kernels/q4x_hc_tc.cu src/kernels/q4x_gdn_conv.cu src/kernels/q4x_gdn_scan.cu src/kernels/q4x_fold.cu
                                  src/offload/miss_pool.cpp src/offload/gsq_cpu.cpp src/offload/cpu_topo.cpp src/kernels/offload.cu
                                  src/api/q4x_serve.cpp)
    # the C API serves qwen4exp models through api/q4x_serve.cpp
    target_compile_definitions(aicore PUBLIC AI_Q4X_FORWARD)
    # decode-once GSQ CPU kernels: AVX2 + FMA + F16C (no -march=native, so the binary stays portable)
    # q4x_experts.cu runs ggml's vec_dot code and op_silu: build it with the same math flags as ggml-cuda
    set_source_files_properties(src/kernels/q4x_experts.cu PROPERTIES COMPILE_OPTIONS "-use_fast_math")
    # q4x_hc.cu / q4x_gdn_conv.cu / q4x_gdn_scan.cu reproduce ggml's hc-fuse.cu and ssm_conv + SiLU (built -use_fast_math) bit for bit
    set_source_files_properties(src/kernels/q4x_hc.cu src/kernels/q4x_gdn_conv.cu src/kernels/q4x_gdn_scan.cu PROPERTIES COMPILE_OPTIONS "-use_fast_math")
    # q4x_fold.cu reproduces the fork's gdn_prep_kernel + gated_delta_net_cuda (built -use_fast_math) bit for bit
    set_source_files_properties(src/kernels/q4x_fold.cu PROPERTIES COMPILE_OPTIONS "-use_fast_math")
    set_source_files_properties(src/offload/gsq_cpu.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    target_link_libraries(aicore PUBLIC ${AI_GGML_CPU})
    add_executable(ai-q4x-run tools/q4x_run.cpp)
    target_include_directories(ai-q4x-run PRIVATE ${AI_VENDOR_INCLUDES})
    target_link_libraries(ai-q4x-run PRIVATE aicore)
    set_target_properties(ai-q4x-run PROPERTIES BUILD_RPATH "${AI_VENDOR_BUILD}/bin" RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    # greedy generation through the C API only (the calls the alphallama-serve adapter makes)
    add_executable(ai-api-gen tools/api_gen.cpp)
    target_include_directories(ai-api-gen PRIVATE src)
    target_link_libraries(ai-api-gen PRIVATE aicore)
    set_target_properties(ai-api-gen PROPERTIES BUILD_RPATH "${AI_VENDOR_BUILD}/bin" RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    # grouped / wide expert kernels on a synthetic slot bank
    add_executable(ai-q4x-expbench tools/q4x_expbench.cpp)
    target_include_directories(ai-q4x-expbench PRIVATE ${AI_VENDOR_INCLUDES})
    target_link_libraries(ai-q4x-expbench PRIVATE aicore)
    set_target_properties(ai-q4x-expbench PROPERTIES BUILD_RPATH "${AI_VENDOR_BUILD}/bin" RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_executable(ai-q4x-hcbench tools/q4x_hcbench.cpp)
    target_include_directories(ai-q4x-hcbench PRIVATE src ${AI_VENDOR_INCLUDES})
    target_link_libraries(ai-q4x-hcbench PRIVATE aicore)
    set_target_properties(ai-q4x-hcbench PROPERTIES BUILD_RPATH "${AI_VENDOR_BUILD}/bin" RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_executable(ai-q4x-densebench tools/q4x_densebench.cpp)
    target_include_directories(ai-q4x-densebench PRIVATE ${AI_VENDOR_INCLUDES})
    target_link_libraries(ai-q4x-densebench PRIVATE aicore)
    set_target_properties(ai-q4x-densebench PROPERTIES BUILD_RPATH "${AI_VENDOR_BUILD}/bin" RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_executable(ai-gsq-bench tools/gsq_bench.cpp src/offload/gsq_cpu.cpp)
    target_include_directories(ai-gsq-bench PRIVATE src ${AI_VENDOR_INCLUDES})
    target_link_libraries(ai-gsq-bench PRIVATE ${AI_GGML_CPU} ${AI_GGML_BASE} ${AI_GGML})
    set_target_properties(ai-gsq-bench PROPERTIES BUILD_RPATH "${AI_VENDOR_BUILD}/bin" RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()

# Generic doorbell expert offload for every op::MoeWeights model:
# src/offload/doorbell_moe.cpp, selected at run time with AI_OFFLOAD=doorbell and --expert-slots N
option(AI_DOORBELL "build the generic doorbell expert offload (AI_OFFLOAD=doorbell)" ON)
if (AI_DOORBELL)
    target_compile_definitions(aicore PUBLIC AI_DOORBELL)
    target_sources(aicore PRIVATE src/offload/doorbell_moe.cpp src/offload/cpu_ffn.cpp src/kernels/doorbell.cu)
    if (NOT AI_Q4X_FORWARD)
        target_sources(aicore PRIVATE src/offload/miss_pool.cpp src/offload/gsq_cpu.cpp src/offload/cpu_topo.cpp)
        set_source_files_properties(src/offload/gsq_cpu.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
        target_link_libraries(aicore PUBLIC ${AI_GGML_CPU})
    endif()
endif()

# qwen4exp (Flash-Next) spec + byte budget from GGUF metadata (no GPU)
add_executable(ai-q4x-inspect tools/q4x_inspect.cpp)
target_include_directories(ai-q4x-inspect PRIVATE ${AI_VENDOR_INCLUDES})
target_link_libraries(ai-q4x-inspect PRIVATE aicore)
add_executable(ai-q4x-load tools/q4x_load.cpp)
target_include_directories(ai-q4x-load PRIVATE ${AI_VENDOR_INCLUDES})
target_link_libraries(ai-q4x-load PRIVATE aicore)

set_target_properties(alphainfer ai-ref ai-q4x-inspect ai-q4x-load ai-dflash2-check ai-smallm-check ai-fadq-check ai-requant ai-bench-gemv ai-bench-fixed ai-bench-head-ncols ai-kernel-diff PROPERTIES
    BUILD_RPATH "${AI_VENDOR_BUILD}/bin"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
