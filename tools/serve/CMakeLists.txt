# SPDX-License-Identifier: Apache-2.0
# AlphaServe: Multi-Backend Intelligence Server
# Backends: ggml (default) + FlashInfer (opt-in) + SGLang proxy
# Intelligence layer: BM25 + NER retrieval, backend-agnostic

set(TARGET alphaserve)

set(ALPHASERVE_SOURCES
    alphaserve.cpp
    alphaserve.h
    serve-scheduler.cpp
    serve-scheduler.h
    serve-gpu-engine.cpp
    serve-gpu-engine.h
    serve-backend.cpp
    serve-backend.h
    serve-intelligence.cpp
    serve-intelligence.h
    serve-spsc-queue.h
    serve-types.h
    serve-slot.h
    serve-mm.h
    serve-response.h
)

# ALPHA_FLASHINFER (top-level CMakeLists.txt) adds the FlashInfer dense path and the Marlin W4A16
# AWQ/GPTQ INT4 Tensor-Core path, both gated on ALPHA_HAS_FLASHINFER.

# AlphaInfer backend (ALPHA_ALPHAINFER, top-level CMakeLists.txt). The adapter (serve-ai-engine.{h,cpp}) is fully
# inside #ifdef ALPHA_HAS_ALPHAINFER, so it compiles to nothing when off.
# Serving: `alphallama-serve --backend alphainfer -m <MTP gguf> -np 2 -c 98304` runs AlphaServe's HTTP,
# scheduler and slots in front of AlphaInfer. Prompt chunks are buffered per slot and prefilled whole; generating
# slots run as ai_engine_decode_mtp rounds over the open streams, with the round's extra tokens returned per slot
# (BatchResult::slot_extra); finished streams are reset. Greedy only.

if (ALPHA_ALPHAINFER)
    list(APPEND ALPHASERVE_SOURCES
        serve-ai-engine.cpp
        serve-ai-engine.h
    )
endif()

if (ALPHA_FLASHINFER)
    list(APPEND ALPHASERVE_SOURCES
        serve-paged-kv.cpp
        serve-paged-kv.h
        serve-forward.cpp
        serve-forward.h
        serve-cuda-graph.cpp
        serve-cuda-graph.h
        serve-fi-engine.cpp
        serve-fi-engine.h
        serve-kernels.cu
        serve-flashinfer-attn.cu
        serve-q4k-gemv.cu
        serve-q4k-wmma.cu
        llama-flashinfer-attn.cu
    )
    # serve-marlin.cu is excluded on MSVC: its Marlin<> kernel templates do not resolve under MSVC's
    # frontend. The consumer side is compiled out there anyway (ggml-cuda.cu declares the marlin hooks
    # weak, which MSVC cannot do, so ALPHA_MARLIN_HOOK is 0 and the hook is never called).
    if (NOT MSVC)
        list(APPEND ALPHASERVE_SOURCES serve-marlin.cu)
    endif()
endif()

add_library(${TARGET} STATIC ${ALPHASERVE_SOURCES})

target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/vendor)
# Include server headers for BM25/NER/Redis
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/tools/server)

target_link_libraries(${TARGET} PUBLIC llama-common cpp-httplib ${CMAKE_THREAD_LIBS_INIT})
# image input (--mmproj): the mtmd projector library
target_link_libraries(${TARGET} PUBLIC mtmd)
target_include_directories(${TARGET} PUBLIC ${CMAKE_SOURCE_DIR}/tools/mtmd)

if (ALPHA_ALPHAINFER)
    target_compile_definitions(${TARGET} PUBLIC ALPHA_HAS_ALPHAINFER)
    # aicore is built in-tree against this tree's ggml, so the engine shares one ggml with the server
    target_link_libraries(${TARGET} PUBLIC aicore)
endif()

if (ALPHA_FLASHINFER)
    enable_language(CUDA)
    target_compile_definitions(${TARGET} PUBLIC ALPHA_HAS_FLASHINFER)
    # FlashInfer header-only includes
    target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/vendor/flashinfer/include)
    # CCCL compatibility shim (cuda/cmath) for CUDA 12.2
    target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/compat)
    # cuBLAS for matmuls
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(${TARGET} PUBLIC CUDA::cublas CUDA::cublasLt CUDA::cudart)
    # CUDA properties for .cu files
    set_source_files_properties(serve-kernels.cu PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(serve-flashinfer-attn.cu PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(serve-q4k-gemv.cu PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(serve-q4k-wmma.cu PROPERTIES LANGUAGE CUDA)
    if (NOT MSVC)   # serve-marlin.cu is not built on MSVC (see the exclusion above)
        set_source_files_properties(serve-marlin.cu PROPERTIES LANGUAGE CUDA)
        set_source_files_properties(serve-marlin.cu PROPERTIES
            COMPILE_FLAGS "--expt-relaxed-constexpr -std=c++17")
    endif()
    set_source_files_properties(llama-flashinfer-attn.cu PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(llama-flashinfer-attn.cu PROPERTIES
        COMPILE_FLAGS "--expt-relaxed-constexpr -std=c++17")
    # FlashInfer decode attention needs SM80+ and C++17
    set_source_files_properties(serve-flashinfer-attn.cu PROPERTIES
        COMPILE_FLAGS "--expt-relaxed-constexpr -std=c++17")
    # Set CUDA architecture (matches main ggml-cuda target)
    if (CMAKE_CUDA_ARCHITECTURES_NATIVE)
        set_target_properties(${TARGET} PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES_NATIVE}")
    else()
        set_target_properties(${TARGET} PROPERTIES CUDA_ARCHITECTURES "80;86;89")
    endif()
endif()

# Redis retrieval backend (optional, requires libhiredis)
option(ALPHA_REDIS "Enable Redis retrieval backend (requires libhiredis)" OFF)
if (ALPHA_REDIS)
    find_library(HIREDIS_LIB hiredis HINTS $ENV{HOME}/local/lib /usr/local/lib)
    find_path(HIREDIS_INCLUDE hiredis/hiredis.h HINTS $ENV{HOME}/local/include /usr/local/include)
    if (HIREDIS_LIB AND HIREDIS_INCLUDE)
        target_sources(${TARGET} PRIVATE
            ${CMAKE_SOURCE_DIR}/tools/server/server-redis.cpp
            ${CMAKE_SOURCE_DIR}/tools/server/server-redis.h)
        target_include_directories(${TARGET} PUBLIC ${HIREDIS_INCLUDE})
        target_link_libraries(${TARGET} PUBLIC ${HIREDIS_LIB})
        target_compile_definitions(${TARGET} PUBLIC ALPHA_HAS_REDIS)
        message(STATUS "AlphaServe: Redis enabled (hiredis: ${HIREDIS_LIB})")
    else()
        message(WARNING "ALPHA_REDIS enabled but hiredis not found — Redis disabled")
    endif()
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# HTTP Server binary (multi-backend intelligence server)
add_executable(alphallama-serve serve-main.cpp)
target_include_directories(alphallama-serve PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(alphallama-serve PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(alphallama-serve PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_include_directories(alphallama-serve PRIVATE ${CMAKE_SOURCE_DIR}/vendor)
target_include_directories(alphallama-serve PRIVATE ${CMAKE_SOURCE_DIR}/tools/server)
target_link_libraries(alphallama-serve PRIVATE alphaserve llama-common cpp-httplib ${CMAKE_THREAD_LIBS_INIT})
# serve-main.cpp calls cudaMemGetInfo directly for VRAM detection
if (GGML_CUDA)
    find_package(CUDAToolkit)
    if (CUDAToolkit_FOUND)
        target_link_libraries(alphallama-serve PRIVATE CUDA::cudart)
    endif()
endif()
# Encrypted .apack decryption is provided by libllama (OpenSSL crypto, see src/CMakeLists.txt), so
# alphallama-serve needs no crypto sources of its own.
