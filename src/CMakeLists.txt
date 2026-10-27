llama_add_compile_flags()

#
# libraries
#

# llama

file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")

add_library(llama
            ../include/llama.h
            llama.cpp
            llama-adapter.cpp
            llama-arch.cpp
            llama-batch.cpp
            llama-chat.cpp
            llama-context.cpp
            llama-context-mtp4.cpp
            llama-cparams.cpp
            llama-grammar.cpp
            llama-graph.cpp
            llama-hparams.cpp
            llama-impl.cpp
            llama-io.cpp
            llama-kv-cache.cpp
            llama-kv-cache-iswa.cpp
            llama-memory.cpp
            llama-memory-hybrid.cpp
            llama-memory-hybrid-idx.cpp
            llama-memory-hybrid-iswa.cpp
            llama-memory-recurrent.cpp
            llama-mmap.cpp
            llama-model-loader.cpp
            llama-model-saver.cpp
            llama-model.cpp
            llama-quant.cpp
            llama-sampler.cpp
            llama-tokenizer-json.cpp
            llama-vocab.cpp
            unicode-data.cpp
            unicode.cpp
            unicode.h
            alpha_mole.cpp
            alpha_mole.h
            alpha_mole_autoconfig.cpp
            alpha_mole_autoconfig.h
            alpha_hw_planner.cpp
            alpha_hw_planner.h
            alpha_train_graph.cpp
            alpha_train_graph.h
            alpha_train_full.cpp
            alpha_train_full.h
            alpha_train_pipeline.cpp
            alpha_train_pipeline.h
            alpha_train_qat.cpp
            alpha_train_qat.h
            alpha_registry.cpp
            alpha_registry.h
            alpha_mole_graph.cpp
            alpha_mole_kv_paging.cpp
            alpha_mole_kv_paging.h
            alpha_mole_pipeline.cpp
            alpha_mole_pipeline.h
            alpha_mole_stream.cpp
            alpha_mole_stream.h
            alpha_mole_expert_cache.cpp
            alpha_mole_expert_cache.h
            alpha_crypto.cpp
            alpha_crypto.h
            alpha_mole_moe_optimizer.cpp
            alpha_mole_moe_optimizer.h
            alpha_mole_speculative.cpp
            alpha_mole_speculative.h
            alpha_mole_train_backward.cpp
            alpha_mole_train_backward.h
            alpha_mole_train_engine.cpp
            alpha_mole_train_engine.h
            alpha_train_combined.cpp
            alpha_train_combined.h
            alpha_train_novel.cpp
            alpha_train_novel.h
            alpha_mole_train_pipeline.cpp
            alpha_mole_train_pipeline.h
            alpha_mole_train_optim.cpp
            alpha_mole_train_optim.h
            alpha_train_chunked.cpp
            alpha_train_chunked.h
            alpha_mole_pinned.cpp
            alpha_mole_pinned.h
            alpha_mole_graph.h
            alpha_mole_stack.cpp
            alpha_mole_stack.h
            alpha_router.cpp
            alpha_router.h
            alpha_moe_portfolio.cpp
            alpha_moe_portfolio.h
            alpha_moe_pack.cpp
            alpha_moe_pack.h
            alpha_dense_pipeline.cpp
            alpha_dense_pipeline.h
            alpha_gdn_record.cpp
            alpha_gdn_record.h
            alpha_env_audit.cpp
            alpha_env_audit.h
            ${LLAMA_MODELS_SOURCES}
            )

set_target_properties(llama PROPERTIES
    VERSION ${LLAMA_INSTALL_VERSION}
    SOVERSION 0
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
    WINDOWS_EXPORT_ALL_SYMBOLS ON  # export all alpha_* symbols for Windows DLL
)

target_include_directories(llama PRIVATE . ../vendor)
target_include_directories(llama PUBLIC ../include)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump

target_link_libraries(llama PUBLIC ggml)
if (WIN32)
    target_link_libraries(llama PRIVATE bcrypt)  # AES-256-GCM for .apack decryption
else()
    # Linux/macOS: real ACX (AES-256-GCM) crypto for .apack decryption. Without this, libllama compiles
    # alpha_crypto.cpp as the OpenSSL-less stub (alpha_acx_decrypt_file returns {}); and because
    # the encrypted-state loaders live in libllama, their calls bind to that stub intra-library,
    # so a per-binary real crypto object cannot override them and every engine fails to decrypt
    # encrypted packs. Provide real OpenSSL here so libllama is the single source of truth.
    find_package(OpenSSL QUIET)
    if (OpenSSL_FOUND)
        target_link_libraries(llama PRIVATE OpenSSL::SSL OpenSSL::Crypto)
        target_compile_definitions(llama PRIVATE ALPHA_HAS_OPENSSL)
    else()
        message(WARNING "OpenSSL not found — libllama .apack/license crypto falls back to stubs")
    endif()
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)

if (ALPHA_FLASHINFER)
    target_compile_definitions(llama PUBLIC ALPHA_HAS_FLASHINFER)
    find_package(CUDAToolkit QUIET)
    if (CUDAToolkit_FOUND)
        target_link_libraries(llama PRIVATE CUDA::cudart)
    endif()
endif()
endif()

# the recurrent-state save/load/prescan engine: only in builds that ask for it (production builds carry
# none of its symbols or strings -- docs/production-build-checklist.md)
if (ALPHA_DELTANET)
    target_sources(llama PRIVATE alpha_deltanet_state.cpp alpha_deltanet_state.h)
endif()
