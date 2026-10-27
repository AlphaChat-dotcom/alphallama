include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

# server-context containing the core server logic, used by llama-server and CLI

set(TARGET server-context)

add_library(${TARGET} STATIC
    server-task.cpp
    server-task.h
    server-queue.cpp
    server-queue.h
    server-common.cpp
    server-common.h
    server-context.cpp
    server-context.h
    server-pipeline.cpp
    server-pipeline.h
    server-tools.cpp
    server-tools.h
    server-help.cpp
    server-help.h
    oracle_embed.cpp
    oracle_embed.h
    server-kv-pool.cpp
    server-kv-pool.h
    ${CMAKE_SOURCE_DIR}/src/alpha_crypto.cpp
    ${CMAKE_SOURCE_DIR}/src/alpha_crypto.h
)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Redis retrieval backend (optional, requires libhiredis)
option(ALPHA_REDIS "Enable Redis retrieval backend (requires libhiredis)" OFF)
if (ALPHA_REDIS)
    find_library(HIREDIS_LIB hiredis HINTS $ENV{HOME}/local/lib /usr/local/lib)
    find_path(HIREDIS_INCLUDE hiredis/hiredis.h HINTS $ENV{HOME}/local/include /usr/local/include)
    if (HIREDIS_LIB AND HIREDIS_INCLUDE)
        target_sources(server-context PRIVATE server-redis.cpp server-redis.h)
        target_include_directories(server-context PRIVATE ${HIREDIS_INCLUDE})
        target_link_libraries(server-context PRIVATE ${HIREDIS_LIB})
        # kv-pool.cpp also uses hiredis when ALPHA_HAS_REDIS is defined
        target_compile_definitions(server-context PRIVATE ALPHA_HAS_REDIS)
        message(STATUS "Redis retrieval backend enabled (hiredis: ${HIREDIS_LIB})")
    else()
        message(WARNING "ALPHA_REDIS enabled but hiredis not found — Redis disabled")
    endif()
endif()

# Dense-L0 semantic retrieval backend (optional: bge-base GGUF embedder + cosine store)
if (ALPHA_DENSE_L0)
    target_sources(server-context PRIVATE server-dense.cpp server-dense.h)
    target_compile_definitions(server-context PRIVATE ALPHA_DENSE_L0)
    message(STATUS "Dense-L0 semantic retrieval backend enabled")
endif()

find_package(OpenSSL QUIET)
if (OpenSSL_FOUND)
    target_link_libraries(${TARGET} PUBLIC llama-common mtmd ${CMAKE_THREAD_LIBS_INIT} OpenSSL::SSL OpenSSL::Crypto)
    target_compile_definitions(${TARGET} PRIVATE ALPHA_HAS_OPENSSL)
else()
    message(WARNING "OpenSSL not found — license enforcement and checkpoint encryption disabled")
    target_link_libraries(${TARGET} PUBLIC llama-common mtmd ${CMAKE_THREAD_LIBS_INIT})
endif()


# inference engine

set(TARGET alphallama)

set(TARGET_SRCS
    server.cpp
    server-http.cpp
    server-http.h
    server-models.cpp
    server-models.h
)

option(LLAMA_BUILD_WEBUI "Build the embedded Web UI" ON)

if (LLAMA_BUILD_WEBUI)
    set(PUBLIC_ASSETS
        index.html
        bundle.js
        bundle.css
        loading.html
        alphaicon.png
    )

    foreach(asset ${PUBLIC_ASSETS})
        set(input "${CMAKE_CURRENT_SOURCE_DIR}/public/${asset}")
        set(output "${CMAKE_CURRENT_BINARY_DIR}/${asset}.hpp")
        list(APPEND TARGET_SRCS ${output})
        add_custom_command(
            DEPENDS "${input}"
            OUTPUT "${output}"
            COMMAND "${CMAKE_COMMAND}" "-DINPUT=${input}" "-DOUTPUT=${output}" -P "${PROJECT_SOURCE_DIR}/scripts/xxd.cmake"
        )
        set_source_files_properties(${output} PROPERTIES GENERATED TRUE)
    endforeach()
    add_definitions(-DLLAMA_BUILD_WEBUI)
else()
endif()

add_executable(${TARGET} ${TARGET_SRCS})
install(TARGETS ${TARGET} RUNTIME)

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PRIVATE server-context PUBLIC llama-common cpp-httplib ${CMAKE_THREAD_LIBS_INIT})

# Re-enable RTTI for the server: Jinja chat templates use dynamic_cast,
# which crashes with /GR- (STATUS_STACK_BUFFER_OVERRUN on Windows)
if (MSVC)
    target_compile_options(${TARGET} PRIVATE /GR)
    target_compile_options(server-context PRIVATE /GR)
else()
    target_compile_options(${TARGET} PRIVATE -frtti)
    target_compile_options(server-context PRIVATE -frtti)
endif()

target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Oracle model embedding (optional)
option(ALPHA_EMBED_ORACLE "Embed oracle-0.8b.gguf into the binary" OFF)
set(ALPHA_ORACLE_GGUF "" CACHE FILEPATH "Path to oracle-0.8b.gguf for embedding")

if (ALPHA_EMBED_ORACLE)
    if (NOT ALPHA_ORACLE_GGUF OR NOT EXISTS "${ALPHA_ORACLE_GGUF}")
        message(FATAL_ERROR "ALPHA_EMBED_ORACLE=ON but ALPHA_ORACLE_GGUF='${ALPHA_ORACLE_GGUF}' not found")
    endif()

    message(STATUS "Embedding oracle model: ${ALPHA_ORACLE_GGUF}")
    target_compile_definitions(server-context PRIVATE ALPHA_ORACLE_EMBEDDED)

    if (WIN32)
        # Windows: embed via resource compiler (.rc)
        file(TO_NATIVE_PATH "${ALPHA_ORACLE_GGUF}" ALPHA_ORACLE_GGUF_NATIVE)
        # Escape backslashes for the .rc file
        string(REPLACE "\\" "\\\\" ALPHA_ORACLE_GGUF_NATIVE "${ALPHA_ORACLE_GGUF_NATIVE}")
        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/oracle_embed.rc.in"
            "${CMAKE_CURRENT_BINARY_DIR}/oracle_embed.rc"
            @ONLY
        )
        target_sources(${TARGET} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/oracle_embed.rc")
    else()
        # Unix: embed via .incbin in assembly
        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/oracle_embed_unix.s.in"
            "${CMAKE_CURRENT_BINARY_DIR}/oracle_embed_unix.s"
            @ONLY
        )
        target_sources(${TARGET} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/oracle_embed_unix.s")
    endif()
endif()
