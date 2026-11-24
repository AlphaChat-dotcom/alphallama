# alphallama-latest builds the server only (with the mtmd library it links)

find_package(Threads REQUIRED)

llama_add_compile_flags()

add_subdirectory(mtmd)
if (LLAMA_BUILD_SERVER)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../ui/apps/chat ${CMAKE_BINARY_DIR}/ui/apps/chat)
    add_subdirectory(server)
endif()
