set(TARGET alphachat)
add_executable(${TARGET}
    alphachat.cpp
    ${PROJECT_SOURCE_DIR}/alpha_lora_train.cpp
)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama llama-common cpp-httplib ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/common
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/vendor/cpp-httplib
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Strip symbols in Release
if(NOT MSVC)
    target_link_options(${TARGET} PRIVATE $<$<CONFIG:Release>:-s>)
endif()
if(MSVC)
    # /OPT:REF removes unreferenced functions, /OPT:ICF folds identical ones
    target_link_options(${TARGET} PRIVATE $<$<CONFIG:Release>:/OPT:REF /OPT:ICF>)
endif()
