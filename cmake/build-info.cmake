set(BUILD_NUMBER 0)
set(BUILD_COMMIT "unknown")
set(BUILD_COMPILER "unknown")
set(BUILD_TARGET "unknown")

# Look for git
find_package(Git)
if(NOT Git_FOUND)
    find_program(GIT_EXECUTABLE NAMES git git.exe)
    if(GIT_EXECUTABLE)
        set(Git_FOUND TRUE)
        message(STATUS "Found Git: ${GIT_EXECUTABLE}")
    else()
        message(WARNING "Git not found. Build info will not be accurate.")
    endif()
endif()

# Get the commit count and hash
if(Git_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE HEAD
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE RES
    )
    if (RES EQUAL 0)
        set(BUILD_COMMIT ${HEAD})
    endif()
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-list --count HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE COUNT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE RES
    )
    if (RES EQUAL 0)
        set(BUILD_NUMBER ${COUNT})
    endif()
endif()

# [ALPHA] SOURCE IDENTITY -- never let a build claim an identity it does not have.
#
# Two silent failures used to reach the binary as a confident-looking value:
#   1. no git (e.g. a tree deployed as a tarball)     -> BUILD_COMMIT stayed the literal "unknown"
#   2. git present but the worktree DIRTY             -> BUILD_COMMIT got the CLEAN commit hash,
#                                                        i.e. it named source that is not what built
# (2) is the worse one: it is not missing information, it is wrong information.
#
# Fix: when the commit cannot describe the tree, derive an identity from the CONTENT -- a SHA256 over
# a sorted manifest of "<file-sha256>  <relative-path>". Deterministic, needs no repo, and changes iff
# the source changes. Reported as "src:<hash>" (no repo) or "<commit>-dirty:<hash>" (uncommitted work),
# so a reader can always tell which of the three cases produced a number.
function(alpha_source_identity OUTVAR)
    file(GLOB_RECURSE _alpha_srcs LIST_DIRECTORIES false
         "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
         "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
         "${CMAKE_CURRENT_SOURCE_DIR}/common/*.c"  "${CMAKE_CURRENT_SOURCE_DIR}/common/*.cpp"
         "${CMAKE_CURRENT_SOURCE_DIR}/common/*.h"
         "${CMAKE_CURRENT_SOURCE_DIR}/tools/*.c"   "${CMAKE_CURRENT_SOURCE_DIR}/tools/*.cpp"
         "${CMAKE_CURRENT_SOURCE_DIR}/tools/*.cu"  "${CMAKE_CURRENT_SOURCE_DIR}/tools/*.h"
         "${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/*.c"   "${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/*.cpp"
         "${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/*.cu"  "${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/*.h"
         "${CMAKE_CURRENT_SOURCE_DIR}/ggml/include/*.h"
         "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")
    list(SORT _alpha_srcs)                      # order must not depend on the filesystem
    set(_alpha_manifest "")
    foreach(_f IN LISTS _alpha_srcs)
        file(SHA256 "${_f}" _fh)
        file(RELATIVE_PATH _rel "${CMAKE_CURRENT_SOURCE_DIR}" "${_f}")
        string(APPEND _alpha_manifest "${_fh}  ${_rel}
")
    endforeach()
    list(LENGTH _alpha_srcs _n)
    string(SHA256 _alpha_all "${_alpha_manifest}")
    string(SUBSTRING "${_alpha_all}" 0 12 _alpha_short)
    message(STATUS "Alpha source identity: src:${_alpha_short} (${_n} files hashed)")
    # NOTE: configure_file() runs at CONFIGURE time only. Upstream re-triggers configure via
    # CMAKE_CONFIGURE_DEPENDS on ${GIT_DIR}/index -- which does NOT exist on a tree with no git,
    # i.e. exactly the trees this hash exists for. Without the line below the identity freezes at
    # the first configure and every later `cmake --build` ships a STALE hash that still looks
    # authoritative. Found by using it: a bench-gate ledger row carried the previous build id.
    # Registering the hashed files makes any source edit re-trigger configure (~1s for 979 files).
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_alpha_srcs})
    set(${OUTVAR} "${_alpha_short}" PARENT_SCOPE)
endfunction()

if (BUILD_COMMIT STREQUAL "unknown")
    alpha_source_identity(ALPHA_SRC_ID)
    set(BUILD_COMMIT "src:${ALPHA_SRC_ID}")
    message(WARNING
        "This tree has no usable git identity, so the build is stamped with a SOURCE CONTENT HASH: "
        "${BUILD_COMMIT}. Numbers measured against this binary cannot be reproduced from a commit -- "
        "archive the source alongside them, or deploy from git instead of a tarball.")
else()
    execute_process(
        COMMAND ${GIT_EXECUTABLE} status --porcelain --untracked-files=no
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE ALPHA_DIRTY OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE ALPHA_DRES)
    if (ALPHA_DRES EQUAL 0 AND NOT "${ALPHA_DIRTY}" STREQUAL "")
        alpha_source_identity(ALPHA_SRC_ID)
        set(BUILD_COMMIT "${BUILD_COMMIT}-dirty:${ALPHA_SRC_ID}")
        message(WARNING
            "Worktree has uncommitted changes; build stamped ${BUILD_COMMIT}. The commit alone does NOT "
            "describe this binary.")
    endif()
endif()

set(BUILD_COMPILER "${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")

if(CMAKE_VS_PLATFORM_NAME)
    set(BUILD_TARGET ${CMAKE_VS_PLATFORM_NAME})
else()
    set(BUILD_TARGET "${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
endif()
