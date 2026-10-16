# Tekken Tag Tournament 2 - static recompilation for the web (and native, for debugging).
#
# Two targets share one runtime:
#   * Emscripten (wasm64 + pthreads + WebGPU)  ->  build/web-*/ttt2.{js,wasm} + web shell
#   * Native Windows (clang-cl)                ->  build/native-*/ttt2.exe
#
# Host tools (XenonRecomp etc.) are a separate project in tools/; see scripts/build-tools.ps1.
cmake_minimum_required(VERSION 3.20)
project(TTT2Web C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "Clang is required (XenonRecomp output relies on Clang extensions). Use clang-cl or emcc.")
endif()

set(TTT2_THIRDPARTY ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty)
set(TTT2_PPC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/recomp/ppc)

if (EXISTS ${TTT2_PPC_DIR}/ppc_config.h)
    set(TTT2_HAVE_RECOMP ON)
    message(STATUS "TTT2: recompiled game code found in recomp/ppc")
else()
    set(TTT2_HAVE_RECOMP OFF)
    message(STATUS "TTT2: recomp/ppc is empty - building the placeholder runtime only")
endif()

if (EMSCRIPTEN)
    # Shared by every translation unit, including the recompiled code.
    #   -sMEMORY64      : guest needs a full 4 GB window plus host heap (see docs/ARCHITECTURE.md)
    #   -pthread        : the game is multithreaded (6 hardware threads on Xenon)
    #   -msimd128/-msse4.1 : simde maps the SSE paths in ppc_context.h onto wasm SIMD
    #   -fwasm-exceptions  : native setjmp/longjmp support (SUPPORT_LONGJMP=wasm)
    add_compile_options(-sMEMORY64=1 -pthread -msimd128 -msse4.1 -fwasm-exceptions)
    add_link_options(-sMEMORY64=1 -pthread -fwasm-exceptions)
endif()

if (TTT2_HAVE_RECOMP)
    # Checked indirect calls for every TU (recompiled code and runtime alike); see
    # runtime/kernel/indirect_call.h.
    set(TTT2_INDIRECT_CALL_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/runtime/kernel/indirect_call.h)
    if (MSVC)
        add_compile_options("/FI${TTT2_INDIRECT_CALL_HEADER}")
        # Recompiled functions are extern "C"; /EHsc would assume they never throw, but
        # ExTerminateThread unwinds through them (runtime/kernel/threads.cpp).
        add_compile_options(/EHc-)
    else()
        add_compile_options("SHELL:-include ${TTT2_INDIRECT_CALL_HEADER}")
    endif()

    add_subdirectory(recomp)
endif()
add_subdirectory(runtime)
