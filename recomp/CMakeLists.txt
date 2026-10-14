# Compiles XenonRecomp's output (recomp/ppc) into a static library.
# This is by far the slowest part of the build: expect thousands of large TUs.

file(GLOB TTT2_PPC_SOURCES CONFIGURE_DEPENDS ${TTT2_PPC_DIR}/ppc_recomp.*.cpp)

add_library(TTT2Recomp STATIC
    ${TTT2_PPC_SOURCES}
    ${TTT2_PPC_DIR}/ppc_func_mapping.cpp
)

target_include_directories(TTT2Recomp PUBLIC
    ${TTT2_PPC_DIR}
    ${TTT2_THIRDPARTY}/XenonRecomp/XenonUtils
    ${TTT2_THIRDPARTY}/XenonRecomp/thirdparty/simde
)

target_compile_options(TTT2Recomp PRIVATE
    -Wno-unused-label
    -Wno-unused-variable
    -Wno-unused-but-set-variable
    -fno-strict-aliasing
)

if (EMSCRIPTEN)
    # Browsers reject wasm functions with more than 50,000 locals or bodies over ~7.6 MB.
    # -O2 keeps giant recompiled functions under those limits far more often than -O0.
    target_compile_options(TTT2Recomp PRIVATE $<$<CONFIG:Debug>:-O1>)
endif()

if (MSVC)
    # clang-cl: x86 intrinsics used directly by ppc_context.h
    target_compile_options(TTT2Recomp PRIVATE -msse4.1 -mavx)
endif()
