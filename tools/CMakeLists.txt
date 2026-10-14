# Host-side tooling. Always built natively (never with Emscripten):
#   XenonAnalyse, XenonRecomp  - from thirdparty/XenonRecomp
#   ttt2-scan                  - our XEX inspector (tools/ttt2-scan)
#
# XenosRecomp and extract-xiso are separate CMake projects because their
# thirdparty trees define targets that collide with XenonRecomp's (fmt, xxHash);
# scripts/build-tools.ps1 builds all three.
cmake_minimum_required(VERSION 3.20)
project(TTT2Tools CXX C)

if (EMSCRIPTEN)
    message(FATAL_ERROR "tools/ must be built for the host, not with Emscripten. Use the 'tools' CMake preset.")
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/XenonRecomp XenonRecomp)

add_executable(ttt2-scan ttt2-scan/main.cpp)
target_link_libraries(ttt2-scan PRIVATE XenonUtils)
target_compile_features(ttt2-scan PRIVATE cxx_std_17)
