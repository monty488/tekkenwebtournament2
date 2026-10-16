if (TTT2_HAVE_RECOMP)
    add_executable(ttt2
        main.cpp
        kernel/memory.cpp
        kernel/module.cpp
        kernel/trace.cpp
        kernel/sync.cpp
        kernel/xconfig.cpp
        kernel/xam.cpp
        kernel/xam_ui.cpp
        kernel/dpc.cpp
        kernel/objects.cpp
        kernel/threads.cpp
        kernel/io.cpp
        vfs/vfs.cpp
        hid/input.cpp
        gpu/video.cpp
        gpu/texture.cpp
        gpu/soft_gpu.cpp
        gpu/xenos_wgsl.cpp
        gpu/webgpu_backend.cpp
        gpu/presenter_win.cpp
        apu/audio.cpp
        apu/xma.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../recomp/generated/import_stubs.cpp
    )
    target_include_directories(ttt2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(ttt2 PRIVATE TTT2Recomp)
else()
    add_executable(ttt2 placeholder.cpp)
endif()

if (WIN32)
    target_compile_definitions(ttt2 PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
endif()

# WebGPU: wgpu-native natively (scripts/fetch-wgpu.ps1), emdawnwebgpu in the browser.
if (TTT2_HAVE_RECOMP AND NOT EMSCRIPTEN)
    set(WGPU_NATIVE_DIR ${CMAKE_SOURCE_DIR}/.tools/wgpu-native)
    if (NOT EXISTS ${WGPU_NATIVE_DIR}/include/webgpu/webgpu.h)
        message(FATAL_ERROR "wgpu-native not found in ${WGPU_NATIVE_DIR}: run scripts\fetch-wgpu.ps1")
    endif()
    target_include_directories(ttt2 PRIVATE ${WGPU_NATIVE_DIR}/include)
    target_link_libraries(ttt2 PRIVATE ${WGPU_NATIVE_DIR}/lib/wgpu_native.dll.lib)
    add_custom_command(TARGET ttt2 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${WGPU_NATIVE_DIR}/lib/wgpu_native.dll $<TARGET_FILE_DIR:ttt2>
    )
endif()

if (MSVC AND TTT2_HAVE_RECOMP)
    target_compile_options(ttt2 PRIVATE -msse4.1 -mavx)
    # RelWithDebInfo defaults to /Ob1 (only explicitly inline functions), which leaves the
    # software renderer's small per-pixel helpers as calls; inline like Release does.
    target_compile_options(ttt2 PRIVATE $<$<CONFIG:RelWithDebInfo>:/Ob2>)
endif()

if (EMSCRIPTEN)
    set_target_properties(ttt2 PROPERTIES SUFFIX ".js")
    # emdawnwebgpu provides webgpu/webgpu.h for the WebGPU backend.
    target_compile_options(ttt2 PRIVATE --use-port=emdawnwebgpu)
    target_link_options(ttt2 PRIVATE
        -sENVIRONMENT=web,worker
        -sMODULARIZE=1
        -sEXPORT_ES6=1
        -sEXPORT_NAME=createTTT2
        -sEXPORTED_RUNTIME_METHODS=['FS','ENV']
        # Memory: the 4 GB guest window is one allocation; leave headroom for the host heap.
        -sALLOW_MEMORY_GROWTH=1
        -sINITIAL_MEMORY=256MB
        -sMAXIMUM_MEMORY=8GB
        # Threads: main() runs on a worker so guest code may block.
        -sPROXY_TO_PTHREAD=1
        # ~25 game threads plus the renderer pool and service threads; more start on demand.
        -sPTHREAD_POOL_SIZE=48
        -sDEFAULT_PTHREAD_STACK_SIZE=1MB
        -sSTACK_SIZE=4MB
        # Files: WASMFS so game data can live in OPFS (no 8 GB in-memory filesystem).
        -sWASMFS=1
        -sSUPPORT_LONGJMP=wasm
        -sWASM_BIGINT=1
        --use-port=emdawnwebgpu
        # WebGPU renders on the command processor's worker: the page canvas is transferred
        # to it at thread creation (not to main()'s thread), which needs the GL bookkeeping.
        -sOFFSCREENCANVAS_SUPPORT=1
        "-sOFFSCREENCANVASES_TO_PTHREAD=''"
        "-sDEFAULT_LIBRARY_FUNCS_TO_INCLUDE=$GL"
    )

    # Stage the web shell next to the build output so `node web/serve.mjs <build dir>` just works.
    add_custom_command(TARGET ttt2 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/web/public $<TARGET_FILE_DIR:ttt2>
    )
endif()
