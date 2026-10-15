if (TTT2_HAVE_RECOMP)
    add_executable(ttt2
        main.cpp
        kernel/memory.cpp
        kernel/module.cpp
        kernel/trace.cpp
        kernel/sync.cpp
        kernel/xconfig.cpp
        kernel/xam.cpp
        kernel/dpc.cpp
        kernel/objects.cpp
        kernel/threads.cpp
        kernel/io.cpp
        vfs/vfs.cpp
        hid/input.cpp
        gpu/video.cpp
        gpu/texture.cpp
        gpu/soft_gpu.cpp
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

if (MSVC AND TTT2_HAVE_RECOMP)
    target_compile_options(ttt2 PRIVATE -msse4.1 -mavx)
endif()

if (EMSCRIPTEN)
    set_target_properties(ttt2 PROPERTIES SUFFIX ".js")
    target_link_options(ttt2 PRIVATE
        -sENVIRONMENT=web,worker
        -sMODULARIZE=1
        -sEXPORT_ES6=1
        -sEXPORT_NAME=createTTT2
        -sEXPORTED_RUNTIME_METHODS=['FS']
        # Memory: the 4 GB guest window is one allocation; leave headroom for the host heap.
        -sALLOW_MEMORY_GROWTH=1
        -sINITIAL_MEMORY=256MB
        -sMAXIMUM_MEMORY=8GB
        # Threads: main() runs on a worker so guest code may block.
        -sPROXY_TO_PTHREAD=1
        -sPTHREAD_POOL_SIZE=12
        -sDEFAULT_PTHREAD_STACK_SIZE=1MB
        -sSTACK_SIZE=4MB
        # Files: WASMFS so game data can live in OPFS (no 8 GB in-memory filesystem).
        -sWASMFS=1
        -sSUPPORT_LONGJMP=wasm
        -sWASM_BIGINT=1
        -sOFFSCREENCANVAS_SUPPORT=1
        --use-port=emdawnwebgpu
    )

    # Stage the web shell next to the build output so `node web/serve.mjs <build dir>` just works.
    add_custom_command(TARGET ttt2 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/web/public $<TARGET_FILE_DIR:ttt2>
    )
endif()
