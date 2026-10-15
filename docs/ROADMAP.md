# Roadmap

Rough milestones, in order. Each depends on the previous one.

## M0: Pipeline (this scaffold)
- [x] Submodules: XenonRecomp, XenosRecomp, extract-xiso
- [x] WebAssembly patch for XenonRecomp (`ppc_context.h`: FPSCR, `__rdtsc`)
- [x] `ttt2-scan`: image dump, import list, save/restore helper addresses
- [x] Scripts: bootstrap, build tools, extract, scan, recompile, build web/native
- [x] Runtime skeleton: guest memory, threads, TLS, critical sections, file I/O, input
- [x] Auto-generated import stubs from the game's real import list
- [x] Web shell: capability checks, OPFS import, COOP/COEP dev server
- [x] First successful `scripts/recompile.ps1` on the real XEX
- [ ] First successful native and web builds (generated TUs compile individually; full link not attempted yet)

## M1: Recompilation is clean
- [x] Fill helper addresses (all eight found uniquely by `ttt2-scan`)
- [ ] Find setjmp/longjmp, if TTT2 uses them
- [x] Resolve every XenonRecomp warning: ~70 missing instructions added
      (`patches/XenonRecomp-0002-ttt2-instructions.patch`), zero warnings left
- [x] Jump tables: `tools/find_switch_tables.py` finds 526 (XenonAnalyse found 0) and
      fixes 69 leaf-function boundaries
- [ ] Remaining traps: 4 instructions in data-as-code (`macchwu`, `maclhwu`, `blrl`).
      Confirm they are unreachable, or mark them `invalid_instructions`
- [ ] Unit-test the new instruction implementations against Xenia's PPC tests (XenonTests)

## M2: Boots to the render loop (native first)
- [x] `ttt2.exe` links and runs: CRT and engine init complete, ~20 game threads start
      (`nuscMain`, `nuscFileAsyncRequest`, ...), audio client registers, GPU init begins
- [x] Kernel variable imports (KeTimeStampBundle, XboxHardwareInfo, module handle, ...)
- [x] Static TLS sized and initialised from the XEX TLS directory
- [x] Dispatcher objects: events, mutants, semaphores, thread objects; waits by handle
      and by guest dispatcher header (timers still to do)
- [x] Graphics interrupt at 60 Hz (vblank, source 0) with the interrupt-status register
      reading "vblank"; DPCs dispatched on their own thread
- [x] Ring buffer + minimal command processor: walks PM4, executes memory writes, fence
      events, conditional writes, scratch-register write-back, interrupts and indirect
      buffers; skips draws. `VdSwap` now fires every vblank (1000+ frames, ~60 fps)
- [x] NT wait semantics: FIFO waiter queues with direct hand-off; thread self-suspend;
      ExTerminateThread; non-trapping integer divides; fall-through across split functions
- [ ] Some worker threads wait >2 s on locks the main thread holds: check for a leak
- [ ] Remaining memory, I/O and XAM calls from the stub log (~208 still stubbed)
- [ ] Content/save APIs (XContent*), profile/sign-in (currently: nobody signed in)

## M3: Pictures
- [x] Decode draw/state PM4 packets: register, constant, fetch and shader state reach a
      software reference renderer (`runtime/gpu/soft_gpu.cpp`): Xenos microcode
      interpreter, triangle/rect/quad rasteriser with depth, blending and alpha test,
      EDRAM model, resolves (tiled, endian-swapped) with clears. The loading screen
      and the Bandai Namco splash render correctly in the native window
- [ ] Software renderer speed: ~65 ms/frame on 16 threads (shader interpretation
      dominates); points, lines, stencil, culling, MSAA resolve and several texture
      formats (e.g. depth textures, format 23) are missing
- [ ] The game stops submitting frames after a timing-dependent point (frame 8 to
      ~600; happens with rendering off too): a guest-side race to find
- [ ] Identify the game's statically linked D3D functions
- [ ] WebGPU backend: device, swap chain, render targets, depth
- [ ] XenosRecomp → HLSL → SPIR-V → WGSL shader cache
- [ ] Texture untiling and format conversion (CPU version done for the common formats)
- [ ] Menus render, then a stage plus characters

## M4: Sound
- [ ] Render-driver callback thread and AudioWorklet output
- [ ] XMA decoding (FFmpeg xma2 → wasm)

## M5: Playable in the browser
- [ ] wasm64 build runs in Chrome/Edge
- [ ] Stable 60 fps in matches (profile; enable XenonRecomp's local-register optimisations)
- [ ] Input latency pass (fighting games are sensitive to it)
- [ ] Saves persist in OPFS

## Stretch
- [ ] wasm32 fallback with address remapping (wider browser support)
- [ ] Online play via WebRTC rollback (would need deterministic hooks into the game loop)
