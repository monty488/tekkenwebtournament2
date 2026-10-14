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
- [ ] First successful `scripts/recompile.ps1` on the real XEX
- [ ] First successful native and web builds (none of the C++ has been compiled yet)

## M1: Recompilation is clean
- [ ] Fill helper addresses; find setjmp/longjmp
- [ ] Resolve every XenonRecomp warning (missing instructions, unresolved jump tables)
- [ ] Teach XenonAnalyse TTT2's jump table patterns if needed
- [ ] Mark non-code data in `.text` as `invalid_instructions`

## M2: Boots to the render loop (native first)
- [ ] Dispatcher objects: events, semaphores, mutants, waits, timers
- [ ] Remaining memory, I/O and XAM calls from the stub log
- [ ] Content/save APIs (XContent*), user/profile stubs (signed-in local user)
- [ ] `VdSwap` reached repeatedly (shown in the log)

## M3: Pictures
- [ ] Identify the game's statically linked D3D functions
- [ ] WebGPU backend: device, swap chain, render targets, depth
- [ ] XenosRecomp → HLSL → SPIR-V → WGSL shader cache
- [ ] Texture untiling and format conversion
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
