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
- [x] XAM system UI, enumerators and content calls complete their XOVERLAPPED requests
      (message boxes, device selector, achievement/content enumerators, content
      create/close/delete; `runtime/kernel/xam_ui.cpp`)
- [ ] Remaining memory, I/O and XAM calls from the stub log (145 still stubbed)
- [ ] Save data backed by host files (content creation is rejected cleanly for now; with
      nobody signed in the game does not save), profiles/sign-in (currently: nobody)

## M3: Pictures
- [x] Decode draw/state PM4 packets: register, constant, fetch and shader state reach a
      software reference renderer (`runtime/gpu/soft_gpu.cpp`): Xenos microcode
      interpreter, triangle/rect/quad rasteriser with depth, blending and alpha test,
      EDRAM model, resolves (tiled, endian-swapped) with clears. The loading screen
      and the Bandai Namco splash render correctly in the native window
- [x] Past the boot logos: the intro movie (YUV planes + subtitle strips), the title
      screen and the attract-mode screens ("Get ready for the next battle", the VS
      screen with character portraits) render correctly in dumped frames
- [x] Stencil, face culling (PA_SU_SC_MODE_CNTL, strip winding), averaged MSAA resolves,
      depth/stencil resolves (k_24_8, k_24_8_FLOAT), gamma render targets (PWL encode on
      resolve), texture sign/gamma/exp_adjust, and float/16-bit/depth/DXN/DXT3A/DXT5A
      texture formats. Culling is unverified until a 3D scene is reachable
- [ ] Software renderer gaps: points and lines, mip levels/LOD (mip 0 only), cube and
      3D textures, per-sample coverage (MSAA edges)
- [ ] Software renderer speed: ~45-65 ms/frame on 16 threads (shader interpretation
      dominates)
- [ ] The bottom-corner "PRESS START" overlaps a "FREE PLAY" string: the game submits
      both, fully opaque, in one strip draw (probably game-side state, e.g. a blink
      timer or credit mode; not yet explained)
- [x] The game stops submitting frames after a timing-dependent point: fixed in the
      kernel HLE (audio callback argument, wakeup storms, critical sections)
- [x] Menus and fights are reachable natively: title -> "not signed in" dialog -> main
      menu; Offline > Arcade reaches the VS screen and a Stage 1 fight (HUD, timer, 3D
      stage); Fight Lab loads a Combot match. Unattended:
      `TTT2_AUTOPRESS=start,start,start,none,a,none,none,a,none,a,none,a,... TTT2_AUTOPRESS_DELAY=40000`
- [x] The attract-mode fight crash (824B0DA4 was a missed switch-table case)
- [x] WebGPU backend (`TTT2_GPU=webgpu`, wgpu-native natively): Xenos vertex and pixel
      shaders translated to WGSL at runtime (`runtime/gpu/xenos_wgsl.cpp`, cached by
      their flattened instructions), vertex fetch from guest memory with the viewport
      transform in the shader, render targets keyed by EDRAM base in sample space
      (MSAA views alias like on the console), unorm/float target formats, blending,
      depth/stencil, culling; colour and depth resolves stay on the GPU; the front buffer
      is presented straight to the window. Menus, character select, VS screens, fights
      and post-fight screens render correctly; fights run at 60 fps with the command
      processor busy ~11 of each 16.7 ms
- [x] `TTT2_GPU_VERIFY_FRAME=n`: every draw of a frame rendered by both renderers from
      the same inputs and compared (found the MSAA aliasing and precision bugs)
- [ ] WebGPU gaps: pixel shaders with predicate-driven control flow (colour skipped,
      logged once), rect lists still shaded on the CPU, points/lines, mip levels, cube and
      3D textures, real MSAA (MSAA views are supersampled), 64bpp/32bpp aliasing of the
      same EDRAM, Xenos 7e3 float colour precision
- [ ] Check Kuma's dark face on the post-fight screen with the verifier (lighting or bug)
- [x] The early startup stall (frame 4-8, most starts in the browser): a mutant's owner
      re-acquiring it recursively was queued behind a waiter that was waiting for it
- [ ] Browser: emdawnwebgpu build of the same backend, canvas presentation

## M4: Sound
- [x] Render-driver callback thread and output: each engine's frames are downmixed to
      stereo, mixed one per tick, and played through WASAPI (native) or an AudioWorklet
      reading the ring from wasm memory (browser); the device's clock paces the callbacks.
      The intro movie and logos have sound; menus and fights need XMA
- [x] XMA decoding: FFmpeg's WMA Pro frame decoder, one frame per 512 output samples
      (`runtime/apu/xma_decoder.c`), frames found by their length fields across packets and
      input buffers; a minimal FFmpeg (avcodec + avutil, xma decoders only) is fetched and
      configured by `scripts/fetch-ffmpeg.ps1` for both targets. Menus and fights have sound
- [ ] XMA: loop subframes (loops start on frame boundaries), multi-stream contexts

## M5: Playable in the browser
- [x] wasm64 build runs in Chrome/Edge: boots from game data streamed off the dev server
      (`serve.mjs --game game`, page `?stream`), intro and logos on a 2D canvas with the
      software renderer at ~18 fps (CPU texture decoding is ~10x slower than native)
- [x] WebGPU in the browser (`?stream&TTT2_GPU=webgpu`): the canvas is transferred to the
      command-processor worker as an OffscreenCanvas, the adapter/device arrive through
      spontaneous callbacks (software rendering until then), and the processor runs in
      event-loop slices so frames are shown. The processor is busy ~0.4 ms per frame;
      the intro runs at ~26 swaps/s (native: 60), bound by the game's own threads in wasm
- [ ] GPU readbacks in the browser (resolves the CPU reads, front-buffer screenshots): they
      need an async map, which a synchronous resolve cannot wait for; they return zeros now
- [x] A thread of the game spun on its own event: `sub_82ABFDE8` sets an auto-reset event
      and SignalAndWaits on it until a 64-bit counter changes (~8 M kernel calls/s, all on
      the global wait lock). Waits now detect a run of self-set signals and block up to
      1 ms (a set from another thread still wakes them). Browser intro: ~26 -> 60 fps
- [x] Fights in the browser at 60 fps (Edge, Ryzen 7800X3D; native-like frame times):
      critical sections waited through libc++'s `atomic::wait`, which on emscripten polls
      with growing sleeps (up to 8 ms) and ignores notify: now an emscripten futex. Also:
      ring uploads batched into one queue write per buffer per submit; per-draw stats clock
      reads only with `TTT2_GPU_STATS=1` (each is a JS call); GPU-kept resolves no longer
      force overlapping textures (two depth formats, every frame) to be decoded again
- [ ] The command processor busy-waits between slices in the browser (`sched_yield` does
      nothing there): block on the ring write pointer instead, to free a core
- [ ] Texture change detection hashes ~6 MB per frame in fights
- [ ] Stable 60 fps in matches (profile; enable XenonRecomp's local-register optimisations)
- [ ] Input latency pass (fighting games are sensitive to it)
- [ ] Saves persist in OPFS

## Stretch
- [ ] wasm32 fallback with address remapping (wider browser support)
- [ ] Online play via WebRTC rollback (would need deterministic hooks into the game loop)
