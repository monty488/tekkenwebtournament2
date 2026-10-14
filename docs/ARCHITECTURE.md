# Architecture

This is the reasoning behind the design, especially where the browser forces
choices that differ from native recompilation projects such as Unleashed Recompiled.

## 1. CPU: static recompilation

XenonRecomp turns every PPC function in `default.xex` into a C++ function
`sub_XXXXXXXX(PPCContext& ctx, uint8_t* base)`. Guest memory accesses become
`base + address` with byte swapping. Indirect calls go through a lookup table
stored inside guest memory right after the image (`PPC_LOOKUP_FUNC`), filled at
startup from `PPCFuncMappings` (`runtime/main.cpp`).

What XenonRecomp needs from us (`recomp/config/TTT2.toml`):

* **Register save/restore helper addresses.** `tools/ttt2-scan` finds these by the
  byte patterns from XenonRecomp's README.
* **Jump tables.** `XenonAnalyse` detects them. Its patterns were written for
  Sonic Unleashed, so expect to add detection for TTT2's compiler output if
  `XenonRecomp` reports unresolved `bctr`s.
* **Function boundaries** that the analyser gets wrong (`functions = [...]`).
* **setjmp/longjmp** addresses, if the game uses them.

### WebAssembly-specific CPU issues

`patches/XenonRecomp-0001-wasm-support.patch` adds a wasm branch to `ppc_context.h`:

* **FPSCR.** Wasm has no rounding-mode or denormal-flush control. The patch keeps a
  soft copy of the requested mode so guest reads round-trip. Code that relies on
  non-default rounding will be slightly off. Games rarely do this.
* **`mftb` / `__rdtsc`.** Implemented from the monotonic clock at the console's
  49.875 MHz timebase, which `KeQueryPerformanceFrequency` reports.
* **SIMD.** `ppc_context.h` uses SSE through simde. With `-msimd128 -msse4.1`,
  Emscripten maps most of it straight to wasm SIMD.

Browser engine limits that the recompiled code can run into:

* **Function size.** V8 rejects functions over ~7.6 MB of bytecode or 50,000 locals.
  Huge recompiled functions must be compiled with optimisation (the Debug preset
  uses `-O1` for `recomp/`). If one still fails, split it with an explicit
  `functions` entry in the TOML.
* **Module size.** Expect a 50–150 MB `.wasm`. Use streaming compilation (the
  Emscripten default) and serve it compressed (brotli).

## 2. Memory: why wasm64

The recompiled code computes `base + guest_address` for any 32-bit guest address,
so it needs a **contiguous 4 GB window**. wasm32 linear memory is at most 4 GB in
total, which leaves no room for the runtime's own heap, the GPU upload buffers or
the C++ stack.

Options considered:

1. **wasm64 (memory64). Chosen.** Allocate the 4 GB window with `aligned_alloc`
   in a memory of up to 8 GB. Browsers commit pages lazily, so physical use tracks
   what the game touches (~512 MB, the console's RAM). Costs: memory64 needs a
   recent browser, and V8 adds explicit bounds checks to 64-bit memory accesses
   (roughly 10–20% slower than wasm32).
2. **wasm32 with address remapping.** Override `PPC_LOAD_*`/`PPC_STORE_*` to fold
   the Xbox 360 address map (0x40000000 virtual, 0x8xxxxxxx image, 0xA0000000+
   physical views) into ~1 GB. This works in every browser but adds a
   branch/mask to every load and store, and breaks `PPC_LOOKUP_FUNC`. It remains
   the fallback if memory64 performance disappoints.

Physical memory views (0xA0000000, 0xC0000000 and 0xE0000000 alias the same RAM on
hardware) are **not** aliased. All physical allocations come from the 0xA0000000
window, and `MmGetPhysicalAddress` masks to 29 bits. This holds as long as the
game uses only the pointer it was given, which Unleashed Recompiled found to be
true in practice.

## 3. Threads

Xenon has 6 hardware threads, and TTT2 uses several (render, audio, streaming,
main). Guest threads map 1:1 to pthreads (Web Workers), which requires
`SharedArrayBuffer` and therefore COOP/COEP headers (`web/serve.mjs`).

`-sPROXY_TO_PTHREAD` runs `main()` on a worker, so guest code can block on waits.
The browser main thread only handles DOM events (keyboard callbacks in
`runtime/hid/input.cpp` are registered for it) and presentation.

Each guest thread gets a PCR/TLS/TEB/stack block in guest memory, with `r13`
pointing at the PCR (`runtime/kernel/threads.cpp`).

**Still to do:** kernel dispatcher objects (events, semaphores, mutants,
`KeWaitForSingleObject` / `KeWaitForMultipleObjects`). These are needed early.

## 4. Kernel HLE

Each kernel/XAM import is a `__imp__Name` PPC function. The runtime implements
them with `GUEST_IMPORT(Name)`. `scripts/gen_import_stubs.py` reads the game's
real import list (`game/scan/imports.txt`) and generates a logging stub for every
import not implemented yet, so the build always links. The stub log becomes the
to-do list.

## 5. File system

The disc contents (~7 GB) cannot sit in an in-memory filesystem. With `-sWASMFS`,
the runtime mounts an **OPFS** backend at `/game`. `web/loader.js` copies the
user's extracted folder into OPFS once, streaming it, and later launches read it
directly. `runtime/vfs` maps `game:\`, `d:\` and `\Device\Cdrom0\` to that mount,
with case-insensitive lookups.

Save data (XContent APIs) will also go to OPFS under a separate directory.

## 6. Graphics (the big one)

There are two ways to replace the Xenos GPU:

| | Low level: emulate the GPU | High level: hook the game's D3D (chosen) |
| --- | --- | --- |
| Hook point | PM4 command ring (`VdInitializeRingBuffer`, `VdSwap`) | Statically linked Xbox D3D functions in the game (`D3DDevice_*`) |
| Game specific? | No | Yes: functions must be identified in TTT2 |
| Needs EDRAM / tiling emulation | Yes, in full | Mostly no |
| Browser performance | Poor. This is Xenia's model, which is heavy even natively. | Good. One WebGPU call per D3D call. |

For a 60 fps fighting game in a browser, the **high-level** approach is the only
one likely to hit frame rate. Plan:

1. Identify the statically linked D3D functions. The XEX's library version table
   says which XDK D3D build TTT2 uses. Signature-match those functions (as for any
   statically linked library), then name them in the TOML.
2. Override them with `PPC_FUNC` hooks that record into a command queue consumed by a
   **WebGPU** backend (`emdawnwebgpu`).
3. **Shaders.** Use XenosRecomp to produce HLSL, compile it with DXC to SPIR-V, and
   convert that to WGSL with Tint, all **offline**. WebGPU has no bindless resources
   and no `vk::RawBufferLoad`, so XenosRecomp's `shader_common.h` needs a WebGPU
   variant: constants in uniform buffers and fixed texture/sampler slots per
   stage. See `shaders/README.md`.
4. Untile textures and convert formats on upload (Xenos tiled layouts, big-endian
   texel data, DXT/CTX1 formats). Do it on the GPU with compute shaders where possible.
5. **Presentation.** WebGPU in a worker needs an `OffscreenCanvas` transferred from
   the page (`-sOFFSCREENCANVAS_SUPPORT`). Otherwise, proxy presentation to the main
   thread.

## 7. Audio

The game's XAudio2 is statically linked. It mixes in software and hands 256-sample,
6-channel float frames to the kernel render driver
(`XAudioRegisterRenderDriverClient` / `XAudioSubmitRenderDriverFrame`, stubbed
in `runtime/apu/audio.cpp`). The runtime must:

* call the registered client callback every 256 samples (5.33 ms at 48 kHz),
  from a dedicated guest thread;
* downmix 5.1 to stereo and push into a `SharedArrayBuffer` ring buffer read by an
  **AudioWorklet**;
* decode **XMA**, the hardware decoder exposed through `XMA*` kernel calls.
  FFmpeg's `xma2` decoder, compiled to wasm, is the most direct route.

## 8. Input

`XamInputGetState` maps to the Gamepad API (standard mapping) plus a keyboard
layout. The keyboard layout follows Tekken's 1/2/3/4 buttons
(`runtime/hid/input.cpp`). Rumble via `vibrationActuator` is still to do.

## 9. Native build

The same runtime builds natively with clang-cl (`scripts/build-native.ps1`). Do the
bring-up there first. The debugger, crash dumps and sanitizers are much better
than in the browser. Then move to wasm once the game reaches its render loop.
