# TTT2 Web

An in-progress **static recompilation** of *Tekken Tag Tournament 2* (Xbox 360) to
WebAssembly, so it runs in a browser tab. Same approach as
[Unleashed Recompiled](https://github.com/hedge-dev/UnleashedRecomp): the game's
PowerPC code is translated to C++ ahead of time with
[XenonRecomp](https://github.com/hedge-dev/XenonRecomp), its shaders with
[XenosRecomp](https://github.com/hedge-dev/XenosRecomp), and a hand-written
runtime replaces the Xbox 360 kernel, GPU, audio and input.

> **Status: playable natively, booting in the browser.** The game boots, renders and
> plays: menus, character select and fights work natively at 60 fps on the WebGPU
> backend. In Chrome/Edge it runs from streamed game data and renders on the GPU through
> the browser's WebGPU, with the intro at 60 fps. Still missing: audio (XMA decoding),
> saves and sign-in, GPU readbacks in the browser, and a full play-through in the browser.
> See [docs/ROADMAP.md](docs/ROADMAP.md).

**No game data is included or ever committed.** You need your own copy of the
game. Everything derived from it (`game/`, `recomp/ppc/`, generated shaders) is
gitignored and must not be redistributed.

## How it works

```
your disc image (.iso)
   │  scripts/extract-game.ps1        extract-xiso
   ▼
game/files/default.xex + data
   │  scripts/scan-xex.ps1            ttt2-scan + XenonAnalyse
   ▼
game/scan/image.bin, imports.txt, recomp_addresses.toml
recomp/config/TTT2_switch_tables.toml
   │  (copy addresses into recomp/config/TTT2.toml)
   │  scripts/recompile.ps1           XenonRecomp + gen_import_stubs.py
   ▼
recomp/ppc/*.cpp                      game code as C++
   │  scripts/build-web.ps1           Emscripten: wasm64 + pthreads + WebGPU
   ▼
build/web-release/ttt2.{js,wasm} + web shell
   │  node web/serve.mjs build/web-release
   ▼
browser: import game folder into OPFS, then Launch
```

## Quick start (Windows)

```powershell
git clone --recursive https://github.com/monty488/tekkenwebtournament2.git
cd tekkenwebtournament2

# 1. Toolchain: CMake, Ninja, LLVM, Emscripten. Add -WithMsvc if you don't have Visual Studio.
scripts\bootstrap.ps1 -WithMsvc

# 2. Host tools (new terminal so PATH is updated)
scripts\build-tools.ps1

# 3. Your game
scripts\extract-game.ps1 -Iso "D:\path\to\TTT2.iso"
scripts\scan-xex.ps1
#    -> paste game\scan\recomp_addresses.toml into recomp\config\TTT2.toml
scripts\recompile.ps1

# 4. Build and run
scripts\build-native.ps1          # debug the bring-up natively first
scripts\build-web.ps1
node web\serve.mjs build\web-release
```

Open http://localhost:8080, import your `game` folder, press **Launch**.

For development, skip the ~7 GB import and stream the game folder from the dev server
instead (files are read on demand, in 1 MiB ranges, and never leave your machine):

```powershell
node web\serve.mjs build\web-release 8080 --game game
```

then open http://localhost:8080/?stream (add `&autostart` to launch straight away;
any `TTT2_*` switch can be passed in the URL too, e.g. `&TTT2_GPU_STATS=1`). To render
on the GPU, as natively: http://localhost:8080/?stream&TTT2_GPU=webgpu&autostart

You can build and serve the web target **before** you have the game: it then
builds a placeholder runtime, which is a quick way to check the toolchain and
browser support.

## Debugging the bring-up

The native build has switches for chasing startup problems:

| Variable | Effect |
| --- | --- |
| `TTT2_TRACE=1` | Logs every kernel/XAM call: thread, name, `r3`–`r6`, return address |
| `TTT2_WATCHDOG=<seconds>` | If no kernel call happens for that long (default 5), prints each guest thread's last call and the recompiled functions it is running (`0` disables) |
| `TTT2_SAMPLE_AT=<ms>[,<count>,<interval ms>,<thread>]` | Dumps guest thread stacks at that time (optionally `count` times, `interval` apart, one thread only), even while some threads keep calling the kernel |
| `TTT2_GPU_LOG=1` | Logs swaps, packet counts and the first draws (with a full state dump of the first) |
| `TTT2_GPU_LOG_FRAME=<n>` | Logs every draw of swap `n`, with its textures |
| `TTT2_GPU_STATS=1` | Every 10 swaps, prints software-renderer time per stage (vertex, raster, texture decode, resolve) |
| `TTT2_WAIT_STATS=1` | Every 5 s, prints the guest call sites whose timed waits and sleeps took longest (asked vs. actual time) and how long blocked threads took to wake |
| `TTT2_GPU_STATS=2` | Every 10 swaps, prints only the frame time and command-processor time (no per-draw timing, which is costly in the browser) |
| `TTT2_DUMP_FRAMES=<n,...>` | Writes the presented image of those swaps to `frame_<n>.bmp` in the working directory |
| `TTT2_GPU=webgpu` | Rasterises and shades pixels on the GPU through WebGPU (wgpu-native natively); the software renderer still shades vertices. Falls back to software if no adapter is found |
| `TTT2_GPU_VERIFY_FRAME=<n>` | WebGPU: renders every draw of swap `n` with both the GPU and the software renderer from the same inputs, logs draws whose render targets differ and dumps both images (`verify_*.bmp`) |
| `TTT2_GPU_CPU_VERTEX=1`, `TTT2_GPU_CPU_RESOLVE=1` | WebGPU: shade vertices on the CPU / read resolves back to memory (for narrowing down differences) |
| `TTT2_WGSL_DUMP=1` | Writes every pixel shader translated for the WebGPU backend to `shader_<hash>.wgsl` |
| `TTT2_SOFT_GPU=0` | Skips rendering (draws are only tracked); frames are then black but run at full speed |
| `TTT2_FILE_LOG=1` | Logs every file open and read |

Bad indirect calls (a function pointer or `bctr` to an address that isn't a
recompiled function) and access violations print a symbolised host backtrace
instead of crashing silently.

In the browser, the same switches go in the page URL (`TTT2_AUTOPRESS` included, to drive
menus unattended). To profile the web build, configure it with `-DTTT2_WEB_PROFILING=ON`
so the wasm keeps function names. Note that a profiler attached through DevTools changes
thread scheduling enough to hide some stalls: compare frame times with it detached
(`TTT2_GPU_STATS=2`).

## Browser requirements

| Feature | Why | Support (late 2025) |
| --- | --- | --- |
| WebAssembly memory64 | Guest needs a full 4 GB address window *plus* host heap | Chrome/Edge 133+, Firefox 134+ |
| SharedArrayBuffer + COOP/COEP | Game is multithreaded | All modern browsers when served by `web/serve.mjs` |
| WebGPU | Rendering | Chrome/Edge, Safari 26+, Firefox 141+ (Windows) |
| OPFS | Keep ~7 GB of game data locally | All modern browsers |

Desktop Chrome or Edge is the primary target.

## Repository layout

| Path | What |
| --- | --- |
| `thirdparty/XenonRecomp` | PPC → C++ recompiler (submodule) |
| `thirdparty/XenosRecomp` | Xenos shader → HLSL recompiler (submodule) |
| `thirdparty/extract-xiso` | Xbox disc image extractor (submodule) |
| `patches/` | Our changes to submodules (WebAssembly support), applied by the scripts |
| `tools/ttt2-scan` | Dumps the XEX image, import list, and finds the register save/restore helper addresses |
| `recomp/config` | XenonRecomp configuration for TTT2 |
| `recomp/ppc` | *(generated, ignored)* recompiled game code |
| `runtime/` | The host runtime: kernel HLE, file system, input, video, audio |
| `shaders/` | Shader pipeline notes (XenosRecomp → WGSL) |
| `web/` | Browser shell and dev server |
| `scripts/` | Pipeline scripts (PowerShell) |
| `docs/` | Architecture and roadmap |

## Legal

This project contains no copyrighted game code or assets. It is a tool to run a
copy of the game you own. Tekken is a trademark of Bandai Namco Entertainment.
This project is not affiliated with or endorsed by Bandai Namco or Microsoft.
