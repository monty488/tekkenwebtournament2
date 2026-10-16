// Video kernel interface and a minimal Xenos command processor.
//
// The game's statically linked D3D talks to the GPU through:
//   * a ring buffer of PM4 packets (VdInitializeRingBuffer); the CPU advances the write
//     pointer by storing to the CP_RB_WPTR register, and learns how far the GPU got from
//     a read-pointer write-back slot (VdEnableRingBufferRPtrWriteBack);
//   * an interrupt callback (VdSetGraphicsInterruptCallback) called on every vblank
//     (source 0) and when the command stream asks for an interrupt (source 1).
//
// The command processor executes what the CPU side synchronises on (memory writes, fence
// events, conditional writes, interrupts, indirect buffers), tracks register/constant/
// shader state, and hands draw packets to the software renderer (soft_gpu.cpp), which
// rasterises into an emulated EDRAM and resolves into memory. VdSwap then presents the
// front buffer. The WebGPU backend will replace the software renderer
// (docs/ARCHITECTURE.md, "Graphics").

#include "video.h"
#include "presenter.h"
#include "soft_gpu.h"
#include "texture.h"
#include "../kernel/guest.h"
#include "../kernel/threads.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <thread>

namespace gpu
{
    // GPU register file is memory-mapped at 0x7FC80000 (one dword per register).
    constexpr uint32_t kRegisterBase = 0x7FC80000;
    constexpr uint32_t CP_RB_RPTR = 0x01C4;
    constexpr uint32_t CP_RB_WPTR = 0x01C5;

    static std::atomic<uint64_t> g_frames{ 0 };
    static std::atomic<uint64_t> g_vblanks{ 0 };

    static std::atomic<uint32_t> g_interruptCallback{ 0 };
    static std::atomic<uint32_t> g_interruptData{ 0 };
    static std::mutex g_interruptMutex; // interrupts are delivered one at a time

    static std::atomic<uint32_t> g_ringBase{ 0 };      // guest address
    static std::atomic<uint32_t> g_ringDwords{ 0 };
    static std::atomic<uint32_t> g_rptrWriteback{ 0 }; // guest address, 0 = disabled
    static uint32_t g_rptr = 0;                         // command processor thread only

    // ---- Stall diagnostics: TTT2_GPU_STALL=1 -------------------------------------
    // Every 2 s logs D3D's fences (device = interrupt callback data; dev+11024 -> GPU-written
    // fence, dev+11036 = last issued, dev+11088 = D3D's ring write pointer). If the fence stops
    // short of the issued one for 3 s, dumps device state and the last 512 PM4 packets.
    static const bool g_stallLog = [] { const char* v = std::getenv("TTT2_GPU_STALL"); return v && *v == '1'; }();
    struct TraceEntry { uint32_t pos, header, d0, d1, d2, depth; };
    static TraceEntry g_trace[512]; // CP thread only
    static uint32_t g_traceNext = 0;
    static uint32_t g_fenceAddr = 0;
    static uint32_t g_fenceLastValue = 0;

    uint64_t PresentedFrames() { return g_frames.load(); }

    static uint32_t PhysicalToGuest(uint32_t physical)
    {
        return 0xA0000000u + (physical & 0x1FFFFFFFu);
    }

    // Scratch registers: when a packet writes SCRATCH_REGn and bit n of SCRATCH_UMSK is set,
    // the CP mirrors the value to SCRATCH_ADDR + 4n. D3D uses this for its fences.
    constexpr uint32_t SCRATCH_UMSK = 0x01DC;
    constexpr uint32_t SCRATCH_ADDR = 0x01DD;
    constexpr uint32_t SCRATCH_REG0 = 0x0578;
    constexpr uint32_t SCRATCH_REG7 = 0x057F;

    static uint32_t ReadRegister(uint32_t index) { return guest::Load<uint32_t>(kRegisterBase + index * 4); }

    static void WriteRegister(uint32_t index, uint32_t value)
    {
        guest::Store<uint32_t>(kRegisterBase + index * 4, value);
        if (index >= SCRATCH_REG0 && index <= SCRATCH_REG7)
        {
            const uint32_t n = index - SCRATCH_REG0;
            if (ReadRegister(SCRATCH_UMSK) & (1u << n))
                guest::Store<uint32_t>(PhysicalToGuest(ReadRegister(SCRATCH_ADDR) + n * 4), value);
        }
    }

    // Writes a value the way the GPU does: the low two address bits select the endian swap.
    static void GpuWrite(uint32_t address, uint32_t value)
    {
        const uint32_t target = PhysicalToGuest(address & ~3u);
        if (g_stallLog && target == g_fenceAddr)
            g_fenceLastValue = value;
        switch (address & 3)
        {
        case 0: *guest::ToHost<uint32_t>(target) = value; break;                          // no swap: host order
        case 1: *guest::ToHost<uint32_t>(target) = ((value & 0x00FF00FF) << 8) | ((value >> 8) & 0x00FF00FF); break; // 8in16
        case 2: guest::Store<uint32_t>(target, value); break;                              // 8in32: big-endian
        case 3: *guest::ToHost<uint32_t>(target) = (value << 16) | (value >> 16); break;   // 16in32
        }
    }

    // Runs the interrupt callback as if on hardware thread `cpu`: D3D's handler reads
    // PCR->CurrentCpu to pick which CPU's interrupt bit to acknowledge and which per-CPU
    // queue (and worker thread) a command-stream callback goes to.
    static void DispatchInterrupt(const guest::GuestEnv& env, uint32_t source, uint8_t cpu)
    {
        const uint32_t callback = g_interruptCallback.load();
        if (!callback)
            return;
        std::lock_guard lock(g_interruptMutex);
        *guest::ToHost<uint8_t>(env.pcr + 0x10C) = cpu; // PCR->CurrentCpu
        guest::CallGuest(env, callback, { source, g_interruptData.load() });
    }

    // ---- PM4 -----------------------------------------------------------------

    enum : uint32_t
    {
        PM4_NOP = 0x10,
        PM4_REG_RMW = 0x21,
        PM4_COND_WRITE = 0x45,
        PM4_EVENT_WRITE = 0x46,
        PM4_ME_INIT = 0x48,
        PM4_INDIRECT_BUFFER_PFD = 0x37,
        PM4_WAIT_REG_MEM = 0x3C,
        PM4_MEM_WRITE = 0x3D,
        PM4_INDIRECT_BUFFER = 0x3F,
        PM4_INTERRUPT = 0x54,
        PM4_EVENT_WRITE_SHD = 0x58,
        PM4_EVENT_WRITE_EXT = 0x59,
        PM4_DRAW_INDX = 0x22,
        PM4_DRAW_INDX_2 = 0x36,
        PM4_IM_LOAD = 0x27,
        PM4_IM_LOAD_IMMEDIATE = 0x2B,
        PM4_SET_CONSTANT = 0x2D,
        PM4_LOAD_ALU_CONSTANT = 0x2F,
        PM4_SET_CONSTANT2 = 0x55,
        PM4_SET_SHADER_CONSTANTS = 0x56,
        PM4_XE_SWAP = 0x64, // private: written by our VdSwap (as in Xenia)
    };

    // ---- State the renderer will consume -------------------------------------

    // Shader microcode most recently loaded for each stage (IM_LOAD*): guest address
    // (0 for immediate loads, which are copied into `immediate`) and size in dwords.
    struct ShaderSlot
    {
        uint32_t address = 0;
        uint32_t dwords = 0;
        std::vector<uint32_t> immediate;
    };
    static ShaderSlot g_shaders[2]; // 0 = vertex, 1 = pixel (CP thread only)

    // Register bases for SET_CONSTANT's type field.
    static uint32_t ConstantBase(uint32_t type)
    {
        switch (type)
        {
        case 0: return 0x4000; // ALU constants (float4)
        case 1: return 0x4800; // fetch constants (textures / vertex buffers)
        case 2: return 0x4900; // boolean constants
        case 3: return 0x4908; // loop constants
        default: return 0x2000; // context registers
        }
    }

    // ---- Diagnostics: TTT2_GPU_LOG=1 ---------------------------------------------

    static const bool g_gpuLog = [] { const char* v = std::getenv("TTT2_GPU_LOG"); return v && *v == '1'; }();
    // TTT2_GPU_LOG_FRAME=N: log every draw (with textures) of swap N.
    static const uint64_t g_logFrame = [] { const char* v = std::getenv("TTT2_GPU_LOG_FRAME"); return v ? std::strtoull(v, nullptr, 10) : 0ull; }();
    static uint64_t g_swapCount = 0;
    static uint64_t g_opcodeCounts[128];
    static uint64_t g_drawsThisFrame = 0;
    static int g_drawsLogged = 0;

    // TTT2_DUMP_FRAMES=a,b,...: write the presented image of those swaps to frame_<n>.bmp.
    static bool ShouldDumpFrame(uint64_t swap)
    {
        static const char* list = std::getenv("TTT2_DUMP_FRAMES");
        if (!list)
            return false;
        for (const char* p = list; *p;)
        {
            char* end;
            const unsigned long long n = std::strtoull(p, &end, 10);
            if (end == p)
                break;
            if (n == swap)
                return true;
            p = *end ? end + 1 : end;
        }
        return false;
    }

    static void DumpBmp(uint64_t swap, const uint32_t* bgra, uint32_t width, uint32_t height)
    {
        char name[64];
        std::snprintf(name, sizeof(name), "frame_%llu.bmp", (unsigned long long)swap);
        std::FILE* f = std::fopen(name, "wb");
        if (!f)
            return;
        const uint32_t imageBytes = width * height * 4;
        uint8_t header[54] = { 'B', 'M' };
        auto put32 = [&](int at, uint32_t v) { for (int i = 0; i < 4; i++) header[at + i] = uint8_t(v >> (8 * i)); };
        put32(2, 54 + imageBytes);
        put32(10, 54);
        put32(14, 40);
        put32(18, width);
        put32(22, uint32_t(-int32_t(height))); // top-down
        header[26] = 1;
        header[28] = 32;
        put32(34, imageBytes);
        std::fwrite(header, 1, sizeof(header), f);
        std::fwrite(bgra, 4, size_t(width) * height, f);
        std::fclose(f);
        std::fprintf(stderr, "[gpu] wrote %s\n", name);
    }

    static uint64_t g_cpBusyNs = 0, g_presentNs = 0; // TTT2_GPU_STATS timings (command processor thread)

    static void OnSwap(uint32_t frontBuffer, uint32_t width, uint32_t height, const std::array<uint32_t, 6>& fetch)
    {
        const uint64_t swaps = ++g_swapCount;
        if (g_gpuLog && (swaps <= 3 || (swaps & (swaps - 1)) == 0))
        {
            std::fprintf(stderr, "[gpu] swap %llu: front buffer %08X %ux%u, fetch %08X %08X %08X %08X %08X %08X, %llu draws\n",
                (unsigned long long)swaps, frontBuffer, width, height, fetch[0], fetch[1], fetch[2], fetch[3], fetch[4], fetch[5],
                (unsigned long long)g_drawsThisFrame);
            std::fprintf(stderr, "[gpu]   packets so far:");
            for (uint32_t op = 0; op < 128; op++)
                if (g_opcodeCounts[op])
                    std::fprintf(stderr, " %02X:%llu", op, (unsigned long long)g_opcodeCounts[op]);
            std::fprintf(stderr, "\n");
        }
        g_drawsThisFrame = 0;
        SoftEndFrame();
        if (g_gpuLog && (swaps <= 3 || (swaps & (swaps - 1)) == 0))
            std::fprintf(stderr, "[gpu]   soft renderer: %llu pixels, %llu resolves so far\n",
                (unsigned long long)SoftPixelsWritten(), (unsigned long long)SoftResolves());
        static const bool softStats = [] { const char* v = std::getenv("TTT2_GPU_STATS"); return v && *v == '1'; }();
        if (softStats && swaps % 10 == 0)
        {
            // Command processor time per 10 swaps: executing packets (rendering included)
            // and converting the front buffer for presentation.
            static auto last = std::chrono::steady_clock::now();
            const auto now = std::chrono::steady_clock::now();
            std::fprintf(stderr, "[gpu] 10 swaps in %.1f ms: command processor busy %.1f ms, present %.1f ms\n",
                std::chrono::duration<double, std::milli>(now - last).count(), g_cpBusyNs / 1e6, g_presentNs / 1e6);
            last = now;
            g_cpBusyNs = 0;
            g_presentNs = 0;
            SoftPrintStats();
        }
        struct PresentTimer
        {
            std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
            ~PresentTimer() { g_presentNs += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count()); }
        } presentTimer;

        // Show the front buffer the game just swapped to.
        const TextureInfo info = DecodeTextureFetch(fetch.data());
        static std::vector<uint32_t> pixels;
        if (info.width > 1 && info.width <= 4096 && info.height <= 4096 && ConvertToBgra(info, pixels, SoftParallelFor))
        {
            for (uint32_t& p : pixels)
                p |= 0xFF000000u; // the window shows colour only
            Present(pixels.data(), info.width, info.height);
            if (ShouldDumpFrame(swaps))
                DumpBmp(swaps, pixels.data(), info.width, info.height);
        }
        else if (g_gpuLog && swaps <= 3)
        {
            std::fprintf(stderr, "[gpu] cannot present front buffer: format %u (%s) %ux%u\n", info.format,
                FormatName(info.format), info.width, info.height);
        }
    }

    // TTT2_SOFT_GPU=0 skips rendering (draws are only logged); the software renderer is
    // the default until the WebGPU backend exists.
    static const bool g_softGpu = [] { const char* v = std::getenv("TTT2_SOFT_GPU"); return !v || *v != '0'; }();

    static ShaderCode StageCode(int stage, std::vector<uint32_t>& storage)
    {
        const ShaderSlot& slot = g_shaders[stage];
        if (!slot.address)
            return { slot.immediate.data(), uint32_t(slot.immediate.size()) };
        storage.resize(slot.dwords);
        for (uint32_t i = 0; i < slot.dwords; i++)
            storage[i] = guest::Load<uint32_t>(slot.address + i * 4);
        return { storage.data(), slot.dwords };
    }

    static void Draw(uint32_t initiator, uint32_t indexBase, uint32_t indexSize, std::vector<uint32_t> immediate)
    {
        if (!g_softGpu)
            return;
        static std::vector<uint32_t> vsStorage, psStorage;
        DrawCall call;
        call.primitive = initiator & 0x3F;
        call.indexSource = (initiator >> 6) & 3;
        call.index32 = (initiator >> 11) & 1;
        call.indexCount = initiator >> 16;
        call.indexBase = indexBase ? PhysicalToGuest(indexBase) : 0;
        call.indexEndian = indexSize >> 30;
        call.immediateIndices = std::move(immediate);
        call.vertexShader = StageCode(0, vsStorage);
        call.pixelShader = StageCode(1, psStorage);
        static const bool logVertices = [] { const char* v = std::getenv("TTT2_GPU_LOG_VERTS"); return v && *v == '1'; }();
        g_softLogVertices = logVertices && g_logFrame && g_swapCount + 1 == g_logFrame;
        g_softLogDraw = g_logFrame && g_swapCount + 1 == g_logFrame;
        static const bool dumpTextures = [] { const char* v = std::getenv("TTT2_GPU_DUMP_TEX"); return v && *v == '1'; }();
        g_softDumpTextures = dumpTextures && g_logFrame && g_swapCount + 1 == g_logFrame;
        const uint64_t pixelsBefore = SoftPixelsWritten();
        SoftDraw(call);
        if (g_logFrame && g_swapCount + 1 == g_logFrame)
            std::fprintf(stderr, "[gpu]   -> %llu pixels written\n", (unsigned long long)(SoftPixelsWritten() - pixelsBefore));
    }

    static void LogDraw(uint32_t initiator, uint32_t indexBase, uint32_t indexBytes)
    {
        ++g_drawsThisFrame;
        const bool frameLog = g_logFrame && g_swapCount + 1 == g_logFrame;
        if (!frameLog && (!g_gpuLog || g_drawsLogged >= 24))
            return;
        if (!frameLog)
            ++g_drawsLogged;
        static constexpr const char* kSource[] = { "dma", "immediate", "auto", "?" };
        std::fprintf(stderr,
            "[gpu] draw prim=%u count=%u src=%s idx32=%u ib=%08X/%u | modectl=%08X surface=%08X color0=%08X depth=%08X | vs=%08X(%u) ps=%08X(%u)\n",
            initiator & 0x3F, initiator >> 16, kSource[(initiator >> 6) & 3], (initiator >> 11) & 1, indexBase, indexBytes,
            ReadRegister(0x2208), ReadRegister(0x2000), ReadRegister(0x2001), ReadRegister(0x2002),
            g_shaders[0].address, g_shaders[0].dwords, g_shaders[1].address, g_shaders[1].dwords);
        if (frameLog)
            std::fprintf(stderr, "[gpu]   depthctl=%08X stencil=%08X/%08X blend=%08X colorctl=%08X mask=%08X sc=%08X-%08X wsc=%08X-%08X su=%08X vte=%08X\n",
                ReadRegister(0x2200), ReadRegister(0x210C), ReadRegister(0x210D), ReadRegister(0x2201), ReadRegister(0x2202),
                ReadRegister(0x2104), ReadRegister(0x200E), ReadRegister(0x200F), ReadRegister(0x2081), ReadRegister(0x2082),
                ReadRegister(0x2205), ReadRegister(0x2206));
        // In frame-log mode, dump each distinct shader pair once, with the draw state.
        bool dumpState = g_drawsLogged == 1 && !frameLog;
        if (frameLog)
        {
            static std::vector<std::vector<uint32_t>> seen;
            std::vector<uint32_t> key;
            for (int stage = 0; stage < 2; stage++)
            {
                const ShaderSlot& sh = g_shaders[stage];
                key.push_back(sh.dwords);
                for (uint32_t i = 0; i < sh.dwords; i++)
                    key.push_back(sh.address ? guest::Load<uint32_t>(sh.address + i * 4) : (i < sh.immediate.size() ? sh.immediate[i] : 0));
            }
            dumpState = std::find(seen.begin(), seen.end(), key) == seen.end();
            if (dumpState)
                seen.push_back(std::move(key));
        }
        if (dumpState)
        {
            // Full dump of the first draw: shaders, non-zero context registers, constants.
            for (int stage = 0; stage < 2; stage++)
            {
                const ShaderSlot& sh = g_shaders[stage];
                std::fprintf(stderr, "[gpu]   %s shader (%u dwords):", stage ? "pixel" : "vertex", sh.dwords);
                for (uint32_t i = 0; i < sh.dwords; i++)
                    std::fprintf(stderr, "%s%08X", i % 6 == 0 ? "\n[gpu]     " : " ",
                        sh.address ? guest::Load<uint32_t>(sh.address + i * 4) : (i < sh.immediate.size() ? sh.immediate[i] : 0));
                std::fprintf(stderr, "\n");
            }
            std::fprintf(stderr, "[gpu]   context registers:");
            int n = 0;
            for (uint32_t r = 0x2000; r < 0x2400; r++)
                if (const uint32_t v = ReadRegister(r))
                    std::fprintf(stderr, "%s%04X=%08X", n++ % 6 == 0 ? "\n[gpu]     " : "  ", r, v);
            std::fprintf(stderr, "\n[gpu]   ALU constants (non-zero):");
            n = 0;
            for (uint32_t r = 0x4000; r < 0x4800; r++)
                if (const uint32_t v = ReadRegister(r))
                    std::fprintf(stderr, "%s c%u.%c=%08X", n++ % 6 == 0 ? "\n[gpu]    " : " ", (r - 0x4000) / 4, "xyzw"[r & 3], v);
            std::fprintf(stderr, "\n[gpu]   fetch constants (non-zero):");
            n = 0;
            for (uint32_t r = 0x4800; r < 0x4900; r++)
                if (const uint32_t v = ReadRegister(r))
                    std::fprintf(stderr, "%s%04X=%08X", n++ % 6 == 0 ? "\n[gpu]     " : "  ", r, v);
            std::fprintf(stderr, "\n");
        }
        for (uint32_t slot = 0; slot < 32; slot++)
        {
            const uint32_t d0 = ReadRegister(0x4800 + slot * 6);
            const uint32_t d1 = ReadRegister(0x4800 + slot * 6 + 1);
            const uint32_t d2 = ReadRegister(0x4800 + slot * 6 + 2);
            if ((d0 & 3) == 2) // texture fetch constant
                std::fprintf(stderr, "[gpu]     tex%-2u fmt=%2u base=%08X size=%ux%u pitch=%u tiled=%u\n", slot,
                    d1 & 0x3F, d1 & ~0xFFFu, (d2 & 0x1FFF) + 1, ((d2 >> 13) & 0x1FFF) + 1, ((d0 >> 22) & 0x1FF) * 32,
                    (d0 >> 31) & 1);
        }
    }

    struct Stream
    {
        uint32_t base;   // guest address
        uint32_t dwords; // size (ring size for the ring, length for indirect buffers)
        uint32_t ring;   // ring buffers wrap
        uint32_t Read(uint32_t index) const { return guest::Load<uint32_t>(base + (ring ? index % dwords : index) * 4); }
    };

    static bool CompareFunction(uint32_t function, uint32_t value, uint32_t reference)
    {
        switch (function & 7)
        {
        case 0: return false;
        case 1: return value < reference;
        case 2: return value <= reference;
        case 3: return value == reference;
        case 4: return value != reference;
        case 5: return value >= reference;
        case 6: return value > reference;
        default: return true;
        }
    }

    static void ExecuteStream(const guest::GuestEnv& env, const Stream& s, uint32_t start, uint32_t end, int depth);

    // Executes one packet at `p`; returns its size in dwords.
    static uint32_t ExecutePacket(const guest::GuestEnv& env, const Stream& s, uint32_t p, int depth)
    {
        const uint32_t header = s.Read(p);
        if (g_stallLog)
            g_trace[g_traceNext++ % 512] = { s.ring ? p : (s.base + p * 4), header, s.Read(p + 1), s.Read(p + 2), s.Read(p + 3), uint32_t(depth) };
        switch (header >> 30)
        {
        case 0: // type 0: write `count` registers starting at `base`
        {
            const uint32_t count = ((header >> 16) & 0x3FFF) + 1;
            const uint32_t reg = header & 0x7FFF;
            const bool oneReg = (header >> 15) & 1;
            for (uint32_t i = 0; i < count; i++)
                WriteRegister(oneReg ? reg : reg + i, s.Read(p + 1 + i));
            return 1 + count;
        }
        case 1: // type 1: two registers
            WriteRegister(header & 0x7FF, s.Read(p + 1));
            WriteRegister((header >> 11) & 0x7FF, s.Read(p + 2));
            return 3;
        case 2: // type 2: nop
            return 1;
        default:
            break;
        }

        // type 3
        const uint32_t opcode = (header >> 8) & 0x7F;
        const uint32_t count = ((header >> 16) & 0x3FFF) + 1;
        auto data = [&](uint32_t i) { return s.Read(p + 1 + i); };
        ++g_opcodeCounts[opcode];

        switch (opcode)
        {
        case PM4_SET_CONSTANT: // { type << 16 | index, values... }
        {
            const uint32_t offsetType = data(0);
            const uint32_t base = ConstantBase((offsetType >> 16) & 0xFF) + (offsetType & 0x7FF);
            for (uint32_t i = 1; i < count; i++)
                WriteRegister(base + i - 1, data(i));
            break;
        }

        case PM4_SET_CONSTANT2: // { register index, values... }
        case PM4_SET_SHADER_CONSTANTS:
        {
            const uint32_t base = data(0) & 0xFFFF;
            for (uint32_t i = 1; i < count; i++)
                WriteRegister(base + i - 1, data(i));
            break;
        }

        case PM4_LOAD_ALU_CONSTANT: // { address, type << 16 | index, dwords }
        {
            const uint32_t source = PhysicalToGuest(data(0) & 0x3FFFFFFF);
            const uint32_t offsetType = data(1);
            const uint32_t base = ConstantBase((offsetType >> 16) & 0xFF) + (offsetType & 0x7FF);
            const uint32_t dwords = data(2);
            for (uint32_t i = 0; i < dwords; i++)
                WriteRegister(base + i, guest::Load<uint32_t>(source + i * 4));
            break;
        }

        case PM4_IM_LOAD: // { address | type, start << 16 | dwords }
        {
            ShaderSlot& slot = g_shaders[data(0) & 1];
            slot.address = PhysicalToGuest(data(0) & ~3u);
            slot.dwords = data(1) & 0xFFFF;
            slot.immediate.clear();
            break;
        }

        case PM4_IM_LOAD_IMMEDIATE: // { type, start << 16 | dwords, microcode... }
        {
            ShaderSlot& slot = g_shaders[data(0) & 1];
            slot.address = 0;
            slot.dwords = data(1) & 0xFFFF;
            slot.immediate.resize(slot.dwords);
            for (uint32_t i = 0; i < slot.dwords && i + 2 < count; i++)
                slot.immediate[i] = data(2 + i);
            break;
        }

        case PM4_DRAW_INDX: // { viz query, initiator, [index base, index size] }
            LogDraw(data(1), count > 2 ? data(2) : 0, count > 3 ? data(3) : 0);
            Draw(data(1), count > 2 ? data(2) : 0, count > 3 ? data(3) : 0, {});
            break;

        case PM4_DRAW_INDX_2: // { initiator, [immediate indices] }
        {
            LogDraw(data(0), 0, 0);
            std::vector<uint32_t> immediate;
            for (uint32_t i = 1; i < count; i++)
                immediate.push_back(data(i));
            Draw(data(0), 0, 0, std::move(immediate));
            break;
        }

        case PM4_XE_SWAP:
            OnSwap(data(1), data(2), data(3), [&] {
                std::array<uint32_t, 6> fetch{};
                for (uint32_t i = 0; i < 6; i++)
                    fetch[i] = data(4 + i);
                return fetch;
            }());
            break;

        case PM4_INDIRECT_BUFFER:
        case PM4_INDIRECT_BUFFER_PFD:
            if (depth < 4)
            {
                Stream ib{ PhysicalToGuest(data(0)), data(1) & 0xFFFFF, 0 };
                ExecuteStream(env, ib, 0, ib.dwords, depth + 1);
            }
            break;

        case PM4_MEM_WRITE:
        {
            const uint32_t address = data(0);
            for (uint32_t i = 1; i < count; i++)
                GpuWrite(address + (i - 1) * 4, data(i));
            break;
        }

        case PM4_EVENT_WRITE_SHD: // { initiator, address, value }: fence writes
        {
            const uint32_t initiator = data(0);
            const uint32_t value = (initiator >> 31) ? uint32_t(g_vblanks.load()) : data(2);
            GpuWrite(data(1), value);
            break;
        }

        case PM4_COND_WRITE: // { info, poll address/reg, reference, mask, write address/reg, write data }
        {
            const uint32_t info = data(0);
            const uint32_t polled = (info & 0x10) ? guest::Load<uint32_t>(PhysicalToGuest(data(1) & ~3u)) : ReadRegister(data(1));
            if (CompareFunction(info, polled & data(3), data(2)))
            {
                if (info & 0x100)
                    GpuWrite(data(4), data(5));
                else
                    WriteRegister(data(4), data(5));
            }
            break;
        }

        case PM4_INTERRUPT: // { cpu mask }: interrupt each selected hardware thread (as Xenia)
        {
            const uint32_t cpuMask = data(0);
            static uint64_t masksSeen = 0;
            if (g_stallLog && !(masksSeen & (1ull << (cpuMask & 63))))
            {
                masksSeen |= 1ull << (cpuMask & 63);
                std::fprintf(stderr, "[gpu] interrupt cpu mask %02X (swap %llu)\n", cpuMask, (unsigned long long)g_swapCount);
            }
            for (uint8_t cpu = 0; cpu < 6; cpu++)
                if (cpuMask & (1u << cpu))
                    DispatchInterrupt(env, 1, cpu);
            break;
        }

        case PM4_WAIT_REG_MEM:
            // Waits on values the CPU or this processor produce; everything here completes
            // immediately, so treat the wait as satisfied.
        case PM4_NOP:
        case PM4_REG_RMW:
        case PM4_EVENT_WRITE:
        case PM4_EVENT_WRITE_EXT:
        case PM4_ME_INIT:
        default:
            // Draws, state and constant uploads: the future WebGPU backend's job.
            break;
        }
        return 1 + count;
    }

    static void ExecuteStream(const guest::GuestEnv& env, const Stream& s, uint32_t start, uint32_t end, int depth)
    {
        uint32_t p = start;
        // For the ring, `end` may be numerically below `start` (wrapped); walk modulo size.
        uint32_t remaining = s.ring ? (end + s.dwords - start) % s.dwords : end - start;
        while (remaining > 0)
        {
            const uint32_t size = ExecutePacket(env, s, p, depth);
            const uint32_t step = size > remaining ? remaining : size;
            p = s.ring ? (p + step) % s.dwords : p + step;
            remaining -= step;
        }
    }

    static void CheckStall()
    {
        const uint32_t dev = g_interruptData.load();
        if (!dev)
            return;
        g_fenceAddr = guest::Load<uint32_t>(dev + 11024);
        static auto since = std::chrono::steady_clock::now();
        static uint32_t lastDone = 0;
        static bool reported = false;
        const uint32_t done = g_fenceAddr ? guest::Load<uint32_t>(g_fenceAddr) : 0;
        const uint32_t issued = guest::Load<uint32_t>(dev + 11036);
        const auto now = std::chrono::steady_clock::now();
        static auto lastPrint = now;
        if (now - lastPrint > std::chrono::seconds(2))
        {
            lastPrint = now;
            std::fprintf(stderr, "[gpu] fences: swap %llu done=%08X issued=%08X rptr=%u wptr=%u soft=%u 13600=%08X\n",
                (unsigned long long)g_swapCount, done, issued, g_rptr, ReadRegister(CP_RB_WPTR), guest::Load<uint32_t>(dev + 11088),
                guest::Load<uint32_t>(dev + 13600));
        }
        if (done != lastDone || done == issued)
        {
            lastDone = done;
            since = now;
            reported = false;
            return;
        }
        if (reported || now - since < std::chrono::seconds(3))
            return;
        reported = true;
        std::fprintf(stderr, "[gpu] STALL: swap %llu fence %08X done=%08X issued=%08X lastGpuWrite=%08X rptr=%u wptr=%u softwptr=%u\n",
            (unsigned long long)g_swapCount, g_fenceAddr, done, issued, g_fenceLastValue, g_rptr, ReadRegister(CP_RB_WPTR),
            guest::Load<uint32_t>(dev + 11088));
        std::fprintf(stderr, "[gpu]   dev+11024..: %08X %08X %08X %08X  11068: %08X %08X  13600: %08X %08X\n",
            guest::Load<uint32_t>(g_fenceAddr + 4), guest::Load<uint32_t>(dev + 11028), guest::Load<uint32_t>(dev + 11032),
            guest::Load<uint32_t>(dev + 11040), guest::Load<uint32_t>(dev + 11068), guest::Load<uint32_t>(dev + 11072),
            guest::Load<uint32_t>(dev + 13600), guest::Load<uint32_t>(dev + 13604));
        std::fprintf(stderr, "[gpu]   15300..15340:");
        for (uint32_t o = 15300; o <= 15340; o += 4)
            std::fprintf(stderr, " %08X", guest::Load<uint32_t>(dev + o));
        const uint32_t blk = guest::Load<uint32_t>(dev + 11028);
        std::fprintf(stderr, "\n[gpu]   intr block %08X:", blk);
        for (uint32_t o = 0; o < 32; o += 4)
            std::fprintf(stderr, " %08X", guest::Load<uint32_t>(blk + o));
        std::fprintf(stderr, "\n[gpu]   last packets:\n");
        for (uint32_t i = 0; i < 512; i++)
        {
            const TraceEntry& e = g_trace[(g_traceNext + i) % 512];
            if (e.header)
                std::fprintf(stderr, "[gpu]     %u %08X: %08X | %08X %08X %08X\n", e.depth, e.pos, e.header, e.d0, e.d1, e.d2);
        }
    }

    static void CommandProcessorThread()
    {
        guest::GuestEnv env = guest::CreateGuestEnv(/*cpu*/ 2);
        auto lastWork = std::chrono::steady_clock::now();
        for (;;)
        {
            const uint32_t base = g_ringBase.load();
            const uint32_t size = g_ringDwords.load();
            if (base && size)
            {
                const uint32_t wptr = ReadRegister(CP_RB_WPTR) % size;
                if (wptr != g_rptr)
                {
                    const auto start = std::chrono::steady_clock::now();
                    ExecuteStream(env, Stream{ base, size, 1 }, g_rptr, wptr, 0);
                    lastWork = std::chrono::steady_clock::now();
                    g_cpBusyNs += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(lastWork - start).count());
                    g_rptr = wptr;
                    WriteRegister(CP_RB_RPTR, g_rptr);
                    if (const uint32_t writeback = g_rptrWriteback.load())
                        guest::Store<uint32_t>(writeback, g_rptr);
                    continue;
                }
            }
            if (g_stallLog)
                CheckStall();
            // The game flushes the GPU (fence + wait) dozens of times per frame, so new
            // packets usually follow within microseconds: poll without sleeping for a while
            // after the last work (a Windows sleep lasts at least ~1 ms).
            if (std::chrono::steady_clock::now() - lastWork < std::chrono::milliseconds(4))
                std::this_thread::yield();
            else
                std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
    }

    static void VblankThread()
    {
        guest::GuestEnv env = guest::CreateGuestEnv(/*cpu*/ 2);
        using clock = std::chrono::steady_clock;
        const auto period = std::chrono::nanoseconds(1000000000 / 60);
        auto next = clock::now() + period;
        for (;;)
        {
            std::this_thread::sleep_until(next);
            next += period;
            g_vblanks.fetch_add(1);
            DispatchInterrupt(env, 0, 2);
        }
    }
}

using namespace guest;
using namespace gpu;

// u32 VdInitializeEngines(u32 unk0, u32 callback, u32 arg, u32* pfp, u32* me)
GUEST_IMPORT(VdInitializeEngines)
{
    // Display/EDRAM registers the CPU reads directly (values per Xenia's GraphicsSystem).
    // The interrupt status must read "vblank" or the game's interrupt callback ignores
    // every vblank and its frame counters never advance.
    WriteRegister(0x0F00, 0x08100748); // RB_EDRAM_TIMING
    WriteRegister(0x0F01, 0x0000200E); // RB_BC_CONTROL
    WriteRegister(0x194C, 0x000002D0); // D1MODE_V_COUNTER
    WriteRegister(0x1951, 0x00000001); // interrupt status: vblank
    WriteRegister(0x1961, 0x050002D0); // D1MODE_VIEWPORT_SIZE: 1280x720
    ctx.r3.u64 = 1;
}

GUEST_IMPORT(VdShutdownEngines) {}

// void VdSetGraphicsInterruptCallback(u32 callback, u32 userData)
GUEST_IMPORT(VdSetGraphicsInterruptCallback)
{
    g_interruptCallback = ctx.r3.u32;
    g_interruptData = ctx.r4.u32;
    TTT2_LOG("graphics interrupt callback %08X (data %08X)", ctx.r3.u32, ctx.r4.u32);
    static std::once_flag started;
    std::call_once(started, [] { std::thread(VblankThread).detach(); });
}

// void VdInitializeRingBuffer(u32 physicalAddress, u32 sizeLog2): size = 1 << (sizeLog2 + 3) bytes
GUEST_IMPORT(VdInitializeRingBuffer)
{
    g_ringDwords = (1u << (ctx.r4.u32 + 3)) / 4;
    g_ringBase = PhysicalToGuest(ctx.r3.u32);
    TTT2_LOG("ring buffer at %08X, %u dwords", g_ringBase.load(), g_ringDwords.load());
    static std::once_flag started;
    std::call_once(started, [] { std::thread(CommandProcessorThread).detach(); });
}

// void VdEnableRingBufferRPtrWriteBack(u32 physicalAddress, u32 blockSizeLog2)
GUEST_IMPORT(VdEnableRingBufferRPtrWriteBack)
{
    g_rptrWriteback = PhysicalToGuest(ctx.r3.u32);
}

GUEST_IMPORT(VdSetSystemCommandBufferGpuIdentifierAddress) {}

// void VdGetSystemCommandBuffer(u32* p0, u32* p1)
GUEST_IMPORT(VdGetSystemCommandBuffer)
{
    if (ctx.r3.u32) Store<uint32_t>(ctx.r3.u32, 0xBEEF0000);
    if (ctx.r4.u32) Store<uint32_t>(ctx.r4.u32, 0xBEEF0001);
}

// void VdQueryVideoMode(XVIDEO_MODE* mode)
GUEST_IMPORT(VdQueryVideoMode)
{
    auto* mode = ToHost<XVIDEO_MODE>(ctx.r3.u32);
    *mode = {};
    mode->DisplayWidth = 1280;
    mode->DisplayHeight = 720;
    mode->IsInterlaced = 0;
    mode->IsWidescreen = 1;
    mode->IsHighDefinition = 1;
    mode->RefreshRate = 0x42700000; // 60.0f as raw bits
    mode->VideoStandard = 1;        // NTSC
    mode->Unknown4A = 0x4A;
    mode->Unknown01 = 0x01;
}

// u32 VdQueryVideoFlags(): 1 = widescreen, 2 = 1280-wide (720p) or better
GUEST_IMPORT(VdQueryVideoFlags)
{
    ctx.r3.u64 = 0x3;
}

// void VdGetCurrentDisplayGamma(u32* type, float* power)
GUEST_IMPORT(VdGetCurrentDisplayGamma)
{
    if (ctx.r3.u32) Store<uint32_t>(ctx.r3.u32, 1);
    if (ctx.r4.u32) Store<float>(ctx.r4.u32, 2.22222233f);
}

// void VdGetCurrentDisplayInformation(X_DISPLAY_INFO* info)
GUEST_IMPORT(VdGetCurrentDisplayInformation)
{
    const uint32_t info = ctx.r3.u32;
    std::memset(ToHost(info), 0, 0x58);
    Store<uint16_t>(info + 0x00, 1280);  // front buffer width
    Store<uint16_t>(info + 0x02, 720);   // front buffer height
    Store<uint16_t>(info + 0x10, 1280);  // display width
    Store<uint16_t>(info + 0x12, 720);   // display height
    Store<uint32_t>(info + 0x40, 0x42700000); // refresh rate 60.0f
    Store<uint16_t>(info + 0x48, 1280);
    Store<uint16_t>(info + 0x4A, 720);
}

GUEST_IMPORT(VdSetDisplayMode) { ctx.r3.u64 = 0; }
GUEST_IMPORT(VdSetDisplayModeOverride) { ctx.r3.u64 = 0; }
GUEST_IMPORT(VdCallGraphicsNotificationRoutines) { ctx.r3.u64 = 0; }
GUEST_IMPORT(VdEnableDisableClockGating) { ctx.r3.u64 = 0; }
GUEST_IMPORT(VdRetrainEDRAMWorker) { ctx.r3.u64 = 0; }
GUEST_IMPORT(VdRetrainEDRAM) { ctx.r3.u64 = 0; }
GUEST_IMPORT(VdIsHSIOTrainingSucceeded) { ctx.r3.u64 = 1; }

// void VdPersistDisplay(u32 unk, u32* unk_out)
GUEST_IMPORT(VdPersistDisplay)
{
    if (ctx.r4.u32) Store<uint32_t>(ctx.r4.u32, 0);
    ctx.r3.u64 = 0;
}

// u32 VdInitializeScalerCommandBuffer(sourceXY, sourceWH, outputXY, outputWH, unk4,
//     verticalFilterType, verticalFilterTable, horizontalFilterType, horizontalFilterTable,
//     unk9, destPtr, destCount)
// Twelve arguments (per Xenia): arguments 9+ live in the caller's parameter area at
// r1 + 0x54 + 8 * (n - 9), so destPtr (11th) is at r1+0x64 and destCount (12th) at r1+0x6C.
// Fills the caller's buffer with type-2 NOP packets and returns how many it wrote.
GUEST_IMPORT(VdInitializeScalerCommandBuffer)
{
    const uint32_t dest = Load<uint32_t>(ctx.r1.u32 + 0x64);
    const uint32_t count = Load<uint32_t>(ctx.r1.u32 + 0x6C);
    if (count > 0x4000)
    {
        TTT2_LOG("VdInitializeScalerCommandBuffer: implausible count, ignoring");
        ctx.r3.u64 = 0;
        return;
    }
    for (uint32_t i = 0; i < count; i++)
        Store<uint32_t>(dest + i * 4, 0x80000000);
    ctx.r3.u64 = count;
}

// void VdSwap(u32* buffer, u32* frontBufferFetch, u32 unk2, u32 unk3, u32 unk4,
//             u32 frontBuffer, u32 format, u32 colorSpace, u32 width, u32 height)
// The caller reserves 64 dwords of its command stream for the swap. Like Xenia, write a
// private XE_SWAP packet there so the command processor sees the frame boundary at the
// right point in the stream, with everything it needs to present the front buffer:
//   { 'SWAP', frontBuffer (physical), width, height, fetch constant[6] }, then zeros.
GUEST_IMPORT(VdSwap)
{
    const uint32_t buffer = ctx.r3.u32;
    if (buffer)
    {
        Store<uint32_t>(buffer, 0xC0000000u | ((63u - 1) << 16) | (PM4_XE_SWAP << 8));
        for (uint32_t i = 1; i < 64; i++)
            Store<uint32_t>(buffer + i * 4, 0);
        Store<uint32_t>(buffer + 4, 0x53574150); // 'SWAP'
        Store<uint32_t>(buffer + 8, ctx.r8.u32 & 0x1FFFFFFF);
        Store<uint32_t>(buffer + 12, Load<uint32_t>(ctx.r1.u32 + 0x54)); // width
        Store<uint32_t>(buffer + 16, Load<uint32_t>(ctx.r1.u32 + 0x5C)); // height
        if (ctx.r4.u32)
            for (uint32_t i = 0; i < 6; i++)
                Store<uint32_t>(buffer + 20 + i * 4, Load<uint32_t>(ctx.r4.u32 + i * 4));
    }

    const uint64_t frame = g_frames.fetch_add(1) + 1;
    if ((frame & (frame - 1)) == 0) // log 1, 2, 4, 8, ... so the log stays readable
        TTT2_LOG("VdSwap: frame %llu (vblank %llu)", (unsigned long long)frame, (unsigned long long)g_vblanks.load());
    ctx.r3.u64 = 0;
}
