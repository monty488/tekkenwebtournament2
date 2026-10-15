// Video kernel interface and a minimal Xenos command processor.
//
// The game's statically linked D3D talks to the GPU through:
//   * a ring buffer of PM4 packets (VdInitializeRingBuffer); the CPU advances the write
//     pointer by storing to the CP_RB_WPTR register, and learns how far the GPU got from
//     a read-pointer write-back slot (VdEnableRingBufferRPtrWriteBack);
//   * an interrupt callback (VdSetGraphicsInterruptCallback) called on every vblank
//     (source 0) and when the command stream asks for an interrupt (source 1).
//
// Rendering is not implemented yet. The command processor here only executes what the
// CPU side synchronises on (memory writes, fence events, conditional writes, interrupts,
// indirect buffers) and skips everything else, so frames "complete" immediately. That is
// enough for the game's frame loop to run and reach VdSwap; draw packets are where the
// WebGPU backend will hook in (docs/ARCHITECTURE.md, "Graphics").

#include "video.h"
#include "../kernel/guest.h"
#include "../kernel/threads.h"

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
        switch (address & 3)
        {
        case 0: *guest::ToHost<uint32_t>(target) = value; break;                          // no swap: host order
        case 1: *guest::ToHost<uint32_t>(target) = ((value & 0x00FF00FF) << 8) | ((value >> 8) & 0x00FF00FF); break; // 8in16
        case 2: guest::Store<uint32_t>(target, value); break;                              // 8in32: big-endian
        case 3: *guest::ToHost<uint32_t>(target) = (value << 16) | (value >> 16); break;   // 16in32
        }
    }

    static void DispatchInterrupt(const guest::GuestEnv& env, uint32_t source)
    {
        const uint32_t callback = g_interruptCallback.load();
        if (!callback)
            return;
        std::lock_guard lock(g_interruptMutex);
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
    };

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

        switch (opcode)
        {
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

        case PM4_INTERRUPT:
            DispatchInterrupt(env, 1);
            break;

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

    static void CommandProcessorThread()
    {
        guest::GuestEnv env = guest::CreateGuestEnv(/*cpu*/ 2);
        for (;;)
        {
            const uint32_t base = g_ringBase.load();
            const uint32_t size = g_ringDwords.load();
            if (base && size)
            {
                const uint32_t wptr = ReadRegister(CP_RB_WPTR) % size;
                if (wptr != g_rptr)
                {
                    ExecuteStream(env, Stream{ base, size, 1 }, g_rptr, wptr, 0);
                    g_rptr = wptr;
                    WriteRegister(CP_RB_RPTR, g_rptr);
                    if (const uint32_t writeback = g_rptrWriteback.load())
                        guest::Store<uint32_t>(writeback, g_rptr);
                    continue;
                }
            }
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
            DispatchInterrupt(env, 0);
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

// void VdSwap(u32* buffer, u32 fetch, u32 unk2, u32 unk3, u32 unk4, u32 frontBuffer, ...)
// The caller reserves 64 dwords of its command stream for the swap; fill them with a NOP
// packet so the stream stays well formed.
GUEST_IMPORT(VdSwap)
{
    const uint32_t buffer = ctx.r3.u32;
    if (buffer)
    {
        Store<uint32_t>(buffer, 0xC0000000u | ((63u - 1) << 16) | (PM4_NOP << 8));
        for (uint32_t i = 1; i < 64; i++)
            Store<uint32_t>(buffer + i * 4, 0);
    }

    const uint64_t frame = g_frames.fetch_add(1) + 1;
    if ((frame & (frame - 1)) == 0) // log 1, 2, 4, 8, ... so the log stays readable
        TTT2_LOG("VdSwap: frame %llu (vblank %llu)", (unsigned long long)frame, (unsigned long long)g_vblanks.load());
    ctx.r3.u64 = 0;
}
