// XMA decoder. On the console the audio chip's XMA decoder works on 64-byte "context"
// records in physical memory: XAudio2 (statically linked) fills in input/output buffer
// pointers and valid bits, then pokes per-context bits in the decoder's register file
// (0x7FEA0000) to kick, lock or clear contexts, and polls the records to see input buffers
// consumed and output produced. Layout and register map follow Xenia's XmaContext /
// XmaDecoder (src/xenia/apu/xma_context.h, xma_register_table.inc).
//
// The register file lives in plain guest memory (reads such as the context array address
// just load it), but the recompiled code's MMIO stores to it come here (recomp/mmio.h), so
// kick/lock/clear commands act like they do in Xenia's XmaDecoder::WriteRegister.
//
// Decoding is not implemented: a kicked context consumes one XMA frame of input per 512
// output samples (frames counted from the packet headers) and writes silence, so XMA voices
// (TTT2 uses them for menu sounds) advance and finish at the rate XAudio2 pulls output.
// Real decoding (FFmpeg) would replace the silence in Work.

#include "xma.h"

#include "../kernel/guest.h"
#include "../kernel/memory.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>

namespace apu
{
    constexpr uint32_t kRegisterBase = 0x7FEA0000;
    constexpr uint32_t kContextCount = 320;
    constexpr uint32_t kContextSize = 64;

    // Register indices (one dword each, at kRegisterBase + 4 * index).
    constexpr uint32_t kContextArrayAddress = 0x0600;
    constexpr uint32_t kKick = 0x0650;  // 10 registers, one bit per context
    constexpr uint32_t kLock = 0x0690;
    constexpr uint32_t kClear = 0x06A0;
    constexpr uint32_t kGroups = kContextCount / 32;

    constexpr uint32_t kBytesPerPacket = 2048;
    constexpr uint32_t kBitsPerPacket = kBytesPerPacket * 8;
    constexpr uint32_t kBitsPerPacketHeader = 32;
    constexpr uint32_t kSamplesPerFrame = 512;
    constexpr uint32_t kBytesPerSample = 2;
    constexpr uint32_t kBytesPerOutputBlock = 256;

    struct Context
    {
        std::atomic<bool> allocated{ false };
        std::atomic<bool> enabled{ false };
    };

    static uint32_t g_contextArray = 0; // guest virtual address of context 0
    static Context g_contexts[kContextCount];
    static std::mutex g_mutex; // serialises context work against the guest's commands
    static std::condition_variable g_kicked;

    // The guest accesses these registers with lwbrx/stwbrx, so they hold host-order values.
    static uint32_t* Register(uint32_t index)
    {
        return guest::ToHost<uint32_t>(kRegisterBase + index * 4);
    }

    static uint32_t PhysicalToGuest(uint32_t physical)
    {
        return 0xA0000000u + (physical & 0x1FFFFFFFu);
    }

    static uint32_t ContextAddress(uint32_t id) { return g_contextArray + id * kContextSize; }

    // XMA_CONTEXT_DATA, kept as big-endian dwords in guest memory. Bit positions count from
    // the least significant bit of each dword.
    struct ContextData
    {
        uint32_t d[10];

        static uint32_t Get(uint32_t v, int shift, int bits) { return (v >> shift) & ((1u << bits) - 1); }
        static void Set(uint32_t& v, int shift, int bits, uint32_t x)
        {
            const uint32_t mask = ((1u << bits) - 1) << shift;
            v = (v & ~mask) | ((x << shift) & mask);
        }

        uint32_t inputPacketCount(int b) const { return Get(d[b], 0, 12); }
        uint32_t loopCount() const { return Get(d[0], 12, 8); }
        void setLoopCount(uint32_t x) { Set(d[0], 12, 8, x); }
        bool inputValid(int b) const { return Get(d[0], 20 + b, 1); }
        void setInputValid(int b, bool x) { Set(d[0], 20 + b, 1, x); }
        uint32_t outputBlockCount() const { return Get(d[0], 22, 5); }
        uint32_t outputWriteOffset() const { return Get(d[0], 27, 5); }
        void setOutputWriteOffset(uint32_t x) { Set(d[0], 27, 5, x); }
        bool isStereo() const { return Get(d[1], 29, 1); }
        bool outputValid() const { return Get(d[1], 31, 1); }
        void setOutputValid(bool x) { Set(d[1], 31, 1, x); }
        uint32_t inputReadOffset() const { return Get(d[2], 0, 26); }
        void setInputReadOffset(uint32_t x) { Set(d[2], 0, 26, x); }
        uint32_t loopStart() const { return Get(d[3], 0, 26); }
        uint32_t loopEnd() const { return Get(d[4], 0, 26); }
        int currentBuffer() const { return Get(d[4], 31, 1); }
        void setCurrentBuffer(int x) { Set(d[4], 31, 1, x); }
        uint32_t inputPtr(int b) const { return d[5 + b]; }
        uint32_t outputPtr() const { return d[7]; }
        uint32_t outputReadOffset() const { return Get(d[9], 0, 5); }

        void Load(uint32_t address)
        {
            for (int i = 0; i < 10; ++i)
                d[i] = guest::Load<uint32_t>(address + i * 4);
        }
    };

    static void ClearContext(uint32_t id)
    {
        ContextData data;
        data.Load(ContextAddress(id));
        data.setInputValid(0, false);
        data.setInputValid(1, false);
        data.setOutputValid(false);
        data.setOutputWriteOffset(0);
        ContextData::Set(data.d[9], 0, 5, 0); // output read offset
        for (int i : { 1, 9, 0 })
            guest::Store<uint32_t>(ContextAddress(id) + i * 4, data.d[i]);
    }

    // Xenia's TrySetupNextLoop.
    static bool TrySetupNextLoop(ContextData& data, bool ignoreReadOffset)
    {
        if (data.loopCount() > 0 && data.loopStart() < data.loopEnd() &&
            (ignoreReadOffset || data.inputReadOffset() >= data.loopEnd()))
        {
            data.setInputReadOffset(data.loopStart());
            if (data.loopCount() < 255)
                data.setLoopCount(data.loopCount() - 1);
            return true;
        }
        return false;
    }

    static void SwapInputBuffer(ContextData& data)
    {
        data.setInputValid(data.currentBuffer(), false);
        data.setCurrentBuffer(data.currentBuffer() ^ 1);
        data.setInputReadOffset(0);
    }

    // Advances the input read offset by one frame. Returns false if no frame starts at or
    // after the current offset in this packet (the offset then points at the next packet).
    static bool ConsumeFrame(ContextData& data)
    {
        const int buffer = data.currentBuffer();
        const uint32_t offset = data.inputReadOffset();
        const uint32_t packet = offset / kBitsPerPacket;
        const uint8_t* header = guest::ToHost<uint8_t>(PhysicalToGuest(data.inputPtr(buffer)) + packet * kBytesPerPacket);

        // XMA2 packet header: frame count (6), first frame bit offset (15), metadata (3),
        // packet skip count (8).
        const uint32_t frameCount = header[0] >> 2;
        const uint32_t firstFrame = kBitsPerPacketHeader + (((header[0] & 3u) << 13) | (header[1] << 5) | (header[2] >> 3));
        const uint32_t skip = header[3] == 0xFF ? 0 : header[3];
        const uint32_t nextPacket = (packet + skip + 1) * kBitsPerPacket;

        const uint32_t within = offset % kBitsPerPacket;
        if (frameCount == 0 || firstFrame >= kBitsPerPacket)
        {
            data.setInputReadOffset(nextPacket);
            return false;
        }

        // Frame lengths are not parsed; frames are assumed evenly spread over the packet.
        const uint32_t frameBits = std::max(1u, (kBitsPerPacket - firstFrame) / frameCount);
        const uint32_t index = within < firstFrame ? 0 : (within - firstFrame) / frameBits;
        if (index >= frameCount)
        {
            data.setInputReadOffset(nextPacket);
            return false;
        }
        if (index + 1 >= frameCount)
            data.setInputReadOffset(nextPacket);
        else
            data.setInputReadOffset(packet * kBitsPerPacket + firstFrame + (index + 1) * frameBits);
        return true;
    }

    // Xenia's XmaContext::Decode, minus the decoding: output is silence.
    static void Work(uint32_t id)
    {
        const uint32_t address = ContextAddress(id);
        ContextData data;
        data.Load(address);

        if (!data.outputValid())
            return;

        if (data.inputValid(0) || data.inputValid(1))
        {
            const uint32_t capacity = data.outputBlockCount() * kBytesPerOutputBlock;
            const uint32_t readOffset = data.outputReadOffset() * kBytesPerOutputBlock;
            uint32_t writeOffset = data.outputWriteOffset() * kBytesPerOutputBlock;
            uint8_t* output = guest::ToHost<uint8_t>(PhysicalToGuest(data.outputPtr()));

            // Xenia's RingBuffer: equal offsets mean empty.
            uint32_t space = writeOffset == readOffset ? capacity
                : writeOffset < readOffset ? readOffset - writeOffset
                : capacity - writeOffset + readOffset;
            const uint32_t frameBytes = (kSamplesPerFrame * kBytesPerSample) << data.isStereo();
            space -= space % frameBytes;

            // Bounded so a malformed stream (e.g. an endless loop over packets with no
            // frames) cannot pin this thread.
            for (int guard = 0; space >= frameBytes && guard < 4096; ++guard)
            {
                if (!data.inputValid(0) && !data.inputValid(1))
                    break;
                const int buffer = data.currentBuffer();
                if (!data.inputValid(buffer))
                {
                    SwapInputBuffer(data); // only the other buffer holds data
                    continue;
                }

                bool looped = TrySetupNextLoop(data, false);
                if (data.inputReadOffset() / kBitsPerPacket >= data.inputPacketCount(buffer))
                {
                    if (!looped && !TrySetupNextLoop(data, true))
                        SwapInputBuffer(data);
                    continue;
                }

                if (!ConsumeFrame(data))
                    continue;

                for (uint32_t n = frameBytes; n;)
                {
                    const uint32_t chunk = std::min(n, capacity - writeOffset);
                    std::memset(output + writeOffset, 0, chunk);
                    writeOffset = (writeOffset + chunk) % capacity;
                    n -= chunk;
                }
                space -= frameBytes;

                if (data.inputReadOffset() / kBitsPerPacket >= data.inputPacketCount(buffer) &&
                    !TrySetupNextLoop(data, true))
                    SwapInputBuffer(data);
            }

            data.setOutputWriteOffset(writeOffset / kBytesPerOutputBlock);
            if (writeOffset == readOffset)
                data.setOutputValid(false);
        }
        else
        {
            data.setOutputValid(false);
        }

        // dword 0 (valid bits, write offset) last: the guest polls it for progress.
        for (int i : { 2, 4, 1, 0 })
            guest::Store<uint32_t>(address + i * 4, data.d[i]);
    }

    static void DecoderThread()
    {
        std::unique_lock lock(g_mutex);
        for (;;)
        {
            // Kicks notify; the timeout only bounds the wait if a notify slips past.
            g_kicked.wait_for(lock, std::chrono::milliseconds(5));
            for (uint32_t id = 0; id < kContextCount; ++id)
            {
                Context& context = g_contexts[id];
                if (context.allocated && context.enabled.exchange(false))
                    Work(id);
            }
        }
    }

    // Xenia's XmaDecoder::WriteRegister. `value` is in the guest's register order; the
    // register file keeps the byte-reversed form, matching the guest's lwbrx/stwbrx access.
    static void WriteRegister(uint32_t address, uint32_t value)
    {
        const uint32_t index = (address & 0xFFFF) / 4;
        const uint32_t bits = __builtin_bswap32(value);
        *Register(index) = bits;

        auto forEachContext = [&](uint32_t first, auto&& action) {
            const uint32_t base = (index - first) * 32;
            for (uint32_t b = bits; b; b &= b - 1)
                action(g_contexts[base + __builtin_ctz(b)], base + __builtin_ctz(b));
        };

        if (index >= kKick && index < kKick + kGroups)
        {
            forEachContext(kKick, [](Context& context, uint32_t) { context.enabled = true; });
            g_kicked.notify_one();
        }
        else if (index >= kLock && index < kLock + kGroups)
        {
            // Taking the mutex waits out a Work() in progress, so the guest owns the
            // context once this returns.
            std::lock_guard lock(g_mutex);
            forEachContext(kLock, [](Context& context, uint32_t) { context.enabled = false; });
        }
        else if (index >= kClear && index < kClear + kGroups)
        {
            std::lock_guard lock(g_mutex);
            forEachContext(kClear, [](Context& context, uint32_t id) {
                if (context.allocated)
                    ClearContext(id);
            });
        }
    }

    void InitXma()
    {
        g_contextArray = guest::AllocPhysical(kContextCount * kContextSize);
        *Register(kContextArrayAddress) = g_contextArray & 0x1FFFFFFFu;
        std::thread(DecoderThread).detach();
    }
}

void ttt2_XmaRegisterWrite(uint32_t address, uint32_t value)
{
    apu::WriteRegister(address, value);
}

using namespace guest;

// u32 XMACreateContext(u32* context)
GUEST_IMPORT(XMACreateContext)
{
    std::lock_guard lock(apu::g_mutex);
    for (uint32_t id = 0; id < apu::kContextCount; ++id)
    {
        apu::Context& context = apu::g_contexts[id];
        if (context.allocated)
            continue;
        context.allocated = true;
        context.enabled = false;
        std::memset(ToHost(apu::ContextAddress(id)), 0, apu::kContextSize);
        Store<uint32_t>(ctx.r3.u32, apu::ContextAddress(id));
        ctx.r3.u64 = STATUS_SUCCESS;
        return;
    }
    TTT2_LOG("XMACreateContext: out of contexts");
    ctx.r3.u64 = STATUS_NO_MEMORY;
}

// void XMAReleaseContext(u32 context)
GUEST_IMPORT(XMAReleaseContext)
{
    std::lock_guard lock(apu::g_mutex);
    const uint32_t offset = ctx.r3.u32 - apu::g_contextArray;
    if (offset < apu::kContextCount * apu::kContextSize)
    {
        apu::Context& context = apu::g_contexts[offset / apu::kContextSize];
        context.allocated = false;
        context.enabled = false;
        std::memset(ToHost(ctx.r3.u32 & ~(apu::kContextSize - 1)), 0, apu::kContextSize);
    }
    ctx.r3.u64 = 0;
}
