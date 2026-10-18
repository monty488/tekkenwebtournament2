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
// A kicked context decodes one XMA frame per 512 output samples, as fast as XAudio2 frees
// output space: frames are found by their 15-bit length fields, gathered across packet
// (and input buffer) boundaries, and decoded by FFmpeg's WMA Pro frame decoder
// (xma_decoder.c). Builds without FFmpeg parse the same way and output silence.

#include "xma.h"

#include "../kernel/guest.h"
#include "../kernel/memory.h"

#if TTT2_HAVE_FFMPEG
#include "xma_decoder.h"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

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
#if TTT2_HAVE_FFMPEG
        // Decoder state (overlap between frames) for this context's stream; g_mutex.
        TTT2XmaDecoder* decoder = nullptr;
        uint32_t decoderChannels = 0, decoderRate = 0;
#endif
        // TTT2_XMA_STATS (g_mutex): activity since the last report.
        uint64_t statFrames = 0, statBits = 0, statSkips = 0, statStalls = 0;
        float statPeak = 0;
        uint32_t statChannels = 0, statRate = 0;
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
        uint32_t sampleRate() const
        {
            static constexpr uint32_t kRates[4] = { 24000, 32000, 44100, 48000 };
            return kRates[Get(d[1], 27, 2)];
        }
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

    static void ResetDecoder(uint32_t id)
    {
#if TTT2_HAVE_FFMPEG
        if (g_contexts[id].decoder)
            ttt2_xma_reset(g_contexts[id].decoder);
#else
        (void)id;
#endif
    }

    static void ClearContext(uint32_t id)
    {
        ResetDecoder(id);
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

    // Reads `count` bits (MSB first) at bit `at` of a big-endian bitstream.
    static uint32_t ReadBits(const uint8_t* data, uint32_t at, int count)
    {
        uint32_t value = 0;
        for (int i = 0; i < count; ++i, ++at)
            value = (value << 1) | ((data[at >> 3] >> (7 - (at & 7))) & 1);
        return value;
    }

    static void CopyBits(const uint8_t* from, uint32_t at, uint8_t* to, uint32_t toAt, uint32_t count)
    {
        for (uint32_t i = 0; i < count; ++i, ++at, ++toAt)
        {
            const uint8_t bit = (from[at >> 3] >> (7 - (at & 7))) & 1;
            uint8_t& byte = to[toAt >> 3];
            byte = uint8_t((byte & ~(0x80 >> (toAt & 7))) | (bit << (7 - (toAt & 7))));
        }
    }

    struct PacketHeader
    {
        uint32_t frameCount;  // frames that start in this packet
        uint32_t firstFrame;  // bit offset of the first of them, from the packet start
        uint32_t skip;        // packets of other streams to skip after this one
    };

    static PacketHeader ReadHeader(const uint8_t* packet)
    {
        // XMA2 packet header: frame count (6), first frame bit offset (15), metadata (3),
        // packet skip count (8).
        return { uint32_t(packet[0] >> 2),
                 kBitsPerPacketHeader + (((packet[0] & 3u) << 13) | (packet[1] << 5) | (packet[2] >> 3)),
                 packet[3] == 0xFF ? 0u : packet[3] };
    }

    constexpr uint32_t kFrameLengthBits = 15;
    constexpr uint32_t kMaxFrameBits = 1u << kFrameLengthBits;

    enum class FrameResult { Decoded, Skipped, NeedInput };

    // Takes the frame at the context's read offset: its bits are gathered into `frame`
    // (following it into later packets, and into the other input buffer if it continues
    // there) and the read offset moves past it. Skipped: no frame starts here (the offset
    // moved on); NeedInput: the frame continues into an input buffer the guest has not
    // supplied yet (nothing changes).
    static FrameResult TakeFrame(ContextData& data, std::vector<uint8_t>& frame, uint32_t& frameBits)
    {
        int buffer = data.currentBuffer();
        uint32_t offset = data.inputReadOffset();
        uint32_t packet = offset / kBitsPerPacket;
        const uint8_t* base = guest::ToHost<uint8_t>(PhysicalToGuest(data.inputPtr(buffer)));
        const uint8_t* bytes = base + packet * kBytesPerPacket;
        const PacketHeader header = ReadHeader(bytes);
        uint32_t within = offset % kBitsPerPacket;

        auto nextPacket = [&] { return (packet + header.skip + 1) * kBitsPerPacket; };
        // Frames start where the header says (earlier bits finish a frame from the previous
        // packet) and follow each other by their lengths. The offset is snapped to the first
        // frame start at or after it: offsets the guest sets (loop starts) point one bit
        // before the frame.
        if (header.frameCount == 0 || header.firstFrame >= kBitsPerPacket)
        {
            data.setInputReadOffset(nextPacket());
            return FrameResult::Skipped;
        }
        uint32_t length = 0;
        for (uint32_t start = header.firstFrame;;)
        {
            if (kBitsPerPacket - start < kFrameLengthBits)
            {
                data.setInputReadOffset(nextPacket());
                return FrameResult::Skipped;
            }
            length = ReadBits(bytes, start, kFrameLengthBits);
            if (length == kMaxFrameBits - 1 || length < kFrameLengthBits)
            {
                // All ones: padding to the end of the packet.
                data.setInputReadOffset(nextPacket());
                return FrameResult::Skipped;
            }
            if (start >= within)
            {
                within = start;
                break;
            }
            start += length;
            if (start >= kBitsPerPacket)
            {
                // The offset lies in the last frame, which continues into the next packet.
                data.setInputReadOffset(nextPacket());
                return FrameResult::Skipped;
            }
        }

        frame.assign((length + 7) / 8 + 64, 0); // + FFmpeg's input padding
        frameBits = length;
        uint32_t copied = 0;
        bool switched = false;
        PacketHeader current = header;
        for (;;)
        {
            const uint32_t chunk = std::min(length - copied, kBitsPerPacket - within);
            CopyBits(bytes, within, frame.data(), copied, chunk);
            copied += chunk;
            within += chunk;
            if (copied == length)
                break;
            // Continues after the next packet's header.
            packet += current.skip + 1;
            if (packet >= data.inputPacketCount(buffer))
            {
                const int other = buffer ^ 1;
                if (switched || !data.inputValid(other))
                    return FrameResult::NeedInput;
                buffer = other;
                packet = 0;
                switched = true;
                base = guest::ToHost<uint8_t>(PhysicalToGuest(data.inputPtr(buffer)));
            }
            bytes = base + packet * kBytesPerPacket;
            current = ReadHeader(bytes);
            within = kBitsPerPacketHeader;
        }

        if (switched)
        {
            data.setInputValid(data.currentBuffer(), false);
            data.setCurrentBuffer(buffer);
        }
        data.setInputReadOffset(packet * kBitsPerPacket + within);
        return FrameResult::Decoded;
    }

    // Writes one decoded frame (512 samples per channel) as big-endian 16-bit PCM,
    // interleaved for stereo, into the context's output ring.
    static void WriteFrame(uint8_t* output, uint32_t capacity, uint32_t& writeOffset, const float* left, const float* right)
    {
        int16_t pcm[kSamplesPerFrame * 2];
        uint32_t count = 0;
        for (uint32_t i = 0; i < kSamplesPerFrame; ++i)
        {
            pcm[count++] = int16_t(std::clamp(left[i], -1.0f, 1.0f) * 32767.0f);
            if (right)
                pcm[count++] = int16_t(std::clamp(right[i], -1.0f, 1.0f) * 32767.0f);
        }
        const uint8_t* src = reinterpret_cast<const uint8_t*>(pcm);
        for (uint32_t i = 0; i < count * 2; i += 2)
        {
            output[writeOffset] = src[i + 1];
            output[writeOffset + 1] = src[i];
            writeOffset = (writeOffset + 2) % capacity;
        }
    }

    // Xenia's XmaContext::Decode.
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

                thread_local std::vector<uint8_t> frame;
                uint32_t frameBits = 0;
                const FrameResult result = TakeFrame(data, frame, frameBits);
                if (result == FrameResult::NeedInput)
                {
                    ++g_contexts[id].statStalls;
                    break;
                }
                if (result == FrameResult::Skipped)
                {
                    ++g_contexts[id].statSkips;
                    continue;
                }

                float left[kSamplesPerFrame] = {}, right[kSamplesPerFrame] = {};
                const bool stereo = data.isStereo();
#if TTT2_HAVE_FFMPEG
                Context& context = g_contexts[id];
                const uint32_t channels = stereo ? 2 : 1;
                if (!context.decoder || context.decoderChannels != channels || context.decoderRate != data.sampleRate())
                {
                    ttt2_xma_destroy(context.decoder);
                    context.decoder = ttt2_xma_create(int(channels), int(data.sampleRate()));
                    context.decoderChannels = channels;
                    context.decoderRate = data.sampleRate();
                }
                static uint64_t decoded = 0, corrupt = 0;
                ++decoded;
                if (context.decoder && !ttt2_xma_decode(context.decoder, frame.data(), int(frameBits), left, stereo ? right : nullptr))
                {
                    // Logged sparsely: a stream read from a wrong offset produces many.
                    if (++corrupt <= 10 || (corrupt & (corrupt - 1)) == 0)
                        TTT2_LOG("xma: context %u: corrupt frame (%u bits); %llu of %llu frames so far", id, frameBits,
                            (unsigned long long)corrupt, (unsigned long long)decoded);
                }
#endif
                {
                    Context& c = g_contexts[id];
                    ++c.statFrames;
                    c.statBits += frameBits;
                    c.statChannels = stereo ? 2 : 1;
                    c.statRate = data.sampleRate();
                    for (uint32_t i = 0; i < kSamplesPerFrame; ++i)
                        c.statPeak = std::max({ c.statPeak, std::abs(left[i]), std::abs(right[i]) });
                }
                WriteFrame(output, capacity, writeOffset, left, stereo ? right : nullptr);
                space -= frameBytes;

                // (The frame may have ended in the other buffer.)
                if (data.inputReadOffset() / kBitsPerPacket >= data.inputPacketCount(data.currentBuffer()) &&
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

    // TTT2_XMA_STATS=1: every 5 s, what each active context decoded.
    static void ReportStats()
    {
        static const bool enabled = [] { const char* v = std::getenv("TTT2_XMA_STATS"); return v && *v == '1'; }();
        static auto last = std::chrono::steady_clock::now();
        if (!enabled || std::chrono::steady_clock::now() - last < std::chrono::seconds(5))
            return;
        last = std::chrono::steady_clock::now();
        for (uint32_t id = 0; id < kContextCount; ++id)
        {
            Context& c = g_contexts[id];
            if (!c.statFrames && !c.statSkips && !c.statStalls)
                continue;
            TTT2_LOG("xma ctx %u: %uch %u Hz, %llu frames (avg %llu bits), peak %.3f, %llu skips, %llu stalls", id, c.statChannels, c.statRate,
                (unsigned long long)c.statFrames, (unsigned long long)(c.statFrames ? c.statBits / c.statFrames : 0), c.statPeak,
                (unsigned long long)c.statSkips, (unsigned long long)c.statStalls);
            c.statFrames = c.statBits = c.statSkips = c.statStalls = 0;
            c.statPeak = 0;
        }
    }

    static void DecoderThread()
    {
        std::unique_lock lock(g_mutex);
        for (;;)
        {
            ReportStats();
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
        apu::ResetDecoder(id);
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
        apu::ResetDecoder(offset / apu::kContextSize);
        context.allocated = false;
        context.enabled = false;
        std::memset(ToHost(ctx.r3.u32 & ~(apu::kContextSize - 1)), 0, apu::kContextSize);
    }
    ctx.r3.u64 = 0;
}
