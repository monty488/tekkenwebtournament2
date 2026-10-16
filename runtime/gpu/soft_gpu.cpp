// Software Xenos renderer (see soft_gpu.h).
//
// Pipeline per draw:
//   1. snapshot the state it needs from the register file (constants, render targets,
//      viewport, blend, depth);
//   2. run the vertex shader microcode once per referenced vertex;
//   3. assemble primitives (triangle list/strip/fan, rect list, quad list), clip against
//      the near plane, apply the viewport transform;
//   4. rasterise into the emulated EDRAM, running the pixel shader per covered pixel;
//   or, for copy draws, resolve the rectangle the vertices cover from EDRAM into memory.
//
// Microcode semantics follow XenosRecomp's translator (thirdparty/XenosRecomp) and the
// public Xenos documentation in Xenia.

#include "soft_gpu.h"
#include "soft_internal.h"
#include "texture.h"
#include "../kernel/guest.h"

#include <cstdint>
#include "../../thirdparty/XenosRecomp/XenosRecomp/shader_code.h"

#include <algorithm>
#include <bit>
#include <array>
#include <atomic>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace gpu
{
    namespace
    {
        static_assert(sizeof(AluInstruction) == 12);
        static_assert(sizeof(VertexFetchInstruction) == 12);
        static_assert(sizeof(TextureFetchInstruction) == 12);


        constexpr uint32_t kRegisterBase = 0x7FC80000;
        uint32_t Reg(uint32_t index) { return guest::Load<uint32_t>(kRegisterBase + index * 4); }
        float AsFloat(uint32_t v) { float f; std::memcpy(&f, &v, 4); return f; }
        float RegF(uint32_t index) { return AsFloat(Reg(index)); }
        uint32_t PhysicalToGuest(uint32_t physical) { return 0xA0000000u + (physical & 0x1FFFFFFFu); }

        std::atomic<uint64_t> g_pixels{ 0 };
        CaptureSink* g_sink = nullptr; // hardware backend, if any (see soft_internal.h)

        // Per-draw rejection counts for g_softLogDraw.
        struct DrawDebug { std::atomic<uint64_t> triangles, setups, inside, stencilFail, depthFail; } g_drawDebug;
        std::atomic<uint64_t> g_resolves{ 0 };

        // Time spent per stage since the last SoftPrintStats (ns).
        struct Stats { uint64_t vertex, raster, resolve, texture, parallel, draws, decodes, decodedTexels, hashed; } g_stats{};
        struct ScopedTimer
        {
            uint64_t& sink;
            std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
            ~ScopedTimer() { sink += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count()); }
        };

        // ---- Worker pool -----------------------------------------------------------
        // Runs fn(0..count-1) across all cores (the caller helps) and returns when every
        // job has finished. Used for row bands of rasterisation and resolves.

        thread_local bool t_inPool = false;
        std::atomic<uint64_t> g_poolBusyNs{ 0 }; // summed over threads (TTT2_GPU_STATS)

        class WorkerPool
        {
        public:
            WorkerPool()
            {
                unsigned n = std::max(1u, std::thread::hardware_concurrency());
                if (const char* v = std::getenv("TTT2_SOFT_THREADS"))
                    n = std::max(1u, unsigned(std::strtoul(v, nullptr, 10)));
                for (unsigned i = 1; i < n; i++)
                    std::thread([this] { Worker(); }).detach();
            }

            // Not reentrant across threads: only the command processor thread submits work
            // (nested calls from inside a job run inline).
            void Run(int count, const std::function<void(int)>& fn)
            {
                if (count <= 1 || t_inPool)
                {
                    for (int i = 0; i < count; i++)
                        fn(i);
                    return;
                }
                ScopedTimer timer{ g_stats.parallel };
                auto job = std::make_shared<Job>();
                job->fn = &fn;
                job->count = count;
                {
                    std::lock_guard lock(mutex_);
                    job_ = job;
                    ++generation_;
                }
                wake_.notify_all();
                Drain(*job);
                std::unique_lock lock(mutex_);
                finished_.wait(lock, [&] { return job->done == job->count; });
                job_.reset();
            }

        private:
            // Each submission has its own counters, so a worker that wakes late for a
            // finished job finds no indices left instead of claiming the next job's.
            struct Job
            {
                const std::function<void(int)>* fn = nullptr;
                int count = 0;
                std::atomic<int> next{ 0 };
                int done = 0; // guarded by mutex_
            };

            void Drain(Job& job)
            {
                t_inPool = true;
                int completed = 0;
                const auto start = std::chrono::steady_clock::now();
                for (int i; (i = job.next.fetch_add(1)) < job.count;)
                {
                    (*job.fn)(i);
                    ++completed;
                }
                g_poolBusyNs += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
                t_inPool = false;
                if (completed)
                {
                    std::lock_guard lock(mutex_);
                    job.done += completed;
                    if (job.done == job.count)
                        finished_.notify_all();
                }
            }

            void Worker()
            {
                uint64_t seen = 0;
                for (;;)
                {
                    std::shared_ptr<Job> job;
                    {
                        std::unique_lock lock(mutex_);
                        wake_.wait(lock, [&] { return generation_ != seen && job_; });
                        seen = generation_;
                        job = job_;
                    }
                    Drain(*job);
                }
            }

            std::mutex mutex_;
            std::condition_variable wake_, finished_;
            std::shared_ptr<Job> job_;
            uint64_t generation_ = 0;
        };

        WorkerPool& Pool()
        {
            static WorkerPool pool;
            return pool;
        }

        // ---- Number formats ----------------------------------------------------

        float HalfToFloat(uint16_t h)
        {
            const uint32_t sign = uint32_t(h >> 15) << 31;
            const uint32_t exp = (h >> 10) & 0x1F;
            const uint32_t mant = h & 0x3FF;
            if (exp == 0)
                return (h >> 15 ? -1.0f : 1.0f) * std::ldexp(float(mant), -24);
            // Xenos half floats have no inf/NaN encoding: exponent 31 is an ordinary value.
            const uint32_t bits = sign | ((exp + 112) << 23) | (mant << 13);
            return AsFloat(bits);
        }

        uint16_t FloatToHalf(float f)
        {
            uint32_t bits;
            std::memcpy(&bits, &f, 4);
            const uint16_t sign = uint16_t((bits >> 16) & 0x8000);
            const int exp = int((bits >> 23) & 0xFF) - 112;
            if (exp <= 0)
                return sign;
            if (exp >= 32)
                return uint16_t(sign | 0x7FFF);
            return uint16_t(sign | (exp << 10) | ((bits >> 13) & 0x3FF));
        }

        // Xenos piecewise-linear gamma (as in Xenia): four segments with slopes 4, 2, 1, 1/2.
        float LinearToPwlGamma(float x)
        {
            x = x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x);
            if (x < 0.0625f) return x * 4.0f;
            if (x < 0.125f) return x * 2.0f + 0.125f;
            if (x < 0.5f) return x + 0.25f;
            return x * 0.5f + 0.5f;
        }

        float PwlGammaToLinear(float g)
        {
            g = g < 0.0f ? 0.0f : (g > 1.0f ? 1.0f : g);
            if (g < 0.25f) return g * 0.25f;
            if (g < 0.375f) return (g - 0.125f) * 0.5f;
            if (g < 0.75f) return g - 0.25f;
            return (g - 0.5f) * 2.0f;
        }

        // 20e4 depth floats (k_24_8_FLOAT): 4-bit exponent (bias 15), 20-bit mantissa, [0, 2).
        float Float20e4To32(uint32_t v)
        {
            uint32_t mantissa = v & 0xFFFFF;
            uint32_t exponent = (v >> 20) & 0xF;
            if (!exponent)
            {
                if (!mantissa)
                    return 0.0f;
                // Denormal: normalise.
                uint32_t shift = 0;
                while (!(mantissa & 0x100000))
                {
                    mantissa <<= 1;
                    ++shift;
                }
                exponent = 1 - shift;
                mantissa &= 0xFFFFF;
            }
            return AsFloat(((exponent + 112) << 23) | (mantissa << 3));
        }

        uint32_t Float32To20e4(float f)
        {
            uint32_t bits;
            std::memcpy(&bits, &f, 4);
            if (int32_t(bits) <= 0)
                return 0;
            if (bits >= 0x40000000u) // >= 2.0
                return 0xFFFFFF;
            if (bits < 0x38800000u) // below the normal range: denormal
            {
                const uint32_t shift = std::min(113u - (bits >> 23), 24u);
                bits = (0x800000u | (bits & 0x7FFFFFu)) >> shift;
            }
            else
            {
                bits -= 0x38000000u; // rebias the exponent from 127 to 15
            }
            return ((bits + 3 + ((bits >> 3) & 1)) >> 3) & 0xFFFFFF;
        }

        uint32_t Swap32(uint32_t v, uint32_t endian)
        {
            switch (endian & 3)
            {
            case 1: return ((v & 0x00FF00FFu) << 8) | ((v >> 8) & 0x00FF00FFu);
            case 2: return (v >> 24) | ((v >> 8) & 0xFF00u) | ((v << 8) & 0xFF0000u) | (v << 24);
            case 3: return (v << 16) | (v >> 16);
            default: return v;
            }
        }

        uint32_t ReadSwapped(uint32_t address, uint32_t endian)
        {
            return Swap32(*guest::ToHost<uint32_t>(address), endian);
        }

        float Saturate(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

        // Xenos multiplies follow D3D9 rules: 0 * anything (including inf/NaN) = 0.
        float Mul(float a, float b) { return (a == 0.0f || b == 0.0f) ? 0.0f : a * b; }

        bool Compare(uint32_t func, float a, float b)
        {
            switch (func & 7)
            {
            case 0: return false;
            case 1: return a < b;
            case 2: return a == b;
            case 3: return a <= b;
            case 4: return a > b;
            case 5: return a != b;
            case 6: return a >= b;
            default: return true;
            }
        }

        // ---- EDRAM ---------------------------------------------------------------
        // 10 MiB as 2048 tiles of 80x16 32-bit samples (40x16 for 64-bit formats). Colour
        // is kept as float4 per sample and depth as float, both addressed by tile so that
        // render targets placed at different bases alias the way they do on hardware.


        std::vector<F4> g_edramColor(size_t(kEdramTiles) * kTileSamples, F4{ 0, 0, 0, 0 });
        std::vector<float> g_edramDepth(size_t(kEdramTiles) * kTileSamples, 1.0f);
        std::vector<uint8_t> g_edramStencil(size_t(kEdramTiles) * kTileSamples, 0);

        bool Is64bppColor(uint32_t format)
        {
            // k_16_16_16_16, k_16_16_16_16_FLOAT, k_32_32_FLOAT
            return format == 5 || format == 7 || format == 15;
        }

        bool IsUnormColor(uint32_t format)
        {
            return format == 0 || format == 1 || format == 2 || format == 10;
        }


        // ---- Textures ------------------------------------------------------------


        struct FetchKey
        {
            uint32_t d[6];
            bool operator==(const FetchKey& o) const { return std::memcmp(d, o.d, sizeof(d)) == 0; }
        };
        struct FetchKeyHash
        {
            size_t operator()(const FetchKey& k) const
            {
                size_t h = 1469598103934665603ull;
                for (uint32_t v : k.d)
                    h = (h ^ v) * 1099511628211ull;
                return h;
            }
        };
        std::unordered_map<FetchKey, Texture, FetchKeyHash> g_textureCache;
        uint64_t g_frame = 1;

        uint64_t HashBytes(const uint8_t* p, size_t bytes)
        {
            uint64_t h0 = 0x9E3779B97F4A7C15ull, h1 = 0xC2B2AE3D27D4EB4Full;
            size_t i = 0;
            for (; i + 16 <= bytes; i += 16)
            {
                uint64_t a, b;
                std::memcpy(&a, p + i, 8);
                std::memcpy(&b, p + i + 8, 8);
                h0 = (h0 ^ a) * 0x100000001B3ull;
                h1 = (h1 ^ b) * 0x100000001B3ull;
            }
            for (; i < bytes; i++)
                h0 = (h0 ^ p[i]) * 0x100000001B3ull;
            return h0 ^ (h1 * 31);
        }

        // Hashes guest memory in 256 KiB chunks across the pool (textures are hashed every
        // frame to detect CPU rewrites).
        uint64_t HashMemory(uint32_t address, uint32_t bytes)
        {
            const uint8_t* p = guest::ToHost<const uint8_t>(address);
            constexpr size_t kChunk = 256 * 1024;
            const int chunks = int((size_t(bytes) + kChunk - 1) / kChunk);
            if (chunks <= 1)
                return HashBytes(p, bytes);
            std::vector<uint64_t> parts(chunks);
            Pool().Run(chunks, [&](int c) {
                const size_t begin = size_t(c) * kChunk;
                parts[c] = HashBytes(p + begin, std::min(kChunk, size_t(bytes) - begin));
            });
            uint64_t h = 0xCBF29CE484222325ull;
            for (uint64_t part : parts)
                h = (h ^ part) * 0x100000001B3ull;
            return h;
        }

        // Block size (texels per side) and bytes per block of a texture format; false if unknown.
        bool FormatBlock(uint32_t format, uint32_t& blockW, uint32_t& blockBytes)
        {
            blockW = 1;
            switch (format)
            {
            case 2: blockBytes = 1; return true;
            case 3: case 4: case 10: case 15: case 24: case 30: blockBytes = 2; return true;
            case 6: case 7: case 14: case 16: case 17: case 22: case 23: case 25: case 31: case 33: case 36: case 50: blockBytes = 4; return true;
            case 26: case 32: case 34: case 37: blockBytes = 8; return true;
            case 35: case 38: blockBytes = 16; return true;
            case 18: case 51: case 58: case 59: blockW = 4; blockBytes = 8; return true;
            case 19: case 20: case 49: case 52: case 53: blockW = 4; blockBytes = 16; return true;
            default: return false;
            }
        }

        // Bytes of guest memory mip 0 occupies (0 if the format is unknown).
        uint32_t TextureByteSize(const TextureInfo& info)
        {
            uint32_t blockW, blockBytes;
            if (!FormatBlock(info.format, blockW, blockBytes))
                return 0;
            const uint32_t blocksX = (info.width + blockW - 1) / blockW;
            uint32_t blocksY = (info.height + blockW - 1) / blockW;
            uint32_t pitch = std::max((info.pitch ? info.pitch : info.width) / blockW, blocksX);
            if (info.tiled)
            {
                pitch = (pitch + 31) & ~31u;
                blocksY = (blocksY + 31) & ~31u;
            }
            return pitch * blocksY * blockBytes;
        }

        // Scales a raw component by the fetch constant's per-component sign mode:
        // 0 unsigned, 1 signed, 2 unsigned biased (x2-1), 3 gamma.
        float NormalizeComponent(uint32_t raw, uint32_t bits, uint32_t sign, bool integer)
        {
            if (sign == 1)
            {
                const int32_t s = int32_t(raw << (32 - bits)) >> (32 - bits);
                return integer ? float(s) : std::max(float(s) / float((1u << (bits - 1)) - 1), -1.0f);
            }
            if (integer)
                return float(raw);
            const float v = float(double(raw) / double((1ull << bits) - 1));
            return sign == 2 ? v * 2.0f - 1.0f : (sign == 3 ? PwlGammaToLinear(v) : v);
        }

        // BC4-style 8-byte block (DXT5A, each half of DXN): 16 values in [0, 1].
        void DecodeBc4(const uint8_t* b, float out[16])
        {
            const uint32_t a0 = b[0], a1 = b[1];
            uint32_t v[8] = { a0, a1 };
            for (int i = 2; i < 8; i++)
                v[i] = a0 > a1 ? ((8 - i) * a0 + (i - 1) * a1) / 7 : (i < 6 ? ((6 - i) * a0 + (i - 1) * a1) / 5 : (i == 6 ? 0 : 255));
            uint64_t bits = 0;
            for (int i = 0; i < 6; i++)
                bits |= uint64_t(b[2 + i]) << (8 * i);
            for (int i = 0; i < 16; i++)
                out[i] = float(v[(bits >> (3 * i)) & 7]) / 255.0f;
        }

        // Formats ConvertToBgra does not handle (or would lose precision on), decoded straight
        // to format components (comp0 = X ...). Single-component formats replicate X to all
        // four, two-component formats replicate Y to Z and W (as Xenia's host swizzles do).
        bool DecodeComponents(const TextureInfo& info, const uint32_t fetch[6], std::vector<F4>& out)
        {
            uint32_t blockW, blockBytes;
            switch (info.format)
            {
            case 7: case 16: case 17: case 22: case 23: case 24: case 25: case 26: case 30: case 31: case 32:
            case 33: case 34: case 35: case 36: case 37: case 38: case 49: case 58: case 59:
                break;
            default:
                return false;
            }
            FormatBlock(info.format, blockW, blockBytes);
            uint32_t sign[4];
            for (int k = 0; k < 4; k++)
                sign[k] = (fetch[0] >> (2 + 2 * k)) & 3;
            const bool integer = fetch[3] & 1;

            const uint32_t log2Bpb = blockBytes == 1 ? 0 : blockBytes == 2 ? 1 : blockBytes == 4 ? 2 : blockBytes == 8 ? 3 : 4;
            const uint32_t blocksX = (info.width + blockW - 1) / blockW;
            const uint32_t blocksY = (info.height + blockW - 1) / blockW;
            uint32_t pitchBlocks = (info.pitch ? info.pitch : info.width) / blockW;
            if (info.tiled)
                pitchBlocks = (pitchBlocks + 31) & ~31u;
            pitchBlocks = std::max(pitchBlocks, blocksX);
            out.assign(size_t(info.width) * info.height, F4{ 0, 0, 0, 0 });
            const uint8_t* base = guest::ToHost<const uint8_t>(info.base);

            auto readBlock = [&](uint32_t offset, uint8_t* dst) {
                // Undo the GPU endian swap so the bytes read as little-endian data.
                std::memcpy(dst, base + offset, blockBytes);
                if (info.endian == 1)
                    for (uint32_t i = 0; i + 1 < blockBytes; i += 2) std::swap(dst[i], dst[i + 1]);
                else if (info.endian == 2)
                    for (uint32_t i = 0; i + 3 < blockBytes; i += 4) { std::swap(dst[i], dst[i + 3]); std::swap(dst[i + 1], dst[i + 2]); }
                else if (info.endian == 3)
                    for (uint32_t i = 0; i + 3 < blockBytes; i += 4) { std::swap(dst[i], dst[i + 2]); std::swap(dst[i + 1], dst[i + 3]); }
            };
            auto word = [](const uint8_t* p, int i) { uint32_t v; std::memcpy(&v, p + i * 4, 4); return v; };
            auto half = [](const uint8_t* p, int i) { uint16_t v; std::memcpy(&v, p + i * 2, 2); return v; };
            auto norm = [&](uint32_t raw, uint32_t bits, int k) { return NormalizeComponent(raw, bits, sign[k], integer); };

            Pool().Run(int(blocksY), [&](int byIndex) {
                const uint32_t by = uint32_t(byIndex);
                uint8_t b[16];
                for (uint32_t bx = 0; bx < blocksX; bx++)
                {
                    const uint32_t offset = info.tiled ? TiledOffset2D(bx, by, pitchBlocks, log2Bpb) : (by * pitchBlocks + bx) * blockBytes;
                    readBlock(offset, b);
                    if (blockW == 4)
                    {
                        float r[16], g[16];
                        if (info.format == 58) // DXT3A: 4-bit values
                            for (int i = 0; i < 16; i++)
                                r[i] = float((b[i / 2] >> ((i & 1) * 4)) & 15) / 15.0f;
                        else
                            DecodeBc4(b, r);
                        if (info.format == 49) // DXN: X block then Y block
                            DecodeBc4(b + 8, g);
                        for (uint32_t t = 0; t < 16; t++)
                        {
                            const uint32_t x = bx * 4 + t % 4, y = by * 4 + t / 4;
                            if (x >= info.width || y >= info.height)
                                continue;
                            F4& o = out[size_t(y) * info.width + x];
                            if (info.format == 49)
                            {
                                const float cx = sign[0] == 1 || sign[0] == 2 ? r[t] * 2.0f - 1.0f : r[t];
                                const float cy = sign[1] == 1 || sign[1] == 2 ? g[t] * 2.0f - 1.0f : g[t];
                                o = { cx, cy, cy, cy };
                            }
                            else
                            {
                                o = { r[t], r[t], r[t], r[t] };
                            }
                        }
                        continue;
                    }
                    if (bx >= info.width || by >= info.height)
                        continue;
                    F4 c{ 0, 0, 0, 0 };
                    switch (info.format)
                    {
                    case 7: { const uint32_t v = word(b, 0); c = { norm(v & 0x3FF, 10, 0), norm((v >> 10) & 0x3FF, 10, 1), norm((v >> 20) & 0x3FF, 10, 2), norm(v >> 30, 2, 3) }; break; }
                    case 16: { const uint32_t v = word(b, 0); c = { norm(v & 0x3FF, 10, 0), norm((v >> 10) & 0x7FF, 11, 1), norm(v >> 21, 11, 2), 1.0f }; break; }
                    case 17: { const uint32_t v = word(b, 0); c = { norm(v & 0x7FF, 11, 0), norm((v >> 11) & 0x7FF, 11, 1), norm(v >> 22, 10, 2), 1.0f }; break; }
                    case 22: { const uint32_t v = word(b, 0); const float d = float(v >> 8) / float(0xFFFFFF); c = { d, d, d, d }; break; }
                    case 23: { const float d = Float20e4To32(word(b, 0) >> 8); c = { d, d, d, d }; break; }
                    case 24: { const float v = norm(half(b, 0), 16, 0); c = { v, v, v, v }; break; }
                    case 25: { const float x = norm(half(b, 0), 16, 0), y = norm(half(b, 1), 16, 1); c = { x, y, y, y }; break; }
                    case 26: c = { norm(half(b, 0), 16, 0), norm(half(b, 1), 16, 1), norm(half(b, 2), 16, 2), norm(half(b, 3), 16, 3) }; break;
                    case 30: { const float v = HalfToFloat(half(b, 0)); c = { v, v, v, v }; break; }
                    case 31: { const float x = HalfToFloat(half(b, 0)), y = HalfToFloat(half(b, 1)); c = { x, y, y, y }; break; }
                    case 32: c = { HalfToFloat(half(b, 0)), HalfToFloat(half(b, 1)), HalfToFloat(half(b, 2)), HalfToFloat(half(b, 3)) }; break;
                    case 33: { const float v = norm(word(b, 0), 32, 0); c = { v, v, v, v }; break; }
                    case 34: { const float x = norm(word(b, 0), 32, 0), y = norm(word(b, 1), 32, 1); c = { x, y, y, y }; break; }
                    case 35: c = { norm(word(b, 0), 32, 0), norm(word(b, 1), 32, 1), norm(word(b, 2), 32, 2), norm(word(b, 3), 32, 3) }; break;
                    case 36: { const float v = AsFloat(word(b, 0)); c = { v, v, v, v }; break; }
                    case 37: { const float x = AsFloat(word(b, 0)), y = AsFloat(word(b, 1)); c = { x, y, y, y }; break; }
                    case 38: c = { AsFloat(word(b, 0)), AsFloat(word(b, 1)), AsFloat(word(b, 2)), AsFloat(word(b, 3)) }; break;
                    }
                    out[size_t(by) * info.width + bx] = c;
                }
            });
            return true;
        }

        void DumpTexture(const Texture& tex, const TextureInfo& info)
        {
            char name[96];
            std::snprintf(name, sizeof(name), "tex_%08X_f%u_%ux%u.bmp", info.base, info.format, tex.width, tex.height);
            std::FILE* f = std::fopen(name, "wb");
            if (!f)
                return;
            const uint32_t imageBytes = tex.width * tex.height * 4;
            uint8_t header[54] = { 'B', 'M' };
            auto put32 = [&](int at, uint32_t v) { for (int i = 0; i < 4; i++) header[at + i] = uint8_t(v >> (8 * i)); };
            put32(2, 54 + imageBytes);
            put32(10, 54);
            put32(14, 40);
            put32(18, tex.width);
            put32(22, uint32_t(-int32_t(tex.height)));
            header[26] = 1;
            header[28] = 32;
            put32(34, imageBytes);
            std::fwrite(header, 1, sizeof(header), f);
            // Swizzled components as RGB over a checkerboard where alpha < 1.
            for (uint32_t y = 0; y < tex.height; y++)
                for (uint32_t x = 0; x < tex.width; x++)
                {
                    const F4& t = tex.texels[size_t(y) * tex.width + x];
                    const float bg = ((x / 8 + y / 8) & 1) ? 0.6f : 0.3f;
                    const float a = Saturate(t[3]);
                    uint8_t px[4];
                    for (int k = 0; k < 3; k++)
                        px[2 - k] = uint8_t(std::lround(Saturate(t[k] * a + bg * (1.0f - a)) * 255.0f));
                    px[3] = 255;
                    std::fwrite(px, 1, 4, f);
                }
            std::fclose(f);
        }

        void DecodeTexture(Texture& tex, const uint32_t fetch[6])
        {
            ScopedTimer timer{ g_stats.texture };
            static std::atomic<uint64_t> versions{ 0 };
            tex.version = ++versions;
            tex.ok = false;
            tex.stale = false;
            if ((fetch[0] & 3) != 2)
                return;
            const TextureInfo info = DecodeTextureFetch(fetch);
            if (info.width > 8192 || info.height > 8192)
                return;
            tex.base = info.base;
            tex.bytes = TextureByteSize(info);
            tex.hash = tex.bytes ? HashMemory(tex.base, tex.bytes) : 0;
            tex.verifiedFrame = g_frame;

            ++g_stats.decodes;
            g_stats.decodedTexels += uint64_t(info.width) * info.height;
            const uint32_t swizzle = (fetch[3] >> 1) & 0xFFF;
            const int expAdjust = int32_t(fetch[3] << 13) >> 26; // bits 13-18, signed
            const float expScale = std::ldexp(1.0f, expAdjust);
            uint32_t sign[4];
            for (int k = 0; k < 4; k++)
                sign[k] = (fetch[0] >> (2 + 2 * k)) & 3;
            const bool integer = fetch[3] & 1;

            uint32_t select[4];
            for (int k = 0; k < 4; k++)
                select[k] = (swizzle >> (k * 3)) & 7;
            // Applies the fetch swizzle and exponent bias to format components (comp0 = X ...).
            auto swizzleTexel = [&](const F4& comp, F4& out) {
                for (int k = 0; k < 4; k++)
                    out[k] = select[k] < 4 ? comp[select[k]] * expScale : (select[k] == 5 ? 1.0f : 0.0f);
            };
            constexpr size_t kChunk = 16384;

            // Reused across decodes (movie frames are decoded every frame).
            // Bound to plain references: lambdas run on pool threads would otherwise see
            // their own thread's (empty) copies.
            thread_local std::vector<F4> compsStorage;
            thread_local std::vector<uint32_t> bgraStorage;
            std::vector<F4>& comps = compsStorage;
            std::vector<uint32_t>& bgra = bgraStorage;
            if (DecodeComponents(info, fetch, comps))
            {
                tex.texels.resize(comps.size());
                Pool().Run(int((comps.size() + kChunk - 1) / kChunk), [&](int chunk) {
                    const size_t end = std::min(comps.size(), size_t(chunk + 1) * kChunk);
                    for (size_t i = size_t(chunk) * kChunk; i < end; i++)
                        swizzleTexel(comps[i], tex.texels[i]);
                });
            }
            else
            {
                // ConvertToBgra keeps 8_8_8_8 in native component order (X in the low byte)
                // and packs every other format as B,G,R,A bytes with R = component 0. Formats
                // with 8-bit components keep their raw bytes, so signed modes can be applied.
                const bool native8888 = info.format == 6 || info.format == 14 || info.format == 50;
                const bool twoComp = info.format == 10; // k_8_8: comp0 = low byte, comp1 = high byte
                const bool oneComp = info.format == 2;  // k_8
                const bool rawBytes = native8888 || twoComp || oneComp;
                const bool anySign = sign[0] || sign[1] || sign[2] || sign[3] || integer;
                auto convert = [&](size_t begin, size_t end) {
                    for (size_t i = begin; i < end; i++)
                    {
                        const uint32_t c = bgra[i];
                        uint32_t raw[4];
                        if (native8888)
                            raw[0] = c & 0xFF, raw[1] = (c >> 8) & 0xFF, raw[2] = (c >> 16) & 0xFF, raw[3] = c >> 24;
                        else if (twoComp)
                            raw[0] = (c >> 16) & 0xFF, raw[1] = c >> 24, raw[2] = raw[1], raw[3] = raw[1];
                        else if (oneComp)
                            raw[0] = raw[1] = raw[2] = raw[3] = c & 0xFF;
                        else
                            raw[0] = (c >> 16) & 0xFF, raw[1] = (c >> 8) & 0xFF, raw[2] = c & 0xFF, raw[3] = c >> 24;
                        F4 comp;
                        for (int k = 0; k < 4; k++)
                        {
                            if (!anySign)
                                comp[k] = float(raw[k]) / 255.0f;
                            else if (rawBytes || sign[k] != 1)
                                comp[k] = NormalizeComponent(raw[k], 8, sign[k], integer);
                            else // signed mode on an expanded format: approximate from unorm
                                comp[k] = float(raw[k]) / 255.0f * 2.0f - 1.0f;
                        }
                        swizzleTexel(comp, tex.texels[i]);
                    }
                };
                // Each band of block rows is converted to texels right after it is untiled,
                // while it is still in cache.
                uint32_t blockW = 1, blockBytes = 0;
                FormatBlock(info.format, blockW, blockBytes);
                tex.texels.resize(size_t(info.width) * info.height);
                const ParallelFor untileAndConvert = [&](uint32_t count, const std::function<void(uint32_t)>& fn) {
                    constexpr uint32_t kRows = 8;
                    Pool().Run(int((count + kRows - 1) / kRows), [&](int band) {
                        const uint32_t r0 = uint32_t(band) * kRows, r1 = std::min(count, r0 + kRows);
                        for (uint32_t row = r0; row < r1; row++)
                            fn(row);
                        const uint32_t y0 = std::min(info.height, r0 * blockW), y1 = std::min(info.height, r1 * blockW);
                        convert(size_t(y0) * info.width, size_t(y1) * info.width);
                    });
                };
                if (!ConvertToBgra(info, bgra, untileAndConvert))
                {
                    static bool warned[64];
                    if (!warned[info.format & 63])
                    {
                        warned[info.format & 63] = true;
                        std::fprintf(stderr, "[gpu] soft: unsupported texture format %u\n", info.format);
                    }
                    return;
                }
            }

            tex.width = info.width;
            tex.height = info.height;
            tex.clampX = (fetch[0] >> 10) & 7;
            tex.clampY = (fetch[0] >> 13) & 7;
            tex.magFilter = (fetch[3] >> 19) & 3;
            tex.minFilter = (fetch[3] >> 21) & 3;
            tex.ok = true;
            if (g_softDumpTextures)
                DumpTexture(tex, info);
        }

        const Texture& LoadTexture(const uint32_t fetch[6])
        {
            FetchKey key;
            std::memcpy(key.d, fetch, sizeof(key.d));
            auto [it, inserted] = g_textureCache.try_emplace(key);
            Texture& tex = it->second;
            if (inserted || tex.stale)
            {
                DecodeTexture(tex, fetch);
            }
            else if (tex.verifiedFrame != g_frame)
            {
                // First use this frame: the CPU may have rewritten the data (movie frames).
                ScopedTimer timer{ g_stats.texture };
                g_stats.hashed += tex.bytes;
                const bool same = tex.bytes && HashMemory(tex.base, tex.bytes) == tex.hash;
                if (same)
                    tex.verifiedFrame = g_frame;
                else
                    DecodeTexture(tex, fetch);
            }
            return tex;
        }

        // Marks decodes whose memory a resolve just overwrote (keeping their storage, which
        // the next decode reuses: render targets sampled every frame are large).
        void InvalidateTextures(uint32_t base, uint32_t bytes)
        {
            for (auto& [key, t] : g_textureCache)
                if (t.bytes == 0 || (t.base < base + bytes && base < t.base + t.bytes))
                    t.stale = true;
        }

        int WrapCoord(int i, int n, uint32_t mode)
        {
            switch (mode)
            {
            case 0: // repeat
                i %= n;
                return i < 0 ? i + n : i;
            case 1: // mirrored repeat
            {
                const int period = 2 * n;
                i %= period;
                if (i < 0)
                    i += period;
                return i < n ? i : period - 1 - i;
            }
            default: // clamp variants
                return i < 0 ? 0 : (i >= n ? n - 1 : i);
            }
        }

        F4 SampleTexture(const Texture& tex, float u, float v, bool linear)
        {
            const int w = int(tex.width), h = int(tex.height);
            if (!linear)
            {
                const int x = WrapCoord(int(std::floor(u)), w, tex.clampX);
                const int y = WrapCoord(int(std::floor(v)), h, tex.clampY);
                return tex.texels[size_t(y) * w + x];
            }
            const float fu = u - 0.5f, fv = v - 0.5f;
            const float x0f = std::floor(fu), y0f = std::floor(fv);
            const float ax = fu - x0f, ay = fv - y0f;
            const int x0 = WrapCoord(int(x0f), w, tex.clampX), x1 = WrapCoord(int(x0f) + 1, w, tex.clampX);
            const int y0 = WrapCoord(int(y0f), h, tex.clampY), y1 = WrapCoord(int(y0f) + 1, h, tex.clampY);
            const F4& t00 = tex.texels[size_t(y0) * w + x0];
            const F4& t10 = tex.texels[size_t(y0) * w + x1];
            const F4& t01 = tex.texels[size_t(y1) * w + x0];
            const F4& t11 = tex.texels[size_t(y1) * w + x1];
            F4 r;
            for (int k = 0; k < 4; k++)
            {
                const float top = t00[k] + (t10[k] - t00[k]) * ax;
                const float bottom = t01[k] + (t11[k] - t01[k]) * ax;
                r[k] = top + (bottom - top) * ay;
            }
            return r;
        }

        // ---- Draw state snapshot ---------------------------------------------------

        struct DrawState
        {
            F4 constants[512];      // VS: 0-255, PS: 256-511
            uint32_t bools[8];
            uint32_t loops[32];
            uint32_t fetch[32 * 6]; // fetch constants (texture slots; vertex slots alias)
            std::atomic<const Texture*> textures[32] = {};
            std::mutex textureMutex;

            // Called from raster workers: the first use of a slot decodes under the lock.
            const Texture& TextureSlot(uint32_t slot)
            {
                slot &= 31;
                if (const Texture* t = textures[slot].load(std::memory_order_acquire))
                    return *t;
                std::lock_guard lock(textureMutex);
                if (!textures[slot].load())
                    textures[slot].store(&LoadTexture(&fetch[slot * 6]), std::memory_order_release);
                return *textures[slot].load();
            }

            void Load()
            {
                for (uint32_t i = 0; i < 512; i++)
                    for (uint32_t k = 0; k < 4; k++)
                        constants[i][k] = RegF(0x4000 + i * 4 + k);
                for (uint32_t i = 0; i < 8; i++)
                    bools[i] = Reg(0x4900 + i);
                for (uint32_t i = 0; i < 32; i++)
                    loops[i] = Reg(0x4908 + i);
                for (uint32_t i = 0; i < 32 * 6; i++)
                    fetch[i] = Reg(0x4800 + i);
            }
        };

        // ---- Shader interpreter ------------------------------------------------------

        struct Invocation
        {
            F4 r[128];
            F4 exports[64];
            uint64_t exported = 0; // bit per export register
            float ps = 0.0f;
            bool p0 = false;
            bool killed = false;
            int a0 = 0;
            int aL = 0;
            // Vertex fetch state shared with mini-fetches.
            uint32_t fetchSlot = 0;
            float fetchIndex = 0.0f;
            uint32_t fetchStride = 0;
        };

        struct Program
        {
            const uint32_t* code = nullptr;
            uint32_t count = 0;
            bool pixel = false;
        };

        // An ALU instruction with its operands decoded and aL folded into the indices it
        // offsets, so executing it per pixel does no bitfield or control-flow work.

        AluOp DecodeAlu(const uint32_t* ins, int aL)
        {
            AluOp op;
            AluInstruction& a = op.a;
            std::memcpy(&a, ins, 12);
            auto decode = [&](uint32_t reg, bool isRegister, uint32_t swizzle, bool negate, bool constRelative, bool relativeSwizzle) {
                AluOp::Source src;
                src.isRegister = isRegister;
                src.negate = negate;
                if (isRegister)
                {
                    src.abs = (reg & 0x80) != 0;
                    src.index = (reg & 0x3F) + ((reg & 0x40) ? uint32_t(aL) : 0u);
                }
                else
                {
                    src.abs = a.absConstants;
                    src.index = reg;
                    if (constRelative)
                    {
                        if (a.constAddressRegisterRelative)
                            src.relativeA0 = true;
                        else
                            src.index += uint32_t(aL);
                    }
                }
                for (int i = 0; i < 4; i++)
                    src.comp[i] = uint8_t(relativeSwizzle ? (((swizzle >> (i * 2)) + i) & 3) : i);
                return src;
            };
            // Constant operands take const0Relative/const1Relative in source order.
            int constIndex = 0;
            auto source = [&](uint32_t reg, bool select, uint32_t swizzle, bool negate) {
                bool relative = false;
                if (!select)
                    relative = (constIndex++ == 0) ? a.const0Relative : a.const1Relative;
                return decode(reg, select, swizzle, negate, relative, true);
            };
            op.src[0] = source(a.src1Register, a.src1Select, a.src1Swizzle, a.src1Negate);
            op.src[1] = source(a.src2Register, a.src2Select, a.src2Swizzle, a.src2Negate);
            const uint32_t sop = uint32_t(a.scalarOpcode);
            op.scalarConst = sop >= 42 && sop <= 47;
            if (op.scalarConst)
            {
                // const (src3Register) op GPR (encoded in the opcode/select/swizzle bits)
                const uint32_t swz = a.src3Swizzle;
                op.src[2] = decode(a.src3Register, false, 0xE4, a.src3Negate, a.const0Relative, false);
                op.scalarConstComp = ((swz >> 6) + 3) & 3;
                op.scalarGpr = ((sop & 1) | (uint32_t(a.src3Select) << 1) | (swz & 0x3C)) & 127;
                op.scalarGprComp = swz & 3;
            }
            else
            {
                op.src[2] = source(a.src3Register, a.src3Select, a.src3Swizzle, a.src3Negate);
            }
            op.vectorDest = a.vectorDest + (a.vectorDestRelative ? uint32_t(aL) : 0u);
            op.scalarDest = a.scalarDest + (a.scalarDestRelative ? uint32_t(aL) : 0u);
            return op;
        }

        // Vector ALU op for one invocation. pushPredicate/newP0 report a predicate push,
        // kill a kill condition (applied by the caller); newP0 must hold the current p0.
        void VectorOp(AluVectorOpcode vop, const F4& src1, const F4& src2, const F4& src3, F4& vr,
            bool& pushPredicate, bool& newP0, bool& kill)
        {
            switch (vop)
            {
            case AluVectorOpcode::Add: for (int i = 0; i < 4; i++) vr[i] = src1[i] + src2[i]; break;
            case AluVectorOpcode::Mul: for (int i = 0; i < 4; i++) vr[i] = Mul(src1[i], src2[i]); break;
            case AluVectorOpcode::Max:
            case AluVectorOpcode::MaxA: for (int i = 0; i < 4; i++) vr[i] = src1[i] >= src2[i] ? src1[i] : src2[i]; break;
            case AluVectorOpcode::Min: for (int i = 0; i < 4; i++) vr[i] = src1[i] < src2[i] ? src1[i] : src2[i]; break;
            case AluVectorOpcode::Seq: for (int i = 0; i < 4; i++) vr[i] = src1[i] == src2[i] ? 1.0f : 0.0f; break;
            case AluVectorOpcode::Sgt: for (int i = 0; i < 4; i++) vr[i] = src1[i] > src2[i] ? 1.0f : 0.0f; break;
            case AluVectorOpcode::Sge: for (int i = 0; i < 4; i++) vr[i] = src1[i] >= src2[i] ? 1.0f : 0.0f; break;
            case AluVectorOpcode::Sne: for (int i = 0; i < 4; i++) vr[i] = src1[i] != src2[i] ? 1.0f : 0.0f; break;
            case AluVectorOpcode::Frc: for (int i = 0; i < 4; i++) vr[i] = src1[i] - std::floor(src1[i]); break;
            case AluVectorOpcode::Trunc: for (int i = 0; i < 4; i++) vr[i] = std::trunc(src1[i]); break;
            case AluVectorOpcode::Floor: for (int i = 0; i < 4; i++) vr[i] = std::floor(src1[i]); break;
            case AluVectorOpcode::Mad: for (int i = 0; i < 4; i++) vr[i] = Mul(src1[i], src2[i]) + src3[i]; break;
            case AluVectorOpcode::CndEq: for (int i = 0; i < 4; i++) vr[i] = src1[i] == 0.0f ? src2[i] : src3[i]; break;
            case AluVectorOpcode::CndGe: for (int i = 0; i < 4; i++) vr[i] = src1[i] >= 0.0f ? src2[i] : src3[i]; break;
            case AluVectorOpcode::CndGt: for (int i = 0; i < 4; i++) vr[i] = src1[i] > 0.0f ? src2[i] : src3[i]; break;
            case AluVectorOpcode::Dp4:
            {
                const float d = Mul(src1[0], src2[0]) + Mul(src1[1], src2[1]) + Mul(src1[2], src2[2]) + Mul(src1[3], src2[3]);
                vr = { d, d, d, d };
                break;
            }
            case AluVectorOpcode::Dp3:
            {
                const float d = Mul(src1[0], src2[0]) + Mul(src1[1], src2[1]) + Mul(src1[2], src2[2]);
                vr = { d, d, d, d };
                break;
            }
            case AluVectorOpcode::Dp2Add:
            {
                const float d = Mul(src1[0], src2[0]) + Mul(src1[1], src2[1]) + src3[0];
                vr = { d, d, d, d };
                break;
            }
            case AluVectorOpcode::Cube:
            {
                // src1 = r.zzxy, src2 = r.yxzz (as the D3D compiler emits it).
                const float x = src1[2], y = src2[0], z = src1[0];
                const float ax = std::fabs(x), ay = std::fabs(y), az = std::fabs(z);
                float sc, tc, ma, face;
                if (az >= ax && az >= ay) { sc = z < 0 ? -x : x; tc = -y; ma = az; face = z < 0 ? 5.0f : 4.0f; }
                else if (ay >= ax) { sc = x; tc = y < 0 ? -z : z; ma = ay; face = y < 0 ? 3.0f : 2.0f; }
                else { sc = x < 0 ? z : -z; tc = -y; ma = ax; face = x < 0 ? 1.0f : 0.0f; }
                vr = { tc, sc, 2.0f * ma, face };
                break;
            }
            case AluVectorOpcode::Max4:
            {
                const float m = std::max(std::max(src1[0], src1[1]), std::max(src1[2], src1[3]));
                vr = { m, m, m, m };
                break;
            }
            case AluVectorOpcode::SetpEqPush:
            case AluVectorOpcode::SetpNePush:
            case AluVectorOpcode::SetpGtPush:
            case AluVectorOpcode::SetpGePush:
            {
                const float b = src2[3];
                bool cond = false;
                switch (vop)
                {
                case AluVectorOpcode::SetpEqPush: cond = b == 0.0f; break;
                case AluVectorOpcode::SetpNePush: cond = b != 0.0f; break;
                case AluVectorOpcode::SetpGtPush: cond = b > 0.0f; break;
                default: cond = b >= 0.0f; break;
                }
                pushPredicate = true;
                newP0 = src1[3] == 0.0f && cond;
                const float v = newP0 ? 0.0f : src1[0] + 1.0f;
                vr = { v, v, v, v };
                break;
            }
            case AluVectorOpcode::KillEq:
            case AluVectorOpcode::KillGt:
            case AluVectorOpcode::KillGe:
            case AluVectorOpcode::KillNe:
            {
                for (int i = 0; i < 4; i++)
                {
                    switch (vop)
                    {
                    case AluVectorOpcode::KillEq: kill |= src1[i] == src2[i]; break;
                    case AluVectorOpcode::KillGt: kill |= src1[i] > src2[i]; break;
                    case AluVectorOpcode::KillGe: kill |= src1[i] >= src2[i]; break;
                    default: kill |= src1[i] != src2[i]; break;
                    }
                }
                const float v = kill ? 1.0f : 0.0f;
                vr = { v, v, v, v };
                break;
            }
            case AluVectorOpcode::Dst:
                vr = { 1.0f, Mul(src1[1], src2[1]), src1[2], src2[3] };
                break;
            default:
                break;
            }
        }

        // Scalar ALU op for one invocation: returns the result (before saturation) and
        // updates a0, newP0 (holding the current p0), kill and whether ps is written.
        float ScalarOp(AluScalarOpcode sop, float sa, float sb, float ps, int& a0, bool& newP0, bool& kill, bool& scalarWrites)
        {
            float sr = ps;
            scalarWrites = sop != AluScalarOpcode::RetainPrev;
            switch (sop)
            {
            case AluScalarOpcode::Adds: sr = sa + sb; break;
            case AluScalarOpcode::AddsPrev: sr = sa + ps; break;
            case AluScalarOpcode::Muls: sr = Mul(sa, sb); break;
            case AluScalarOpcode::MulsPrev:
            case AluScalarOpcode::MulsPrev2: sr = Mul(sa, ps); break;
            case AluScalarOpcode::Maxs: sr = sa >= sb ? sa : sb; break;
            case AluScalarOpcode::Mins: sr = sa < sb ? sa : sb; break;
            case AluScalarOpcode::Seqs: sr = sa == 0.0f ? 1.0f : 0.0f; break;
            case AluScalarOpcode::Sgts: sr = sa > 0.0f ? 1.0f : 0.0f; break;
            case AluScalarOpcode::Sges: sr = sa >= 0.0f ? 1.0f : 0.0f; break;
            case AluScalarOpcode::Snes: sr = sa != 0.0f ? 1.0f : 0.0f; break;
            case AluScalarOpcode::Frcs: sr = sa - std::floor(sa); break;
            case AluScalarOpcode::Truncs: sr = std::trunc(sa); break;
            case AluScalarOpcode::Floors: sr = std::floor(sa); break;
            case AluScalarOpcode::Exp: sr = std::exp2(sa); break;
            case AluScalarOpcode::Logc: sr = std::log2(sa); if (std::isinf(sr) && sr < 0) sr = -FLT_MAX; break;
            case AluScalarOpcode::Log: sr = std::log2(sa); break;
            case AluScalarOpcode::Rcpc: sr = 1.0f / sa; if (std::isinf(sr)) sr = std::copysign(FLT_MAX, sr); break;
            case AluScalarOpcode::Rcpf: sr = 1.0f / sa; if (std::isinf(sr)) sr = std::copysign(0.0f, sr); break;
            case AluScalarOpcode::Rcp: sr = 1.0f / sa; break;
            case AluScalarOpcode::Rsqc: sr = 1.0f / std::sqrt(sa); if (std::isinf(sr)) sr = std::copysign(FLT_MAX, sr); break;
            case AluScalarOpcode::Rsqf: sr = 1.0f / std::sqrt(sa); if (std::isinf(sr)) sr = std::copysign(0.0f, sr); break;
            case AluScalarOpcode::Rsq: sr = 1.0f / std::sqrt(sa); break;
            case AluScalarOpcode::MaxAs:
                a0 = int(std::clamp(std::floor(sa + 0.5f), -256.0f, 255.0f));
                sr = sa >= sb ? sa : sb;
                break;
            case AluScalarOpcode::MaxAsf:
                a0 = int(std::clamp(std::floor(sa), -256.0f, 255.0f));
                sr = sa >= sb ? sa : sb;
                break;
            case AluScalarOpcode::Subs: sr = sa - sb; break;
            case AluScalarOpcode::SubsPrev: sr = sa - ps; break;
            case AluScalarOpcode::SetpEq: newP0 = sa == 0.0f; sr = newP0 ? 0.0f : 1.0f; break;
            case AluScalarOpcode::SetpNe: newP0 = sa != 0.0f; sr = newP0 ? 0.0f : 1.0f; break;
            case AluScalarOpcode::SetpGt: newP0 = sa > 0.0f; sr = newP0 ? 0.0f : 1.0f; break;
            case AluScalarOpcode::SetpGe: newP0 = sa >= 0.0f; sr = newP0 ? 0.0f : 1.0f; break;
            case AluScalarOpcode::SetpInv:
                newP0 = sa == 1.0f;
                sr = newP0 ? 0.0f : (sa == 0.0f ? 1.0f : sa);
                break;
            case AluScalarOpcode::SetpPop:
                newP0 = sa - 1.0f <= 0.0f;
                sr = newP0 ? 0.0f : sa - 1.0f;
                break;
            case AluScalarOpcode::SetpClr: newP0 = false; sr = FLT_MAX; break;
            case AluScalarOpcode::SetpRstr: newP0 = sa == 0.0f; sr = sa; break;
            case AluScalarOpcode::KillsEq:
            case AluScalarOpcode::KillsGt:
            case AluScalarOpcode::KillsGe:
            case AluScalarOpcode::KillsNe:
            case AluScalarOpcode::KillsOne:
            {
                switch (sop)
                {
                case AluScalarOpcode::KillsEq: kill = sa == 0.0f; break;
                case AluScalarOpcode::KillsGt: kill = sa > 0.0f; break;
                case AluScalarOpcode::KillsGe: kill = sa >= 0.0f; break;
                case AluScalarOpcode::KillsNe: kill = sa != 0.0f; break;
                default: kill = sa == 1.0f; break;
                }
                sr = kill ? 1.0f : 0.0f;
                break;
            }
            case AluScalarOpcode::Sqrt: sr = std::sqrt(sa); break;
            case AluScalarOpcode::Mulsc0:
            case AluScalarOpcode::Mulsc1: sr = Mul(sa, sb); break;
            case AluScalarOpcode::Addsc0:
            case AluScalarOpcode::Addsc1: sr = sa + sb; break;
            case AluScalarOpcode::Subsc0:
            case AluScalarOpcode::Subsc1: sr = sa - sb; break;
            case AluScalarOpcode::Sin: sr = std::sin(sa); break;
            case AluScalarOpcode::Cos: sr = std::cos(sa); break;
            default: scalarWrites = false; break;
            }
            return sr;
        }

        // Executes decoded instructions for one invocation (shared by the interpreter and
        // the compiled, batched pixel path).
        struct Executor
        {
            DrawState& state;
            bool pixel;
            uint32_t constBase;

            F4 Read(const AluOp::Source& src, const Invocation& s) const
            {
                const F4& raw = src.isRegister ? s.r[src.index & 127]
                    : state.constants[constBase + ((src.index + (src.relativeA0 ? uint32_t(s.a0) : 0u)) & 255)];
                F4 out;
                for (int i = 0; i < 4; i++)
                {
                    float v = raw[src.comp[i]];
                    if (src.abs)
                        v = std::fabs(v);
                    out[i] = src.negate ? -v : v;
                }
                return out;
            }

            void Alu(const AluOp& op, Invocation& s) const
            {
                const AluInstruction& a = op.a;
                if (a.isPredicated && s.p0 != bool(a.predicateCondition))
                    return;

                const auto vop = a.vectorOpcode;
                const auto sop = a.scalarOpcode;
                const F4 src1 = Read(op.src[0], s);
                const F4 src2 = Read(op.src[1], s);

                // Scalar operands come from src3: a = its swizzled .w, b = its swizzled .x.
                F4 src3{};
                float sa = 0.0f, sb = 0.0f;
                if (op.scalarConst)
                {
                    sa = Read(op.src[2], s)[op.scalarConstComp];
                    float v = s.r[op.scalarGpr][op.scalarGprComp];
                    if (a.absConstants)
                        v = std::fabs(v);
                    sb = a.src3Negate ? -v : v;
                }
                else
                {
                    src3 = Read(op.src[2], s);
                    // The relative swizzle was applied per component; the scalar operands
                    // are picked by their absolute positions (components 3 and 0).
                    sa = src3[3];
                    sb = src3[0];
                }

                // ---- vector op
                F4 vr{};
                bool pushPredicate = false, newP0 = s.p0, vectorKill = false;
                VectorOp(vop, src1, src2, src3, vr, pushPredicate, newP0, vectorKill);
                if (vectorKill && pixel)
                    s.killed = true;
                if (vop == AluVectorOpcode::MaxA)
                    s.a0 = int(std::clamp(std::floor(src1[3] + 0.5f), -256.0f, 255.0f));
                if (a.vectorSaturate)
                    for (float& v : vr)
                        v = Saturate(v);

                // ---- scalar op
                bool scalarWrites = true, kill = false;
                float sr = ScalarOp(sop, sa, sb, s.ps, s.a0, newP0, kill, scalarWrites);
                if (kill && pixel)
                    s.killed = true;
                if (a.scalarSaturate)
                    sr = Saturate(sr);
                const bool scalarSetsPredicate = uint32_t(sop) >= 27 && uint32_t(sop) <= 34;
                if (pushPredicate || scalarSetsPredicate)
                    s.p0 = newP0;
                if (scalarWrites)
                    s.ps = sr;

                // ---- writes
                const uint32_t vmask = a.vectorWriteMask, smask = a.scalarWriteMask;
                if (a.exportData)
                {
                    const uint32_t dest = a.vectorDest & 63;
                    F4& e = s.exports[dest];
                    const uint32_t ones = vmask & smask;
                    const uint32_t zeros = a.scalarDestRelative ? (0xFu & ~(vmask | smask)) : 0u;
                    for (int i = 0; i < 4; i++)
                    {
                        const uint32_t bit = 1u << i;
                        if (ones & bit) e[i] = 1.0f;
                        else if (vmask & bit) e[i] = vr[i];
                        else if (smask & bit) e[i] = s.ps;
                        else if (zeros & bit) e[i] = 0.0f;
                    }
                    if (vmask | smask | zeros)
                        s.exported |= 1ull << dest;
                }
                else
                {
                    if (vmask)
                    {
                        F4& r = s.r[op.vectorDest & 127];
                        for (int i = 0; i < 4; i++)
                            if (vmask & (1u << i))
                                r[i] = vr[i];
                    }
                    if (smask)
                    {
                        F4& r = s.r[op.scalarDest & 127];
                        for (int i = 0; i < 4; i++)
                            if (smask & (1u << i))
                                r[i] = s.ps;
                    }
                }
            }

            static void ApplyDestSwizzle(F4& dst, const F4& value, uint32_t dstSwizzle)
            {
                for (int i = 0; i < 4; i++)
                {
                    const uint32_t sel = (dstSwizzle >> (i * 3)) & 7;
                    if (sel < 4) dst[i] = value[sel];
                    else if (sel == 4) dst[i] = 0.0f;
                    else if (sel == 5) dst[i] = 1.0f;
                }
            }

            void VertexFetch(const VertexFetchInstruction& f, Invocation& s) const
            {
                if (f.isPredicated && s.p0 != bool(f.predicateCondition))
                    return;
                if (!f.isMiniFetch)
                {
                    s.fetchSlot = f.constIndex * 3 + f.constIndexSelect;
                    s.fetchIndex = s.r[f.srcRegister & 63][f.srcSwizzle & 3];
                    s.fetchStride = f.stride;
                }
                const int index = int(f.isIndexRounded ? std::floor(s.fetchIndex + 0.5f) : std::floor(s.fetchIndex));
                const uint32_t d0 = state.fetch[s.fetchSlot * 2], d1 = state.fetch[s.fetchSlot * 2 + 1];
                const uint32_t base = PhysicalToGuest(d0 & 0xFFFFFFFCu);
                const uint32_t endian = d1 & 3;
                const uint32_t address = base + uint32_t((int64_t(index) * s.fetchStride + f.offset) * 4);

                F4 v{ 0.0f, 0.0f, 0.0f, 1.0f };
                const bool isSigned = f.formatCompAll;
                const bool isInteger = f.numFormatAll;
                auto normalise = [&](uint32_t raw, uint32_t bits) {
                    if (isSigned)
                    {
                        const int32_t sv = int32_t(raw << (32 - bits)) >> (32 - bits);
                        if (isInteger)
                            return float(sv);
                        const float maxv = float((1u << (bits - 1)) - 1);
                        return f.signedRfModeAll ? (float(sv) * 2.0f + 1.0f) / (maxv * 2.0f + 1.0f) : std::max(float(sv) / maxv, -1.0f);
                    }
                    if (isInteger)
                        return float(raw);
                    return float(raw) / float(bits == 32 ? 4294967295.0 : double((1ull << bits) - 1));
                };
                auto bitsAt = [](uint32_t value, uint32_t offset, uint32_t bits) {
                    return bits == 32 ? value : (value >> offset) & ((1u << bits) - 1);
                };

                bool isFloat = false;
                switch (f.format)
                {
                case 6: // k_8_8_8_8
                {
                    const uint32_t d = ReadSwapped(address, endian);
                    for (int i = 0; i < 4; i++)
                        v[i] = normalise(bitsAt(d, i * 8, 8), 8);
                    break;
                }
                case 7: // k_2_10_10_10
                {
                    const uint32_t d = ReadSwapped(address, endian);
                    v = { normalise(bitsAt(d, 0, 10), 10), normalise(bitsAt(d, 10, 10), 10), normalise(bitsAt(d, 20, 10), 10), normalise(bitsAt(d, 30, 2), 2) };
                    break;
                }
                case 16: // k_10_11_11
                {
                    const uint32_t d = ReadSwapped(address, endian);
                    v = { normalise(bitsAt(d, 0, 10), 10), normalise(bitsAt(d, 10, 11), 11), normalise(bitsAt(d, 21, 11), 11), 1.0f };
                    break;
                }
                case 17: // k_11_11_10
                {
                    const uint32_t d = ReadSwapped(address, endian);
                    v = { normalise(bitsAt(d, 0, 11), 11), normalise(bitsAt(d, 11, 11), 11), normalise(bitsAt(d, 22, 10), 10), 1.0f };
                    break;
                }
                case 25: // k_16_16 (one packed dword per pair, X in the low half)
                case 26: // k_16_16_16_16
                {
                    const uint32_t dwords = f.format == 25 ? 1 : 2;
                    for (uint32_t d = 0; d < dwords; d++)
                    {
                        const uint32_t w = ReadSwapped(address + d * 4, endian);
                        v[d * 2] = normalise(w & 0xFFFF, 16);
                        v[d * 2 + 1] = normalise(w >> 16, 16);
                    }
                    break;
                }
                case 31: // k_16_16_FLOAT
                case 32: // k_16_16_16_16_FLOAT
                {
                    isFloat = true;
                    const uint32_t dwords = f.format == 31 ? 1 : 2;
                    for (uint32_t d = 0; d < dwords; d++)
                    {
                        const uint32_t w = ReadSwapped(address + d * 4, endian);
                        v[d * 2] = HalfToFloat(uint16_t(w & 0xFFFF));
                        v[d * 2 + 1] = HalfToFloat(uint16_t(w >> 16));
                    }
                    break;
                }
                case 33: case 34: case 35: // k_32, k_32_32, k_32_32_32_32
                {
                    const uint32_t n = f.format - 32;
                    for (uint32_t i = 0; i < n; i++)
                        v[i] = normalise(ReadSwapped(address + i * 4, endian), 32);
                    break;
                }
                case 36: case 37: case 38: case 57: // k_32_FLOAT, k_32_32_FLOAT, k_32_32_32_32_FLOAT, k_32_32_32_FLOAT
                {
                    isFloat = true;
                    const uint32_t n = f.format == 57 ? 3 : f.format - 35;
                    for (uint32_t i = 0; i < n; i++)
                        v[i] = AsFloat(ReadSwapped(address + i * 4, endian));
                    break;
                }
                default:
                {
                    static bool warned[64];
                    if (!warned[f.format & 63])
                    {
                        warned[f.format & 63] = true;
                        std::fprintf(stderr, "[gpu] soft: unsupported vertex format %u\n", uint32_t(f.format));
                    }
                    break;
                }
                }
                if (!isFloat && f.expAdjust)
                    for (float& c : v)
                        c = std::ldexp(c, f.expAdjust);
                ApplyDestSwizzle(s.r[f.dstRegister & 63], v, f.dstSwizzle);
            }

            void TextureFetch(const TextureFetchInstruction& f, Invocation& s) const
            {
                if (f.isPredicated && s.p0 != bool(f.predCondition))
                    return;
                if (f.opcode != FetchOpcode::TextureFetch)
                {
                    if (f.opcode == FetchOpcode::GetTextureWeights || f.opcode == FetchOpcode::GetTextureComputedLod ||
                        f.opcode == FetchOpcode::GetTextureGradients || f.opcode == FetchOpcode::GetTextureBorderColorFrac)
                        ApplyDestSwizzle(s.r[f.dstRegister & 63], F4{ 0, 0, 0, 0 }, f.dstSwizzle);
                    return;
                }
                const F4& src = s.r[f.srcRegister & 63];
                const F4 result = FetchTexel(f, state.TextureSlot(f.constIndex), src[f.srcSwizzle & 3], src[(f.srcSwizzle >> 2) & 3]);
                ApplyDestSwizzle(s.r[f.dstRegister & 63], result, f.dstSwizzle);
            }

            // A texture fetch's result for coordinates (u, v) from its source register.
            static F4 FetchTexel(const TextureFetchInstruction& f, const Texture& tex, float u, float v)
            {
                F4 result{ 0.0f, 0.0f, 0.0f, 0.0f };
                if (tex.ok && tex.width && tex.height)
                {
                    if (!f.texCoordDenorm)
                    {
                        u *= float(tex.width);
                        v *= float(tex.height);
                    }
                    u += float(f.offsetX) * 0.5f;
                    v += float(f.offsetY) * 0.5f;
                    const uint32_t filter = f.magFilter == 3 ? tex.magFilter : f.magFilter;
                    result = SampleTexture(tex, u, v, filter == 1);
                }
                return result;
            }
        };

        bool BoolConstant(const DrawState& state, uint32_t index) { return (state.bools[(index >> 5) & 7] >> (index & 31)) & 1; }

        // Walks the control flow program for one invocation, decoding as it goes.
        struct Interpreter
        {
            DrawState& state;
            const Program& program;
            Invocation& s;
            uint32_t constBase;

            bool BoolConstant(uint32_t index) const { return gpu::BoolConstant(state, index); }

            void ExecClause(uint32_t address, uint32_t count, uint32_t sequence)
            {
                const Executor exec{ state, program.pixel, constBase };
                for (uint32_t k = 0; k < count; k++)
                {
                    const uint32_t at = (address + k) * 3;
                    if (at + 3 > program.count)
                        return;
                    const uint32_t* ins = program.code + at;
                    if ((sequence >> (k * 2)) & 1)
                    {
                        if ((ins[0] & 0x1F) == uint32_t(FetchOpcode::VertexFetch))
                        {
                            VertexFetchInstruction f;
                            std::memcpy(&f, ins, 12);
                            exec.VertexFetch(f, s);
                        }
                        else
                        {
                            TextureFetchInstruction f;
                            std::memcpy(&f, ins, 12);
                            exec.TextureFetch(f, s);
                        }
                    }
                    else
                    {
                        exec.Alu(DecodeAlu(ins, s.aL), s);
                    }
                }
            }

            void Run()
            {
                struct Loop { uint32_t count, iteration; int savedAL; int step; };
                Loop loops[4];
                int loopDepth = 0;
                uint32_t calls[4];
                int callDepth = 0;
                uint32_t pc = 0;

                for (int guard = 0; guard < 65536; guard++)
                {
                    const uint32_t base = (pc / 2) * 3;
                    if (base + 3 > program.count)
                        return;
                    uint32_t w0, w1;
                    if (pc & 1)
                    {
                        w0 = (program.code[base + 1] >> 16) | (program.code[base + 2] << 16);
                        w1 = program.code[base + 2] >> 16;
                    }
                    else
                    {
                        w0 = program.code[base];
                        w1 = program.code[base + 1] & 0xFFFF;
                    }
                    const uint32_t opcode = (w1 >> 12) & 0xF;
                    const bool condition = (w1 >> 10) & 1;
                    uint32_t next = pc + 1;

                    switch (ControlFlowOpcode(opcode))
                    {
                    case ControlFlowOpcode::Nop:
                    case ControlFlowOpcode::Alloc:
                    case ControlFlowOpcode::MarkVsFetchDone:
                        break;

                    case ControlFlowOpcode::Exec:
                    case ControlFlowOpcode::ExecEnd:
                    case ControlFlowOpcode::CondExec:
                    case ControlFlowOpcode::CondExecEnd:
                    case ControlFlowOpcode::CondExecPred:
                    case ControlFlowOpcode::CondExecPredEnd:
                    case ControlFlowOpcode::CondExecPredClean:
                    case ControlFlowOpcode::CondExecPredCleanEnd:
                    {
                        const auto op = ControlFlowOpcode(opcode);
                        bool run = true;
                        if (op == ControlFlowOpcode::CondExec || op == ControlFlowOpcode::CondExecEnd)
                            run = BoolConstant((w1 >> 2) & 0xFF) == condition;
                        else if (op != ControlFlowOpcode::Exec && op != ControlFlowOpcode::ExecEnd)
                            run = s.p0 == condition;
                        if (run)
                            ExecClause(w0 & 0xFFF, (w0 >> 12) & 7, (w0 >> 16) & 0xFFF);
                        if (op == ControlFlowOpcode::ExecEnd || op == ControlFlowOpcode::CondExecEnd ||
                            op == ControlFlowOpcode::CondExecPredEnd || op == ControlFlowOpcode::CondExecPredCleanEnd)
                            return;
                        break;
                    }

                    case ControlFlowOpcode::LoopStart:
                    {
                        const uint32_t constant = state.loops[(w0 >> 16) & 31];
                        const uint32_t count = constant & 0xFF;
                        if (loopDepth == 4)
                            return;
                        loops[loopDepth++] = { count, 0, s.aL, int(int8_t(constant >> 16)) };
                        s.aL = int((constant >> 8) & 0xFF);
                        if (count == 0)
                        {
                            s.aL = loops[--loopDepth].savedAL;
                            next = w0 & 0x1FFF;
                        }
                        break;
                    }

                    case ControlFlowOpcode::LoopEnd:
                    {
                        if (loopDepth == 0)
                            break;
                        Loop& loop = loops[loopDepth - 1];
                        const bool breakNow = ((w0 >> 21) & 1) && s.p0 == condition;
                        if (!breakNow && ++loop.iteration < loop.count)
                        {
                            s.aL += loop.step;
                            next = w0 & 0x1FFF;
                        }
                        else
                        {
                            s.aL = loop.savedAL;
                            --loopDepth;
                        }
                        break;
                    }

                    case ControlFlowOpcode::CondCall:
                    case ControlFlowOpcode::CondJmp:
                    {
                        bool take;
                        if ((w0 >> 13) & 1)
                            take = true;
                        else if ((w0 >> 14) & 1)
                            take = s.p0 == condition;
                        else
                            take = BoolConstant((w1 >> 2) & 0xFF) == condition;
                        if (!take)
                            break;
                        if (ControlFlowOpcode(opcode) == ControlFlowOpcode::CondCall)
                        {
                            if (callDepth == 4)
                                return;
                            calls[callDepth++] = pc + 1;
                        }
                        next = w0 & 0x1FFF;
                        break;
                    }

                    case ControlFlowOpcode::Return:
                        if (callDepth == 0)
                            return;
                        next = calls[--callDepth];
                        break;

                    default:
                        return;
                    }
                    pc = next;
                }
            }
        };

        // ---- Compiled pixel shaders ------------------------------------------------------
        // Control flow that depends only on per-draw state (bool and loop constants,
        // unconditional jumps and calls) is resolved once per draw, leaving a straight list
        // of decoded instructions with aL folded in. Pixels then run through the list in
        // batches, so decoding and control flow cost once per batch instead of once per
        // pixel. Programs whose flow depends on the predicate (predicated jumps, calls or
        // loop breaks) are not compiled and keep using the Interpreter.



        // Mirrors Interpreter::Run with the per-invocation state that control flow can
        // depend on (aL) tracked statically.
        CompiledShader CompileFlow(const DrawState& state, const Program& program);

        CompiledShader Compile(const DrawState& state, const Program& program)
        {
            CompiledShader cs = CompileFlow(state, program);
            for (const CompiledOp& op : cs.ops)
                if (op.kind == CompiledOp::Kind::VertexFetch)
                    return {};
            return cs;
        }

        CompiledShader CompileFlow(const DrawState& state, const Program& program)
        {
            CompiledShader cs;
            constexpr size_t kMaxOps = 4096;
            int aL = 0;
            auto clause = [&](uint32_t address, uint32_t count, uint32_t sequence, int predicate) {
                CompiledOp head;
                head.clausePredicate = int8_t(predicate);
                cs.ops.push_back(head);
                for (uint32_t k = 0; k < count; k++)
                {
                    const uint32_t at = (address + k) * 3;
                    if (at + 3 > program.count)
                        return;
                    const uint32_t* ins = program.code + at;
                    CompiledOp op;
                    if ((sequence >> (k * 2)) & 1)
                    {
                        if ((ins[0] & 0x1F) == uint32_t(FetchOpcode::VertexFetch))
                        {
                            op.kind = CompiledOp::Kind::VertexFetch; // rejected by Compile
                        }
                        else
                        {
                            op.kind = CompiledOp::Kind::TextureFetch;
                            std::memcpy(&op.textureFetch, ins, 12);
                        }
                    }
                    else
                    {
                        op.kind = CompiledOp::Kind::Alu;
                        op.alu = DecodeAlu(ins, aL);
                    }
                    cs.ops.push_back(op);
                }
            };

            struct Loop { uint32_t count, iteration; int savedAL; int step; };
            Loop loops[4];
            int loopDepth = 0;
            uint32_t calls[4];
            int callDepth = 0;
            uint32_t pc = 0;
            cs.ok = true; // every exit below is a normal end of the program unless it says otherwise
            for (int guard = 0; guard < 65536; guard++)
            {
                if (cs.ops.size() > kMaxOps)
                    return {};
                const uint32_t base = (pc / 2) * 3;
                if (base + 3 > program.count)
                    return cs;
                uint32_t w0, w1;
                if (pc & 1)
                {
                    w0 = (program.code[base + 1] >> 16) | (program.code[base + 2] << 16);
                    w1 = program.code[base + 2] >> 16;
                }
                else
                {
                    w0 = program.code[base];
                    w1 = program.code[base + 1] & 0xFFFF;
                }
                const uint32_t opcode = (w1 >> 12) & 0xF;
                const bool condition = (w1 >> 10) & 1;
                uint32_t next = pc + 1;

                switch (ControlFlowOpcode(opcode))
                {
                case ControlFlowOpcode::Nop:
                case ControlFlowOpcode::Alloc:
                case ControlFlowOpcode::MarkVsFetchDone:
                    break;

                case ControlFlowOpcode::Exec:
                case ControlFlowOpcode::ExecEnd:
                case ControlFlowOpcode::CondExec:
                case ControlFlowOpcode::CondExecEnd:
                case ControlFlowOpcode::CondExecPred:
                case ControlFlowOpcode::CondExecPredEnd:
                case ControlFlowOpcode::CondExecPredClean:
                case ControlFlowOpcode::CondExecPredCleanEnd:
                {
                    const auto op = ControlFlowOpcode(opcode);
                    if (op == ControlFlowOpcode::CondExec || op == ControlFlowOpcode::CondExecEnd)
                    {
                        if (BoolConstant(state, (w1 >> 2) & 0xFF) == condition)
                            clause(w0 & 0xFFF, (w0 >> 12) & 7, (w0 >> 16) & 0xFFF, -1);
                    }
                    else if (op == ControlFlowOpcode::Exec || op == ControlFlowOpcode::ExecEnd)
                    {
                        clause(w0 & 0xFFF, (w0 >> 12) & 7, (w0 >> 16) & 0xFFF, -1);
                    }
                    else
                    {
                        clause(w0 & 0xFFF, (w0 >> 12) & 7, (w0 >> 16) & 0xFFF, condition ? 1 : 0);
                    }
                    if (op == ControlFlowOpcode::ExecEnd || op == ControlFlowOpcode::CondExecEnd ||
                        op == ControlFlowOpcode::CondExecPredEnd || op == ControlFlowOpcode::CondExecPredCleanEnd)
                        return cs;
                    break;
                }

                case ControlFlowOpcode::LoopStart:
                {
                    const uint32_t constant = state.loops[(w0 >> 16) & 31];
                    const uint32_t count = constant & 0xFF;
                    if (loopDepth == 4)
                        return cs;
                    loops[loopDepth++] = { count, 0, aL, int(int8_t(constant >> 16)) };
                    aL = int((constant >> 8) & 0xFF);
                    if (count == 0)
                    {
                        aL = loops[--loopDepth].savedAL;
                        next = w0 & 0x1FFF;
                    }
                    break;
                }

                case ControlFlowOpcode::LoopEnd:
                {
                    if (loopDepth == 0)
                        break;
                    if ((w0 >> 21) & 1)
                        return {}; // breaks on the predicate
                    Loop& loop = loops[loopDepth - 1];
                    if (++loop.iteration < loop.count)
                    {
                        aL += loop.step;
                        next = w0 & 0x1FFF;
                    }
                    else
                    {
                        aL = loop.savedAL;
                        --loopDepth;
                    }
                    break;
                }

                case ControlFlowOpcode::CondCall:
                case ControlFlowOpcode::CondJmp:
                {
                    bool take;
                    if ((w0 >> 13) & 1)
                        take = true;
                    else if ((w0 >> 14) & 1)
                        return {}; // taken on the predicate
                    else
                        take = BoolConstant(state, (w1 >> 2) & 0xFF) == condition;
                    if (!take)
                        break;
                    if (ControlFlowOpcode(opcode) == ControlFlowOpcode::CondCall)
                    {
                        if (callDepth == 4)
                            return cs;
                        calls[callDepth++] = pc + 1;
                    }
                    next = w0 & 0x1FFF;
                    break;
                }

                case ControlFlowOpcode::Return:
                    if (callDepth == 0)
                        return cs;
                    next = calls[--callDepth];
                    break;

                default:
                    return cs;
                }
                pc = next;
            }
            return cs;
        }

        // ---- Batched execution ---------------------------------------------------------
        // Compiled pixel shaders run on up to kLanes pixels at a time with registers stored
        // component-major (each component of a register holds all lanes contiguously), so
        // every instruction is a handful of short loops over the batch. Results match
        // Executor exactly: the opcode math is shared through VectorOp and ScalarOp.

        constexpr int kLanes = 16;
        using LaneFloats = float[kLanes];

        struct LaneBatch
        {
            alignas(64) float r[128][4][kLanes];
            alignas(64) float exports[64][4][kLanes];
            uint64_t exported[kLanes];
            float ps[kLanes];
            int a0[kLanes];
            bool p0[kLanes], killed[kLanes], clauseRun[kLanes];

            F4 Export(uint32_t i, int l) const { return { exports[i][0][l], exports[i][1][l], exports[i][2][l], exports[i][3][l] }; }

            // The lane as an Invocation (TTT2_SOFT_VERIFY).
            void ToInvocation(int l, Invocation& inv) const
            {
                for (int i = 0; i < 128; i++)
                    for (int c = 0; c < 4; c++)
                        inv.r[i][c] = r[i][c][l];
                for (int i = 0; i < 64; i++)
                    for (int c = 0; c < 4; c++)
                        inv.exports[i][c] = exports[i][c][l];
                inv.exported = exported[l];
                inv.ps = ps[l];
                inv.a0 = a0[l];
                inv.aL = 0;
                inv.p0 = p0[l];
                inv.killed = killed[l];
            }
        };

        struct BatchExecutor
        {
            DrawState& state;
            uint32_t constBase;
            LaneBatch& b;
            int n;

            void Read(const AluOp::Source& src, LaneFloats out[4]) const
            {
                if (src.isRegister)
                {
                    const auto& reg = b.r[src.index & 127];
                    for (int c = 0; c < 4; c++)
                    {
                        const float* in = reg[src.comp[c]];
                        float* o = out[c];
                        if (!src.abs && !src.negate)
                            std::memcpy(o, in, sizeof(LaneFloats));
                        else
                            for (int l = 0; l < kLanes; l++)
                            {
                                float v = in[l];
                                if (src.abs)
                                    v = std::fabs(v);
                                o[l] = src.negate ? -v : v;
                            }
                    }
                }
                else if (!src.relativeA0)
                {
                    const F4& raw = state.constants[constBase + (src.index & 255)];
                    for (int c = 0; c < 4; c++)
                    {
                        float v = raw[src.comp[c]];
                        if (src.abs)
                            v = std::fabs(v);
                        v = src.negate ? -v : v;
                        for (int l = 0; l < kLanes; l++)
                            out[c][l] = v;
                    }
                }
                else
                {
                    for (int l = 0; l < n; l++)
                    {
                        const F4& raw = state.constants[constBase + ((src.index + uint32_t(b.a0[l])) & 255)];
                        for (int c = 0; c < 4; c++)
                        {
                            float v = raw[src.comp[c]];
                            if (src.abs)
                                v = std::fabs(v);
                            out[c][l] = src.negate ? -v : v;
                        }
                    }
                }
            }

            // Lanes the instruction runs for; returns whether that is all of them.
            bool Active(const bool* clause, bool predicated, bool condition, bool active[kLanes]) const
            {
                bool all = true;
                for (int l = 0; l < n; l++)
                {
                    active[l] = (!clause || clause[l]) && (!predicated || b.p0[l] == condition);
                    all &= active[l];
                }
                return all;
            }

            void Alu(const AluOp& op, const bool* clause) const
            {
                const AluInstruction& a = op.a;
                bool active[kLanes];
                const bool all = Active(clause, a.isPredicated, a.predicateCondition, active);
                const auto vop = a.vectorOpcode;
                const auto sop = a.scalarOpcode;

                alignas(64) LaneFloats s1[4], s2[4], s3[4];
                LaneFloats sa, sb;
                Read(op.src[0], s1);
                Read(op.src[1], s2);
                if (op.scalarConst)
                {
                    LaneFloats c[4];
                    Read(op.src[2], c);
                    const float* gpr = b.r[op.scalarGpr][op.scalarGprComp];
                    for (int l = 0; l < kLanes; l++)
                    {
                        sa[l] = c[op.scalarConstComp][l];
                        float v = gpr[l];
                        if (a.absConstants)
                            v = std::fabs(v);
                        sb[l] = a.src3Negate ? -v : v;
                    }
                    std::memset(s3, 0, sizeof(s3));
                }
                else
                {
                    Read(op.src[2], s3);
                    for (int l = 0; l < kLanes; l++)
                    {
                        sa[l] = s3[3][l];
                        sb[l] = s3[0][l];
                    }
                }

                // ---- vector op (common opcodes as lane loops, the rest per lane)
                alignas(64) LaneFloats vr[4];
                bool pushPredicate = false;
                bool newP0[kLanes], kill[kLanes] = {};
                std::memcpy(newP0, b.p0, sizeof(newP0));
                auto each = [&](auto&& fn) {
                    for (int c = 0; c < 4; c++)
                        for (int l = 0; l < kLanes; l++)
                            vr[c][l] = fn(s1[c][l], s2[c][l], s3[c][l]);
                };
                switch (vop)
                {
                case AluVectorOpcode::Add: each([](float x, float y, float) { return x + y; }); break;
                case AluVectorOpcode::Mul: each([](float x, float y, float) { return Mul(x, y); }); break;
                case AluVectorOpcode::Max:
                case AluVectorOpcode::MaxA: each([](float x, float y, float) { return x >= y ? x : y; }); break;
                case AluVectorOpcode::Min: each([](float x, float y, float) { return x < y ? x : y; }); break;
                case AluVectorOpcode::Mad: each([](float x, float y, float z) { return Mul(x, y) + z; }); break;
                case AluVectorOpcode::CndEq: each([](float x, float y, float z) { return x == 0.0f ? y : z; }); break;
                case AluVectorOpcode::CndGe: each([](float x, float y, float z) { return x >= 0.0f ? y : z; }); break;
                case AluVectorOpcode::CndGt: each([](float x, float y, float z) { return x > 0.0f ? y : z; }); break;
                case AluVectorOpcode::Dp4:
                case AluVectorOpcode::Dp3:
                    for (int l = 0; l < kLanes; l++)
                    {
                        float d = Mul(s1[0][l], s2[0][l]) + Mul(s1[1][l], s2[1][l]) + Mul(s1[2][l], s2[2][l]);
                        if (vop == AluVectorOpcode::Dp4)
                            d += Mul(s1[3][l], s2[3][l]);
                        vr[0][l] = vr[1][l] = vr[2][l] = vr[3][l] = d;
                    }
                    break;
                default:
                    for (int l = 0; l < n; l++)
                    {
                        const F4 x{ s1[0][l], s1[1][l], s1[2][l], s1[3][l] };
                        const F4 y{ s2[0][l], s2[1][l], s2[2][l], s2[3][l] };
                        const F4 z{ s3[0][l], s3[1][l], s3[2][l], s3[3][l] };
                        F4 v{};
                        VectorOp(vop, x, y, z, v, pushPredicate, newP0[l], kill[l]);
                        for (int c = 0; c < 4; c++)
                            vr[c][l] = v[c];
                    }
                    break;
                }
                for (int l = 0; l < n; l++)
                    if (active[l])
                    {
                        if (kill[l])
                            b.killed[l] = true;
                        if (vop == AluVectorOpcode::MaxA)
                            b.a0[l] = int(std::clamp(std::floor(s1[3][l] + 0.5f), -256.0f, 255.0f));
                    }
                if (a.vectorSaturate)
                    for (int c = 0; c < 4; c++)
                        for (int l = 0; l < kLanes; l++)
                            vr[c][l] = Saturate(vr[c][l]);

                // ---- scalar op
                LaneFloats sr;
                bool scalarWrites = false;
                if (sop != AluScalarOpcode::RetainPrev)
                {
                    for (int l = 0; l < n; l++)
                    {
                        if (!active[l])
                            continue;
                        bool scalarKill = false;
                        sr[l] = ScalarOp(sop, sa[l], sb[l], b.ps[l], b.a0[l], newP0[l], scalarKill, scalarWrites);
                        if (a.scalarSaturate)
                            sr[l] = Saturate(sr[l]);
                        if (scalarKill)
                            b.killed[l] = true;
                    }
                }
                const bool scalarSetsPredicate = uint32_t(sop) >= 27 && uint32_t(sop) <= 34;
                for (int l = 0; l < n; l++)
                {
                    if (!active[l])
                        continue;
                    if (pushPredicate || scalarSetsPredicate)
                        b.p0[l] = newP0[l];
                    if (scalarWrites)
                        b.ps[l] = sr[l];
                }

                // ---- writes (lanes past n are scratch, so full batches skip the mask)
                const uint32_t vmask = a.vectorWriteMask, smask = a.scalarWriteMask;
                auto write = [&](float (*dst)[kLanes], int c, const float* value) {
                    if (all)
                        std::memcpy(dst[c], value, sizeof(LaneFloats));
                    else
                        for (int l = 0; l < n; l++)
                            if (active[l])
                                dst[c][l] = value[l];
                };
                auto fill = [&](float (*dst)[kLanes], int c, float value) {
                    for (int l = 0; l < n; l++)
                        if (active[l])
                            dst[c][l] = value;
                };
                if (a.exportData)
                {
                    const uint32_t dest = a.vectorDest & 63;
                    auto& e = b.exports[dest];
                    const uint32_t ones = vmask & smask;
                    const uint32_t zeros = a.scalarDestRelative ? (0xFu & ~(vmask | smask)) : 0u;
                    for (int c = 0; c < 4; c++)
                    {
                        const uint32_t bit = 1u << c;
                        if (ones & bit) fill(e, c, 1.0f);
                        else if (vmask & bit) write(e, c, vr[c]);
                        else if (smask & bit) write(e, c, b.ps);
                        else if (zeros & bit) fill(e, c, 0.0f);
                    }
                    if (vmask | smask | zeros)
                        for (int l = 0; l < n; l++)
                            if (active[l])
                                b.exported[l] |= 1ull << dest;
                }
                else
                {
                    if (vmask)
                        for (int c = 0; c < 4; c++)
                            if (vmask & (1u << c))
                                write(b.r[op.vectorDest & 127], c, vr[c]);
                    if (smask)
                        for (int c = 0; c < 4; c++)
                            if (smask & (1u << c))
                                write(b.r[op.scalarDest & 127], c, b.ps);
                }
            }

            void TextureFetch(const TextureFetchInstruction& f, const bool* clause) const
            {
                bool active[kLanes];
                Active(clause, f.isPredicated, f.predCondition, active);
                auto store = [&](int l, const F4& value) {
                    auto& dst = b.r[f.dstRegister & 63];
                    for (int c = 0; c < 4; c++)
                    {
                        const uint32_t sel = (f.dstSwizzle >> (c * 3)) & 7;
                        if (sel < 4) dst[c][l] = value[sel];
                        else if (sel == 4) dst[c][l] = 0.0f;
                        else if (sel == 5) dst[c][l] = 1.0f;
                    }
                };
                if (f.opcode != FetchOpcode::TextureFetch)
                {
                    if (f.opcode == FetchOpcode::GetTextureWeights || f.opcode == FetchOpcode::GetTextureComputedLod ||
                        f.opcode == FetchOpcode::GetTextureGradients || f.opcode == FetchOpcode::GetTextureBorderColorFrac)
                        for (int l = 0; l < n; l++)
                            if (active[l])
                                store(l, F4{ 0, 0, 0, 0 });
                    return;
                }
                const Texture& tex = state.TextureSlot(f.constIndex);
                const auto& src = b.r[f.srcRegister & 63];
                for (int l = 0; l < n; l++)
                    if (active[l])
                        store(l, Executor::FetchTexel(f, tex, src[f.srcSwizzle & 3][l], src[(f.srcSwizzle >> 2) & 3][l]));
            }

            void Run(const CompiledShader& cs) const
            {
                const bool* clause = nullptr; // set while a predicated clause runs
                for (const CompiledOp& op : cs.ops)
                {
                    switch (op.kind)
                    {
                    case CompiledOp::Kind::Clause:
                        clause = nullptr;
                        if (op.clausePredicate >= 0)
                        {
                            for (int l = 0; l < n; l++)
                                b.clauseRun[l] = b.p0[l] == (op.clausePredicate != 0);
                            clause = b.clauseRun;
                        }
                        break;
                    case CompiledOp::Kind::Alu:
                        Alu(op.alu, clause);
                        break;
                    case CompiledOp::Kind::TextureFetch:
                        TextureFetch(op.textureFetch, clause);
                        break;
                    case CompiledOp::Kind::VertexFetch:
                        break; // Compile rejects these
                    }
                }
            }
        };

        // ---- Vertices and primitives -------------------------------------------------

        struct Vertex
        {
            F4 position;
            F4 interpolators[16];
        };

        struct ScreenVertex
        {
            float x, y, z, invW;
            const Vertex* v;
        };


        float BlendFactor(uint32_t factor, const F4& src, const F4& dst, const F4& constant, int c)
        {
            switch (factor)
            {
            case 0: return 0.0f;
            case 1: return 1.0f;
            case 4: return src[c];
            case 5: return 1.0f - src[c];
            case 6: return src[3];
            case 7: return 1.0f - src[3];
            case 8: return dst[c];
            case 9: return 1.0f - dst[c];
            case 10: return dst[3];
            case 11: return 1.0f - dst[3];
            case 12: return constant[c];
            case 13: return 1.0f - constant[c];
            case 14: return constant[3];
            case 15: return 1.0f - constant[3];
            case 16: return c == 3 ? 1.0f : std::min(src[3], 1.0f - dst[3]);
            default: return 1.0f;
            }
        }

        F4 Blend(const F4& src, const F4& dst, uint32_t control, const F4& constant)
        {
            F4 out;
            for (int c = 0; c < 4; c++)
            {
                const uint32_t shift = c == 3 ? 16 : 0;
                const uint32_t srcFactor = (control >> shift) & 0x1F;
                const uint32_t func = (control >> (shift + 5)) & 7;
                const uint32_t dstFactor = (control >> (shift + 8)) & 0x1F;
                const float s = src[c] * BlendFactor(srcFactor, src, dst, constant, c);
                const float d = dst[c] * BlendFactor(dstFactor, src, dst, constant, c);
                switch (func)
                {
                case 0: out[c] = s + d; break;
                case 1: out[c] = s - d; break;
                case 2: out[c] = std::min(src[c], dst[c]); break;
                case 3: out[c] = std::max(src[c], dst[c]); break;
                case 4: out[c] = d - s; break;
                default: out[c] = s + d; break;
                }
            }
            return out;
        }

        bool BlendIsReplace(uint32_t control)
        {
            // src ONE, dst ZERO, ADD for both colour and alpha.
            return (control & 0x1FFF) == 0x0001 && ((control >> 16) & 0x1FFF) == 0x0001;
        }

        struct Renderer
        {
            DrawState& state;
            RasterState& rs;
            Program ps;
            const CompiledShader* compiled = nullptr; // ps, compiled (null: interpret per pixel)
            uint64_t pixels = 0;

            // Triangles of the current draw, set up and binned, rasterised by Flush().
            struct Setup
            {
                ScreenVertex v[3];
                struct Edge { float ax, ay, dx, dy; bool owns; } edges[3];
                float invArea;
                bool front;
                int x0, x1, y0, y1;
            };
            std::vector<Setup> setups;
            std::deque<Vertex> extraVertices; // clipped / completed vertices (stable addresses)

            ScreenVertex ToScreen(const Vertex& v) const
            {
                const F4& p = v.position;
                float x = p[0], y = p[1], z = p[2];
                const float w = p[3];
                // VTX_W0_FMT (bit 10) only says the rasteriser gets 1/W; the shader always
                // exports W, so the divides below use its reciprocal either way.
                const float invW = w != 0.0f ? 1.0f / w : 1.0f;
                if (!(rs.vte & (1u << 8)))
                {
                    x *= invW;
                    y *= invW;
                }
                if (!(rs.vte & (1u << 9)))
                    z *= invW;
                x = x * rs.scale[0] + rs.offset[0];
                y = y * rs.scale[1] + rs.offset[1];
                z = z * rs.scale[2] + rs.offset[2];
                if (rs.windowOffset)
                {
                    x += rs.windowX;
                    y += rs.windowY;
                }
                return { x, y, z, (rs.vte & (1u << 8)) ? 1.0f : invW, &v };
            }

            static uint8_t StencilOp(uint32_t op, uint8_t v, uint8_t ref)
            {
                switch (op)
                {
                case 1: return 0;
                case 2: return ref;
                case 3: return v == 255 ? 255 : uint8_t(v + 1);
                case 4: return v == 0 ? 0 : uint8_t(v - 1);
                case 5: return uint8_t(~v);
                case 6: return uint8_t(v + 1);
                case 7: return uint8_t(v - 1);
                default: return v;
                }
            }

            void UpdateStencil(int px, int py, const RasterState::StencilFace& sf, uint32_t op)
            {
                if (!op || !sf.writeMask)
                    return;
                rs.depth.ForEachSample(uint32_t(px), uint32_t(py), [&](size_t i) {
                    const uint8_t v = g_edramStencil[i];
                    g_edramStencil[i] = uint8_t((v & ~sf.writeMask) | (StencilOp(op, v, uint8_t(sf.ref)) & sf.writeMask));
                });
            }

            // Early stencil and depth tests (sample 0; depth exported by the shader is still
            // tested against the interpolated value), applying the fail/zfail stencil ops.
            bool EarlyTests(int px, int py, float z, bool front)
            {
                if (!rs.stencil && !rs.depthTest)
                    return true;
                const size_t depthIndex = rs.depth.Index(uint32_t(px) * rs.depth.scaleX, uint32_t(py) * rs.depth.scaleY);
                const RasterState::StencilFace& sf = rs.stencilFace[front ? 0 : 1];
                if (rs.stencil && !Compare(sf.func, float(sf.ref & sf.mask), float(g_edramStencil[depthIndex] & sf.mask)))
                {
                    if (g_softLogDraw)
                        ++g_drawDebug.stencilFail;
                    UpdateStencil(px, py, sf, sf.fail);
                    return false;
                }
                if (rs.depthTest && !Compare(rs.depthFunc, z, g_edramDepth[depthIndex]))
                {
                    if (g_softLogDraw)
                        ++g_drawDebug.depthFail;
                    if (rs.stencil)
                        UpdateStencil(px, py, sf, sf.zfail);
                    return false;
                }
                return true;
            }

            // Fresh invocation state and the interpolated pixel shader inputs.
            void SetupInputs(Invocation& inv, int px, int py, const ScreenVertex* const t[3], const float bary[3]) const
            {
                inv.exported = 0;
                inv.ps = 0.0f;
                inv.p0 = false;
                inv.killed = false;
                inv.a0 = 0;
                inv.aL = 0;
                // Perspective-correct interpolation.
                const float w0 = bary[0] * t[0]->invW, w1 = bary[1] * t[1]->invW, w2 = bary[2] * t[2]->invW;
                const float sum = w0 + w1 + w2;
                const float k = sum != 0.0f ? 1.0f / sum : 0.0f;
                for (uint32_t i = 0; i < 16; i++)
                {
                    if (!(rs.interpolatorMask & (1u << i)))
                    {
                        inv.r[i] = { 0, 0, 0, 0 };
                        continue;
                    }
                    const F4& a = t[0]->v->interpolators[i];
                    const F4& b = t[1]->v->interpolators[i];
                    const F4& c = t[2]->v->interpolators[i];
                    for (int j = 0; j < 4; j++)
                        inv.r[i][j] = (a[j] * w0 + b[j] * w1 + c[j] * w2) * k;
                }
                if (rs.paramGen && rs.paramGenRegister < 128)
                    inv.r[rs.paramGenRegister] = { float(px), float(py), 0.0f, 0.0f };
                for (int i = 0; i < 4; i++)
                    inv.exports[i] = { 0, 0, 0, 0 };
            }

            // The same for one lane of a batch.
            // Fresh state for a batch: what SetupInputs below does not write for each lane.
            void BeginBatch(LaneBatch& b) const
            {
                std::memset(b.exported, 0, sizeof(b.exported));
                std::memset(b.ps, 0, sizeof(b.ps));
                std::memset(b.a0, 0, sizeof(b.a0));
                std::memset(b.p0, 0, sizeof(b.p0));
                std::memset(b.killed, 0, sizeof(b.killed));
                for (uint32_t i = 0; i < 16; i++)
                    if (!(rs.interpolatorMask & (1u << i)))
                        std::memset(b.r[i], 0, sizeof(b.r[i]));
                std::memset(b.exports, 0, sizeof(b.exports[0]) * 4);
            }

            // The same as SetupInputs(Invocation&) for one lane of a batch (after BeginBatch).
            void SetupInputs(LaneBatch& b, int l, int px, int py, const ScreenVertex* const t[3], const float bary[3]) const
            {
                const float w0 = bary[0] * t[0]->invW, w1 = bary[1] * t[1]->invW, w2 = bary[2] * t[2]->invW;
                const float sum = w0 + w1 + w2;
                const float k = sum != 0.0f ? 1.0f / sum : 0.0f;
                for (uint32_t mask = rs.interpolatorMask & 0xFFFF; mask; mask &= mask - 1)
                {
                    const uint32_t i = uint32_t(std::countr_zero(mask));
                    const F4& a = t[0]->v->interpolators[i];
                    const F4& bb = t[1]->v->interpolators[i];
                    const F4& c = t[2]->v->interpolators[i];
                    for (int j = 0; j < 4; j++)
                        b.r[i][j][l] = (a[j] * w0 + bb[j] * w1 + c[j] * w2) * k;
                }
                if (rs.paramGen && rs.paramGenRegister < 128)
                {
                    b.r[rs.paramGenRegister][0][l] = float(px);
                    b.r[rs.paramGenRegister][1][l] = float(py);
                    b.r[rs.paramGenRegister][2][l] = 0.0f;
                    b.r[rs.paramGenRegister][3][l] = 0.0f;
                }
            }

            bool WriteOutputs(int px, int py, const Invocation& inv, float z, bool front)
            {
                return WriteOutputs(px, py, inv.exported, inv.killed, inv.exports, inv.exports[61][0], z, front);
            }

            bool WriteOutputs(int px, int py, const LaneBatch& b, int l, float z, bool front)
            {
                const F4 color[4] = { b.Export(0, l), b.Export(1, l), b.Export(2, l), b.Export(3, l) };
                return WriteOutputs(px, py, b.exported[l], b.killed[l], color, b.exports[61][0][l], z, front);
            }

            // After the pixel shader: kill and alpha test, then stencil, depth and colour
            // writes. Returns whether the pixel was written.
            bool WriteOutputs(int px, int py, uint64_t exported, bool killed, const F4 color[4], float depthExport, float z, bool front)
            {
                if (rs.mode == 4 && ps.count)
                {
                    if (killed)
                        return false;
                    if (rs.alphaTest && !Compare(rs.alphaFunc, color[0][3], rs.alphaRef))
                        return false;
                }

                if (exported & (1ull << 61))
                    z = depthExport;
                if (rs.stencil)
                    UpdateStencil(px, py, rs.stencilFace[front ? 0 : 1], rs.stencilFace[front ? 0 : 1].zpass);
                if (rs.depthWrite)
                {
                    const float dz = std::clamp(z, 0.0f, 1.0f);
                    rs.depth.ForEachSample(uint32_t(px), uint32_t(py), [&](size_t i) { g_edramDepth[i] = dz; });
                }

                if (rs.mode != 4)
                    return true;
                for (int rt = 0; rt < 4; rt++)
                {
                    const uint32_t mask = (rs.colorMask >> (rt * 4)) & 0xF;
                    if (!mask || !(exported & (1ull << rt)))
                        continue;
                    const Surface& surface = rs.color[rt];
                    const F4 src = color[rt];
                    const bool unorm = IsUnormColor(surface.format);
                    const bool replace = BlendIsReplace(rs.blend[rt]);
                    surface.ForEachSample(uint32_t(px), uint32_t(py), [&](size_t i) {
                        F4& dst = g_edramColor[i];
                        F4 out = replace ? src : Blend(src, dst, rs.blend[rt], rs.blendConstant);
                        for (int c = 0; c < 4; c++)
                            if (mask & (1u << c))
                                dst[c] = unorm ? Saturate(out[c]) : out[c];
                    });
                }
                return true;
            }

            // One pixel through the interpreter (programs that could not be compiled).
            bool ShadePixel(int px, int py, const ScreenVertex* const t[3], const float bary[3], float z, bool front)
            {
                if (!EarlyTests(px, py, z, front))
                    return false;
                Invocation inv;
                SetupInputs(inv, px, py, t, bary);
                if (rs.mode == 4 && ps.count)
                {
                    Interpreter it{ state, ps, inv, 256 };
                    it.Run();
                }
                return WriteOutputs(px, py, inv, z, front);
            }

            void Triangle(const ScreenVertex& a, const ScreenVertex& b, const ScreenVertex& c, bool cull = false)
            {
                if (g_softLogDraw)
                    ++g_drawDebug.triangles;
                Setup su;
                su.v[0] = a; su.v[1] = b; su.v[2] = c;
                float area = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
                if (area == 0.0f || !std::isfinite(area))
                    return;
                // area > 0: clockwise on screen. Rectangles always count as front faces.
                su.front = !cull || (area > 0.0f) == rs.frontCW;
                if (cull && (su.front ? rs.cullFront : rs.cullBack))
                    return;
                if (g_sink)
                {
                    g_sink->Triangle({ a.x, a.y, a.z, a.invW, a.v->interpolators }, { b.x, b.y, b.z, b.invW, b.v->interpolators },
                        { c.x, c.y, c.z, c.invW, c.v->interpolators }, su.front);
                    return;
                }
                if (area < 0.0f)
                {
                    std::swap(su.v[1], su.v[2]);
                    area = -area;
                }

                const float minX = std::min({ a.x, b.x, c.x }), maxX = std::max({ a.x, b.x, c.x });
                const float minY = std::min({ a.y, b.y, c.y }), maxY = std::max({ a.y, b.y, c.y });
                su.x0 = std::max(rs.sx0, int(std::floor(minX - rs.pixelCenter)));
                su.x1 = std::min(rs.sx1 - 1, int(std::ceil(maxX - rs.pixelCenter)));
                su.y0 = std::max(rs.sy0, int(std::floor(minY - rs.pixelCenter)));
                su.y1 = std::min(rs.sy1 - 1, int(std::ceil(maxY - rs.pixelCenter)));
                if (su.x0 > su.x1 || su.y0 > su.y1)
                    return;

                // Edge e is opposite vertex e. Top-left rule: an edge owns its pixels when it
                // runs downwards, or exactly horizontally to the left.
                for (int e = 0; e < 3; e++)
                {
                    const ScreenVertex& p = su.v[(e + 1) % 3];
                    const ScreenVertex& q = su.v[(e + 2) % 3];
                    const float dx = q.x - p.x, dy = q.y - p.y;
                    su.edges[e] = { p.x, p.y, dx, dy, dy > 0.0f || (dy == 0.0f && dx < 0.0f) };
                }
                su.invArea = 1.0f / area;
                if (g_softLogDraw)
                    ++g_drawDebug.setups;
                setups.push_back(su);
            }

            uint64_t RasteriseBand(const Setup& su, int y0, int y1)
            {
                const ScreenVertex* const t[3] = { &su.v[0], &su.v[1], &su.v[2] };
                uint64_t written = 0;
                // Compiled shaders run on batches of up to kLanes pixels from one row of one
                // triangle; those pixels are distinct, so testing them all before writing any
                // gives the same result as shading them one by one.
                thread_local std::unique_ptr<LaneBatch> lanesStorage;
                if (compiled && !lanesStorage)
                    lanesStorage.reset(new LaneBatch{}); // zeroed: idle lanes hold harmless values
                LaneBatch* lanes = lanesStorage.get();
                struct Pending { int px; float z; } pending[kLanes];
                int batch = 0;
                auto flush = [&](int py) {
                    // TTT2_SOFT_VERIFY=1: also interpret every pixel and report the first mismatch.
                    static const bool verify = [] { const char* v = std::getenv("TTT2_SOFT_VERIFY"); return v && *v == '1'; }();
                    thread_local std::unique_ptr<Invocation[]> reference;
                    if (verify)
                    {
                        if (!reference)
                            reference.reset(new Invocation[kLanes]);
                        for (int i = 0; i < batch; i++)
                        {
                            lanes->ToInvocation(i, reference[i]);
                            Interpreter{ state, ps, reference[i], 256 }.Run();
                        }
                    }
                    BatchExecutor{ state, 256, *lanes, batch }.Run(*compiled);
                    if (verify)
                        for (int i = 0; i < batch; i++)
                        {
                            const Invocation& r = reference[i];
                            bool same = lanes->exported[i] == r.exported && lanes->killed[i] == r.killed;
                            for (int e = 0; e < 4; e++)
                                {
                                const F4 mine = lanes->Export(e, i);
                                same &= std::memcmp(&r.exports[e], &mine, sizeof(F4)) == 0 || !(r.exported & (1ull << e));
                            }
                            static std::atomic<int> reported{ 0 };
                            if (!same && reported++ < 4)
                            {
                                const F4 o0 = lanes->Export(0, i);
                                std::fprintf(stderr, "[gpu] VERIFY mismatch px %d,%d: exported %llX/%llX killed %d/%d o0 (%g %g %g %g)/(%g %g %g %g)\n",
                                    pending[i].px, py, (unsigned long long)lanes->exported[i], (unsigned long long)r.exported, lanes->killed[i], r.killed,
                                    o0[0], o0[1], o0[2], o0[3], r.exports[0][0], r.exports[0][1], r.exports[0][2], r.exports[0][3]);
                                std::fprintf(stderr, "[gpu]   ps:");
                                for (uint32_t k = 0; k < ps.count; k++)
                                    std::fprintf(stderr, " %08X", ps.code[k]);
                                std::fprintf(stderr, "\n");
                            }
                        }
                    for (int i = 0; i < batch; i++)
                        written += WriteOutputs(pending[i].px, py, *lanes, i, pending[i].z, su.front);
                    batch = 0;
                };
                for (int py = y0; py <= y1; py++)
                {
                    const float sy = float(py) + rs.pixelCenter;
                    // Narrow the row to where every edge function can be non-negative (with
                    // a margin; each pixel is still tested exactly below).
                    int xs = su.x0, xe = su.x1;
                    for (int e = 0; e < 3; e++)
                    {
                        const auto& E = su.edges[e];
                        if (E.dy == 0.0f)
                            continue;
                        const float boundary = E.ax + E.dx * (sy - E.ay) / E.dy - rs.pixelCenter;
                        if (!std::isfinite(boundary) || std::fabs(boundary) > 1.0e6f)
                            continue;
                        if (E.dy > 0.0f)
                            xe = std::min(xe, int(std::floor(boundary)) + 2);
                        else
                            xs = std::max(xs, int(std::ceil(boundary)) - 2);
                    }
                    for (int px = xs; px <= xe; px++)
                    {
                        const float sx = float(px) + rs.pixelCenter;
                        float bary[3];
                        bool inside = true;
                        for (int e = 0; e < 3 && inside; e++)
                        {
                            const auto& E = su.edges[e];
                            const float w = E.dx * (sy - E.ay) - E.dy * (sx - E.ax);
                            inside = w > 0.0f || (w == 0.0f && E.owns);
                            bary[e] = w * su.invArea;
                        }
                        if (!inside)
                            continue;
                        const float z = bary[0] * t[0]->z + bary[1] * t[1]->z + bary[2] * t[2]->z;
                        if (g_softLogDraw)
                            ++g_drawDebug.inside;
                        if (rs.mode != 4 || !ps.count)
                        {
                            // No pixel shader runs: nothing reads the inputs.
                            static const F4 kNoColor[4] = {};
                            if (EarlyTests(px, py, z, su.front))
                                written += WriteOutputs(px, py, 0, false, kNoColor, 0.0f, z, su.front);
                            continue;
                        }
                        if (!compiled)
                        {
                            written += ShadePixel(px, py, t, bary, z, su.front);
                            continue;
                        }
                        if (!EarlyTests(px, py, z, su.front))
                            continue;
                        if (batch == 0)
                            BeginBatch(*lanes);
                        SetupInputs(*lanes, batch, px, py, t, bary);
                        pending[batch++] = { px, z };
                        if (batch == kLanes)
                            flush(py);
                    }
                    if (batch)
                        flush(py);
                }
                return written;
            }

            // Rasterises everything set up so far. Rows are split into bands; each band
            // walks all triangles in submission order, so overlapping triangles still
            // blend in draw order.
            void Flush()
            {
                if (setups.empty())
                    return;
                int minY = INT32_MAX, maxY = INT32_MIN;
                uint64_t area = 0;
                for (const Setup& su : setups)
                {
                    minY = std::min(minY, su.y0);
                    maxY = std::max(maxY, su.y1);
                    area += uint64_t(su.x1 - su.x0 + 1) * uint64_t(su.y1 - su.y0 + 1);
                }
                constexpr int kBandRows = 8;
                const int bands = (maxY - minY) / kBandRows + 1;
                std::atomic<uint64_t> written{ 0 };
                auto band = [&](int index) {
                    const int b0 = minY + index * kBandRows, b1 = b0 + kBandRows - 1;
                    uint64_t n = 0;
                    for (const Setup& su : setups)
                    {
                        const int y0 = std::max(su.y0, b0), y1 = std::min(su.y1, b1);
                        if (y0 <= y1)
                            n += RasteriseBand(su, y0, y1);
                    }
                    written += n;
                };
                if (area < 4096)
                {
                    for (int i = 0; i < bands; i++)
                        band(i);
                }
                else
                {
                    Pool().Run(bands, band);
                }
                pixels += written.load();
                setups.clear();
            }

            // Clips a clip-space triangle against w > 0 and z >= 0, then rasterises it.
            void ClipAndDraw(const Vertex& a, const Vertex& b, const Vertex& c)
            {
                const bool needsClip = rs.clip && !(rs.vte & (1u << 8)) &&
                    (a.position[3] <= 0 || b.position[3] <= 0 || c.position[3] <= 0 ||
                     a.position[2] < 0 || b.position[2] < 0 || c.position[2] < 0);
                if (!needsClip)
                {
                    Triangle(ToScreen(a), ToScreen(b), ToScreen(c), true);
                    return;
                }

                std::vector<Vertex> poly = { a, b, c };
                auto clipPlane = [&](auto distance) {
                    std::vector<Vertex> out;
                    for (size_t i = 0; i < poly.size(); i++)
                    {
                        const Vertex& p = poly[i];
                        const Vertex& q = poly[(i + 1) % poly.size()];
                        const float dp = distance(p), dq = distance(q);
                        if (dp >= 0)
                            out.push_back(p);
                        if ((dp >= 0) != (dq >= 0))
                        {
                            const float t = dp / (dp - dq);
                            Vertex m;
                            for (int j = 0; j < 4; j++)
                                m.position[j] = p.position[j] + (q.position[j] - p.position[j]) * t;
                            for (int k = 0; k < 16; k++)
                                for (int j = 0; j < 4; j++)
                                    m.interpolators[k][j] = p.interpolators[k][j] + (q.interpolators[k][j] - p.interpolators[k][j]) * t;
                            out.push_back(m);
                        }
                    }
                    poly.swap(out);
                };
                clipPlane([](const Vertex& v) { return v.position[3] - 1e-6f; });
                clipPlane([](const Vertex& v) { return v.position[2]; });
                if (poly.size() < 3)
                    return;
                std::vector<ScreenVertex> sv;
                for (const Vertex& v : poly)
                    sv.push_back(ToScreen(extraVertices.emplace_back(v)));
                for (size_t i = 1; i + 1 < sv.size(); i++)
                    Triangle(sv[0], sv[i], sv[i + 1], true);
            }
        };

        // Rect lists give three corners; the fourth completes the parallelogram across the
        // longest edge (the diagonal), in screen space.
        Vertex CompleteRect(const Vertex& v0, const Vertex& v1, const Vertex& v2, const ScreenVertex s[3], int order[3])
        {
            auto len = [&](int i, int j) { const float dx = s[i].x - s[j].x, dy = s[i].y - s[j].y; return dx * dx + dy * dy; };
            const float d01 = len(0, 1), d12 = len(1, 2), d20 = len(2, 0);
            int corner;
            if (d01 >= d12 && d01 >= d20) corner = 2;
            else if (d12 >= d20) corner = 0;
            else corner = 1;
            const Vertex* v[3] = { &v0, &v1, &v2 };
            const int i = (corner + 1) % 3, j = (corner + 2) % 3;
            order[0] = corner; order[1] = i; order[2] = j;
            Vertex out;
            for (int k = 0; k < 4; k++)
                out.position[k] = v[i]->position[k] + v[j]->position[k] - v[corner]->position[k];
            for (int n = 0; n < 16; n++)
                for (int k = 0; k < 4; k++)
                    out.interpolators[n][k] = v[i]->interpolators[n][k] + v[j]->interpolators[n][k] - v[corner]->interpolators[n][k];
            return out;
        }

        // ---- Resolve -----------------------------------------------------------------

        uint32_t ToUnorm(float v, uint32_t bits)
        {
            // lround of a value in [0, max], without the library call.
            const float x = Saturate(v) * float((1u << bits) - 1);
            const uint32_t r = uint32_t(x);
            return x - float(r) >= 0.5f ? r + 1 : r;
        }

        // Packs one pixel in a texture format; returns bytes per pixel (0 = unsupported).
        uint32_t PackPixel(const F4& c, uint32_t format, uint32_t out[4])
        {
            switch (format)
            {
            case 2: // k_8
                out[0] = ToUnorm(c[0], 8);
                return 1;
            case 4: // k_5_6_5
                out[0] = (ToUnorm(c[0], 5) << 11) | (ToUnorm(c[1], 6) << 5) | ToUnorm(c[2], 5);
                return 2;
            case 6: // k_8_8_8_8
                out[0] = ToUnorm(c[0], 8) | (ToUnorm(c[1], 8) << 8) | (ToUnorm(c[2], 8) << 16) | (ToUnorm(c[3], 8) << 24);
                return 4;
            case 7: // k_2_10_10_10
                out[0] = ToUnorm(c[0], 10) | (ToUnorm(c[1], 10) << 10) | (ToUnorm(c[2], 10) << 20) | (ToUnorm(c[3], 2) << 30);
                return 4;
            case 10: // k_8_8
                out[0] = ToUnorm(c[0], 8) | (ToUnorm(c[1], 8) << 8);
                return 2;
            case 22: // k_24_8: depth in the high 24 bits, stencil (c[1], 0-255) in the low 8
                out[0] = (ToUnorm(c[0], 24) << 8) | (uint32_t(c[1]) & 0xFF);
                return 4;
            case 23: // k_24_8_FLOAT
                out[0] = (Float32To20e4(c[0]) << 8) | (uint32_t(c[1]) & 0xFF);
                return 4;
            case 25: // k_16_16
                out[0] = ToUnorm(c[0], 16) | (ToUnorm(c[1], 16) << 16);
                return 4;
            case 26: // k_16_16_16_16
                out[0] = ToUnorm(c[0], 16) | (ToUnorm(c[1], 16) << 16);
                out[1] = ToUnorm(c[2], 16) | (ToUnorm(c[3], 16) << 16);
                return 8;
            case 31: // k_16_16_FLOAT
                out[0] = FloatToHalf(c[0]) | (uint32_t(FloatToHalf(c[1])) << 16);
                return 4;
            case 32: // k_16_16_16_16_FLOAT
                out[0] = FloatToHalf(c[0]) | (uint32_t(FloatToHalf(c[1])) << 16);
                out[1] = FloatToHalf(c[2]) | (uint32_t(FloatToHalf(c[3])) << 16);
                return 8;
            case 36: case 37: case 38: // k_32_FLOAT, k_32_32_FLOAT, k_32_32_32_32_FLOAT
            {
                const uint32_t n = format - 35;
                for (uint32_t i = 0; i < n; i++)
                    std::memcpy(&out[i], &c[i], 4);
                return n * 4;
            }
            default:
                return 0;
            }
        }

        void WritePixel(uint32_t address, const uint32_t* words, uint32_t bytes, uint32_t endian)
        {
            if (bytes >= 4)
            {
                for (uint32_t i = 0; i < bytes / 4; i++)
                    *guest::ToHost<uint32_t>(address + i * 4) = Swap32(words[i], endian);
                return;
            }
            // Sub-dword pixels: find where the swap moves them inside their dword.
            if (bytes == 2)
            {
                uint16_t v = uint16_t(words[0]);
                uint32_t at = address;
                if (endian == 1 || endian == 2)
                    v = uint16_t((v >> 8) | (v << 8));
                if (endian == 2 || endian == 3)
                    at ^= 2;
                *guest::ToHost<uint16_t>(at) = v;
                return;
            }
            uint32_t at = address;
            if (endian == 1) at ^= 1;
            else if (endian == 2) at ^= 3;
            else if (endian == 3) at ^= 2;
            *guest::ToHost<uint8_t>(at) = uint8_t(words[0]);
        }

        void Resolve(const RasterState& rs, const ScreenVertex* corners, size_t count)
        {
            ScopedTimer timer{ g_stats.resolve };
            ++g_resolves;
            float minX = FLT_MAX, minY = FLT_MAX, maxX = -FLT_MAX, maxY = -FLT_MAX;
            for (size_t i = 0; i < count; i++)
            {
                minX = std::min(minX, corners[i].x); maxX = std::max(maxX, corners[i].x);
                minY = std::min(minY, corners[i].y); maxY = std::max(maxY, corners[i].y);
            }
            // Corners are in EDRAM space (window offset applied); the destination uses the
            // coordinates before the window offset, which is how tiled rendering places
            // each tile in the final surface.
            const int ex0 = std::max(0, int(std::lround(minX))), ey0 = std::max(0, int(std::lround(minY)));
            const int ex1 = int(std::lround(maxX)), ey1 = int(std::lround(maxY));
            if (ex1 <= ex0 || ey1 <= ey0 || ex1 > 8192 || ey1 > 8192)
                return;
            const int offsetX = rs.windowOffset ? int(rs.windowX) : 0;
            const int offsetY = rs.windowOffset ? int(rs.windowY) : 0;

            const uint32_t copyControl = Reg(0x2318);
            const uint32_t source = copyControl & 7;
            const uint32_t sampleSelect = (copyControl >> 4) & 7;
            const bool colorClear = (copyControl >> 8) & 1;
            const bool depthClear = (copyControl >> 9) & 1;
            const uint32_t command = (copyControl >> 20) & 3;
            const uint32_t destBase = PhysicalToGuest(Reg(0x2319));
            const uint32_t destPitchReg = Reg(0x231A);
            const uint32_t destPitch = destPitchReg & 0x3FFF;
            const uint32_t destHeight = (destPitchReg >> 16) & 0x3FFF;
            const uint32_t destInfo = Reg(0x231B);
            const uint32_t endian = destInfo & 7;
            const uint32_t format = (destInfo >> 7) & 0x3F;
            const int expBias = int32_t(destInfo << 10) >> 26; // bits 16-21, signed
            const bool swapRB = (destInfo >> 24) & 1;

            const int rows = ey1 - ey0;
            auto parallelRows = [&](auto&& row) {
                constexpr int kBand = 16;
                Pool().Run((rows + kBand - 1) / kBand, [&](int band) {
                    const int y0 = ey0 + band * kBand, y1 = std::min(ey1, y0 + kBand);
                    for (int y = y0; y < y1; y++)
                        row(y);
                });
            };

            uint32_t probe[4];
            const uint32_t bytes = PackPixel(F4{ 0, 0, 0, 0 }, format, probe);
            if (command != 3 && destPitch && Reg(0x2319) && !bytes)
            {
                static bool warned[64];
                if (!warned[format & 63])
                {
                    warned[format & 63] = true;
                    std::fprintf(stderr, "[gpu] soft: unsupported resolve format %u\n", format);
                }
            }
            else if (command != 3 && destPitch && Reg(0x2319))
            {
                // A hardware backend owns the render targets: read the rectangle back first.
                std::vector<F4> sinkColor;
                std::vector<float> sinkDepth;
                std::vector<uint8_t> sinkStencil;
                if (g_sink)
                {
                    if (source < 4)
                        g_sink->ReadColor(rs, source, ex0, ey0, ex1, ey1, sinkColor);
                    else
                        g_sink->ReadDepth(rs, ex0, ey0, ex1, ey1, sinkDepth, sinkStencil);
                }
                const size_t sinkPitch = size_t(ex1 - ex0);
                const uint32_t pitchAligned = (destPitch + 31) & ~31u;
                const float scale = std::ldexp(1.0f, expBias);
                const uint32_t log2 = bytes == 1 ? 0 : bytes == 2 ? 1 : bytes == 4 ? 2 : bytes == 8 ? 3 : 4;
                parallelRows([&](int y) {
                    const int dy = y - offsetY;
                    if (dy < 0 || (destHeight && uint32_t(dy) >= destHeight))
                        return;
                    for (int x = ex0; x < ex1; x++)
                    {
                        const int dx = x - offsetX;
                        if (dx < 0 || uint32_t(dx) >= destPitch)
                            continue;
                        F4 c;
                        if (g_sink)
                        {
                            const size_t at = size_t(y - ey0) * sinkPitch + size_t(x - ex0);
                            if (source < 4)
                            {
                                c = at < sinkColor.size() ? sinkColor[at] : F4{ 0, 0, 0, 0 };
                                if (rs.color[source].format == 1)
                                    for (int k = 0; k < 3; k++)
                                        c[k] = LinearToPwlGamma(c[k]);
                            }
                            else
                            {
                                const float d = at < sinkDepth.size() ? sinkDepth[at] : 0.0f;
                                c = { d, float(at < sinkStencil.size() ? sinkStencil[at] : 0), d, d };
                            }
                        }
                        else if (source < 4)
                        {
                            const Surface& s = rs.color[source];
                            const uint32_t samples = s.scaleX * s.scaleY;
                            if (sampleSelect >= 4 && samples > 1)
                            {
                                // Averaged resolve (01, 23 and 0123 all reduce to the full
                                // pixel here: samples are written with the same coverage).
                                c = { 0, 0, 0, 0 };
                                s.ForEachSample(uint32_t(x), uint32_t(y), [&](size_t i) {
                                    for (int k = 0; k < 4; k++)
                                        c[k] += g_edramColor[i][k];
                                });
                                for (float& v : c)
                                    v /= float(samples);
                            }
                            else
                            {
                                // Sample n: 2x stacks samples vertically, 4x is 2x2.
                                const uint32_t n = sampleSelect < samples ? sampleSelect : 0;
                                const uint32_t sx = s.scaleX == 2 ? (n & 1) : 0;
                                const uint32_t sy = s.scaleX == 2 ? (n >> 1) : n;
                                c = g_edramColor[s.Index(uint32_t(x) * s.scaleX + sx, uint32_t(y) * s.scaleY + sy)];
                            }
                            // EDRAM keeps linear values; k_8_8_8_8_GAMMA targets hold them
                            // gamma-encoded, which is what the copy writes out.
                            if (s.format == 1)
                                for (int k = 0; k < 3; k++)
                                    c[k] = LinearToPwlGamma(c[k]);
                        }
                        else
                        {
                            const size_t i = rs.depth.Index(uint32_t(x) * rs.depth.scaleX, uint32_t(y) * rs.depth.scaleY);
                            const float d = g_edramDepth[i];
                            c = { d, float(g_edramStencil[i]), d, d }; // y = stencil for k_24_8 packing
                        }
                        if (expBias)
                            for (float& v : c)
                                v *= scale;
                        if (swapRB)
                            std::swap(c[0], c[2]);
                        uint32_t words[4];
                        PackPixel(c, format, words);
                        WritePixel(destBase + TiledOffset2D(uint32_t(dx), uint32_t(dy), pitchAligned, log2), words, bytes, endian);
                    }
                });
            }

            if (colorClear && source < 4)
            {
                const Surface& s = rs.color[source];
                const uint32_t v = Reg(0x231E);
                F4 c;
                if (s.format == 2 || s.format == 10)
                    c = { float(v & 0x3FF) / 1023.0f, float((v >> 10) & 0x3FF) / 1023.0f, float((v >> 20) & 0x3FF) / 1023.0f, float(v >> 30) / 3.0f };
                else if (s.format == 6) // k_16_16_FLOAT: both halves in RB_COLOR_CLEAR
                    c = { HalfToFloat(uint16_t(v)), HalfToFloat(uint16_t(v >> 16)), 0.0f, 0.0f };
                else if (s.format == 7) // k_16_16_16_16_FLOAT: low dword in RB_COLOR_CLEAR_LO
                {
                    const uint32_t lo = Reg(0x231F);
                    c = { HalfToFloat(uint16_t(lo)), HalfToFloat(uint16_t(lo >> 16)), HalfToFloat(uint16_t(v)), HalfToFloat(uint16_t(v >> 16)) };
                }
                else if (s.format == 14)
                    c = { AsFloat(v), 0, 0, 0 };
                else
                    c = { float(v & 0xFF) / 255.0f, float((v >> 8) & 0xFF) / 255.0f, float((v >> 16) & 0xFF) / 255.0f, float(v >> 24) / 255.0f };
                if (g_sink)
                    g_sink->ClearColor(rs, source, ex0, ey0, ex1, ey1, c);
                else
                    parallelRows([&](int y) {
                        for (int x = ex0; x < ex1; x++)
                            s.ForEachSample(uint32_t(x), uint32_t(y), [&](size_t i) { g_edramColor[i] = c; });
                    });
            }
            if (depthClear)
            {
                const uint32_t clear = Reg(0x231D);
                const float d = (Reg(0x2002) >> 16) & 1 ? Float20e4To32(clear >> 8) : float(clear >> 8) / float(0xFFFFFF);
                const uint8_t st = uint8_t(clear & 0xFF);
                if (g_sink)
                    g_sink->ClearDepth(rs, ex0, ey0, ex1, ey1, d, st);
                else
                    parallelRows([&](int y) {
                        for (int x = ex0; x < ex1; x++)
                            rs.depth.ForEachSample(uint32_t(x), uint32_t(y), [&](size_t i) { g_edramDepth[i] = d; g_edramStencil[i] = st; });
                    });
            }

            // Textures sampled from the destination must be decoded again.
            if (command != 3 && destPitch && Reg(0x2319) && bytes)
            {
                const uint32_t pitchAligned = (destPitch + 31) & ~31u;
                const uint32_t heightAligned = ((destHeight ? destHeight : uint32_t(ey1 - offsetY)) + 31) & ~31u;
                InvalidateTextures(destBase, pitchAligned * heightAligned * bytes);
            }
        }

        // Texture fetch constant slots a program's exec clauses use (for decoding up front,
        // on the command processor thread, before rasterisation fans out).
        uint32_t TextureSlotsUsed(const uint32_t* code, uint32_t count)
        {
            uint32_t slots = 0;
            uint32_t cfLimit = (count / 3) * 2; // control flow precedes the first clause
            for (uint32_t pc = 0; pc < cfLimit; pc++)
            {
                const uint32_t base = (pc / 2) * 3;
                if (base + 3 > count)
                    break;
                const uint32_t w0 = (pc & 1) ? (code[base + 1] >> 16) | (code[base + 2] << 16) : code[base];
                const uint32_t w1 = (pc & 1) ? code[base + 2] >> 16 : code[base + 1] & 0xFFFF;
                const auto op = ControlFlowOpcode((w1 >> 12) & 0xF);
                const bool isExec = op == ControlFlowOpcode::Exec || op == ControlFlowOpcode::ExecEnd ||
                    (op >= ControlFlowOpcode::CondExec && op <= ControlFlowOpcode::CondExecPredEnd) ||
                    op == ControlFlowOpcode::CondExecPredClean || op == ControlFlowOpcode::CondExecPredCleanEnd;
                if (!isExec)
                    continue;
                const uint32_t address = w0 & 0xFFF, n = (w0 >> 12) & 7, sequence = (w0 >> 16) & 0xFFF;
                cfLimit = std::min(cfLimit, address * 2);
                for (uint32_t k = 0; k < n; k++)
                {
                    const uint32_t at = (address + k) * 3;
                    if (at + 3 <= count && ((sequence >> (k * 2)) & 1) && (code[at] & 0x1F) == uint32_t(FetchOpcode::TextureFetch))
                        slots |= 1u << ((code[at] >> 20) & 0x1F);
                }
            }
            return slots;
        }

        std::vector<uint32_t> LoadCode(const ShaderCode& code)
        {
            return std::vector<uint32_t>(code.dwords, code.dwords + code.count);
        }
    }

    bool g_softLogVertices = false;
    bool g_softLogDraw = false;
    bool g_softDumpTextures = false;

    void RasterState::Load()
    {
        mode = Reg(0x2208) & 7;
        vte = Reg(0x2206);
        scale[0] = (vte & 0x01) ? RegF(0x210F) : 1.0f;
        offset[0] = (vte & 0x02) ? RegF(0x2110) : 0.0f;
        scale[1] = (vte & 0x04) ? RegF(0x2111) : 1.0f;
        offset[1] = (vte & 0x08) ? RegF(0x2112) : 0.0f;
        scale[2] = (vte & 0x10) ? RegF(0x2113) : 1.0f;
        offset[2] = (vte & 0x20) ? RegF(0x2114) : 0.0f;

        const uint32_t modeCntl = Reg(0x2205);
        const uint32_t windowOffsetReg = Reg(0x2080);
        windowOffset = (modeCntl >> 16) & 1;
        cullFront = modeCntl & 1;
        cullBack = (modeCntl >> 1) & 1;
        frontCW = (modeCntl >> 2) & 1;
        windowX = float(int32_t(windowOffsetReg << 17) >> 17);
        windowY = float(int32_t((windowOffsetReg >> 16) << 17) >> 17);
        pixelCenter = (Reg(0x2302) & 1) ? 0.5f : 0.0f;
        clip = !((Reg(0x2204) >> 16) & 1);

        const uint32_t tl = Reg(0x2081), br = Reg(0x2082);
        sx0 = int(tl & 0x7FFF); sy0 = int((tl >> 16) & 0x7FFF);
        sx1 = int(br & 0x7FFF); sy1 = int((br >> 16) & 0x7FFF);
        if (!(tl >> 31))
        {
            const int ox = int(windowX), oy = int(windowY);
            sx0 += ox; sx1 += ox; sy0 += oy; sy1 += oy;
        }
        const uint32_t stl = Reg(0x200E), sbr = Reg(0x200F);
        sx0 = std::max(sx0, int(stl & 0x7FFF)); sy0 = std::max(sy0, int((stl >> 16) & 0x7FFF));
        if (sbr)
        {
            sx1 = std::min(sx1, int(sbr & 0x7FFF));
            sy1 = std::min(sy1, int((sbr >> 16) & 0x7FFF));
        }
        sx0 = std::max(sx0, 0); sy0 = std::max(sy0, 0);

        const uint32_t surfaceInfo = Reg(0x2000);
        static constexpr uint32_t kColorInfo[4] = { 0x2001, 0x2003, 0x2004, 0x2005 };
        static constexpr uint32_t kBlend[4] = { 0x2201, 0x2209, 0x220A, 0x220B };
        for (int i = 0; i < 4; i++)
        {
            const uint32_t info = Reg(kColorInfo[i]);
            const uint32_t format = (info >> 16) & 0xF;
            color[i].Init(info & 0xFFF, surfaceInfo, format, Is64bppColor(format));
            blend[i] = Reg(kBlend[i]);
        }
        // Keep pixels inside the surface pitch.
        sx1 = std::min(sx1, int(color[0].pitch ? color[0].pitch : 8192));
        colorMask = Reg(0x2104);
        blendConstant = { RegF(0x2105), RegF(0x2106), RegF(0x2107), RegF(0x2108) };

        const uint32_t depthInfo = Reg(0x2002);
        depth.Init(depthInfo & 0xFFF, surfaceInfo, 0, false);
        const uint32_t depthControl = Reg(0x2200);
        depthTest = (depthControl >> 1) & 1;
        depthWrite = depthTest && ((depthControl >> 2) & 1);
        depthFunc = (depthControl >> 4) & 7;
        stencil = depthControl & 1;
        const bool backfaceStencil = (depthControl >> 7) & 1;
        for (int f = 0; f < 2; f++)
        {
            StencilFace& sf = stencilFace[f];
            const uint32_t shift = (f == 1 && backfaceStencil) ? 12 : 0;
            sf.func = (depthControl >> (8 + shift)) & 7;
            sf.fail = (depthControl >> (11 + shift)) & 7;
            sf.zpass = (depthControl >> (14 + shift)) & 7;
            sf.zfail = (depthControl >> (17 + shift)) & 7;
            const uint32_t refMask = Reg((f == 1 && backfaceStencil) ? 0x210C : 0x210D);
            sf.ref = refMask & 0xFF;
            sf.mask = (refMask >> 8) & 0xFF;
            sf.writeMask = (refMask >> 16) & 0xFF;
        }

        const uint32_t colorControl = Reg(0x2202);
        alphaTest = (colorControl >> 3) & 1;
        alphaFunc = colorControl & 7;
        alphaRef = RegF(0x210E);

        const uint32_t programCntl = Reg(0x2180);
        paramGen = (programCntl >> 18) & 1;
        paramGenRegister = (Reg(0x2181) >> 8) & 0xFF;
    }

    void SoftSetCaptureSink(CaptureSink* sink) { g_sink = sink; }

    void SoftParallelFor(uint32_t count, const std::function<void(uint32_t)>& fn)
    {
        constexpr uint32_t kRows = 8;
        Pool().Run(int((count + kRows - 1) / kRows), [&](int band) {
            for (uint32_t row = uint32_t(band) * kRows; row < std::min(count, uint32_t(band + 1) * kRows); row++)
                fn(row);
        });
    }

    uint64_t SoftPixelsWritten() { return g_pixels.load(); }
    uint64_t SoftResolves() { return g_resolves.load(); }

    void SoftEndFrame()
    {
        ++g_frame;
        if (g_textureCache.size() > 256)
            g_textureCache.clear();
    }

    void SoftPrintStats()
    {
        std::fprintf(stderr, "[gpu]   soft: %llu draws, vertex %.1f ms, raster %.1f ms (incl. texture decode %.1f ms), resolve %.1f ms, in worker pool %.1f ms (%.1f thread-ms busy); %llu texture decodes (%.1f Mtexels), %.1f MB hashed\n",
            (unsigned long long)g_stats.draws, g_stats.vertex / 1e6, g_stats.raster / 1e6, g_stats.texture / 1e6, g_stats.resolve / 1e6, g_stats.parallel / 1e6, g_poolBusyNs.exchange(0) / 1e6,
            (unsigned long long)g_stats.decodes, g_stats.decodedTexels / 1e6, g_stats.hashed / 1e6);
        g_stats = {};
    }

    void SoftDraw(const DrawCall& draw)
    {
        static DrawState state;
        RasterState rs;
        rs.Load();
        if (rs.mode != 4 && rs.mode != 5 && rs.mode != 6)
            return;
        const uint32_t prim = draw.primitive;
        if (prim != 4 && prim != 5 && prim != 6 && prim != 8 && prim != 13)
            return; // points and lines are not drawn yet
        if (!draw.vertexShader.count || draw.indexCount == 0)
            return;

        ++g_stats.draws;
        if (g_softDumpTextures)
            g_textureCache.clear(); // decode (and so dump) everything this draw samples
        state.Load();
        for (auto& t : state.textures)
            t.store(nullptr);

        const std::vector<uint32_t> vsCode = LoadCode(draw.vertexShader);
        const std::vector<uint32_t> psCode = LoadCode(draw.pixelShader);
        const Program vs{ vsCode.data(), uint32_t(vsCode.size()), false };

        // ---- indices
        const uint32_t indexOffset = Reg(0x2102);
        // Primitive restart (PA_SU_SC_MODE_CNTL.MULTI_PRIM_IB_ENA): the reset index
        // (VGT_MULTI_PRIM_IB_RESET_INDX) in a DMA index buffer ends the current strip or fan.
        const bool restartEnabled = draw.indexSource == 0 && ((Reg(0x2205) >> 21) & 1);
        const uint32_t resetIndex = Reg(0x2103) & (draw.index32 ? 0xFFFFFFFFu : 0xFFFFu);
        std::vector<uint8_t> restart(draw.indexCount, 0);
        std::vector<uint32_t> indices(draw.indexCount);
        for (uint32_t i = 0; i < draw.indexCount; i++)
        {
            uint32_t index = i;
            if (draw.indexSource == 0 && draw.indexBase)
            {
                if (draw.index32)
                    index = ReadSwapped(draw.indexBase + i * 4, draw.indexEndian);
                else
                {
                    const uint32_t word = ReadSwapped(draw.indexBase + (i / 2) * 4, draw.indexEndian);
                    index = (i & 1) ? word >> 16 : word & 0xFFFF;
                }
            }
            else if (draw.indexSource == 1)
            {
                if (draw.index32)
                    index = i < draw.immediateIndices.size() ? draw.immediateIndices[i] : 0;
                else
                {
                    const uint32_t word = (i / 2) < draw.immediateIndices.size() ? draw.immediateIndices[i / 2] : 0;
                    index = (i & 1) ? word >> 16 : word & 0xFFFF;
                }
            }
            if (restartEnabled && index == resetIndex)
                restart[i] = 1;
            indices[i] = index + indexOffset;
        }

        // ---- vertex shading (once per distinct index)
        auto vertexStart = std::chrono::steady_clock::now();
        std::unordered_map<uint32_t, uint32_t> slotOf;
        std::vector<uint32_t> unique; // distinct indices in first-use order
        std::vector<uint32_t> slots(indices.size());
        constexpr uint32_t kNoSlot = UINT32_MAX; // restart markers have no vertex
        for (size_t i = 0; i < indices.size(); i++)
        {
            if (restart[i])
            {
                slots[i] = kNoSlot;
                continue;
            }
            auto [it, inserted] = slotOf.emplace(indices[i], uint32_t(unique.size()));
            if (inserted)
                unique.push_back(indices[i]);
            slots[i] = it->second;
        }
        // Vertices are independent: shade them across the pool.
        std::vector<Vertex> vertices(unique.size());
        std::vector<uint32_t> exportedMasks(unique.size());
        constexpr size_t kVerticesPerJob = 64;
        Pool().Run(int((unique.size() + kVerticesPerJob - 1) / kVerticesPerJob), [&](int job) {
            const size_t end = std::min(unique.size(), size_t(job + 1) * kVerticesPerJob);
            for (size_t u = size_t(job) * kVerticesPerJob; u < end; u++)
            {
                Invocation inv;
                for (F4& e : inv.exports)
                    e = { 0, 0, 0, 0 };
                inv.r[0] = { float(unique[u]), 0, 0, 0 };
                Interpreter interp{ state, vs, inv, 0 };
                interp.Run();
                Vertex& v = vertices[u];
                v.position = inv.exports[62];
                for (int k = 0; k < 16; k++)
                    v.interpolators[k] = inv.exports[k];
                exportedMasks[u] = uint32_t(inv.exported & 0xFFFF);
            }
        });
        uint32_t interpolatorMask = 0;
        for (uint32_t mask : exportedMasks)
            interpolatorMask |= mask;
        rs.interpolatorMask = interpolatorMask;
        if (g_softLogVertices)
        {
            for (size_t i = 0; i < indices.size(); i++)
            {
                if (slots[i] == kNoSlot)
                    continue;
                const Vertex& v = vertices[slots[i]];
                std::fprintf(stderr, "[gpu]       v%zu idx=%u pos=(%g %g %g %g)", i, indices[i], v.position[0], v.position[1], v.position[2], v.position[3]);
                for (int k = 0; k < 2; k++)
                    if (interpolatorMask & (1u << k))
                        std::fprintf(stderr, " o%d=(%g %g %g %g)", k, v.interpolators[k][0], v.interpolators[k][1], v.interpolators[k][2], v.interpolators[k][3]);
                std::fprintf(stderr, "\n");
            }
        }
        g_stats.vertex += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - vertexStart).count());

        Renderer renderer{ state, rs, Program{ psCode.data(), uint32_t(psCode.size()), true } };
        // TTT2_SOFT_INTERP=1 interprets every pixel (reference for the compiled path).
        static const bool interpretOnly = [] { const char* v = std::getenv("TTT2_SOFT_INTERP"); return v && *v == '1'; }();
        CompiledShader compiledPs;
        if (g_sink && rs.mode != 6)
        {
            // The backend shades pixels: flatten the pixel shader's control flow for it.
            if (rs.mode == 4 && renderer.ps.count)
            {
                compiledPs = CompileFlow(state, renderer.ps);
                for (const CompiledOp& op : compiledPs.ops)
                    if (op.kind == CompiledOp::Kind::VertexFetch)
                        compiledPs.ok = false;
            }
            CaptureDraw capture;
            capture.rs = &rs;
            capture.ps = compiledPs.ok ? &compiledPs : nullptr;
            capture.psConstants = &state.constants[256];
            capture.texture = [](uint32_t slot) -> const Texture& { return state.TextureSlot(slot); };
            g_sink->BeginDraw(capture);
        }
        else if (rs.mode == 4 && renderer.ps.count && !interpretOnly)
        {
            compiledPs = Compile(state, renderer.ps);
            if (compiledPs.ok)
                renderer.compiled = &compiledPs;
        }

        // ---- copy: resolve the covered rectangle
        if (rs.mode == 6)
        {
            std::vector<ScreenVertex> corners;
            for (const Vertex& v : vertices)
                corners.push_back(renderer.ToScreen(v));
            Resolve(rs, corners.data(), corners.size());
            return;
        }

        // ---- decode the pixel shader's textures before fanning out
        if (rs.mode == 4 && !g_sink)
        {
            const uint32_t slots = TextureSlotsUsed(psCode.data(), uint32_t(psCode.size()));
            for (uint32_t slot = 0; slot < 32; slot++)
                if (slots & (1u << slot))
                    state.TextureSlot(slot);
        }

        // ---- primitive assembly
        ScopedTimer rasterTimer{ g_stats.raster };
        auto tri = [&](uint32_t a, uint32_t b, uint32_t c) {
            if (slots[a] == kNoSlot || slots[b] == kNoSlot || slots[c] == kNoSlot)
                return; // a list primitive containing a restart marker is dropped
            renderer.ClipAndDraw(vertices[slots[a]], vertices[slots[b]], vertices[slots[c]]);
        };
        // Strips and fans restart after each marker: [begin, end) runs of real vertices.
        auto forEachRun = [&](auto&& fn) {
            size_t begin = 0;
            for (size_t i = 0; i <= slots.size(); i++)
                if (i == slots.size() || slots[i] == kNoSlot)
                {
                    if (i > begin)
                        fn(begin, i);
                    begin = i + 1;
                }
        };
        const size_t n = slots.size();
        switch (prim)
        {
        case 4: // triangle list
            for (size_t i = 0; i + 2 < n; i += 3)
                tri(uint32_t(i), uint32_t(i + 1), uint32_t(i + 2));
            break;
        case 5: // triangle fan
            forEachRun([&](size_t begin, size_t end) {
                for (size_t i = begin + 1; i + 1 < end; i++)
                    tri(uint32_t(begin), uint32_t(i), uint32_t(i + 1));
            });
            break;
        case 6: // triangle strip (odd triangles swap their first two vertices to keep the winding)
            forEachRun([&](size_t begin, size_t end) {
                for (size_t i = begin; i + 2 < end; i++)
                {
                    if ((i - begin) & 1)
                        tri(uint32_t(i + 1), uint32_t(i), uint32_t(i + 2));
                    else
                        tri(uint32_t(i), uint32_t(i + 1), uint32_t(i + 2));
                }
            });
            break;
        case 13: // quad list
            for (size_t i = 0; i + 3 < n; i += 4)
            {
                tri(uint32_t(i), uint32_t(i + 1), uint32_t(i + 2));
                tri(uint32_t(i), uint32_t(i + 2), uint32_t(i + 3));
            }
            break;
        case 8: // rect list
            for (size_t i = 0; i + 2 < n; i += 3)
            {
                if (slots[i] == kNoSlot || slots[i + 1] == kNoSlot || slots[i + 2] == kNoSlot)
                    continue;
                const Vertex& v0 = vertices[slots[i]];
                const Vertex& v1 = vertices[slots[i + 1]];
                const Vertex& v2 = vertices[slots[i + 2]];
                const ScreenVertex s[3] = { renderer.ToScreen(v0), renderer.ToScreen(v1), renderer.ToScreen(v2) };
                int order[3];
                const Vertex& v3 = renderer.extraVertices.emplace_back(CompleteRect(v0, v1, v2, s, order));
                const ScreenVertex s3 = renderer.ToScreen(v3);
                renderer.Triangle(s[order[0]], s[order[1]], s[order[2]]);
                renderer.Triangle(s3, s[order[1]], s[order[2]]);
            }
            break;
        }
        if (g_sink)
        {
            g_sink->EndDraw();
            return;
        }
        renderer.Flush();
        g_pixels += renderer.pixels;
        if (g_softLogDraw)
        {
            float zMin = FLT_MAX, zMax = -FLT_MAX;
            for (const Vertex& v : vertices)
            {
                const ScreenVertex sv = renderer.ToScreen(v);
                zMin = std::min(zMin, sv.z);
                zMax = std::max(zMax, sv.z);
            }
            const size_t centre = rs.depth.Index(uint32_t(rs.sx1 / 2) * rs.depth.scaleX, uint32_t(rs.sy1 / 2) * rs.depth.scaleY);
            std::fprintf(stderr, "[gpu]   debug: %llu triangles, %llu set up, %llu pixels inside, %llu stencil fails, %llu depth fails; screen z %g..%g; depth at centre %g stencil %u; depth test %d func %u write %d\n",
                (unsigned long long)g_drawDebug.triangles.exchange(0), (unsigned long long)g_drawDebug.setups.exchange(0),
                (unsigned long long)g_drawDebug.inside.exchange(0), (unsigned long long)g_drawDebug.stencilFail.exchange(0),
                (unsigned long long)g_drawDebug.depthFail.exchange(0), zMin, zMax, g_edramDepth[centre], g_edramStencil[centre],
                rs.depthTest, rs.depthFunc, rs.depthWrite);
        }
    }
}
