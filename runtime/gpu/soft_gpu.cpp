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
#include "texture.h"
#include "../kernel/guest.h"

#include <cstdint>
#include "../../thirdparty/XenosRecomp/XenosRecomp/shader_code.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdio>
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

        using F4 = std::array<float, 4>;

        constexpr uint32_t kRegisterBase = 0x7FC80000;
        uint32_t Reg(uint32_t index) { return guest::Load<uint32_t>(kRegisterBase + index * 4); }
        float AsFloat(uint32_t v) { float f; std::memcpy(&f, &v, 4); return f; }
        float RegF(uint32_t index) { return AsFloat(Reg(index)); }
        uint32_t PhysicalToGuest(uint32_t physical) { return 0xA0000000u + (physical & 0x1FFFFFFFu); }

        std::atomic<uint64_t> g_pixels{ 0 };
        std::atomic<uint64_t> g_resolves{ 0 };

        // Time spent per stage since the last SoftPrintStats (ns).
        struct Stats { uint64_t vertex, raster, resolve, texture, draws, decodes, decodedTexels, hashed; } g_stats{};
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

        class WorkerPool
        {
        public:
            WorkerPool()
            {
                const unsigned n = std::max(1u, std::thread::hardware_concurrency());
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
                for (int i; (i = job.next.fetch_add(1)) < job.count;)
                {
                    (*job.fn)(i);
                    ++completed;
                }
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

        constexpr uint32_t kEdramTiles = 2048;
        constexpr uint32_t kTileSamples = 80 * 16;

        std::vector<F4> g_edramColor(size_t(kEdramTiles) * kTileSamples, F4{ 0, 0, 0, 0 });
        std::vector<float> g_edramDepth(size_t(kEdramTiles) * kTileSamples, 1.0f);

        bool Is64bppColor(uint32_t format)
        {
            // k_16_16_16_16, k_16_16_16_16_FLOAT, k_32_32_FLOAT
            return format == 5 || format == 7 || format == 15;
        }

        bool IsUnormColor(uint32_t format)
        {
            return format == 0 || format == 1 || format == 2 || format == 10;
        }

        struct Surface
        {
            uint32_t baseTile = 0;
            uint32_t pitch = 0;      // pixels
            uint32_t scaleX = 1, scaleY = 1;
            uint32_t tileWidth = 80;  // samples per tile row
            uint32_t tilesPerRow = 1;
            uint32_t format = 0;

            void Init(uint32_t base, uint32_t surfaceInfo, uint32_t fmt, bool wide)
            {
                baseTile = base & 0xFFF;
                pitch = surfaceInfo & 0x3FFF;
                const uint32_t msaa = (surfaceInfo >> 16) & 3;
                scaleY = msaa >= 1 ? 2 : 1;
                scaleX = msaa >= 2 ? 2 : 1;
                format = fmt;
                tileWidth = wide ? 40 : 80;
                tilesPerRow = std::max(1u, (pitch * scaleX + tileWidth - 1) / tileWidth);
            }

            size_t Index(uint32_t sx, uint32_t sy) const
            {
                const uint32_t tile = (baseTile + (sy / 16) * tilesPerRow + sx / tileWidth) % kEdramTiles;
                return size_t(tile) * kTileSamples + (sy % 16) * tileWidth + sx % tileWidth;
            }

            template <typename Fn>
            void ForEachSample(uint32_t x, uint32_t y, Fn&& fn) const
            {
                for (uint32_t j = 0; j < scaleY; j++)
                    for (uint32_t i = 0; i < scaleX; i++)
                        fn(Index(x * scaleX + i, y * scaleY + j));
            }
        };

        // ---- Textures ------------------------------------------------------------

        struct Texture
        {
            bool ok = false;
            uint32_t width = 0, height = 0;
            uint32_t clampX = 0, clampY = 0;
            uint32_t magFilter = 0, minFilter = 0;
            std::vector<F4> texels; // after the fetch constant's swizzle
            // Source memory and its hash, to reuse the decode while the data is unchanged.
            uint32_t base = 0, bytes = 0;
            uint64_t hash = 0;
            uint64_t verifiedFrame = 0;
        };

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

        uint64_t HashMemory(uint32_t address, uint32_t bytes)
        {
            const uint8_t* p = guest::ToHost<const uint8_t>(address);
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

        // Bytes of guest memory mip 0 occupies (0 if the format is unknown).
        uint32_t TextureByteSize(const TextureInfo& info)
        {
            uint32_t blockW = 1, blockBytes;
            switch (info.format)
            {
            case 2: blockBytes = 1; break;
            case 3: case 4: case 10: case 15: blockBytes = 2; break;
            case 6: case 14: case 50: blockBytes = 4; break;
            case 18: case 51: blockW = 4; blockBytes = 8; break;
            case 19: case 20: case 52: case 53: blockW = 4; blockBytes = 16; break;
            default: return 0;
            }
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

        void DecodeTexture(Texture& tex, const uint32_t fetch[6])
        {
            ScopedTimer timer{ g_stats.texture };
            tex.ok = false;
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
            std::vector<uint32_t> bgra;
            const ParallelFor parallel = [](uint32_t count, const std::function<void(uint32_t)>& fn) {
                constexpr uint32_t kRows = 8;
                Pool().Run(int((count + kRows - 1) / kRows), [&](int band) {
                    for (uint32_t row = uint32_t(band) * kRows; row < std::min(count, uint32_t(band + 1) * kRows); row++)
                        fn(row);
                });
            };
            if (!ConvertToBgra(info, bgra, parallel))
            {
                static bool warned[64];
                if (!warned[info.format & 63])
                {
                    warned[info.format & 63] = true;
                    std::fprintf(stderr, "[gpu] soft: unsupported texture format %u\n", info.format);
                }
                return;
            }

            // ConvertToBgra keeps 8_8_8_8 in native component order (X in the low byte) and
            // packs every other format as B,G,R,A bytes with R = component 0.
            const bool native8888 = info.format == 6 || info.format == 14 || info.format == 50;
            const bool twoComp = info.format == 10; // k_8_8: comp0 = low byte, comp1 = high byte
            const uint32_t swizzle = (fetch[3] >> 1) & 0xFFF;

            tex.width = info.width;
            tex.height = info.height;
            tex.clampX = (fetch[0] >> 10) & 7;
            tex.clampY = (fetch[0] >> 13) & 7;
            tex.magFilter = (fetch[3] >> 19) & 3;
            tex.minFilter = (fetch[3] >> 21) & 3;
            tex.texels.resize(bgra.size());
            constexpr size_t kChunk = 16384;
            Pool().Run(int((bgra.size() + kChunk - 1) / kChunk), [&](int chunk) {
                const size_t end = std::min(bgra.size(), size_t(chunk + 1) * kChunk);
                for (size_t i = size_t(chunk) * kChunk; i < end; i++)
                {
                    const uint32_t c = bgra[i];
                    const float b0 = float(c & 0xFF) / 255.0f, b1 = float((c >> 8) & 0xFF) / 255.0f;
                    const float b2 = float((c >> 16) & 0xFF) / 255.0f, b3 = float(c >> 24) / 255.0f;
                    F4 comp;
                    if (native8888)
                        comp = { b0, b1, b2, b3 };
                    else if (twoComp)
                        comp = { b2, b3, 0.0f, 0.0f };
                    else
                        comp = { b2, b1, b0, b3 };
                    F4& out = tex.texels[i];
                    for (int k = 0; k < 4; k++)
                    {
                        const uint32_t sel = (swizzle >> (k * 3)) & 7;
                        out[k] = sel < 4 ? comp[sel] : (sel == 5 ? 1.0f : 0.0f);
                    }
                }
            });
            tex.ok = true;
        }

        const Texture& LoadTexture(const uint32_t fetch[6])
        {
            FetchKey key;
            std::memcpy(key.d, fetch, sizeof(key.d));
            auto [it, inserted] = g_textureCache.try_emplace(key);
            Texture& tex = it->second;
            if (inserted)
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

        // Drops decodes whose memory a resolve just overwrote.
        void InvalidateTextures(uint32_t base, uint32_t bytes)
        {
            for (auto it = g_textureCache.begin(); it != g_textureCache.end();)
            {
                const Texture& t = it->second;
                const bool overlaps = t.bytes == 0 || (t.base < base + bytes && base < t.base + t.bytes);
                it = overlaps ? g_textureCache.erase(it) : std::next(it);
            }
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

        struct Interpreter
        {
            DrawState& state;
            const Program& program;
            Invocation& s;
            uint32_t constBase;

            F4 ReadSource(uint32_t reg, bool isRegister, uint32_t swizzle, bool negate, bool absConstants,
                bool constRelative, bool addressRelative, bool relativeSwizzle) const
            {
                F4 raw;
                bool abs;
                if (isRegister)
                {
                    abs = (reg & 0x80) != 0;
                    uint32_t index = reg & 0x3F;
                    if (reg & 0x40)
                        index += uint32_t(s.aL);
                    raw = s.r[index & 127];
                }
                else
                {
                    abs = absConstants;
                    int index = int(reg);
                    if (constRelative)
                        index += addressRelative ? s.a0 : s.aL;
                    raw = state.constants[constBase + (uint32_t(index) & 255)];
                }
                F4 out;
                for (int i = 0; i < 4; i++)
                {
                    float v = raw[relativeSwizzle ? (((swizzle >> (i * 2)) + i) & 3) : i];
                    if (abs)
                        v = std::fabs(v);
                    out[i] = negate ? -v : v;
                }
                return out;
            }

            void Alu(const uint32_t* ins)
            {
                AluInstruction a;
                std::memcpy(&a, ins, 12);
                if (a.isPredicated && s.p0 != bool(a.predicateCondition))
                    return;

                // Constant operands take const0Relative/const1Relative in source order.
                int constIndex = 0;
                auto source = [&](uint32_t reg, bool select, uint32_t swizzle, bool negate) {
                    bool relative = false;
                    if (!select)
                        relative = (constIndex++ == 0) ? a.const0Relative : a.const1Relative;
                    return ReadSource(reg, select, swizzle, negate, a.absConstants, relative,
                        a.constAddressRegisterRelative, true);
                };

                const auto vop = a.vectorOpcode;
                const auto sop = a.scalarOpcode;
                const F4 src1 = source(a.src1Register, a.src1Select, a.src1Swizzle, a.src1Negate);
                const F4 src2 = source(a.src2Register, a.src2Select, a.src2Swizzle, a.src2Negate);

                // Scalar operands come from src3: a = its swizzled .w, b = its swizzled .x.
                const bool scalarConst = uint32_t(sop) >= 42 && uint32_t(sop) <= 47;
                F4 src3{};
                float sa = 0.0f, sb = 0.0f;
                if (scalarConst)
                {
                    // const (src3Register) op GPR (encoded in the opcode/select/swizzle bits)
                    const uint32_t swz = a.src3Swizzle;
                    const F4 c = ReadSource(a.src3Register, false, 0xE4 /* identity, unused */, a.src3Negate,
                        a.absConstants, a.const0Relative, a.constAddressRegisterRelative, false);
                    sa = c[((swz >> 6) + 3) & 3];
                    const uint32_t gpr = (uint32_t(sop) & 1) | (uint32_t(a.src3Select) << 1) | (swz & 0x3C);
                    float v = s.r[gpr & 127][swz & 3];
                    if (a.absConstants)
                        v = std::fabs(v);
                    sb = a.src3Negate ? -v : v;
                }
                else
                {
                    src3 = source(a.src3Register, a.src3Select, a.src3Swizzle, a.src3Negate);
                    // `source` applied the relative swizzle per component; undo it to pick
                    // the scalar operands by their absolute positions (components 3 and 0).
                    sa = src3[3];
                    sb = src3[0];
                }

                // ---- vector op
                F4 vr{};
                bool pushPredicate = false, newP0 = s.p0;
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
                    bool kill = false;
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
                    if (kill && program.pixel)
                        s.killed = true;
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
                if (vop == AluVectorOpcode::MaxA)
                    s.a0 = int(std::clamp(std::floor(src1[3] + 0.5f), -256.0f, 255.0f));
                if (a.vectorSaturate)
                    for (float& v : vr)
                        v = Saturate(v);

                // ---- scalar op
                float sr = s.ps;
                bool scalarWrites = sop != AluScalarOpcode::RetainPrev;
                switch (sop)
                {
                case AluScalarOpcode::Adds: sr = sa + sb; break;
                case AluScalarOpcode::AddsPrev: sr = sa + s.ps; break;
                case AluScalarOpcode::Muls: sr = Mul(sa, sb); break;
                case AluScalarOpcode::MulsPrev:
                case AluScalarOpcode::MulsPrev2: sr = Mul(sa, s.ps); break;
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
                    s.a0 = int(std::clamp(std::floor(sa + 0.5f), -256.0f, 255.0f));
                    sr = sa >= sb ? sa : sb;
                    break;
                case AluScalarOpcode::MaxAsf:
                    s.a0 = int(std::clamp(std::floor(sa), -256.0f, 255.0f));
                    sr = sa >= sb ? sa : sb;
                    break;
                case AluScalarOpcode::Subs: sr = sa - sb; break;
                case AluScalarOpcode::SubsPrev: sr = sa - s.ps; break;
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
                    bool kill;
                    switch (sop)
                    {
                    case AluScalarOpcode::KillsEq: kill = sa == 0.0f; break;
                    case AluScalarOpcode::KillsGt: kill = sa > 0.0f; break;
                    case AluScalarOpcode::KillsGe: kill = sa >= 0.0f; break;
                    case AluScalarOpcode::KillsNe: kill = sa != 0.0f; break;
                    default: kill = sa == 1.0f; break;
                    }
                    if (kill && program.pixel)
                        s.killed = true;
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
                        uint32_t dest = a.vectorDest;
                        if (a.vectorDestRelative)
                            dest += uint32_t(s.aL);
                        F4& r = s.r[dest & 127];
                        for (int i = 0; i < 4; i++)
                            if (vmask & (1u << i))
                                r[i] = vr[i];
                    }
                    if (smask)
                    {
                        uint32_t dest = a.scalarDest;
                        if (a.scalarDestRelative)
                            dest += uint32_t(s.aL);
                        F4& r = s.r[dest & 127];
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

            void VertexFetch(const uint32_t* ins)
            {
                VertexFetchInstruction f;
                std::memcpy(&f, ins, 12);
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

            void TextureFetch(const uint32_t* ins)
            {
                TextureFetchInstruction f;
                std::memcpy(&f, ins, 12);
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
                float u = src[f.srcSwizzle & 3];
                float v = src[(f.srcSwizzle >> 2) & 3];
                const Texture& tex = state.TextureSlot(f.constIndex);
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
                ApplyDestSwizzle(s.r[f.dstRegister & 63], result, f.dstSwizzle);
            }

            bool BoolConstant(uint32_t index) const { return (state.bools[(index >> 5) & 7] >> (index & 31)) & 1; }

            void ExecClause(uint32_t address, uint32_t count, uint32_t sequence)
            {
                for (uint32_t k = 0; k < count; k++)
                {
                    const uint32_t at = (address + k) * 3;
                    if (at + 3 > program.count)
                        return;
                    const uint32_t* ins = program.code + at;
                    if ((sequence >> (k * 2)) & 1)
                    {
                        if ((ins[0] & 0x1F) == uint32_t(FetchOpcode::VertexFetch))
                            VertexFetch(ins);
                        else
                            TextureFetch(ins);
                    }
                    else
                    {
                        Alu(ins);
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

        struct RasterState
        {
            uint32_t mode = 0;
            // viewport / transform
            uint32_t vte = 0;
            float scale[3] = { 1, 1, 1 }, offset[3] = { 0, 0, 0 };
            float windowX = 0, windowY = 0;
            bool windowOffset = false;
            float pixelCenter = 0.0f;
            bool clip = true;
            // scissor (pixels, exclusive max)
            int sx0 = 0, sy0 = 0, sx1 = 0, sy1 = 0;
            // targets
            Surface color[4];
            uint32_t colorMask = 0;
            uint32_t blend[4] = {};
            F4 blendConstant{};
            Surface depth;
            bool depthTest = false, depthWrite = false;
            uint32_t depthFunc = 7;
            bool alphaTest = false;
            uint32_t alphaFunc = 7;
            float alphaRef = 0.0f;
            // pixel shader inputs
            uint32_t interpolatorMask = 0;
            bool paramGen = false;
            uint32_t paramGenRegister = 0;

            void Load()
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

                const uint32_t colorControl = Reg(0x2202);
                alphaTest = (colorControl >> 3) & 1;
                alphaFunc = colorControl & 7;
                alphaRef = RegF(0x210E);

                const uint32_t programCntl = Reg(0x2180);
                paramGen = (programCntl >> 18) & 1;
                paramGenRegister = (Reg(0x2181) >> 8) & 0xFF;
            }
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
            uint64_t pixels = 0;

            // Triangles of the current draw, set up and binned, rasterised by Flush().
            struct Setup
            {
                ScreenVertex v[3];
                struct Edge { float ax, ay, dx, dy; bool owns; } edges[3];
                float invArea;
                int x0, x1, y0, y1;
            };
            std::vector<Setup> setups;
            std::deque<Vertex> extraVertices; // clipped / completed vertices (stable addresses)

            ScreenVertex ToScreen(const Vertex& v) const
            {
                const F4& p = v.position;
                float x = p[0], y = p[1], z = p[2];
                const float w = p[3];
                const float invW = (rs.vte & (1u << 10)) ? w : (w != 0.0f ? 1.0f / w : 1.0f);
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

            bool ShadePixel(int px, int py, const ScreenVertex* const t[3], const float bary[3], float z)
            {
                // Early depth test (depth exported by the shader is tested after it runs).
                const size_t depthIndex = rs.depth.Index(uint32_t(px) * rs.depth.scaleX, uint32_t(py) * rs.depth.scaleY);
                if (rs.depthTest && !Compare(rs.depthFunc, z, g_edramDepth[depthIndex]))
                    return false;

                Invocation inv;
                inv.ps = 0.0f;
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

                if (rs.mode == 4 && ps.count)
                {
                    Interpreter it{ state, ps, inv, 256 };
                    it.Run();
                    if (inv.killed)
                        return false;
                    if (rs.alphaTest && !Compare(rs.alphaFunc, inv.exports[0][3], rs.alphaRef))
                        return false;
                }

                if (inv.exported & (1ull << 61))
                    z = inv.exports[61][0];
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
                    if (!mask || !(inv.exported & (1ull << rt)))
                        continue;
                    const Surface& surface = rs.color[rt];
                    const F4 src = inv.exports[rt];
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

            void Triangle(const ScreenVertex& a, const ScreenVertex& b, const ScreenVertex& c)
            {
                Setup su;
                su.v[0] = a; su.v[1] = b; su.v[2] = c;
                float area = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
                if (area == 0.0f || !std::isfinite(area))
                    return;
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
                setups.push_back(su);
            }

            uint64_t RasteriseBand(const Setup& su, int y0, int y1)
            {
                const ScreenVertex* const t[3] = { &su.v[0], &su.v[1], &su.v[2] };
                uint64_t written = 0;
                for (int py = y0; py <= y1; py++)
                {
                    const float sy = float(py) + rs.pixelCenter;
                    for (int px = su.x0; px <= su.x1; px++)
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
                        written += ShadePixel(px, py, t, bary, z);
                    }
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
                    Triangle(ToScreen(a), ToScreen(b), ToScreen(c));
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
                    Triangle(sv[0], sv[i], sv[i + 1]);
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
            const float max = float((1u << bits) - 1);
            return uint32_t(std::lround(Saturate(v) * max));
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
                        if (source < 4)
                        {
                            const Surface& s = rs.color[source];
                            c = g_edramColor[s.Index(uint32_t(x) * s.scaleX, uint32_t(y) * s.scaleY)];
                        }
                        else
                        {
                            const float d = g_edramDepth[rs.depth.Index(uint32_t(x) * rs.depth.scaleX, uint32_t(y) * rs.depth.scaleY)];
                            c = { d, d, d, d };
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
                else if (s.format == 6 || s.format == 7)
                {
                    const uint32_t lo = Reg(0x231F);
                    c = { HalfToFloat(uint16_t(lo)), HalfToFloat(uint16_t(lo >> 16)), HalfToFloat(uint16_t(v)), HalfToFloat(uint16_t(v >> 16)) };
                }
                else if (s.format == 14)
                    c = { AsFloat(v), 0, 0, 0 };
                else
                    c = { float(v & 0xFF) / 255.0f, float((v >> 8) & 0xFF) / 255.0f, float((v >> 16) & 0xFF) / 255.0f, float(v >> 24) / 255.0f };
                parallelRows([&](int y) {
                    for (int x = ex0; x < ex1; x++)
                        s.ForEachSample(uint32_t(x), uint32_t(y), [&](size_t i) { g_edramColor[i] = c; });
                });
            }
            if (depthClear)
            {
                const float d = float(Reg(0x231D) >> 8) / float(0xFFFFFF);
                parallelRows([&](int y) {
                    for (int x = ex0; x < ex1; x++)
                        rs.depth.ForEachSample(uint32_t(x), uint32_t(y), [&](size_t i) { g_edramDepth[i] = d; });
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
        std::fprintf(stderr, "[gpu]   soft: %llu draws, vertex %.1f ms, raster %.1f ms (incl. texture decode %.1f ms), resolve %.1f ms; %llu texture decodes (%.1f Mtexels), %.1f MB hashed\n",
            (unsigned long long)g_stats.draws, g_stats.vertex / 1e6, g_stats.raster / 1e6, g_stats.texture / 1e6, g_stats.resolve / 1e6,
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
        state.Load();
        for (auto& t : state.textures)
            t.store(nullptr);

        const std::vector<uint32_t> vsCode = LoadCode(draw.vertexShader);
        const std::vector<uint32_t> psCode = LoadCode(draw.pixelShader);
        const Program vs{ vsCode.data(), uint32_t(vsCode.size()), false };

        // ---- indices
        const uint32_t indexOffset = Reg(0x2102);
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
            indices[i] = index + indexOffset;
        }

        // ---- vertex shading (once per distinct index)
        auto vertexStart = std::chrono::steady_clock::now();
        std::unordered_map<uint32_t, uint32_t> slotOf;
        std::vector<Vertex> vertices;
        std::vector<uint32_t> slots(indices.size());
        uint32_t interpolatorMask = 0;
        for (size_t i = 0; i < indices.size(); i++)
        {
            auto [it, inserted] = slotOf.emplace(indices[i], uint32_t(vertices.size()));
            if (inserted)
            {
                Invocation inv;
                for (F4& e : inv.exports)
                    e = { 0, 0, 0, 0 };
                inv.r[0] = { float(indices[i]), 0, 0, 0 };
                Interpreter interp{ state, vs, inv, 0 };
                interp.Run();
                Vertex v;
                v.position = inv.exports[62];
                for (int k = 0; k < 16; k++)
                    v.interpolators[k] = inv.exports[k];
                interpolatorMask |= uint32_t(inv.exported & 0xFFFF);
                vertices.push_back(v);
            }
            slots[i] = it->second;
        }
        rs.interpolatorMask = interpolatorMask;
        g_stats.vertex += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - vertexStart).count());

        Renderer renderer{ state, rs, Program{ psCode.data(), uint32_t(psCode.size()), true } };

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
        if (rs.mode == 4)
        {
            const uint32_t slots = TextureSlotsUsed(psCode.data(), uint32_t(psCode.size()));
            for (uint32_t slot = 0; slot < 32; slot++)
                if (slots & (1u << slot))
                    state.TextureSlot(slot);
        }

        // ---- primitive assembly
        ScopedTimer rasterTimer{ g_stats.raster };
        auto tri = [&](uint32_t a, uint32_t b, uint32_t c) {
            renderer.ClipAndDraw(vertices[slots[a]], vertices[slots[b]], vertices[slots[c]]);
        };
        const size_t n = slots.size();
        switch (prim)
        {
        case 4: // triangle list
            for (size_t i = 0; i + 2 < n; i += 3)
                tri(uint32_t(i), uint32_t(i + 1), uint32_t(i + 2));
            break;
        case 5: // triangle fan
            for (size_t i = 1; i + 1 < n; i++)
                tri(0, uint32_t(i), uint32_t(i + 1));
            break;
        case 6: // triangle strip
            for (size_t i = 0; i + 2 < n; i++)
                tri(uint32_t(i), uint32_t(i + 1), uint32_t(i + 2));
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
        renderer.Flush();
        g_pixels += renderer.pixels;
    }
}
