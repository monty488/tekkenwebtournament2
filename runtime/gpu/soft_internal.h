#pragma once

// Types shared between the software renderer (soft_gpu.cpp) and the hardware backends
// that reuse its front end (vertex shading, primitive assembly, clipping, resolve
// packing, texture decoding, pixel-shader control-flow flattening).

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <cstdint>
#include <vector>
#include "../../thirdparty/XenosRecomp/XenosRecomp/shader_code.h"

namespace gpu
{
    using F4 = std::array<float, 4>;

    constexpr uint32_t kEdramTiles = 2048;
    constexpr uint32_t kTileSamples = 80 * 16;

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
            // Constant divisors (the tile width is 80 or 40) avoid a hardware divide.
            const uint32_t tx = tileWidth == 80 ? sx / 80 : sx / 40;
            const uint32_t tile = (baseTile + (sy / 16) * tilesPerRow + tx) % kEdramTiles;
            return size_t(tile) * kTileSamples + (sy % 16) * tileWidth + (sx - tx * tileWidth);
        }

        template <typename Fn>
        void ForEachSample(uint32_t x, uint32_t y, Fn&& fn) const
        {
            // A pixel's samples (at most 2x2, starting on even coordinates) never straddle
            // a tile, whose width and height are even: they sit side by side in one tile.
            const size_t first = Index(x * scaleX, y * scaleY);
            for (uint32_t j = 0; j < scaleY; j++)
                for (uint32_t i = 0; i < scaleX; i++)
                    fn(first + j * tileWidth + i);
        }
    };

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
        bool stale = false; // a resolve overwrote the source: decode again on next use
        uint64_t version = 0; // changes on every decode (hardware backends re-upload on change)
    };

    struct AluOp
    {
        struct Source
        {
            uint32_t index = 0;      // register (taken & 127) or constant (taken & 255)
            bool isRegister = false;
            bool relativeA0 = false; // constants: index += a0 when read
            bool abs = false, negate = false;
            uint8_t comp[4] = { 0, 1, 2, 3 };
        };
        AluInstruction a;
        Source src[3];
        // Scalar ops 42-47 take constant src3 .w-ish (scalarConstComp) and a GPR component.
        bool scalarConst = false;
        uint32_t scalarConstComp = 0, scalarGpr = 0, scalarGprComp = 0;
        uint32_t vectorDest = 0, scalarDest = 0; // GPR destinations, aL applied
    };

    struct CompiledOp
    {
        enum class Kind : uint8_t { Clause, Alu, VertexFetch, TextureFetch };
        Kind kind = Kind::Clause;
        int8_t clausePredicate = -1; // Clause: -1 always runs, else only where p0 == it
        AluOp alu;
        VertexFetchInstruction vertexFetch;
        TextureFetchInstruction textureFetch;
    };

    struct CompiledShader
    {
        std::vector<CompiledOp> ops;
        bool ok = false;
    };

    struct RasterState
    {
        uint32_t mode = 0;
        // viewport / transform
        uint32_t vte = 0;
        float scale[3] = { 1, 1, 1 }, offset[3] = { 0, 0, 0 };
        float windowX = 0, windowY = 0;
        bool windowOffset = false;
        // PA_SU_SC_MODE_CNTL: face culling; front faces wind clockwise in window
        // space (y down) when frontCW is set, counter-clockwise otherwise.
        bool cullFront = false, cullBack = false, frontCW = false;
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
        // Stencil (RB_DEPTHCONTROL / RB_STENCILREFMASK[_BF]): [0] front, [1] back faces.
        struct StencilFace { uint32_t func = 7, fail = 0, zpass = 0, zfail = 0, ref = 0, mask = 0xFF, writeMask = 0xFF; };
        bool stencil = false;
        StencilFace stencilFace[2];
        uint32_t depthFunc = 7;
        bool alphaTest = false;
        uint32_t alphaFunc = 7;
        float alphaRef = 0.0f;
        // pixel shader inputs
        uint32_t interpolatorMask = 0;
        bool paramGen = false;
        uint32_t paramGenRegister = 0;

        void Load(); // soft_gpu.cpp
    };

    // ---- Capture: hardware backends ------------------------------------------------
    // With a sink installed, SoftDraw still runs the vertex shader, assembles, clips and
    // culls primitives and decodes register state, but hands the resulting screen-space
    // triangles to the sink instead of rasterising them, and resolves read and clear
    // the sink's render targets instead of the emulated EDRAM.

    struct CaptureVertex
    {
        float x, y, z, invW;        // window (EDRAM pixel) coordinates, depth, 1/w
        const F4* interpolators;    // 16 entries
    };

    struct CaptureDraw
    {
        const RasterState* rs = nullptr;
        const CompiledShader* ps = nullptr; // control flow flattened for this draw (null: none)
        const F4* psConstants = nullptr;    // c256..c511
        const uint32_t* fetch = nullptr;    // fetch constants (32 slots x 6 dwords)
        std::function<const Texture&(uint32_t slot)> texture;
    };

    // A resolve (EDRAM -> memory copy) as decoded from the RB_COPY_* registers.
    struct ResolveRequest
    {
        const RasterState* rs = nullptr;
        uint32_t source = 0;          // colour render target 0-3 (4 = depth)
        uint32_t sampleSelect = 0;
        int x0 = 0, y0 = 0, x1 = 0, y1 = 0; // EDRAM pixel rectangle
        int offsetX = 0, offsetY = 0;       // destination = EDRAM position - offset
        uint32_t destBase = 0;        // guest address
        uint32_t destPitch = 0, destHeight = 0;
        uint32_t destFormat = 0;      // texture format
        uint32_t endian = 0;
        int expBias = 0;
        bool swapRB = false;
    };

    class CaptureSink
    {
    public:
        virtual ~CaptureSink() = default;
        virtual void BeginDraw(const CaptureDraw& draw) = 0;
        virtual void Triangle(const CaptureVertex& a, const CaptureVertex& b, const CaptureVertex& c, bool front) = 0;
        virtual void EndDraw() = 0;
        // Resolve support. Rectangles are [x0, x1) x [y0, y1) in EDRAM pixel coordinates;
        // reads fill `out` row-major.
        virtual void ReadColor(const RasterState& rs, uint32_t rt, int x0, int y0, int x1, int y1, std::vector<F4>& out) = 0;
        virtual void ReadDepth(const RasterState& rs, int x0, int y0, int x1, int y1, std::vector<float>& depth, std::vector<uint8_t>& stencil) = 0;
        virtual void ClearColor(const RasterState& rs, uint32_t rt, int x0, int y0, int x1, int y1, const F4& color) = 0;
        virtual void ClearDepth(const RasterState& rs, int x0, int y0, int x1, int y1, float depth, uint8_t stencil) = 0;
        // Performs the copy without touching guest memory (the backend keeps the image and
        // binds it where the game samples that address). False: use ReadColor/ReadDepth
        // and write memory as usual.
        virtual bool ResolveToGpu(const ResolveRequest&) { return false; }
    };

    void SoftSetCaptureSink(CaptureSink* sink);
}
