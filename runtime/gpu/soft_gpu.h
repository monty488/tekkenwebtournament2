#pragma once

// Software reference renderer for Xenos draws: shader microcode interpreter, rasteriser
// into an emulated EDRAM, and resolves (EDRAM -> memory). Slow, but simple and checkable;
// it defines the behaviour the WebGPU backend must reproduce.
//
// All state (context registers, constants, fetch constants) is read from the GPU register
// file at the time of the draw, so SoftDraw must run on the command processor thread in
// packet order.

#include <cstdint>
#include <vector>

namespace gpu
{
    struct ShaderCode
    {
        const uint32_t* dwords = nullptr; // host-order microcode
        uint32_t count = 0;
    };

    struct DrawCall
    {
        uint32_t primitive = 0;        // VGT_DRAW_INITIATOR prim type
        uint32_t indexCount = 0;
        uint32_t indexSource = 2;      // 0 = DMA, 1 = immediate, 2 = auto
        uint32_t indexBase = 0;        // guest address of the index buffer (DMA)
        bool index32 = false;
        uint32_t indexEndian = 0;
        std::vector<uint32_t> immediateIndices; // raw packet dwords (immediate)
        ShaderCode vertexShader;
        ShaderCode pixelShader;
    };

    // Executes one draw against the current register state. Copy draws (RB_MODECONTROL
    // = copy) perform a resolve instead of rasterising.
    void SoftDraw(const DrawCall& draw);

    // Called on every swap: drops cached texture decodes (the CPU may rewrite textures).
    void SoftEndFrame();

    // Debug: when set, SoftDraw prints every vertex's exported position and interpolators.
    extern bool g_softLogVertices;
    // Debug: when set, every texture decoded is also written to tex_<address>_<format>.bmp.
    extern bool g_softDumpTextures;

    // Debug counters.
    uint64_t SoftPixelsWritten();
    uint64_t SoftResolves();
    void SoftPrintStats(); // per-stage timings since the previous call
}
