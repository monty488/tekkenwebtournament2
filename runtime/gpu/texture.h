#pragma once

// Xenos texture helpers: fetch-constant decoding, tiled address math and format
// conversion to host BGRA8. Used by the presenter now and the renderer later.

#include <cstdint>
#include <functional>
#include <vector>

namespace gpu
{
    // Texture fetch constant (6 dwords at SHADER_CONSTANT_FETCH 0x4800 + slot * 6).
    struct TextureInfo
    {
        uint32_t base = 0;      // guest address of mip 0
        uint32_t width = 0, height = 0;
        uint32_t pitch = 0;     // in texels (block-aligned for compressed formats)
        uint32_t format = 0;    // xenos::TextureFormat
        uint32_t endian = 0;    // 0 none, 1 8in16, 2 8in32, 3 16in32
        bool tiled = false;
    };
    TextureInfo DecodeTextureFetch(const uint32_t fetch[6]);

    // Byte offset of texel/block (x, y) inside a tiled 2D surface of `pitch` blocks.
    uint32_t TiledOffset2D(uint32_t x, uint32_t y, uint32_t pitch, uint32_t log2BytesPerBlock);

    // Runs fn(0..count-1), possibly in parallel.
    using ParallelFor = std::function<void(uint32_t count, const std::function<void(uint32_t)>& fn)>;

    // Converts mip 0 to BGRA8 (A in the high byte). Returns false for unsupported formats.
    // Block rows are independent; `parallel`, if given, spreads them across threads.
    bool ConvertToBgra(const TextureInfo& info, std::vector<uint32_t>& out, const ParallelFor& parallel = {});

    const char* FormatName(uint32_t format);
}
