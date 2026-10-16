#include "texture.h"
#include "../kernel/guest.h"

#include <cstring>
#include <utility>

namespace gpu
{
    enum : uint32_t
    {
        k_8 = 2, k_1_5_5_5 = 3, k_5_6_5 = 4, k_8_8_8_8 = 6, k_8_8 = 10, k_8_8_8_8_A = 14, k_4_4_4_4 = 15,
        k_DXT1 = 18, k_DXT2_3 = 19, k_DXT4_5 = 20,
        k_8_8_8_8_AS_16_16_16_16 = 50, k_DXT1_AS_16_16_16_16 = 51, k_DXT2_3_AS_16_16_16_16 = 52, k_DXT4_5_AS_16_16_16_16 = 53,
    };

    const char* FormatName(uint32_t format)
    {
        switch (format)
        {
        case k_8: return "8";
        case k_1_5_5_5: return "1_5_5_5";
        case k_5_6_5: return "5_6_5";
        case k_8_8_8_8: case k_8_8_8_8_A: case k_8_8_8_8_AS_16_16_16_16: return "8_8_8_8";
        case k_8_8: return "8_8";
        case k_4_4_4_4: return "4_4_4_4";
        case k_DXT1: case k_DXT1_AS_16_16_16_16: return "DXT1";
        case k_DXT2_3: case k_DXT2_3_AS_16_16_16_16: return "DXT2_3";
        case k_DXT4_5: case k_DXT4_5_AS_16_16_16_16: return "DXT4_5";
        default: return "?";
        }
    }

    TextureInfo DecodeTextureFetch(const uint32_t fetch[6])
    {
        TextureInfo info;
        info.tiled = (fetch[0] >> 31) & 1;
        info.pitch = ((fetch[0] >> 22) & 0x1FF) * 32;
        info.format = fetch[1] & 0x3F;
        info.endian = (fetch[1] >> 6) & 3;
        info.base = 0xA0000000u + ((fetch[1] & ~0xFFFu) & 0x1FFFFFFF);
        info.width = (fetch[2] & 0x1FFF) + 1;
        info.height = ((fetch[2] >> 13) & 0x1FFF) + 1;
        return info;
    }

    // Xenos 2D tiling (as in Xenia's texture conversion / crunch's X360 untiling).
    // `pitch` is in blocks and must be a multiple of 32.
    uint32_t TiledOffset2D(uint32_t x, uint32_t y, uint32_t pitch, uint32_t log2Bpb)
    {
        const uint32_t macroOuter = ((y >> 5) * (pitch >> 5)) << (log2Bpb + 7);
        const uint32_t microOuter = ((y & 6) << 2) << log2Bpb;
        const uint32_t outer = macroOuter + ((microOuter & ~0xFu) << 1) + (microOuter & 0xF) + ((y & 8) << (3 + log2Bpb)) + ((y & 1) << 4);

        const uint32_t macro = (x >> 5) << (log2Bpb + 7);
        const uint32_t micro = (x & 7) << log2Bpb;
        const uint32_t offset = outer + (macro + ((micro & ~0xFu) << 1) + (micro & 0xF));
        const uint32_t inner = ((offset & ~0x1FFu) << 3) + ((offset & 0x1C0) << 2) + (offset & 0x3F) +
            ((y & 16) << 7) + (((((y & 8) >> 2) + (x >> 3)) & 3) << 6);
        return (inner >> log2Bpb) << log2Bpb;
    }

    // Copies one block and undoes the GPU endian swap so it reads as little-endian data.
    static void ReadBlock(const uint8_t* src, uint32_t bytes, uint32_t endian, uint8_t* dst)
    {
        std::memcpy(dst, src, bytes);
        switch (endian)
        {
        case 1: // 8in16
            for (uint32_t i = 0; i + 1 < bytes; i += 2)
                std::swap(dst[i], dst[i + 1]);
            break;
        case 2: // 8in32
            for (uint32_t i = 0; i + 3 < bytes; i += 4)
            {
                std::swap(dst[i], dst[i + 3]);
                std::swap(dst[i + 1], dst[i + 2]);
            }
            break;
        case 3: // 16in32
            for (uint32_t i = 0; i + 3 < bytes; i += 4)
            {
                std::swap(dst[i], dst[i + 2]);
                std::swap(dst[i + 1], dst[i + 3]);
            }
            break;
        default:
            break;
        }
    }

    static uint32_t Bgra(uint32_t r, uint32_t g, uint32_t b, uint32_t a)
    {
        return (a << 24) | (r << 16) | (g << 8) | b;
    }

    static uint32_t From565(uint16_t c)
    {
        const uint32_t r = (c >> 11) & 31, g = (c >> 5) & 63, b = c & 31;
        return Bgra((r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2), 255);
    }

    // DXT colour block (8 bytes, little-endian after ReadBlock) -> 16 BGRA texels.
    static void DecodeColorBlock(const uint8_t* b, bool dxt1, uint32_t out[16])
    {
        const uint16_t c0 = uint16_t(b[0] | (b[1] << 8));
        const uint16_t c1 = uint16_t(b[2] | (b[3] << 8));
        uint32_t palette[4] = { From565(c0), From565(c1), 0, 0 };
        auto mix = [](uint32_t a, uint32_t b, int wa, int wb, int div) {
            uint32_t r = 0;
            for (int s = 0; s < 24; s += 8)
                r |= ((((a >> s) & 0xFF) * wa + ((b >> s) & 0xFF) * wb) / div) << s;
            return r | 0xFF000000u;
        };
        if (!dxt1 || c0 > c1)
        {
            palette[2] = mix(palette[0], palette[1], 2, 1, 3);
            palette[3] = mix(palette[0], palette[1], 1, 2, 3);
        }
        else
        {
            palette[2] = mix(palette[0], palette[1], 1, 1, 2);
            palette[3] = 0; // transparent black
        }
        const uint32_t indices = uint32_t(b[4] | (b[5] << 8) | (b[6] << 16) | (uint32_t(b[7]) << 24));
        for (int i = 0; i < 16; i++)
            out[i] = palette[(indices >> (2 * i)) & 3];
    }

    bool ConvertToBgra(const TextureInfo& info, std::vector<uint32_t>& out, const ParallelFor& parallel)
    {
        uint32_t blockW = 1, blockBytes = 4;
        switch (info.format)
        {
        case k_8_8_8_8: case k_8_8_8_8_A: case k_8_8_8_8_AS_16_16_16_16: blockBytes = 4; break;
        case k_5_6_5: case k_1_5_5_5: case k_4_4_4_4: case k_8_8: blockBytes = 2; break;
        case k_8: blockBytes = 1; break;
        case k_DXT1: case k_DXT1_AS_16_16_16_16: blockW = 4; blockBytes = 8; break;
        case k_DXT2_3: case k_DXT4_5: case k_DXT2_3_AS_16_16_16_16: case k_DXT4_5_AS_16_16_16_16: blockW = 4; blockBytes = 16; break;
        default: return false;
        }

        const uint32_t log2Bpb = blockBytes == 1 ? 0 : blockBytes == 2 ? 1 : blockBytes == 4 ? 2 : blockBytes == 8 ? 3 : 4;
        const uint32_t blocksX = (info.width + blockW - 1) / blockW;
        const uint32_t blocksY = (info.height + blockW - 1) / blockW;
        uint32_t pitchBlocks = (info.pitch ? info.pitch : info.width) / blockW;
        if (info.tiled)
            pitchBlocks = (pitchBlocks + 31) & ~31u;
        if (pitchBlocks < blocksX)
            pitchBlocks = blocksX;

        out.resize(size_t(info.width) * info.height); // every texel is written below
        const uint8_t* base = guest::ToHost<const uint8_t>(info.base);

        // Single-texel blocks of 1, 2 or 4 bytes, read as one value with the endian swap
        // done arithmetically (same result as ReadBlock).
        auto texelRow = [&]<uint32_t Bytes>(uint32_t by) {
            uint32_t* row = out.data() + size_t(by) * info.width;
            for (uint32_t bx = 0; bx < blocksX; bx++)
            {
                const uint32_t offset = info.tiled ? TiledOffset2D(bx, by, pitchBlocks, log2Bpb) : (by * pitchBlocks + bx) * Bytes;
                const uint8_t* p = base + offset;
                uint32_t c;
                if constexpr (Bytes == 4)
                {
                    uint32_t v;
                    std::memcpy(&v, p, 4);
                    switch (info.endian)
                    {
                    case 1: v = ((v & 0x00FF00FFu) << 8) | ((v >> 8) & 0x00FF00FFu); break;
                    case 2: v = (v >> 24) | ((v >> 8) & 0xFF00u) | ((v << 8) & 0xFF0000u) | (v << 24); break;
                    case 3: v = (v << 16) | (v >> 16); break;
                    default: break;
                    }
                    c = v;
                }
                else if constexpr (Bytes == 2)
                {
                    uint16_t v;
                    std::memcpy(&v, p, 2);
                    if (info.endian == 1)
                        v = uint16_t((v >> 8) | (v << 8));
                    if (info.format == k_5_6_5) c = From565(v);
                    else if (info.format == k_4_4_4_4) c = Bgra(((v >> 8) & 15) * 17, ((v >> 4) & 15) * 17, (v & 15) * 17, ((v >> 12) & 15) * 17);
                    else if (info.format == k_8_8) c = Bgra(v & 0xFF, v & 0xFF, v & 0xFF, v >> 8);
                    else c = Bgra(((v >> 10) & 31) << 3, ((v >> 5) & 31) << 3, (v & 31) << 3, (v >> 15) ? 255 : 0);
                }
                else
                {
                    c = Bgra(p[0], p[0], p[0], 255);
                }
                if (bx < info.width && by < info.height)
                    row[bx] = c;
            }
        };

        auto decodeRow = [&](uint32_t by) {
            if (blockW == 1)
            {
                if (blockBytes == 4) texelRow.template operator()<4>(by);
                else if (blockBytes == 2) texelRow.template operator()<2>(by);
                else texelRow.template operator()<1>(by);
                return;
            }
            uint8_t block[16];
            uint32_t texels[16];
            for (uint32_t bx = 0; bx < blocksX; bx++)
            {
                const uint32_t offset = info.tiled ? TiledOffset2D(bx, by, pitchBlocks, log2Bpb) : (by * pitchBlocks + bx) * blockBytes;
                ReadBlock(base + offset, blockBytes, info.endian, block);

                if (blockW == 1)
                {
                    uint32_t c = 0;
                    switch (blockBytes)
                    {
                    case 4:
                    {
                        // Xenos 8_8_8_8 holds X in the low byte; D3D's ARGB maps X->B, Y->G, Z->R, W->A.
                        const uint32_t v = uint32_t(block[0] | (block[1] << 8) | (block[2] << 16) | (uint32_t(block[3]) << 24));
                        c = v;
                        break;
                    }
                    case 2:
                    {
                        const uint16_t v = uint16_t(block[0] | (block[1] << 8));
                        if (info.format == k_5_6_5) c = From565(v);
                        else if (info.format == k_4_4_4_4) c = Bgra(((v >> 8) & 15) * 17, ((v >> 4) & 15) * 17, (v & 15) * 17, ((v >> 12) & 15) * 17);
                        else if (info.format == k_8_8) c = Bgra(v & 0xFF, v & 0xFF, v & 0xFF, v >> 8);
                        else c = Bgra(((v >> 10) & 31) << 3, ((v >> 5) & 31) << 3, (v & 31) << 3, (v >> 15) ? 255 : 0);
                        break;
                    }
                    default:
                        c = Bgra(block[0], block[0], block[0], 255);
                        break;
                    }
                    if (bx < info.width && by < info.height)
                        out[size_t(by) * info.width + bx] = c;
                    continue;
                }

                // DXT: colour block is last 8 bytes for DXT2-5, alpha first.
                const bool dxt1 = blockBytes == 8;
                DecodeColorBlock(dxt1 ? block : block + 8, dxt1, texels);
                if (!dxt1)
                {
                    if (info.format == k_DXT2_3 || info.format == k_DXT2_3_AS_16_16_16_16)
                    {
                        for (int i = 0; i < 16; i++)
                        {
                            const uint32_t a = (block[i / 2] >> ((i & 1) * 4)) & 15;
                            texels[i] = (texels[i] & 0xFFFFFF) | ((a * 17) << 24);
                        }
                    }
                    else
                    {
                        const uint32_t a0 = block[0], a1 = block[1];
                        uint32_t alphas[8] = { a0, a1 };
                        for (int i = 2; i < 8; i++)
                            alphas[i] = a0 > a1 ? ((8 - i) * a0 + (i - 1) * a1) / 7 : (i < 6 ? ((6 - i) * a0 + (i - 1) * a1) / 5 : (i == 6 ? 0 : 255));
                        uint64_t bits = 0;
                        for (int i = 0; i < 6; i++)
                            bits |= uint64_t(block[2 + i]) << (8 * i);
                        for (int i = 0; i < 16; i++)
                            texels[i] = (texels[i] & 0xFFFFFF) | (alphas[(bits >> (3 * i)) & 7] << 24);
                    }
                }
                for (uint32_t ty = 0; ty < 4; ty++)
                    for (uint32_t tx = 0; tx < 4; tx++)
                    {
                        const uint32_t x = bx * 4 + tx, y = by * 4 + ty;
                        if (x < info.width && y < info.height)
                            out[size_t(y) * info.width + x] = texels[ty * 4 + tx];
                    }
            }
        };
        if (parallel)
            parallel(blocksY, decodeRow);
        else
            for (uint32_t by = 0; by < blocksY; by++)
                decodeRow(by);
        return true;
    }
}
