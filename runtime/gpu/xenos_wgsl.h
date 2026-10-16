#pragma once

// Translates Xenos shaders, with control flow already flattened for one draw
// (soft_gpu.cpp's CompileFlow), into one WGSL module per vertex/pixel shader pair for
// the WebGPU backend.
//
// The vertex stage is either the translated vertex shader (vertex data fetched from
// guest memory uploaded to a storage buffer, then the Xenos viewport transform) or, when
// `vs` is null, a pass-through for vertices the software front end already shaded.
//
// Semantics mirror the software interpreter (Executor in soft_gpu.cpp), including the
// Xenos rule that 0 * anything = 0 in multiplies.

#include "soft_internal.h"

#include <cstdint>
#include <string>

namespace gpu
{
    // Per-draw data shared by every shader (group 0, binding 0, dynamic offset):
    //   misc  vec4f       x = alpha reference
    //   xf    vec4f x 3   viewport transform for translated vertex shaders:
    //                     [0] = (x scale, x offset, y scale, y offset) in pixels, window
    //                           offset and pixel-centre shift included
    //                     [1] = (z scale, z offset, target width, target height)
    //                     [2] = (VTX_XY_FMT, VTX_Z_FMT, VTX_W0_FMT, 0) as 0/1
    //   c     vec4f x 512 ALU constants (vertex 0-255, pixel 256-511)
    //   vf    vec4u x 96  vertex fetch slots: (word offset in M, endian, 0, 0)
    // Group 0 binding 1 = M, guest vertex memory (array<u32>, raw bytes).
    constexpr uint32_t kDrawDataBytes = 16 + 48 + 512 * 16 + 96 * 16;
    extern const char* const kWgslDrawData;

    struct WgslOptions
    {
        uint32_t interpolatorMask = 0; // pass-through vertex stage: interpolators provided
        bool paramGen = false;         // SQ_PROGRAM_CNTL.PARAM_GEN: pixel position in a GPR
        uint32_t paramGenRegister = 0;
        uint32_t alphaFunc = 7;        // alpha test (7 = always passes)
        uint32_t colorTargets = 0;     // bit per bound render target (outputs @location(k))
        // Fetch-constant swizzle to apply after sampling, per slot (3 bits per component:
        // 0-3 = component, 4 = 0, 5 = 1). Identity (0x688) for textures decoded on the CPU,
        // which already have it applied; GPU-resolved images need it here.
        uint16_t slotSwizzle[32] = { 0x688, 0x688, 0x688, 0x688, 0x688, 0x688, 0x688, 0x688, 0x688, 0x688, 0x688,
            0x688, 0x688, 0x688, 0x688, 0x688, 0x688, 0x688, 0x688, 0x688, 0x688, 0x688, 0x688, 0x688, 0x688, 0x688,
            0x688, 0x688, 0x688, 0x688, 0x688, 0x688 };
    };

    struct WgslShader
    {
        std::string code;
        uint32_t textureSlots = 0;   // fetch constant slots sampled (bindings 2k, 2k+1 in group 1)
        uint32_t colorExports = 0;   // colour exports written by the pixel shader
        bool depthExport = false;
        uint32_t interpolators = 0;  // interpolators passed from the vertex to the pixel stage
        uint64_t vertexSlots[2] = {}; // vertex fetch slots (0-95) the vertex shader reads
        // Per slot: dwords per vertex index and the dword range read relative to
        // index * stride (so a draw uploads only the vertices its indices reach).
        struct SlotRange { uint32_t stride = 0; int32_t first = INT32_MAX; int32_t end = INT32_MIN; };
        SlotRange slotRange[96];
    };

    // `vs` null: pass-through vertex stage (vertex buffer: position + interpolators in
    // options.interpolatorMask). `ps` null: the fragment stage writes nothing.
    WgslShader TranslateToWgsl(const CompiledShader* vs, const CompiledShader* ps, const WgslOptions& options);
}
