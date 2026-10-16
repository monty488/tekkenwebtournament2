#pragma once

// Translates a Xenos pixel shader, with its control flow already flattened for one draw
// (soft_gpu.cpp's CompileFlow), into a WGSL module for the WebGPU backend. The module
// also carries a pass-through vertex stage: vertices arrive already shaded, clipped and
// in clip space from the software front end.
//
// Semantics mirror the software interpreter (Executor in soft_gpu.cpp), including the
// Xenos rule that 0 * anything = 0 in multiplies.

#include "soft_internal.h"

#include <cstdint>
#include <string>

namespace gpu
{
    struct WgslOptions
    {
        uint32_t interpolatorMask = 0; // interpolators the vertex stage provides
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
        uint32_t colorExports = 0;   // colour exports written by the shader
        bool depthExport = false;
    };

    // Bindings: group 0 binding 0 = DrawData (dynamic offset): misc vec4 (x = alpha ref)
    // followed by the 256 pixel shader constants. Group 1 = textures and samplers.
    // `ps` may be null (no pixel shader): the fragment stage then writes nothing.
    WgslShader TranslateToWgsl(const CompiledShader* ps, const WgslOptions& options);
}
