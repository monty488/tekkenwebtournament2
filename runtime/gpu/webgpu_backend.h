#pragma once

// WebGPU backend (TTT2_GPU=webgpu). The software renderer's front end shades vertices,
// assembles, clips and culls primitives; this backend rasterises them on the GPU with
// the pixel shader translated to WGSL, keeps render targets in GPU textures keyed by
// their EDRAM location, and reads them back for resolves.

#include "soft_internal.h"

namespace gpu
{
    // Creates the device and returns the sink to install with SoftSetCaptureSink, or
    // null if no WebGPU adapter is available.
    CaptureSink* CreateWebGpuBackend();

    // Submits the frame's outstanding GPU work (called on every swap).
    void WebGpuEndFrame();

    // Debug: per-frame timings and counters since the previous call.
    void WebGpuPrintStats();
}
