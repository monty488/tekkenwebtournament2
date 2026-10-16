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

    // The front buffer as BGRA if the backend holds it as a GPU-resolved image (guest
    // memory then is stale). False: present from memory as usual.
    bool WebGpuReadFrontBuffer(const uint32_t fetch[6], std::vector<uint32_t>& bgra, uint32_t& width, uint32_t& height);
    // Presents the front buffer straight from the GPU to the window. False: present from
    // memory (or a readback) as usual.
    bool WebGpuPresentFrontBuffer(const uint32_t fetch[6]);

    bool WebGpuActive();
    // Writes every GPU-kept resolve to guest memory and forgets them (before verification).
    void WebGpuFlushResolves();

    // Debug: per-frame timings and counters since the previous call.
    void WebGpuPrintStats();
}
