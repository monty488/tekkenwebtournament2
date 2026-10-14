// Video imports. Rendering itself is not implemented yet; see docs/ARCHITECTURE.md
// ("Graphics") for the plan. VdSwap is counted so the shell can show whether the
// game has reached its render loop, which is the first big bring-up milestone.

#include "video.h"
#include "../kernel/guest.h"

#include <atomic>

namespace gpu
{
    static std::atomic<uint64_t> g_frames{ 0 };

    uint64_t PresentedFrames() { return g_frames.load(); }
}

using namespace guest;

// void VdQueryVideoMode(XVIDEO_MODE* mode)
GUEST_IMPORT(VdQueryVideoMode)
{
    auto* mode = ToHost<XVIDEO_MODE>(ctx.r3.u32);
    *mode = {};
    mode->DisplayWidth = 1280;
    mode->DisplayHeight = 720;
    mode->IsInterlaced = 0;
    mode->IsWidescreen = 1;
    mode->IsHighDefinition = 1;
    mode->RefreshRate = 0x42700000; // 60.0f as raw bits
    mode->VideoStandard = 1;        // NTSC
    mode->Unknown4A = 0x4A;
    mode->Unknown01 = 0x01;
}

GUEST_IMPORT(VdSwap)
{
    const uint64_t frame = gpu::g_frames.fetch_add(1) + 1;
    if ((frame & (frame - 1)) == 0) // log 1, 2, 4, 8, ... so the log stays readable
        TTT2_LOG("VdSwap: frame %llu", (unsigned long long)frame);
    ctx.r3.u64 = 0;
}
