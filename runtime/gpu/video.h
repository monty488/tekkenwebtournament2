#pragma once

#include <cstdint>

namespace gpu
{
    // Number of VdSwap calls so far (frames the game has tried to present).
    uint64_t PresentedFrames();
}
