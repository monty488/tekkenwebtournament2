#pragma once

#include <cstdint>

namespace gpu
{
    // Opens the output window (native) on first use and shows a BGRA image.
    // Safe to call from any thread; the latest frame wins.
    void Present(const uint32_t* bgra, uint32_t width, uint32_t height);

    // Number of frames shown so far.
    uint64_t PresentedToScreen();
}
