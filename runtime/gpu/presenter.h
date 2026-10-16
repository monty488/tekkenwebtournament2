#pragma once

#include <cstdint>

namespace gpu
{
    // Opens the output window (native) on first use and shows a BGRA image.
    // Safe to call from any thread; the latest frame wins.
    void Present(const uint32_t* bgra, uint32_t width, uint32_t height);

    // Number of frames shown so far.
    uint64_t PresentedToScreen();

    // The output window (HWND on Windows), created on first use; null where there is none.
    void* PresenterWindow();
    // While set, a GPU swap chain draws the window and the GDI path stays out of the way.
    void PresenterSetGpu(bool gpu);
    // Counts a frame the GPU presented (for PresentedToScreen).
    void PresenterCountFrame();
}
