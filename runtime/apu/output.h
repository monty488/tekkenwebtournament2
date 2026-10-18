#pragma once

// Host audio output. The render driver hands over 48 kHz stereo frames; they go through a
// lock-free ring buffer to the platform's audio device: WASAPI natively, an AudioWorklet in
// the browser (which reads the ring straight out of wasm memory).

#include <cstdint>

namespace apu
{
    constexpr uint32_t kOutputRate = 48000;

    // Starts the device (once). Returns false when there is no output (the caller then
    // paces itself by the clock alone).
    bool OutputStart();

    // Appends interleaved stereo samples. Drops what does not fit.
    void OutputPush(const float* stereo, uint32_t frames);

    // Stereo frames queued and not yet played (0 without a device).
    uint32_t OutputQueued();
}
