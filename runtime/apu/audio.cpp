// Audio imports. The game's XAudio2 is statically linked and mixes in software;
// it hands finished frames to the kernel render driver registered here. Output
// is not wired up yet (planned: Web Audio AudioWorklet fed from a ring buffer;
// see docs/ARCHITECTURE.md, "Audio").

#include "../kernel/guest.h"

#include <atomic>

namespace apu
{
    constexpr uint32_t kDriverHandle = 0x41550000;

    static std::atomic<uint32_t> g_clientCallback{ 0 };
    static std::atomic<uint32_t> g_clientParam{ 0 };
}

using namespace guest;

// u32 XAudioRegisterRenderDriverClient(u32* callbackInfo /* {func, param} */, u32* driver)
GUEST_IMPORT(XAudioRegisterRenderDriverClient)
{
    apu::g_clientCallback = Load<uint32_t>(ctx.r3.u32 + 0);
    apu::g_clientParam = Load<uint32_t>(ctx.r3.u32 + 4);
    Store<uint32_t>(ctx.r4.u32, apu::kDriverHandle);
    TTT2_LOG("audio client registered: callback 0x%08X", apu::g_clientCallback.load());
    ctx.r3.u64 = 0;
}

GUEST_IMPORT(XAudioUnregisterRenderDriverClient)
{
    apu::g_clientCallback = 0;
    ctx.r3.u64 = 0;
}

// u32 XAudioSubmitRenderDriverFrame(u32 driver, float* samples /* 6 ch x 256, big-endian */)
GUEST_IMPORT(XAudioSubmitRenderDriverFrame)
{
    // TODO: byteswap, downmix 5.1 -> stereo, push to the AudioWorklet ring buffer.
    ctx.r3.u64 = 0;
}
