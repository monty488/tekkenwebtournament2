// Audio imports. The game's XAudio2 is statically linked and mixes in software; it hands
// finished frames to the kernel render driver registered here. The console calls the
// registered client every 256 samples (48 kHz, 6 channels); titles use that cadence as
// their audio clock (movie playback syncs to it), so the callback must run even before
// sound output exists.
//
// Output is not wired up yet (planned: Web Audio AudioWorklet fed from a ring buffer;
// see docs/ARCHITECTURE.md, "Audio").

#include "../kernel/guest.h"
#include "../kernel/memory.h"
#include "../kernel/threads.h"

#include <atomic>
#include <chrono>
#include <thread>

namespace apu
{
    constexpr uint32_t kDriverHandle = 0x41550000;
    constexpr uint32_t kSampleRate = 48000;
    constexpr uint32_t kFrameSamples = 256;

    static std::atomic<uint32_t> g_clientCallback{ 0 };
    // The client callback receives a pointer to a guest dword holding the registered
    // parameter, not the parameter itself (as on the console and in Xenia); TTT2's
    // XAudio2 callback dereferences it to find its engine object.
    static std::atomic<uint32_t> g_clientParamPtr{ 0 };
    static std::atomic<uint64_t> g_framesSubmitted{ 0 };

    static void RenderDriverThread()
    {
        guest::GuestEnv env = guest::CreateGuestEnv(/*cpu*/ 4);
        using clock = std::chrono::steady_clock;
        const auto period = std::chrono::nanoseconds(1000000000ull * kFrameSamples / kSampleRate);
        auto next = clock::now() + period;
        for (;;)
        {
            std::this_thread::sleep_until(next);
            next += period;
            if (clock::now() > next + period * 8)
                next = clock::now() + period; // fell far behind (debugger, load spike): resync
            if (const uint32_t callback = g_clientCallback.load())
                guest::CallGuest(env, callback, { g_clientParamPtr.load() });
        }
    }
}

using namespace guest;

// u32 XAudioRegisterRenderDriverClient(u32* callbackInfo /* {func, param} */, u32* driver)
GUEST_IMPORT(XAudioRegisterRenderDriverClient)
{
    static const uint32_t paramPtr = AllocVirtual(4);
    Store<uint32_t>(paramPtr, Load<uint32_t>(ctx.r3.u32 + 4));
    apu::g_clientParamPtr = paramPtr;
    apu::g_clientCallback = Load<uint32_t>(ctx.r3.u32 + 0);
    Store<uint32_t>(ctx.r4.u32, apu::kDriverHandle);
    TTT2_LOG("audio client registered: callback 0x%08X", apu::g_clientCallback.load());
    static std::once_flag started;
    std::call_once(started, [] { std::thread(apu::RenderDriverThread).detach(); });
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
    // TODO: byteswap, downmix 5.1 -> stereo, push to the output ring buffer.
    apu::g_framesSubmitted.fetch_add(1, std::memory_order_relaxed);
    ctx.r3.u64 = 0;
}

// u32 XAudioGetVoiceCategoryVolume(u32 category, float* volume)
GUEST_IMPORT(XAudioGetVoiceCategoryVolume)
{
    if (ctx.r4.u32)
        Store<float>(ctx.r4.u32, 1.0f);
    ctx.r3.u64 = 0;
}
