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
#include <mutex>
#include <thread>

namespace apu
{
    // Driver handles are kDriverHandle | client index, as in Xenia.
    constexpr uint32_t kDriverHandle = 0x41550000;
    constexpr uint32_t kMaxClients = 8;
    constexpr uint32_t kSampleRate = 48000;
    constexpr uint32_t kFrameSamples = 256;

    // Each XAudio2 engine registers its own client (TTT2 creates a second engine when
    // leaving the intro movie), and every registered client must keep getting called: an
    // engine whose callback stops never processes its voices again.
    struct Client
    {
        std::atomic<uint32_t> callback{ 0 };
        // The callback receives a pointer to a guest dword holding the registered parameter,
        // not the parameter itself (as on the console and in Xenia); TTT2's XAudio2 callback
        // dereferences it to find its engine object.
        uint32_t paramPtr = 0;
    };
    static Client g_clients[kMaxClients];
    static std::mutex g_clientsMutex; // registration only
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
            for (Client& client : g_clients)
                if (const uint32_t callback = client.callback.load())
                    guest::CallGuest(env, callback, { client.paramPtr });
        }
    }
}

using namespace guest;

// u32 XAudioRegisterRenderDriverClient(u32* callbackInfo /* {func, param} */, u32* driver)
GUEST_IMPORT(XAudioRegisterRenderDriverClient)
{
    std::lock_guard lock(apu::g_clientsMutex);
    for (uint32_t index = 0; index < apu::kMaxClients; ++index)
    {
        apu::Client& client = apu::g_clients[index];
        if (client.callback.load())
            continue;
        if (!client.paramPtr)
            client.paramPtr = AllocVirtual(4);
        Store<uint32_t>(client.paramPtr, Load<uint32_t>(ctx.r3.u32 + 4));
        client.callback = Load<uint32_t>(ctx.r3.u32 + 0);
        Store<uint32_t>(ctx.r4.u32, apu::kDriverHandle | index);
        TTT2_LOG("audio client %u registered: callback 0x%08X", index, client.callback.load());
        static std::once_flag started;
        std::call_once(started, [] { std::thread(apu::RenderDriverThread).detach(); });
        ctx.r3.u64 = 0;
        return;
    }
    TTT2_LOG("XAudioRegisterRenderDriverClient: out of clients");
    ctx.r3.u64 = 0x8007000E; // E_OUTOFMEMORY
}

// u32 XAudioUnregisterRenderDriverClient(u32 driver)
GUEST_IMPORT(XAudioUnregisterRenderDriverClient)
{
    std::lock_guard lock(apu::g_clientsMutex);
    const uint32_t index = ctx.r3.u32 & 0xFFFF;
    if ((ctx.r3.u32 & 0xFFFF0000) == apu::kDriverHandle && index < apu::kMaxClients)
        apu::g_clients[index].callback = 0;
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
