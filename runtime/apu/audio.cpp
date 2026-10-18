// Audio imports. The game's XAudio2 is statically linked and mixes in software; it hands
// finished frames to the kernel render driver registered here. The console calls the
// registered client every 256 samples (48 kHz, 6 channels); titles use that cadence as
// their audio clock (movie playback syncs to it), so the callback must run even before
// sound output exists.
//
// Submitted frames are downmixed to stereo and played through apu/output.cpp (WASAPI
// natively, an AudioWorklet in the browser). The device's clock paces the callbacks, so
// the game mixes exactly as fast as the sound plays.

#include "../kernel/guest.h"
#include "../kernel/memory.h"
#include "../kernel/threads.h"
#include "output.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <deque>
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
        // Frames this client's engine submitted (stereo, already downmixed), mixed with the
        // other clients' one per tick.
        std::mutex framesMutex;
        std::deque<std::array<float, 256 * 2>> frames;
    };
    static Client g_clients[kMaxClients];
    static std::mutex g_clientsMutex; // registration only
    static std::atomic<uint64_t> g_framesSubmitted{ 0 };

    static void MixTick();

    // Frames queued for the device that pacing aims for (~32 ms: low latency for a fighting
    // game, enough slack for a late callback).
    constexpr uint32_t kTargetQueued = 1536;

    static void RenderDriverThread()
    {
        guest::GuestEnv env = guest::CreateGuestEnv(/*cpu*/ 4);
        const bool output = OutputStart();
        using clock = std::chrono::steady_clock;
        const auto period = std::chrono::nanoseconds(1000000000ull * kFrameSamples / kSampleRate);
        auto next = clock::now() + period;
        // The device paces the callbacks while it is consuming; when it is not (no device,
        // or a browser page whose audio waits for a click) the clock alone does, so the
        // game's audio clock (movies sync to it) never stops.
        uint32_t lastQueued = 0;
        auto lastConsumed = clock::now();
        for (;;)
        {
            std::this_thread::sleep_until(next);
            next += period;
            if (clock::now() > next + period * 8)
                next = clock::now() + period; // fell far behind (debugger, load spike): resync

            int calls = 1;
            if (output)
            {
                const uint32_t queued = OutputQueued();
                if (queued < lastQueued || queued == 0)
                    lastConsumed = clock::now();
                lastQueued = queued;
                if (clock::now() - lastConsumed < std::chrono::milliseconds(200))
                {
                    if (queued > kTargetQueued + kFrameSamples)
                        calls = 0; // ahead of the device: skip a beat
                    else if (queued + kFrameSamples < kTargetQueued)
                        calls = 2; // behind: catch up
                }
            }
            for (int call = 0; call < calls; call++)
            {
                for (Client& client : g_clients)
                    if (const uint32_t callback = client.callback.load())
                        guest::CallGuest(env, callback, { client.paramPtr });
                MixTick();
            }
        }
    }

    // TTT2_AUDIO_DUMP=<file.wav>: also writes the stereo output to a WAV file.
    static void DumpWav(const float* stereo, uint32_t frames)
    {
        static FILE* file = [] {
            const char* path = std::getenv("TTT2_AUDIO_DUMP");
            FILE* f = path ? std::fopen(path, "wb") : nullptr;
            if (f)
            {
                const uint8_t header[44] = {}; // written properly as the file grows
                std::fwrite(header, 1, sizeof(header), f);
            }
            return f;
        }();
        if (!file)
            return;
        static uint32_t dataBytes = 0;
        int16_t pcm[kFrameSamples * 2];
        for (uint32_t i = 0; i < frames * 2; i++)
            pcm[i] = int16_t(std::clamp(stereo[i], -1.0f, 1.0f) * 32767.0f);
        std::fwrite(pcm, sizeof(int16_t), frames * 2, file);
        dataBytes += frames * 4;
        // Rewrite the header every second so the file is valid whenever the game stops.
        if (dataBytes % (kSampleRate * 4) < frames * 4)
        {
            auto put32 = [&](uint32_t v) { std::fwrite(&v, 4, 1, file); };
            auto put16 = [&](uint16_t v) { std::fwrite(&v, 2, 1, file); };
            std::fseek(file, 0, SEEK_SET);
            std::fwrite("RIFF", 1, 4, file); put32(36 + dataBytes);
            std::fwrite("WAVEfmt ", 1, 8, file); put32(16); put16(1); put16(2); put32(kSampleRate); put32(kSampleRate * 4); put16(4); put16(16);
            std::fwrite("data", 1, 4, file); put32(dataBytes);
            std::fseek(file, 0, SEEK_END);
            std::fflush(file);
        }
    }

    // Sums one frame from each client that has one and sends it out.
    static void MixTick()
    {
        float mix[kFrameSamples * 2] = {};
        bool any = false;
        for (Client& client : g_clients)
        {
            std::lock_guard lock(client.framesMutex);
            if (client.frames.empty())
                continue;
            for (uint32_t i = 0; i < kFrameSamples * 2; i++)
                mix[i] += client.frames.front()[i];
            client.frames.pop_front();
            any = true;
        }
        if (!any)
            return;
        OutputPush(mix, kFrameSamples);
        DumpWav(mix, kFrameSamples);
    }

    // A frame is 6 planar channels of 256 big-endian floats: FL, FR, FC, LFE, SL, SR.
    static void SubmitFrame(uint32_t driver, uint32_t samples)
    {
        const uint32_t index = driver & 0xFFFF;
        if ((driver & 0xFFFF0000) != kDriverHandle || index >= kMaxClients)
            return;
        std::array<float, kFrameSamples * 2> stereo;
        auto channel = [&](uint32_t c, uint32_t i) { return guest::Load<float>(samples + (c * kFrameSamples + i) * 4); };
        constexpr float kCentre = 0.7071f, kSurround = 0.7071f;
        for (uint32_t i = 0; i < kFrameSamples; i++)
        {
            const float c = channel(2, i) * kCentre;
            stereo[i * 2] = channel(0, i) + c + channel(4, i) * kSurround;
            stereo[i * 2 + 1] = channel(1, i) + c + channel(5, i) * kSurround;
        }
        Client& client = g_clients[index];
        std::lock_guard lock(client.framesMutex);
        if (client.frames.size() < 8) // a client far ahead of the mixer drops frames
            client.frames.push_back(stereo);
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
    if (ctx.r4.u32)
        apu::SubmitFrame(ctx.r3.u32, ctx.r4.u32);
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
