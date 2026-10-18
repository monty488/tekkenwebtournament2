// Host audio output: a single-producer/single-consumer ring of interleaved stereo floats,
// drained by WASAPI (native) or an AudioWorklet (browser).

#include "output.h"

#include "../kernel/guest.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#pragma comment(lib, "ole32.lib")
#elif defined(__EMSCRIPTEN__)
#include <emscripten.h>
#endif

namespace apu
{
    namespace
    {
        // Indices count frames and wrap freely; capacity is a power of two. The browser's
        // AudioWorklet reads this layout directly (web/public/audio-worklet.js).
        struct Ring
        {
            std::atomic<uint32_t> write{ 0 };
            std::atomic<uint32_t> read{ 0 };
            uint32_t capacity = 0; // frames
            uint32_t pad = 0;
            float data[1];         // capacity * 2
        };
        static_assert(offsetof(Ring, data) == 16);

        constexpr uint32_t kRingFrames = 8192; // ~170 ms
        Ring* g_ring = nullptr;
        bool g_started = false;

        Ring* CreateRing()
        {
            const size_t bytes = offsetof(Ring, data) + sizeof(float) * 2 * kRingFrames;
            void* memory = std::calloc(1, bytes);
            Ring* ring = new (memory) Ring();
            ring->capacity = kRingFrames;
            return ring;
        }

        // Consumer side (device thread): copies up to `frames` frames, zero-fills the rest.
        // Returns the number of frames that came from the ring.
        uint32_t Pull(float* out, uint32_t frames)
        {
            Ring& r = *g_ring;
            const uint32_t read = r.read.load(std::memory_order_relaxed);
            const uint32_t available = r.write.load(std::memory_order_acquire) - read;
            const uint32_t n = std::min(frames, available);
            for (uint32_t i = 0; i < n; i++)
            {
                const uint32_t at = (read + i) & (r.capacity - 1);
                out[i * 2] = r.data[at * 2];
                out[i * 2 + 1] = r.data[at * 2 + 1];
            }
            std::memset(out + n * 2, 0, sizeof(float) * 2 * (frames - n));
            r.read.store(read + n, std::memory_order_release);
            return n;
        }

#ifdef _WIN32
        void DeviceThread()
        {
            CoInitializeEx(nullptr, COINIT_MULTITHREADED);
            IMMDeviceEnumerator* enumerator = nullptr;
            IMMDevice* device = nullptr;
            IAudioClient* client = nullptr;
            IAudioRenderClient* render = nullptr;
            HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
            auto fail = [&](const char* what, HRESULT hr) {
                TTT2_LOG("audio: %s failed (0x%08lX); no sound", what, (unsigned long)hr);
            };
            HRESULT hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, __uuidof(IMMDeviceEnumerator),
                reinterpret_cast<void**>(&enumerator));
            if (FAILED(hr)) return fail("device enumerator", hr);
            hr = enumerator->GetDefaultAudioEndpoint(eRender, eConsole, &device);
            if (FAILED(hr)) return fail("default output device", hr);
            hr = device->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, reinterpret_cast<void**>(&client));
            if (FAILED(hr)) return fail("IAudioClient", hr);

            WAVEFORMATEX format{};
            format.wFormatTag = WAVE_FORMAT_IEEE_FLOAT;
            format.nChannels = 2;
            format.nSamplesPerSec = kOutputRate;
            format.wBitsPerSample = 32;
            format.nBlockAlign = 8;
            format.nAvgBytesPerSec = kOutputRate * 8;
            // Shared mode converts to the device's mix format and rate for us.
            const DWORD flags = AUDCLNT_STREAMFLAGS_EVENTCALLBACK | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY;
            hr = client->Initialize(AUDCLNT_SHAREMODE_SHARED, flags, 30 * 10000 /* 30 ms */, 0, &format, nullptr);
            if (FAILED(hr)) return fail("IAudioClient::Initialize", hr);
            client->SetEventHandle(event);
            UINT32 bufferFrames = 0;
            client->GetBufferSize(&bufferFrames);
            hr = client->GetService(__uuidof(IAudioRenderClient), reinterpret_cast<void**>(&render));
            if (FAILED(hr)) return fail("IAudioRenderClient", hr);
            client->Start();
            TTT2_LOG("audio: WASAPI output, %u-frame device buffer", bufferFrames);

            for (;;)
            {
                WaitForSingleObject(event, 100);
                UINT32 padding = 0;
                if (FAILED(client->GetCurrentPadding(&padding)))
                    continue;
                const UINT32 frames = bufferFrames - padding;
                if (!frames)
                    continue;
                BYTE* buffer = nullptr;
                if (FAILED(render->GetBuffer(frames, &buffer)))
                    continue;
                Pull(reinterpret_cast<float*>(buffer), frames);
                render->ReleaseBuffer(frames, 0);
            }
        }
#endif
    }

    bool OutputStart()
    {
        static std::once_flag once;
        std::call_once(once, [] {
            if (const char* v = std::getenv("TTT2_AUDIO"); v && *v == '0')
                return;
            g_ring = CreateRing();
#ifdef _WIN32
            std::thread(DeviceThread).detach();
            g_started = true;
#elif defined(__EMSCRIPTEN__)
            // The page owns the AudioContext (it needs a user gesture to start); it attaches
            // an AudioWorklet that reads this ring from wasm memory (web/public/loader.js).
            MAIN_THREAD_ASYNC_EM_ASM({
                if (Module['ttt2StartAudio'])
                    Module['ttt2StartAudio'](wasmMemory.buffer, Number($0));
            }, double(uintptr_t(g_ring)));
            g_started = true;
#endif
        });
        return g_started;
    }

    void OutputPush(const float* stereo, uint32_t frames)
    {
        if (!g_ring)
            return;
        Ring& r = *g_ring;
        const uint32_t write = r.write.load(std::memory_order_relaxed);
        const uint32_t used = write - r.read.load(std::memory_order_acquire);
        const uint32_t n = std::min(frames, r.capacity - std::min(used, r.capacity));
        for (uint32_t i = 0; i < n; i++)
        {
            const uint32_t at = (write + i) & (r.capacity - 1);
            r.data[at * 2] = stereo[i * 2];
            r.data[at * 2 + 1] = stereo[i * 2 + 1];
        }
        r.write.store(write + n, std::memory_order_release);
    }

    uint32_t OutputQueued()
    {
        if (!g_ring)
            return 0;
        return g_ring->write.load(std::memory_order_acquire) - g_ring->read.load(std::memory_order_acquire);
    }
}
