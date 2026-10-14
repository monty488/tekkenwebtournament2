// Controller input -> XamInputGetState.
//
// Web:    Gamepad API (standard mapping) + a keyboard layout for playing without a pad.
// Native: XInput.
//
// Default Tekken layout on Xbox 360: X = Left Punch, Y = Right Punch,
// A = Left Kick, B = Right Kick, RB = Tag.

#include "input.h"
#include "../kernel/guest.h"

#include <atomic>
#include <cstring>
#include <mutex>

#if defined(__EMSCRIPTEN__)
#include <emscripten/html5.h>
#elif defined(_WIN32)
#include <windows.h>
#include <xinput.h>
#pragma comment(lib, "xinput.lib")
#endif

namespace hid
{
    struct PadState
    {
        uint16_t buttons = 0;
        uint8_t leftTrigger = 0, rightTrigger = 0;
        int16_t lx = 0, ly = 0, rx = 0, ry = 0;
        bool connected = false;
    };

#if defined(__EMSCRIPTEN__)
    // Keyboard state is written by callbacks on the browser main thread and read
    // by guest threads, hence the atomic bitmask.
    static std::atomic<uint16_t> g_keyboardButtons{ 0 };

    static uint16_t KeyToButton(const char* code)
    {
        struct Mapping { const char* code; uint16_t button; };
        static constexpr Mapping kMap[] =
        {
            { "KeyW", XAMINPUT_GAMEPAD_DPAD_UP },  { "KeyS", XAMINPUT_GAMEPAD_DPAD_DOWN },
            { "KeyA", XAMINPUT_GAMEPAD_DPAD_LEFT }, { "KeyD", XAMINPUT_GAMEPAD_DPAD_RIGHT },
            { "KeyU", XAMINPUT_GAMEPAD_X }, // Left Punch
            { "KeyI", XAMINPUT_GAMEPAD_Y }, // Right Punch
            { "KeyJ", XAMINPUT_GAMEPAD_A }, // Left Kick
            { "KeyK", XAMINPUT_GAMEPAD_B }, // Right Kick
            { "KeyO", XAMINPUT_GAMEPAD_RIGHT_SHOULDER }, // Tag
            { "KeyL", XAMINPUT_GAMEPAD_LEFT_SHOULDER },
            { "Enter", XAMINPUT_GAMEPAD_START }, { "Backspace", XAMINPUT_GAMEPAD_BACK },
            { "ArrowUp", XAMINPUT_GAMEPAD_DPAD_UP },  { "ArrowDown", XAMINPUT_GAMEPAD_DPAD_DOWN },
            { "ArrowLeft", XAMINPUT_GAMEPAD_DPAD_LEFT }, { "ArrowRight", XAMINPUT_GAMEPAD_DPAD_RIGHT },
        };
        for (const auto& m : kMap)
            if (std::strcmp(code, m.code) == 0)
                return m.button;
        return 0;
    }

    static EM_BOOL OnKey(int type, const EmscriptenKeyboardEvent* e, void*)
    {
        const uint16_t button = KeyToButton(e->code);
        if (!button)
            return EM_FALSE;
        if (type == EMSCRIPTEN_EVENT_KEYDOWN)
            g_keyboardButtons.fetch_or(button);
        else
            g_keyboardButtons.fetch_and(uint16_t(~button));
        return EM_TRUE;
    }

    void Init()
    {
        // Run on the browser main thread: guest threads never return to an event loop.
        emscripten_set_keydown_callback_on_thread(EMSCRIPTEN_EVENT_TARGET_WINDOW, nullptr, true, OnKey,
            EM_CALLBACK_THREAD_CONTEXT_MAIN_RUNTIME_THREAD);
        emscripten_set_keyup_callback_on_thread(EMSCRIPTEN_EVENT_TARGET_WINDOW, nullptr, true, OnKey,
            EM_CALLBACK_THREAD_CONTEXT_MAIN_RUNTIME_THREAD);
    }

    static int16_t Axis(double v) { return int16_t(v * 32767.0); }

    PadState Poll(uint32_t user)
    {
        PadState pad;
        if (user == 0)
        {
            pad.buttons = g_keyboardButtons.load();
            pad.connected = true; // keyboard always counts as player 1
        }

        static std::mutex sampleMutex;
        std::lock_guard lock(sampleMutex);
        if (emscripten_sample_gamepad_data() != EMSCRIPTEN_RESULT_SUCCESS)
            return pad;

        EmscriptenGamepadEvent gp;
        if (int(user) >= emscripten_get_num_gamepads() ||
            emscripten_get_gamepad_status(int(user), &gp) != EMSCRIPTEN_RESULT_SUCCESS || !gp.connected)
            return pad;

        // W3C "standard" mapping indices.
        static constexpr uint16_t kButtons[17] =
        {
            XAMINPUT_GAMEPAD_A, XAMINPUT_GAMEPAD_B, XAMINPUT_GAMEPAD_X, XAMINPUT_GAMEPAD_Y,
            XAMINPUT_GAMEPAD_LEFT_SHOULDER, XAMINPUT_GAMEPAD_RIGHT_SHOULDER, 0, 0,
            XAMINPUT_GAMEPAD_BACK, XAMINPUT_GAMEPAD_START,
            XAMINPUT_GAMEPAD_LEFT_THUMB, XAMINPUT_GAMEPAD_RIGHT_THUMB,
            XAMINPUT_GAMEPAD_DPAD_UP, XAMINPUT_GAMEPAD_DPAD_DOWN,
            XAMINPUT_GAMEPAD_DPAD_LEFT, XAMINPUT_GAMEPAD_DPAD_RIGHT, 0,
        };
        for (int i = 0; i < gp.numButtons && i < 17; ++i)
            if (gp.digitalButton[i])
                pad.buttons |= kButtons[i];

        pad.leftTrigger = uint8_t(gp.analogButton[6] * 255.0);
        pad.rightTrigger = uint8_t(gp.analogButton[7] * 255.0);
        if (gp.numAxes >= 4)
        {
            pad.lx = Axis(gp.axis[0]); pad.ly = Axis(-gp.axis[1]);
            pad.rx = Axis(gp.axis[2]); pad.ry = Axis(-gp.axis[3]);
        }
        pad.connected = true;
        return pad;
    }
#elif defined(_WIN32)
    void Init() {}

    PadState Poll(uint32_t user)
    {
        PadState pad;
        XINPUT_STATE state{};
        if (user >= XUSER_MAX_COUNT || XInputGetState(user, &state) != ERROR_SUCCESS)
            return pad;
        // XInput's button bits match the Xbox 360's.
        pad.buttons = state.Gamepad.wButtons;
        pad.leftTrigger = state.Gamepad.bLeftTrigger;
        pad.rightTrigger = state.Gamepad.bRightTrigger;
        pad.lx = state.Gamepad.sThumbLX; pad.ly = state.Gamepad.sThumbLY;
        pad.rx = state.Gamepad.sThumbRX; pad.ry = state.Gamepad.sThumbRY;
        pad.connected = true;
        return pad;
    }
#else
    void Init() {}
    PadState Poll(uint32_t) { return {}; }
#endif
}

using namespace guest;

// u32 XamInputGetState(u32 userIndex, u32 flags, XAMINPUT_STATE* state)
GUEST_IMPORT(XamInputGetState)
{
    static std::atomic<uint32_t> packet{ 0 };
    const hid::PadState pad = hid::Poll(ctx.r3.u32);
    if (!pad.connected)
    {
        ctx.r3.u64 = ERROR_DEVICE_NOT_CONNECTED;
        return;
    }

    const uint32_t s = ctx.r5.u32;
    Store<uint32_t>(s + 0x0, packet.fetch_add(1));
    Store<uint16_t>(s + 0x4, pad.buttons);
    *ToHost<uint8_t>(s + 0x6) = pad.leftTrigger;
    *ToHost<uint8_t>(s + 0x7) = pad.rightTrigger;
    Store<int16_t>(s + 0x8, pad.lx);
    Store<int16_t>(s + 0xA, pad.ly);
    Store<int16_t>(s + 0xC, pad.rx);
    Store<int16_t>(s + 0xE, pad.ry);
    ctx.r3.u64 = ERROR_SUCCESS;
}

// u32 XamInputGetCapabilities(u32 userIndex, u32 flags, XAMINPUT_CAPABILITIES* caps)
GUEST_IMPORT(XamInputGetCapabilities)
{
    if (!hid::Poll(ctx.r3.u32).connected)
    {
        ctx.r3.u64 = ERROR_DEVICE_NOT_CONNECTED;
        return;
    }
    const uint32_t caps = ctx.r5.u32;
    std::memset(ToHost(caps), 0, sizeof(XAMINPUT_CAPABILITIES));
    *ToHost<uint8_t>(caps + 0) = 1; // XINPUT_DEVTYPE_GAMEPAD
    *ToHost<uint8_t>(caps + 1) = 1; // XINPUT_DEVSUBTYPE_GAMEPAD
    ctx.r3.u64 = ERROR_SUCCESS;
}

// u32 XamInputSetState(u32 userIndex, u32 flags, XAMINPUT_VIBRATION* vibration)
GUEST_IMPORT(XamInputSetState)
{
    // Rumble is ignored for now (Gamepad API: vibrationActuator.playEffect).
    ctx.r3.u64 = hid::Poll(ctx.r3.u32).connected ? ERROR_SUCCESS : ERROR_DEVICE_NOT_CONNECTED;
}
