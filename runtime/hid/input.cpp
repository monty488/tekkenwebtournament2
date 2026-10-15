// Controller input -> XamInputGetState.
//
// Web:    Gamepad API (standard mapping) + a keyboard layout for playing without a pad.
// Native: XInput.
//
// Default Tekken layout on Xbox 360: X = Left Punch, Y = Right Punch,
// A = Left Kick, B = Right Kick, RB = Tag.

#include "input.h"
#include "../kernel/guest.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <atomic>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

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

    // Testing aid: TTT2_AUTOPRESS=start|a|b taps that button for 150 ms every 2 s on player 1.
    // A comma-separated list (e.g. start,start,a) names the button for each press in turn
    // ("none" skips one); the last entry repeats.
    static uint16_t AutoPressButtons()
    {
        static const std::vector<uint16_t> buttons = [] {
            std::vector<uint16_t> list;
            for (const char* p = std::getenv("TTT2_AUTOPRESS"); p && *p;)
            {
                const char* end = std::strchr(p, ',');
                const std::string name(p, end ? size_t(end - p) : std::strlen(p));
                list.push_back(name == "start" ? XAMINPUT_GAMEPAD_START
                    : name == "a"              ? XAMINPUT_GAMEPAD_A
                    : name == "b"              ? XAMINPUT_GAMEPAD_B
                                               : 0);
                p = end ? end + 1 : nullptr;
            }
            return list;
        }();
        if (buttons.empty())
            return 0;
        // TTT2_AUTOPRESS_DELAY=<ms>: start pressing only that long after the first poll.
        // TTT2_AUTOPRESS_COUNT=<n>: stop after n presses.
        static const auto start = std::chrono::steady_clock::now();
        static const long long delay = [] { const char* v = std::getenv("TTT2_AUTOPRESS_DELAY"); return v ? std::atoll(v) : 0ll; }();
        static const long long count = [] { const char* v = std::getenv("TTT2_AUTOPRESS_COUNT"); return v ? std::atoll(v) : 0ll; }();
        const long long elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count() - delay;
        if (elapsed < 0 || (count && elapsed / 2000 >= count))
            return 0;
        const size_t press = std::min(size_t(elapsed / 2000), buttons.size() - 1);
        return (elapsed % 2000) < 150 ? buttons[press] : 0;
    }

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

    // Same layout as the web build. Read only while a window of this process has focus.
    static uint16_t KeyboardButtons()
    {
        DWORD focusPid = 0;
        GetWindowThreadProcessId(GetForegroundWindow(), &focusPid);
        if (focusPid != GetCurrentProcessId())
            return 0;

        struct Mapping { int key; uint16_t button; };
        static constexpr Mapping kMap[] =
        {
            { 'W', XAMINPUT_GAMEPAD_DPAD_UP },  { 'S', XAMINPUT_GAMEPAD_DPAD_DOWN },
            { 'A', XAMINPUT_GAMEPAD_DPAD_LEFT }, { 'D', XAMINPUT_GAMEPAD_DPAD_RIGHT },
            { VK_UP, XAMINPUT_GAMEPAD_DPAD_UP }, { VK_DOWN, XAMINPUT_GAMEPAD_DPAD_DOWN },
            { VK_LEFT, XAMINPUT_GAMEPAD_DPAD_LEFT }, { VK_RIGHT, XAMINPUT_GAMEPAD_DPAD_RIGHT },
            { 'U', XAMINPUT_GAMEPAD_X }, { 'I', XAMINPUT_GAMEPAD_Y },
            { 'J', XAMINPUT_GAMEPAD_A }, { 'K', XAMINPUT_GAMEPAD_B },
            { 'O', XAMINPUT_GAMEPAD_RIGHT_SHOULDER }, { 'L', XAMINPUT_GAMEPAD_LEFT_SHOULDER },
            { VK_RETURN, XAMINPUT_GAMEPAD_START }, { VK_BACK, XAMINPUT_GAMEPAD_BACK },
        };
        uint16_t buttons = 0;
        for (const auto& m : kMap)
            if (GetAsyncKeyState(m.key) & 0x8000)
                buttons |= m.button;
        return buttons;
    }

    PadState Poll(uint32_t user)
    {
        PadState pad;
        XINPUT_STATE state{};
        if (user < XUSER_MAX_COUNT && XInputGetState(user, &state) == ERROR_SUCCESS)
        {
            // XInput's button bits match the Xbox 360's.
            pad.buttons = state.Gamepad.wButtons;
            pad.leftTrigger = state.Gamepad.bLeftTrigger;
            pad.rightTrigger = state.Gamepad.bRightTrigger;
            pad.lx = state.Gamepad.sThumbLX; pad.ly = state.Gamepad.sThumbLY;
            pad.rx = state.Gamepad.sThumbRX; pad.ry = state.Gamepad.sThumbRY;
            pad.connected = true;
        }
        static const bool noKeyboard = [] { const char* v = std::getenv("TTT2_NO_KEYBOARD"); return v && *v == '1'; }();
        if (user == 0 && !noKeyboard)
        {
            pad.buttons |= KeyboardButtons() | AutoPressButtons();
            pad.connected = true; // the keyboard always counts as player 1
        }
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

// ---- Keystrokes ----------------------------------------------------------------
// XamInputGetKeystrokeEx reports button transitions as virtual-key events. They are
// derived here by diffing successive polls per user.

namespace hid
{
    struct Keystroke
    {
        uint16_t virtualKey;
        uint16_t flags; // 1 = key down, 2 = key up
    };

    static std::mutex g_keyMutex;
    static uint16_t g_lastButtons[4];
    static std::deque<Keystroke> g_keys[4];

    static uint16_t VirtualKeyFor(uint16_t button)
    {
        switch (button)
        {
        case XAMINPUT_GAMEPAD_A: return 0x5800;               // VK_PAD_A
        case XAMINPUT_GAMEPAD_B: return 0x5801;
        case XAMINPUT_GAMEPAD_X: return 0x5802;
        case XAMINPUT_GAMEPAD_Y: return 0x5803;
        case XAMINPUT_GAMEPAD_RIGHT_SHOULDER: return 0x5804;
        case XAMINPUT_GAMEPAD_LEFT_SHOULDER: return 0x5805;
        case XAMINPUT_GAMEPAD_DPAD_UP: return 0x5810;
        case XAMINPUT_GAMEPAD_DPAD_DOWN: return 0x5811;
        case XAMINPUT_GAMEPAD_DPAD_LEFT: return 0x5812;
        case XAMINPUT_GAMEPAD_DPAD_RIGHT: return 0x5813;
        case XAMINPUT_GAMEPAD_START: return 0x5814;
        case XAMINPUT_GAMEPAD_BACK: return 0x5815;
        case XAMINPUT_GAMEPAD_LEFT_THUMB: return 0x5816;
        case XAMINPUT_GAMEPAD_RIGHT_THUMB: return 0x5817;
        default: return 0;
        }
    }

    // Returns the next keystroke for `user`, polling the device first.
    static bool NextKeystroke(uint32_t user, Keystroke& out, bool& connected)
    {
        const PadState pad = Poll(user);
        connected = pad.connected;
        std::lock_guard lock(g_keyMutex);
        const uint16_t changed = uint16_t(pad.buttons ^ g_lastButtons[user]);
        for (uint16_t bit = 1; bit; bit = uint16_t(bit << 1))
        {
            if ((changed & bit) && VirtualKeyFor(bit))
                g_keys[user].push_back({ VirtualKeyFor(bit), uint16_t((pad.buttons & bit) ? 1 : 2) });
        }
        g_lastButtons[user] = pad.buttons;
        if (g_keys[user].empty())
            return false;
        out = g_keys[user].front();
        g_keys[user].pop_front();
        return true;
    }
}

// u32 XamInputGetKeystrokeEx(u32* userIndex /* in: 0-3 or 0xFF = any; out: user */, u32 flags,
//                            XINPUT_KEYSTROKE* keystroke)
GUEST_IMPORT(XamInputGetKeystrokeEx)
{
    constexpr uint32_t ERROR_EMPTY_ = 0x10D2;
    const uint32_t pUser = ctx.r3.u32;
    const uint32_t requested = pUser ? Load<uint32_t>(pUser) : 0xFF;
    const uint32_t keystroke = ctx.r5.u32;

    bool anyConnected = false;
    for (uint32_t user = 0; user < 4; user++)
    {
        if (requested != 0xFF && requested != user)
            continue;
        hid::Keystroke key;
        bool connected = false;
        if (hid::NextKeystroke(user, key, connected))
        {
            Store<uint16_t>(keystroke + 0, key.virtualKey);
            Store<uint16_t>(keystroke + 2, 0);          // Unicode
            Store<uint16_t>(keystroke + 4, key.flags);
            *ToHost<uint8_t>(keystroke + 6) = uint8_t(user);
            *ToHost<uint8_t>(keystroke + 7) = 0;        // HidCode
            if (pUser)
                Store<uint32_t>(pUser, user);
            ctx.r3.u64 = ERROR_SUCCESS;
            return;
        }
        anyConnected = anyConnected || connected;
    }
    ctx.r3.u64 = anyConnected ? ERROR_EMPTY_ : ERROR_DEVICE_NOT_CONNECTED;
}

// u32 XamInputGetKeystroke(u32 userIndex, u32 flags, XINPUT_KEYSTROKE* keystroke)
GUEST_IMPORT(XamInputGetKeystroke)
{
    // Same as the Ex variant with a by-value user index.
    constexpr uint32_t ERROR_EMPTY_ = 0x10D2;
    hid::Keystroke key;
    bool connected = false;
    const uint32_t user = ctx.r3.u32 == 0xFF ? 0 : ctx.r3.u32;
    if (user < 4 && hid::NextKeystroke(user, key, connected))
    {
        Store<uint16_t>(ctx.r5.u32 + 0, key.virtualKey);
        Store<uint16_t>(ctx.r5.u32 + 2, 0);
        Store<uint16_t>(ctx.r5.u32 + 4, key.flags);
        *ToHost<uint8_t>(ctx.r5.u32 + 6) = uint8_t(user);
        *ToHost<uint8_t>(ctx.r5.u32 + 7) = 0;
        ctx.r3.u64 = ERROR_SUCCESS;
        return;
    }
    ctx.r3.u64 = connected ? ERROR_EMPTY_ : ERROR_DEVICE_NOT_CONNECTED;
}

// u32 XamInputSetState(u32 userIndex, u32 flags, XAMINPUT_VIBRATION* vibration)
GUEST_IMPORT(XamInputSetState)
{
    // Rumble is ignored for now (Gamepad API: vibrationActuator.playEffect).
    ctx.r3.u64 = hid::Poll(ctx.r3.u32).connected ? ERROR_SUCCESS : ERROR_DEVICE_NOT_CONNECTED;
}
