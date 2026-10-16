// XAM (system library) and miscellaneous runtime-library imports.
//
// Profiles: the runtime presents a console with no signed-in user for now, so
// profile reads fail with ERROR_NOT_LOGGED_ON. Titles handle that state (it is the
// normal state before anyone signs in).

#include "guest.h"
#include "sync.h"

#include <cstring>
#include <deque>
#include <utility>

using namespace guest;

constexpr uint32_t ERROR_NOT_LOGGED_ON_ = 1245;

// u32 XGetGameRegion(): 0x00FF = North America.
GUEST_IMPORT(XGetGameRegion)
{
    ctx.r3.u64 = 0x00FF;
}

// u32 XAudioGetSpeakerConfig(u32* config)
GUEST_IMPORT(XAudioGetSpeakerConfig)
{
    if (ctx.r3.u32)
        Store<uint32_t>(ctx.r3.u32, 0x00010001); // stereo, as Xenia reports
    ctx.r3.u64 = 0;
}

// u32 XamUserGetSigninState(u32 userIndex): 0 = not signed in.
GUEST_IMPORT(XamUserGetSigninState)
{
    ctx.r3.u64 = 0;
}

constexpr uint32_t ERROR_NO_SUCH_USER_ = 0x525;

// u32 XamUserGetSigninInfo(u32 userIndex, u32 flags, XUSER_SIGNIN_INFO* info /* 0x28 bytes */)
// Nobody is signed in: zero the info (XUID 0 = no user) and fail, as the console does.
// Returning success without filling it made the title read a stale XUID off its stack.
GUEST_IMPORT(XamUserGetSigninInfo)
{
    if (!ctx.r5.u32)
    {
        ctx.r3.u64 = 87; // ERROR_INVALID_PARAMETER
        return;
    }
    std::memset(ToHost<uint8_t>(ctx.r5.u32), 0, 0x28);
    ctx.r3.u64 = ERROR_NO_SUCH_USER_;
}

// u32 XamUserGetName(u32 userIndex, char* buffer, u32 bufferSize)
GUEST_IMPORT(XamUserGetName)
{
    if (ctx.r4.u32 && ctx.r5.u32)
        Store<uint8_t>(ctx.r4.u32, 0);
    ctx.r3.u64 = ERROR_NO_SUCH_USER_;
}

// u32 XamUserGetXUID(u32 userIndex, u32 typeMask, u64* xuid)
GUEST_IMPORT(XamUserGetXUID)
{
    if (ctx.r5.u32)
        Store<uint64_t>(ctx.r5.u32, 0);
    ctx.r3.u64 = ERROR_NO_SUCH_USER_;
}

// ---- Notifications --------------------------------------------------------
//
// XNotify listeners queue (id, param) pairs from the areas in their mask
// (area = (id >> 25) & 0x3F). A listener handle is signalled while it has
// notifications pending. Titles that show system UI (sign-in, message boxes)
// wait for XN_SYS_UI(0), "UI closed", before going on.

namespace
{
    constexpr uint32_t XN_SYS_UI = 0x00000009;
    constexpr uint32_t XN_SYS_SIGNINCHANGED = 0x0000000A;

    struct NotifyListener : Waitable
    {
        uint64_t mask = 0;
        std::deque<std::pair<uint32_t, uint32_t>> queue; // guarded by WaitMutex()
        bool IsSignaled(uint32_t) override { return !queue.empty(); }
    };

    std::mutex g_listenersMutex;
    std::vector<std::weak_ptr<NotifyListener>> g_listeners;
    bool g_notifiedStartup = false;

    void Enqueue(NotifyListener& listener, uint32_t id, uint32_t param)
    {
        if (!(listener.mask & (1ull << ((id >> 25) & 0x3F))))
            return;
        std::lock_guard lock(WaitMutex());
        listener.queue.emplace_back(id, param);
        DeliverSignals(listener);
    }

    void Broadcast(uint32_t id, uint32_t param)
    {
        std::lock_guard lock(g_listenersMutex);
        for (auto it = g_listeners.begin(); it != g_listeners.end();)
        {
            if (auto listener = it->lock())
            {
                Enqueue(*listener, id, param);
                ++it;
            }
            else
            {
                it = g_listeners.erase(it);
            }
        }
    }
}

namespace guest
{
    void BroadcastNotification(uint32_t id, uint32_t param) { Broadcast(id, param); }
}

// u32 XamNotifyCreateListener(u64 areaMask, u32 maxVersion): returns a handle, 0 on failure.
GUEST_IMPORT(XamNotifyCreateListener)
{
    auto listener = std::make_shared<NotifyListener>();
    listener->mask = ctx.r3.u64;
    {
        std::lock_guard lock(g_listenersMutex);
        g_listeners.push_back(listener);
        // Titles expect the dashboard's startup notifications on their first system
        // listener (Xenia does the same): the UI opening and closing, sign-in changes.
        if (!g_notifiedStartup && (listener->mask & 1))
        {
            g_notifiedStartup = true;
            Enqueue(*listener, XN_SYS_UI, 1);
            Enqueue(*listener, XN_SYS_UI, 0);
            Enqueue(*listener, XN_SYS_SIGNINCHANGED, 1);
            Enqueue(*listener, XN_SYS_SIGNINCHANGED, 1);
        }
    }
    ctx.r3.u64 = Handles().Insert(listener);
}

// u32 XNotifyGetNext(u32 listener, u32 matchId, u32* id, u32* param): 1 if one was dequeued.
GUEST_IMPORT(XNotifyGetNext)
{
    const uint32_t pId = ctx.r5.u32, pParam = ctx.r6.u32;
    const uint32_t matchId = ctx.r4.u32;
    auto listener = Handles().Get<NotifyListener>(ctx.r3.u32);
    bool found = false;
    uint32_t id = 0, param = 0;
    if (listener)
    {
        std::lock_guard lock(WaitMutex());
        for (auto it = listener->queue.begin(); it != listener->queue.end(); ++it)
        {
            if (matchId && it->first != matchId)
                continue;
            id = it->first;
            param = it->second;
            listener->queue.erase(it);
            found = true;
            break;
        }
    }
    if (pId)
        Store<uint32_t>(pId, id);
    if (pParam)
        Store<uint32_t>(pParam, param);
    ctx.r3.u64 = found ? 1 : 0;
}

// u32 XamShowSigninUI(u32 paneCount, u32 flags): there are no profiles to pick, so the
// UI opens and closes at once and nobody is signed in afterwards.
GUEST_IMPORT(XamShowSigninUI)
{
    Broadcast(XN_SYS_UI, 1);
    Broadcast(XN_SYS_UI, 0);
    ctx.r3.u64 = 0;
}

// u32 XamUserReadProfileSettings(titleId, userIndex, xuidCount, xuids, settingCount,
//                                settingIds, u32* bufferSize, buffer, overlapped)
GUEST_IMPORT(XamUserReadProfileSettings)
{
    ctx.r3.u64 = ERROR_NOT_LOGGED_ON_;
}

// void RtlTimeToTimeFields(u64* time /* 100 ns since 1601 */, TIME_FIELDS* fields)
GUEST_IMPORT(RtlTimeToTimeFields)
{
    const uint64_t time = Load<uint64_t>(ctx.r3.u32);
    const uint32_t out = ctx.r4.u32;

    uint64_t days = time / 864000000000ull;
    const uint64_t dayTime = time % 864000000000ull;
    const uint16_t weekday = uint16_t((days + 1) % 7); // 1601-01-01 was a Monday

    // Civil-from-days (Howard Hinnant), shifted from the 1601 epoch to 0000-03-01.
    int64_t z = int64_t(days) + 584694;  // days from 0000-03-01 to 1601-01-01
    const int64_t era = z / 146097;
    const int64_t doe = z - era * 146097;
    const int64_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    int64_t year = yoe + era * 400;
    const int64_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const int64_t mp = (5 * doy + 2) / 153;
    const int64_t day = doy - (153 * mp + 2) / 5 + 1;
    const int64_t month = mp < 10 ? mp + 3 : mp - 9;
    if (month <= 2)
        ++year;

    Store<uint16_t>(out + 0x0, uint16_t(year));
    Store<uint16_t>(out + 0x2, uint16_t(month));
    Store<uint16_t>(out + 0x4, uint16_t(day));
    Store<uint16_t>(out + 0x6, uint16_t(dayTime / 36000000000ull));            // hour
    Store<uint16_t>(out + 0x8, uint16_t(dayTime / 600000000ull % 60));          // minute
    Store<uint16_t>(out + 0xA, uint16_t(dayTime / 10000000ull % 60));           // second
    Store<uint16_t>(out + 0xC, uint16_t(dayTime / 10000ull % 1000));            // millisecond
    Store<uint16_t>(out + 0xE, weekday);
}

// void XGetVideoMode(XVIDEO_MODE* mode): same answer as VdQueryVideoMode.
GUEST_IMPORT(XGetVideoMode)
{
    auto* mode = ToHost<XVIDEO_MODE>(ctx.r3.u32);
    *mode = {};
    mode->DisplayWidth = 1280;
    mode->DisplayHeight = 720;
    mode->IsWidescreen = 1;
    mode->IsHighDefinition = 1;
    mode->RefreshRate = 0x42700000; // 60.0f
    mode->VideoStandard = 1;        // NTSC
    mode->Unknown4A = 0x4A;
    mode->Unknown01 = 0x01;
}

// u32 XamUserCheckPrivilege(u32 userIndex, u32 privilege, u32* result)
GUEST_IMPORT(XamUserCheckPrivilege)
{
    if (ctx.r5.u32)
        Store<uint32_t>(ctx.r5.u32, 0);
    ctx.r3.u64 = 0x525; // ERROR_NO_SUCH_USER: nobody is signed in
}

// void RtlInitAnsiString(ANSI_STRING* dest, const char* source)
GUEST_IMPORT(RtlInitAnsiString)
{
    const uint32_t dest = ctx.r3.u32;
    const uint32_t source = ctx.r4.u32;
    const uint16_t length = source ? uint16_t(std::strlen(ToHost<const char>(source))) : 0;
    Store<uint16_t>(dest + 0, length);
    Store<uint16_t>(dest + 2, source ? uint16_t(length + 1) : 0);
    Store<uint32_t>(dest + 4, source);
}

// void RtlInitUnicodeString(UNICODE_STRING* dest, const wchar16* source /* big-endian */)
GUEST_IMPORT(RtlInitUnicodeString)
{
    const uint32_t dest = ctx.r3.u32;
    const uint32_t source = ctx.r4.u32;
    uint16_t chars = 0;
    if (source)
        while (Load<uint16_t>(source + chars * 2) != 0)
            ++chars;
    Store<uint16_t>(dest + 0, uint16_t(chars * 2));
    Store<uint16_t>(dest + 2, source ? uint16_t(chars * 2 + 2) : 0);
    Store<uint32_t>(dest + 4, source);
}

// void RtlFillMemoryUlong(void* dest, u32 length, u32 pattern): fill with a big-endian dword.
GUEST_IMPORT(RtlFillMemoryUlong)
{
    const uint32_t dest = ctx.r3.u32;
    const uint32_t length = ctx.r4.u32 & ~3u;
    for (uint32_t i = 0; i < length; i += 4)
        Store<uint32_t>(dest + i, ctx.r5.u32);
}
