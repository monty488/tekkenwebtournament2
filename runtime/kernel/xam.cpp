// XAM (system library) and miscellaneous runtime-library imports.
//
// Profiles: the runtime presents a console with no signed-in user for now, so
// profile reads fail with ERROR_NOT_LOGGED_ON. Titles handle that state (it is the
// normal state before anyone signs in).

#include "guest.h"

#include <cstring>

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
