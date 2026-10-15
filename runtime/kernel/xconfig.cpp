// Console configuration queries, answered as a retail US console with an English
// dashboard on an HDMI 720p display would. Values follow Xenia's xboxkrnl_xconfig.

#include "guest.h"

#include <cstring>

using namespace guest;

namespace
{
    constexpr uint16_t XCONFIG_SECURED_CATEGORY = 0x0002;
    constexpr uint16_t XCONFIG_USER_CATEGORY = 0x0003;

    // Writes `size` bytes of `value` (big-endian) or zero-fills; returns bytes written.
    uint32_t Setting(uint16_t category, uint16_t setting, uint8_t* out, uint32_t outSize)
    {
        uint32_t value = 0;
        uint32_t size = 4;

        if (category == XCONFIG_SECURED_CATEGORY)
        {
            switch (setting)
            {
            case 0x0002: value = 0x00001000; break;   // AV region: USA/Canada (NTSC-M)
            case 0x0003: value = 0x00000000; break;   // game region
            default: break;
            }
        }
        else if (category == XCONFIG_USER_CATEGORY)
        {
            switch (setting)
            {
            case 0x0009: value = 1; break;            // language: English
            case 0x000A: value = 0x00040000; break;   // video flags: widescreen
            case 0x000B: value = 0x00010000; break;   // audio flags: stereo
            case 0x000C: value = 0x40; break;         // retail flags: dashboard initial setup done
            case 0x000E: size = 1; value = 103; break; // country: United States
            default: break;
            }
        }

        std::memset(out, 0, outSize);
        if (outSize < size)
            return size;
        for (uint32_t i = 0; i < size; i++)
            out[i] = uint8_t(value >> (8 * (size - 1 - i)));
        return size;
    }
}

// NTSTATUS ExGetXConfigSetting(u16 category, u16 setting, void* buffer, u16 bufferSize, u16* requiredSize)
GUEST_IMPORT(ExGetXConfigSetting)
{
    const uint16_t category = ctx.r3.u16;
    const uint16_t setting = ctx.r4.u16;
    const uint32_t buffer = ctx.r5.u32;
    const uint16_t bufferSize = ctx.r6.u16;
    const uint32_t pRequired = ctx.r7.u32;

    uint8_t scratch[64]{};
    const uint32_t written = Setting(category, setting, buffer ? ToHost<uint8_t>(buffer) : scratch,
        buffer ? bufferSize : sizeof(scratch));
    if (pRequired)
        Store<uint16_t>(pRequired, uint16_t(written));
    ctx.r3.u64 = (buffer && bufferSize < written) ? 0xC0000023 /* STATUS_BUFFER_TOO_SMALL */ : STATUS_SUCCESS;
}

// u32 XGetAVPack(): 6 = HDMI.
GUEST_IMPORT(XGetAVPack)
{
    ctx.r3.u64 = 6;
}

// NTSTATUS MmQueryStatistics(MM_STATISTICS* stats). The caller sets stats->Length first.
GUEST_IMPORT(MmQueryStatistics)
{
    const uint32_t stats = ctx.r3.u32;
    const uint32_t length = Load<uint32_t>(stats);
    std::memset(ToHost(stats + 4), 0, length > 4 ? length - 4 : 0);
    // Pages are 4 KB: report 512 MB total with ~256 MB available, as a title sees at boot.
    Store<uint32_t>(stats + 0x04, 0x20000);  // TotalPhysicalPages
    Store<uint32_t>(stats + 0x08, 0x00000);  // KernelPages
    Store<uint32_t>(stats + 0x0C, 0x10000);  // TitleAvailablePages
    Store<uint32_t>(stats + 0x10, 0x20000 * 0x1000); // TitleTotalVirtualMemoryBytes
    ctx.r3.u64 = STATUS_SUCCESS;
}
