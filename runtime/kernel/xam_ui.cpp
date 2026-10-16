// XAM system UI, enumerators and content (save data) imports.
//
// Most of these are asynchronous on the console: they take an XOVERLAPPED, return
// ERROR_IO_PENDING, and later fill in the overlapped result and signal its event.
// Here every request completes before the call returns, which titles handle fine as
// long as the overlapped block is completed. (A stub that returns 0 without completing
// it leaves the title polling forever.)
//
// The console presented is one with no signed-in profile and one hard drive with no
// saved content. Without a profile, TTT2 does not save, so content creation is
// rejected cleanly rather than backed by host files for now.

#include "guest.h"
#include "objects.h"
#include "sync.h"

#include <cstring>
#include <string>
#include <vector>

using namespace guest;

namespace guest
{
    // xam.cpp: queues a notification on every listener whose area mask matches.
    void BroadcastNotification(uint32_t id, uint32_t param);
}

namespace
{
    constexpr uint32_t ERROR_SUCCESS_ = 0;
    constexpr uint32_t ERROR_FILE_NOT_FOUND_ = 2;
    constexpr uint32_t ERROR_NO_MORE_FILES_ = 0x12;
    constexpr uint32_t ERROR_INVALID_PARAMETER_ = 87;
    constexpr uint32_t ERROR_IO_PENDING_ = 997;
    constexpr uint32_t ERROR_NO_SUCH_USER_ = 0x525;

    constexpr uint32_t XN_SYS_UI = 0x00000009;

    // Stack arguments: the 9th argument onwards lives in the caller's parameter save
    // area, 8 bytes per slot starting at r1+0x50 (32-bit values in the low word).
    uint32_t StackArg(const PPCContext& ctx, int index /* 9-based */)
    {
        return Load<uint32_t>(ctx.r1.u32 + 0x54 + (index - 9) * 8);
    }

    // XOVERLAPPED: InternalLow (result), InternalHigh (length), InternalContext,
    // hEvent, pCompletionRoutine, dwCompletionContext, dwExtendedError.
    void CompleteOverlapped(uint32_t overlapped, uint32_t result, uint32_t length = 0, uint32_t extendedError = 0)
    {
        Store<uint32_t>(overlapped + 0x00, result);
        Store<uint32_t>(overlapped + 0x04, length);
        Store<uint32_t>(overlapped + 0x18, extendedError ? extendedError : (result ? 0x80070000u | result : 0));
        if (const uint32_t event = Load<uint32_t>(overlapped + 0x0C))
        {
            if (auto e = std::dynamic_pointer_cast<Event>(WaitableFromHandle(event)))
                SignalObject(e);
        }
        if (Load<uint32_t>(overlapped + 0x10))
            TTT2_LOG("XOVERLAPPED completion routine %08X not called", Load<uint32_t>(overlapped + 0x10));
    }

    // Returns what the import itself should return: ERROR_IO_PENDING for an
    // overlapped request (completed already), otherwise the result.
    uint32_t Finish(uint32_t overlapped, uint32_t result, uint32_t length = 0)
    {
        if (!overlapped)
            return result;
        CompleteOverlapped(overlapped, result, length);
        return ERROR_IO_PENDING_;
    }

    std::string ReadWide(uint32_t address, size_t maxChars = 512)
    {
        std::string out;
        if (!address)
            return out;
        for (size_t i = 0; i < maxChars; i++)
        {
            const uint16_t c = Load<uint16_t>(address + uint32_t(i * 2));
            if (!c)
                break;
            out += c < 0x80 ? char(c) : '?';
        }
        return out;
    }

    void BroadcastUi()
    {
        // The UI "opens" and "closes" at once.
        BroadcastNotification(XN_SYS_UI, 1);
        BroadcastNotification(XN_SYS_UI, 0);
    }

    // An enumerator over fixed-size items (achievements, stats, content: all empty for
    // now). XamEnumerate copies as many as fit, then reports ERROR_NO_MORE_FILES.
    struct Enumerator : KernelObject
    {
        uint32_t itemSize = 0;
        uint32_t itemsPerCall = 1;
        std::vector<std::vector<uint8_t>> items;
        size_t position = 0;
    };

    uint32_t CreateEnumerator(uint32_t itemSize, uint32_t itemsPerCall, uint32_t pBufferSize, uint32_t pHandle)
    {
        if (!pHandle)
            return ERROR_INVALID_PARAMETER_;
        auto e = std::make_shared<Enumerator>();
        e->itemSize = itemSize;
        e->itemsPerCall = itemsPerCall ? itemsPerCall : 1;
        if (pBufferSize)
            Store<uint32_t>(pBufferSize, itemSize * e->itemsPerCall);
        Store<uint32_t>(pHandle, Handles().Insert(e));
        return ERROR_SUCCESS_;
    }
}

namespace
{
    // XMSGBOX: the title waits for the box to close and reads which button was chosen.
    uint32_t ShowMessageBox(uint32_t title, uint32_t text, uint32_t buttonCount, uint32_t buttons,
        uint32_t focus, uint32_t pResult, uint32_t overlapped)
    {
        std::string labels;
        for (uint32_t i = 0; i < buttonCount && buttons; i++)
            labels += (i ? " | " : "") + ReadWide(Load<uint32_t>(buttons + i * 4), 64);
        TTT2_LOG("message box \"%s\": %s [%s] -> button %u", ReadWide(title, 128).c_str(),
            ReadWide(text).c_str(), labels.c_str(), focus);
        BroadcastUi();
        if (pResult)
            Store<uint32_t>(pResult, focus < buttonCount ? focus : 0);
        return Finish(overlapped, ERROR_SUCCESS_);
    }
}

// u32 XamShowMessageBoxUI(user, title, text, buttonCount, buttons, focusButton, flags,
//                         u32* result, XOVERLAPPED* overlapped)
GUEST_IMPORT(XamShowMessageBoxUI)
{
    ctx.r3.u64 = ShowMessageBox(ctx.r4.u32, ctx.r5.u32, ctx.r6.u32, ctx.r7.u32, ctx.r8.u32, ctx.r10.u32, StackArg(ctx, 9));
}

// u32 XamShowMessageBoxUIEx(user, title, text, buttonCount, buttons, focusButton, flags,
//                           unused, u32* result, XOVERLAPPED* overlapped)
GUEST_IMPORT(XamShowMessageBoxUIEx)
{
    ctx.r3.u64 = ShowMessageBox(ctx.r4.u32, ctx.r5.u32, ctx.r6.u32, ctx.r7.u32, ctx.r8.u32, StackArg(ctx, 9), StackArg(ctx, 10));
}

// u32 XamShowDeviceSelectorUI(user, contentType, contentFlags, u64 totalRequested,
//                             u32* deviceId, XOVERLAPPED* overlapped)
GUEST_IMPORT(XamShowDeviceSelectorUI)
{
    BroadcastUi();
    if (ctx.r7.u32)
        Store<uint32_t>(ctx.r7.u32, 1); // the hard drive
    ctx.r3.u64 = Finish(ctx.r8.u32, ERROR_SUCCESS_);
}

// Other system UI the title may open (friends, gamer card, marketplace, keyboard):
// it opens and is dismissed straight away.
// u32 XamShowKeyboardUI(user, flags, default, title, description, u16* buffer, count, overlapped)
GUEST_IMPORT(XamShowKeyboardUI)
{
    BroadcastUi();
    ctx.r3.u64 = Finish(ctx.r10.u32, 1223 /* ERROR_CANCELLED */);
}

GUEST_IMPORT(XamShowFriendsUI) { BroadcastUi(); ctx.r3.u64 = ERROR_SUCCESS_; }
GUEST_IMPORT(XamShowGamerCardUIForXUID) { BroadcastUi(); ctx.r3.u64 = ERROR_SUCCESS_; }
GUEST_IMPORT(XamShowMarketplaceUI) { BroadcastUi(); ctx.r3.u64 = ERROR_SUCCESS_; }
GUEST_IMPORT(XamShowMarketplaceDownloadItemsUI) { BroadcastUi(); ctx.r3.u64 = ERROR_SUCCESS_; }

// ---- Enumerators ---------------------------------------------------------------------

// u32 XamUserCreateAchievementEnumerator(titleId, user, u64 xuid, detailFlags, startIndex,
//                                        count, u32* bufferSize, u32* handle)
GUEST_IMPORT(XamUserCreateAchievementEnumerator)
{
    // XACHIEVEMENT_DETAILS (0x24) plus room for its label/description strings.
    ctx.r3.u64 = CreateEnumerator(0x24 + 3 * 0x200, ctx.r8.u32, ctx.r9.u32, ctx.r10.u32);
}

// u32 XamUserCreateStatsEnumerator(titleId, user?, u64 xuid, rankStart, viewCount, views,
//                                  u32* bufferSize, u32* handle)
GUEST_IMPORT(XamUserCreateStatsEnumerator)
{
    ctx.r3.u64 = ERROR_NO_SUCH_USER_; // leaderboards need Live
}

// u32 XamContentCreateEnumerator(user, deviceId, contentType, contentFlags, itemCount,
//                                u32* bufferSize, u32* handle)
GUEST_IMPORT(XamContentCreateEnumerator)
{
    ctx.r3.u64 = CreateEnumerator(0x134 /* XCONTENT_DATA */, ctx.r7.u32, ctx.r8.u32, ctx.r9.u32);
}

// u32 XamEnumerate(handle, flags, buffer, bufferSize, u32* itemsReturned, XOVERLAPPED*)
GUEST_IMPORT(XamEnumerate)
{
    const uint32_t buffer = ctx.r5.u32, bufferSize = ctx.r6.u32;
    const uint32_t pItems = ctx.r7.u32, overlapped = ctx.r8.u32;
    auto e = Handles().Get<Enumerator>(ctx.r3.u32);
    if (!e)
    {
        ctx.r3.u64 = 6; // ERROR_INVALID_HANDLE
        return;
    }
    uint32_t count = 0;
    while (e->position < e->items.size() && count < e->itemsPerCall && (count + 1) * e->itemSize <= bufferSize)
    {
        std::memcpy(ToHost<uint8_t>(buffer + count * e->itemSize), e->items[e->position].data(), e->itemSize);
        ++e->position;
        ++count;
    }
    const uint32_t result = count ? ERROR_SUCCESS_ : ERROR_NO_MORE_FILES_;
    if (pItems)
        Store<uint32_t>(pItems, count);
    ctx.r3.u64 = Finish(overlapped, result, count);
}

// ---- Content (save data) ---------------------------------------------------------------

// u32 XamContentGetDeviceState(deviceId, XOVERLAPPED*)
GUEST_IMPORT(XamContentGetDeviceState)
{
    ctx.r3.u64 = Finish(ctx.r4.u32, ctx.r3.u32 == 1 ? ERROR_SUCCESS_ : 1167 /* ERROR_DEVICE_NOT_CONNECTED */);
}

// u32 XamContentCreateEx(user, rootName, XCONTENT_DATA*, flags, u32* disposition,
//                        u32* license, cacheSize, u64 contentSize, XOVERLAPPED*)
GUEST_IMPORT(XamContentCreateEx)
{
    const uint32_t contentData = ctx.r5.u32;
    char fileName[43] = {};
    if (contentData)
        std::memcpy(fileName, ToHost<const char>(contentData + 0x108), 42);
    TTT2_LOG("XamContentCreateEx: %s:%s (flags %X) rejected: content is not supported yet",
        ToHost<const char>(ctx.r4.u32) ? ToHost<const char>(ctx.r4.u32) : "?", fileName, ctx.r6.u32);
    if (ctx.r7.u32)
        Store<uint32_t>(ctx.r7.u32, 0);
    ctx.r3.u64 = Finish(StackArg(ctx, 9), ERROR_FILE_NOT_FOUND_);
}

// u32 XamContentClose(rootName, XOVERLAPPED*) / XamContentFlush(rootName, XOVERLAPPED*)
GUEST_IMPORT(XamContentClose) { ctx.r3.u64 = Finish(ctx.r4.u32, ERROR_SUCCESS_); }
GUEST_IMPORT(XamContentFlush) { ctx.r3.u64 = Finish(ctx.r4.u32, ERROR_SUCCESS_); }

// u32 XamContentDelete(user, XCONTENT_DATA*, XOVERLAPPED*)
GUEST_IMPORT(XamContentDelete) { ctx.r3.u64 = Finish(ctx.r5.u32, ERROR_FILE_NOT_FOUND_); }

// u32 XamContentGetCreator(user, XCONTENT_DATA*, u32* isCreator, u64* xuid, XOVERLAPPED*)
GUEST_IMPORT(XamContentGetCreator)
{
    if (ctx.r5.u32)
        Store<uint32_t>(ctx.r5.u32, 0);
    if (ctx.r6.u32)
        Store<uint64_t>(ctx.r6.u32, 0);
    ctx.r3.u64 = Finish(ctx.r7.u32, ERROR_FILE_NOT_FOUND_);
}
