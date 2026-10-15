#include "guest.h"
#include "objects.h"

namespace guest
{
    HandleTable& Handles()
    {
        static HandleTable table;
        return table;
    }

    void LogUnimplemented(const char* name)
    {
        TTT2_LOG("unimplemented import called: %s", name);
    }
}

GUEST_IMPORT(NtClose)
{
    ctx.r3.u64 = guest::Handles().Close(ctx.r3.u32) ? STATUS_SUCCESS : STATUS_INVALID_HANDLE;
}

// DbgPrint(const char* format, ...). Prints the raw format string; guest
// varargs would need the PPC va_list layout to expand.
GUEST_IMPORT(DbgPrint)
{
    if (const char* format = guest::ToHost<const char>(ctx.r3.u32))
        TTT2_LOG("DbgPrint: %s", format);
    ctx.r3.u64 = STATUS_SUCCESS;
}

// u32 RtlNtStatusToDosError(NTSTATUS status): the common codes titles check for.
GUEST_IMPORT(RtlNtStatusToDosError)
{
    const uint32_t status = ctx.r3.u32;
    uint32_t error;
    switch (status)
    {
    case STATUS_SUCCESS:               error = 0; break;    // ERROR_SUCCESS
    case STATUS_PENDING:               error = 997; break;  // ERROR_IO_PENDING
    case 0x00000102:                   error = 258; break;  // STATUS_TIMEOUT -> WAIT_TIMEOUT
    case STATUS_NOT_IMPLEMENTED:       error = 1; break;    // ERROR_INVALID_FUNCTION
    case STATUS_INVALID_HANDLE:        error = 6; break;    // ERROR_INVALID_HANDLE
    case STATUS_INVALID_PARAMETER:     error = 87; break;   // ERROR_INVALID_PARAMETER
    case STATUS_END_OF_FILE:           error = 38; break;   // ERROR_HANDLE_EOF
    case STATUS_NO_MEMORY:             error = 8; break;    // ERROR_NOT_ENOUGH_MEMORY
    case STATUS_OBJECT_NAME_NOT_FOUND: error = 2; break;    // ERROR_FILE_NOT_FOUND
    case STATUS_OBJECT_PATH_NOT_FOUND: error = 3; break;    // ERROR_PATH_NOT_FOUND
    case 0xC0000225:                   error = 1168; break; // STATUS_NOT_FOUND -> ERROR_NOT_FOUND
    default:
        // Win32-in-NTSTATUS (FACILITY_WIN32) carries the code directly.
        error = ((status & 0xFFFF0000) == 0x80070000 || (status & 0xFFFF0000) == 0xC0070000) ? (status & 0xFFFF) : 317; // ERROR_MR_MID_NOT_FOUND
        break;
    }
    ctx.r3.u64 = error;
}

// void RtlRaiseException(EXCEPTION_RECORD* record)
// Guest SEH is not emulated. The one exception titles raise routinely is MSVC's
// SetThreadName (0x406D1388), which their own __except swallows when no debugger is
// attached, so continuing after the call matches what the console does. Anything
// else (e.g. a C++ throw, 0xE06D7363) is reported and stops the run.
GUEST_IMPORT(RtlRaiseException)
{
    const uint32_t record = ctx.r3.u32;
    const uint32_t code = guest::Load<uint32_t>(record);
    if (code == 0x406D1388)
    {
        const uint32_t name = guest::Load<uint32_t>(record + 0x14 + 4);
        const uint32_t threadId = guest::Load<uint32_t>(record + 0x14 + 8);
        TTT2_LOG("thread %08X named \"%s\"", threadId, name ? guest::ToHost<const char>(name) : "?");
        return;
    }

    TTT2_LOG("guest raised exception 0x%08X at %08X (not supported)", code, guest::Load<uint32_t>(record + 0xC));
    std::fflush(stderr);
    std::abort();
}
