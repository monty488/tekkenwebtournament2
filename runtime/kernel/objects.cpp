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
