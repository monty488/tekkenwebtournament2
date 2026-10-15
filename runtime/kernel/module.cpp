// The executable module as the kernel loader presents it: kernel variable imports,
// the module handle (LDR_DATA_TABLE_ENTRY), and XEX header queries.

#include "module.h"
#include "memory.h"
#include "threads.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <sstream>
#include <thread>
#include <vector>

namespace guest
{
    static uint32_t g_xexHeader = 0;     // guest copy of the XEX headers
    static uint32_t g_xexHeaderSize = 0;
    static uint32_t g_ldrEntry = 0;      // X_LDR_DATA_TABLE_ENTRY for default.xex
    static uint32_t g_timeStampBundle = 0;

    static TlsInfo g_tls;

    uint32_t ExecutableModuleHandle() { return g_ldrEntry; }
    const TlsInfo& GetTlsInfo() { return g_tls; }

    // Returns the guest address of the optional header `key`'s value field, or 0.
    static uint32_t FindOptHeader(uint32_t headerBase, uint32_t key)
    {
        if (!headerBase)
            return 0;
        const uint32_t count = Load<uint32_t>(headerBase + 0x14);
        for (uint32_t i = 0; i < count; i++)
        {
            const uint32_t entry = headerBase + 0x18 + i * 8;
            if (Load<uint32_t>(entry) == key)
                return entry + 4;
        }
        return 0;
    }

    static bool LoadFile(const std::filesystem::path& path, std::vector<uint8_t>& out)
    {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f)
            return false;
        out.resize(size_t(f.tellg()));
        f.seekg(0);
        f.read(reinterpret_cast<char*>(out.data()), std::streamsize(out.size()));
        return bool(f);
    }

    // FILETIME-style 100 ns units since 1601, as the console reports system time.
    static uint64_t SystemTime100ns()
    {
        using namespace std::chrono;
        constexpr uint64_t kEpochDelta = 116444736000000000ull; // 1601 -> 1970
        return kEpochDelta + uint64_t(duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count() / 100);
    }

    static void TimeStampThread()
    {
        using namespace std::chrono;
        const auto start = steady_clock::now();
        for (;;)
        {
            const auto elapsed = steady_clock::now() - start;
            const uint64_t interrupt = uint64_t(duration_cast<nanoseconds>(elapsed).count() / 100);
            Store<uint64_t>(g_timeStampBundle + 0x00, interrupt);          // InterruptTime
            Store<uint64_t>(g_timeStampBundle + 0x08, SystemTime100ns());  // SystemTime
            Store<uint32_t>(g_timeStampBundle + 0x10, uint32_t(duration_cast<milliseconds>(elapsed).count())); // TickCount
            std::this_thread::sleep_for(milliseconds(1));
        }
    }

    bool InitModule(const std::filesystem::path& scanDir)
    {
        // XEX headers -> guest memory.
        std::vector<uint8_t> header;
        if (!LoadFile(scanDir / "xex_header.bin", header))
        {
            TTT2_LOG("missing %s (re-run scripts/scan-xex.ps1)", (scanDir / "xex_header.bin").string().c_str());
            return false;
        }
        g_xexHeaderSize = uint32_t(header.size());
        g_xexHeader = AllocVirtual(g_xexHeaderSize);
        std::memcpy(ToHost(g_xexHeader), header.data(), header.size());

        // X_LDR_DATA_TABLE_ENTRY (layout per Xenia).
        g_ldrEntry = AllocVirtual(0x100);
        for (uint32_t links = 0; links < 0x18; links += 8)          // empty list heads point to themselves
        {
            Store<uint32_t>(g_ldrEntry + links, g_ldrEntry + links);
            Store<uint32_t>(g_ldrEntry + links + 4, g_ldrEntry + links);
        }
        Store<uint32_t>(g_ldrEntry + 0x18, PPC_IMAGE_BASE);         // DllBase
        Store<uint32_t>(g_ldrEntry + 0x1C, PPC_IMAGE_BASE);         // ImageBase
        Store<uint32_t>(g_ldrEntry + 0x20, PPC_IMAGE_SIZE);         // SizeOfImage
        Store<uint32_t>(g_ldrEntry + 0x38, PPC_IMAGE_SIZE);         // FullImageSize
        if (const uint32_t ep = FindOptHeader(g_xexHeader, 0x00010100)) // XEX_HEADER_ENTRY_POINT
            Store<uint32_t>(g_ldrEntry + 0x3C, Load<uint32_t>(ep));
        Store<uint16_t>(g_ldrEntry + 0x40, 1);                      // LoadCount
        Store<uint32_t>(g_ldrEntry + 0x58, g_xexHeader);            // XexHeaderBase

        // XEX_HEADER_TLS_INFO: { slot count, raw data address, data size, raw data size }
        if (const uint32_t tls = FindOptHeader(g_xexHeader, 0x00020104))
        {
            const uint32_t info = g_xexHeader + Load<uint32_t>(tls);
            g_tls.rawAddress = Load<uint32_t>(info + 4);
            g_tls.dataSize = Load<uint32_t>(info + 8);
            g_tls.rawSize = Load<uint32_t>(info + 12);
            TTT2_LOG("static TLS: %u bytes (%u initialised from 0x%08X)", g_tls.dataSize, g_tls.rawSize, g_tls.rawAddress);
        }

        // Variable imports.
        std::ifstream vars(scanDir / "variables.txt");
        if (!vars)
        {
            TTT2_LOG("missing %s (re-run scripts/scan-xex.ps1)", (scanDir / "variables.txt").string().c_str());
            return false;
        }

        std::string line;
        while (std::getline(vars, line))
        {
            std::istringstream fields(line);
            std::string name;
            uint32_t slot = 0;
            fields >> name >> std::hex >> slot;
            if (name.empty() || !slot)
                continue;

            uint32_t storage = 0;
            if (name == "KeTimeStampBundle")
            {
                storage = g_timeStampBundle = AllocPhysical(0x18);
            }
            else if (name == "XboxHardwareInfo")
            {
                storage = AllocPhysical(0x10);
                Store<uint32_t>(storage + 0, 0);       // Flags
                *ToHost<uint8_t>(storage + 4) = 6;     // NumberOfProcessors (hardware threads)
            }
            else if (name == "XboxKrnlVersion")
            {
                storage = AllocPhysical(0x8);
                Store<uint16_t>(storage + 0, 2);       // 2.0.17559.0 (late-retail dashboard)
                Store<uint16_t>(storage + 2, 0);
                Store<uint16_t>(storage + 4, 17559);
                Store<uint16_t>(storage + 6, 0);
            }
            else if (name == "XexExecutableModuleHandle")
            {
                storage = AllocPhysical(0x4);
                Store<uint32_t>(storage, g_ldrEntry);
            }
            else if (name == "ExLoadedCommandLine")
            {
                storage = AllocPhysical(0x400);
                std::strcpy(ToHost<char>(storage), "\"default.xex\"");
            }
            else if (name == "VdGpuClockInMHz")
            {
                storage = AllocPhysical(0x4);
                Store<uint32_t>(storage, 500);
            }
            else if (name == "VdHSIOCalibrationLock")
            {
                storage = AllocPhysical(0x1C);         // RTL_CRITICAL_SECTION
                auto* cs = ToHost<XRTL_CRITICAL_SECTION>(storage);
                cs->LockCount = -1;
            }
            else
            {
                // KeDebugMonitorData, KeCertMonitorData, ExThreadObjectType, VdGlobalDevice, ...:
                // zeroed storage is what a retail console without a debugger presents.
                storage = AllocPhysical(0x100);
            }

            Store<uint32_t>(slot, storage);
        }

        if (g_timeStampBundle)
            std::thread(TimeStampThread).detach();
        return true;
    }
}

using namespace guest;

// u32 RtlImageXexHeaderField(void* xexHeaderBase, u32 key)
GUEST_IMPORT(RtlImageXexHeaderField)
{
    const uint32_t key = ctx.r4.u32;
    const uint32_t value = FindOptHeader(ctx.r3.u32, key);
    if (!value)
    {
        ctx.r3.u64 = 0;
        return;
    }
    switch (key & 0xFF)
    {
    case 0x00: ctx.r3.u64 = Load<uint32_t>(value); break;              // the value itself
    case 0x01: ctx.r3.u64 = value; break;                              // pointer to the value
    default:   ctx.r3.u64 = ctx.r3.u32 + Load<uint32_t>(value); break; // offset into the headers
    }
}

// NTSTATUS XexGetModuleSection(u32 module, const char* name, u32* data, u32* size)
// Looks up an embedded XEX resource (XEX_HEADER_RESOURCE_INFO), e.g. XUI/XLAST data.
GUEST_IMPORT(XexGetModuleSection)
{
    const char* wanted = ToHost<const char>(ctx.r4.u32);
    const uint32_t info = FindOptHeader(g_xexHeader, 0x000002FF);
    if (info && wanted)
    {
        const uint32_t block = g_xexHeader + Load<uint32_t>(info);
        const uint32_t count = (Load<uint32_t>(block) - 4) / 16;
        for (uint32_t i = 0; i < count; i++)
        {
            const uint32_t entry = block + 4 + i * 16;
            if (std::strncmp(ToHost<const char>(entry), wanted, 8) == 0)
            {
                if (ctx.r5.u32) Store<uint32_t>(ctx.r5.u32, Load<uint32_t>(entry + 8));
                if (ctx.r6.u32) Store<uint32_t>(ctx.r6.u32, Load<uint32_t>(entry + 12));
                ctx.r3.u64 = STATUS_SUCCESS;
                return;
            }
        }
    }
    TTT2_LOG("XexGetModuleSection: no resource '%.8s'", wanted ? wanted : "(null)");
    ctx.r3.u64 = 0xC0000225; // STATUS_NOT_FOUND
}

// NTSTATUS XexLoadImage(const char* name, u32 flags, u32 minVersion, u32* handle)
// The game probes xam.xex for optional Xbox Live party/community UI exports. Report the
// module as absent so it takes its offline path rather than calling host-less stubs.
GUEST_IMPORT(XexLoadImage)
{
    const char* name = ToHost<const char>(ctx.r3.u32);
    TTT2_LOG("XexLoadImage(\"%s\"): not available", name ? name : "(null)");
    if (ctx.r6.u32)
        Store<uint32_t>(ctx.r6.u32, 0);
    ctx.r3.u64 = 0xC0000225; // STATUS_NOT_FOUND
}

// NTSTATUS XexGetProcedureAddress(u32 module, u32 ordinal, u32* address)
GUEST_IMPORT(XexGetProcedureAddress)
{
    if (ctx.r5.u32)
        Store<uint32_t>(ctx.r5.u32, 0);
    ctx.r3.u64 = 0xC0000225; // STATUS_NOT_FOUND
}

GUEST_IMPORT(XexUnloadImage)
{
    ctx.r3.u64 = STATUS_SUCCESS;
}

// u32 XexCheckExecutablePrivilege(u32 privilege): bit N of XEX_HEADER_SYSTEM_FLAGS.
GUEST_IMPORT(XexCheckExecutablePrivilege)
{
    const uint32_t bit = ctx.r3.u32;
    const uint32_t flags = FindOptHeader(g_xexHeader, 0x00030000);
    ctx.r3.u64 = (flags && bit < 32) ? (Load<uint32_t>(flags) >> bit) & 1 : 0;
}

// void KeQuerySystemTime(u64* time)
GUEST_IMPORT(KeQuerySystemTime)
{
    if (ctx.r3.u32)
        Store<uint64_t>(ctx.r3.u32, SystemTime100ns());
}
