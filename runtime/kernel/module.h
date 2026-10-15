#pragma once

#include "guest.h"

#include <filesystem>

namespace guest
{
    // Maps the XEX headers, builds the module's LDR_DATA_TABLE_ENTRY and fills every
    // variable import slot (from scan/variables.txt). Call after the image is loaded.
    bool InitModule(const std::filesystem::path& scanDir);

    uint32_t ExecutableModuleHandle();

    // Static TLS (__declspec(thread)) from XEX_HEADER_TLS_INFO: every thread gets
    // `dataSize` bytes, the first `rawSize` copied from `rawAddress`, the rest zeroed.
    struct TlsInfo
    {
        uint32_t rawAddress = 0;
        uint32_t dataSize = 0;
        uint32_t rawSize = 0;
    };
    const TlsInfo& GetTlsInfo();
}
