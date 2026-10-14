// File I/O imports, backed by vfs (host files natively, OPFS in the browser).
// All I/O is synchronous; asynchronous requests complete before returning.

#include "guest.h"
#include "objects.h"
#include "../vfs/vfs.h"

#include <cstdio>
#include <string>

namespace guest
{
    // Game archives can exceed 2 GB, so plain fseek/ftell (32-bit long on Windows) won't do.
    static int Seek64(std::FILE* f, uint64_t offset)
    {
#ifdef _WIN32
        return _fseeki64(f, int64_t(offset), SEEK_SET);
#else
        return fseeko(f, off_t(offset), SEEK_SET);
#endif
    }

    static uint64_t Tell64(std::FILE* f)
    {
#ifdef _WIN32
        return uint64_t(_ftelli64(f));
#else
        return uint64_t(ftello(f));
#endif
    }

    struct GuestFile : KernelObject
    {
        std::FILE* file = nullptr;
        uint64_t size = 0;
        std::string path;

        ~GuestFile() override
        {
            if (file)
                std::fclose(file);
        }
    };

    static std::string ReadObjectName(uint32_t objectAttributes)
    {
        auto* attributes = ToHost<XOBJECT_ATTRIBUTES>(objectAttributes);
        if (!attributes)
            return {};
        const XANSI_STRING* name = attributes->Name.get();
        if (!name || !name->Buffer.get())
            return {};
        return std::string(name->Buffer.get(), name->Length.get());
    }

    static void SetIoStatus(uint32_t ioStatusBlock, uint32_t status, uint32_t information)
    {
        if (auto* io = ToHost<XIO_STATUS_BLOCK>(ioStatusBlock))
        {
            io->Status = status;
            io->Information = information;
        }
    }

    // Shared by NtCreateFile/NtOpenFile. Only read access is supported: the disc is read-only,
    // and save data is expected to go through XContent (not implemented yet).
    static uint32_t OpenFile(uint32_t pHandle, uint32_t objectAttributes, uint32_t ioStatusBlock)
    {
        const std::string guestPath = ReadObjectName(objectAttributes);
        const auto hostPath = vfs::Resolve(guestPath);

        auto file = std::make_shared<GuestFile>();
        file->path = guestPath;
        file->file = std::fopen(hostPath.string().c_str(), "rb");
        if (!file->file)
        {
            TTT2_LOG("open failed: %s", guestPath.c_str());
            SetIoStatus(ioStatusBlock, STATUS_OBJECT_NAME_NOT_FOUND, 0);
            return STATUS_OBJECT_NAME_NOT_FOUND;
        }
        std::fseek(file->file, 0, SEEK_END);
        file->size = Tell64(file->file);
        Seek64(file->file, 0);

        Store<uint32_t>(pHandle, Handles().Insert(file));
        SetIoStatus(ioStatusBlock, STATUS_SUCCESS, 1 /* FILE_OPENED */);
        return STATUS_SUCCESS;
    }
}

using namespace guest;

// NTSTATUS NtCreateFile(u32* handle, u32 access, OBJECT_ATTRIBUTES* attrs, IO_STATUS_BLOCK* io,
//                       u64* allocationSize, u32 fileAttributes, u32 shareAccess, u32 disposition)
GUEST_IMPORT(NtCreateFile)
{
    ctx.r3.u64 = OpenFile(ctx.r3.u32, ctx.r5.u32, ctx.r6.u32);
}

// NTSTATUS NtOpenFile(u32* handle, u32 access, OBJECT_ATTRIBUTES* attrs, IO_STATUS_BLOCK* io, u32 options)
GUEST_IMPORT(NtOpenFile)
{
    ctx.r3.u64 = OpenFile(ctx.r3.u32, ctx.r5.u32, ctx.r6.u32);
}

// NTSTATUS NtReadFile(u32 handle, u32 event, u32 apcRoutine, u32 apcContext,
//                     IO_STATUS_BLOCK* io, void* buffer, u32 length, s64* byteOffset)
GUEST_IMPORT(NtReadFile)
{
    auto file = Handles().Get<GuestFile>(ctx.r3.u32);
    const uint32_t ioStatusBlock = ctx.r7.u32;
    const uint32_t buffer = ctx.r8.u32;
    const uint32_t length = ctx.r9.u32;
    const uint32_t pOffset = ctx.r10.u32;

    if (!file)
    {
        ctx.r3.u64 = STATUS_INVALID_HANDLE;
        return;
    }

    if (pOffset)
    {
        // -2 (FILE_USE_FILE_POINTER_POSITION) means "current position".
        const int64_t offset = Load<int64_t>(pOffset);
        if (offset >= 0)
            Seek64(file->file, uint64_t(offset));
    }

    const size_t read = std::fread(ToHost(buffer), 1, length, file->file);
    const uint32_t status = (read == 0 && length != 0) ? STATUS_END_OF_FILE : STATUS_SUCCESS;
    SetIoStatus(ioStatusBlock, status, uint32_t(read));

    // TODO: signal ctx.r4 (event) / queue the APC once events are implemented.
    ctx.r3.u64 = status;
}

// NTSTATUS NtQueryInformationFile(u32 handle, IO_STATUS_BLOCK* io, void* info, u32 length, u32 infoClass)
GUEST_IMPORT(NtQueryInformationFile)
{
    constexpr uint32_t FileStandardInformation = 5;
    constexpr uint32_t FilePositionInformation = 14;

    auto file = Handles().Get<GuestFile>(ctx.r3.u32);
    const uint32_t ioStatusBlock = ctx.r4.u32;
    const uint32_t info = ctx.r5.u32;
    const uint32_t infoClass = ctx.r7.u32;

    if (!file)
    {
        ctx.r3.u64 = STATUS_INVALID_HANDLE;
        return;
    }

    switch (infoClass)
    {
    case FileStandardInformation:
        Store<uint64_t>(info + 0x00, (file->size + 0xFFF) & ~0xFFFull); // AllocationSize
        Store<uint64_t>(info + 0x08, file->size);                       // EndOfFile
        Store<uint32_t>(info + 0x10, 1);                                // NumberOfLinks
        *ToHost<uint8_t>(info + 0x14) = 0;                              // DeletePending
        *ToHost<uint8_t>(info + 0x15) = 0;                              // Directory
        SetIoStatus(ioStatusBlock, STATUS_SUCCESS, 0x18);
        ctx.r3.u64 = STATUS_SUCCESS;
        return;

    case FilePositionInformation:
        Store<uint64_t>(info, Tell64(file->file));
        SetIoStatus(ioStatusBlock, STATUS_SUCCESS, 8);
        ctx.r3.u64 = STATUS_SUCCESS;
        return;

    default:
        TTT2_LOG("NtQueryInformationFile: unhandled class %u for %s", infoClass, file->path.c_str());
        ctx.r3.u64 = STATUS_NOT_IMPLEMENTED;
        return;
    }
}

// NTSTATUS NtSetInformationFile(u32 handle, IO_STATUS_BLOCK* io, void* info, u32 length, u32 infoClass)
GUEST_IMPORT(NtSetInformationFile)
{
    constexpr uint32_t FilePositionInformation = 14;

    auto file = Handles().Get<GuestFile>(ctx.r3.u32);
    if (!file)
    {
        ctx.r3.u64 = STATUS_INVALID_HANDLE;
        return;
    }

    if (ctx.r7.u32 == FilePositionInformation)
    {
        Seek64(file->file, Load<uint64_t>(ctx.r5.u32));
        SetIoStatus(ctx.r4.u32, STATUS_SUCCESS, 0);
        ctx.r3.u64 = STATUS_SUCCESS;
        return;
    }

    TTT2_LOG("NtSetInformationFile: unhandled class %u", ctx.r7.u32);
    ctx.r3.u64 = STATUS_NOT_IMPLEMENTED;
}
