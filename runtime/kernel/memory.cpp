// Guest address space and the kernel memory APIs.
//
// The recompiled code addresses guest memory as `base + guest_address` with
// 32-bit guest addresses, so the whole 4 GB (PPC_MEMORY_SIZE) range is reserved
// up front. The function lookup table XenonRecomp uses for indirect calls lives
// inside that range, right after the XEX image.
//
// On the web this requires wasm64 (memory64): a wasm32 module can address only
// 4 GB in total, leaving no room for the host heap. See docs/ARCHITECTURE.md.
//
// Xbox 360 virtual layout (simplified):
//   0x40000000-0x7FFFFFFF  virtual 64 KB pages     -> g_virtualHeap
//   0x82000000-...         XEX image + function table
//   0xA0000000-0xBFFFFFFF  physical memory, 64 KB  -> g_physicalHeap
//   0xC0000000/0xE0000000  other physical views (aliases; not aliased here, see docs)

#include "memory.h"

#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>

#ifdef _WIN32
#include <windows.h>
#endif

namespace guest
{
    uint8_t* g_base = nullptr;

    // First-fit range allocator over a fixed guest address window.
    class RangeHeap
    {
    public:
        RangeHeap(uint32_t begin, uint32_t end) : m_begin(begin), m_end(end)
        {
            m_free[begin] = end - begin;
        }

        uint32_t Allocate(uint32_t size, uint32_t alignment)
        {
            std::lock_guard lock(m_mutex);
            size = AlignUp(size ? size : 1, kPageSize);
            alignment = alignment < kPageSize ? kPageSize : alignment;

            for (auto it = m_free.begin(); it != m_free.end(); ++it)
            {
                const uint32_t blockStart = it->first;
                const uint32_t blockSize = it->second;
                const uint32_t start = AlignUp(blockStart, alignment);
                if (start < blockStart || uint64_t(start - blockStart) + size > blockSize)
                    continue;

                m_free.erase(it);
                if (start > blockStart)
                    m_free[blockStart] = start - blockStart;
                const uint32_t tail = blockStart + blockSize - (start + size);
                if (tail)
                    m_free[start + size] = tail;

                m_used[start] = size;
                std::memset(ToHost(start), 0, size);
                return start;
            }
            return 0;
        }

        // Returns the freed size, or 0 if `address` was not an allocation start.
        uint32_t Free(uint32_t address)
        {
            std::lock_guard lock(m_mutex);
            auto used = m_used.find(address);
            if (used == m_used.end())
                return 0;

            const uint32_t freed = used->second;
            uint32_t start = address;
            uint32_t size = freed;
            m_used.erase(used);

            // Coalesce with neighbours.
            auto next = m_free.find(start + size);
            if (next != m_free.end())
            {
                size += next->second;
                m_free.erase(next);
            }
            auto prev = m_free.lower_bound(start);
            if (prev != m_free.begin())
            {
                --prev;
                if (prev->first + prev->second == start)
                {
                    start = prev->first;
                    size += prev->second;
                    m_free.erase(prev);
                }
            }
            m_free[start] = size;
            return freed;
        }

        uint32_t SizeOf(uint32_t address)
        {
            std::lock_guard lock(m_mutex);
            auto it = m_used.find(address);
            return it == m_used.end() ? 0 : it->second;
        }

        bool Contains(uint32_t address) const { return address >= m_begin && address < m_end; }

    private:
        static constexpr uint32_t kPageSize = 0x1000;
        static uint32_t AlignUp(uint32_t v, uint32_t a) { return (v + a - 1) & ~(a - 1); }

        uint32_t m_begin, m_end;
        std::mutex m_mutex;
        std::map<uint32_t, uint32_t> m_free; // start -> size
        std::map<uint32_t, uint32_t> m_used; // start -> size
    };

    static RangeHeap* g_virtualHeap;
    static RangeHeap* g_physicalHeap;

    bool InitMemory()
    {
#ifdef _WIN32
        g_base = static_cast<uint8_t*>(VirtualAlloc(nullptr, PPC_MEMORY_SIZE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        // wasm64: fresh linear memory from memory.grow is zero-filled and browsers
        // commit it lazily, so this does not cost 4 GB of RAM up front.
        g_base = static_cast<uint8_t*>(std::aligned_alloc(0x10000, PPC_MEMORY_SIZE));
#endif
        if (!g_base)
        {
            TTT2_LOG("failed to reserve the 4 GB guest address space");
            return false;
        }

        g_virtualHeap = new RangeHeap(0x40000000, 0x80000000);
        g_physicalHeap = new RangeHeap(0xA0000000, 0xC0000000);
        return true;
    }

    uint32_t AllocPhysical(uint32_t size, uint32_t alignment)
    {
        return g_physicalHeap->Allocate(size, alignment);
    }

    uint32_t AllocVirtual(uint32_t size, uint32_t alignment)
    {
        return g_virtualHeap->Allocate(size, alignment);
    }

    void Free(uint32_t address)
    {
        if (g_physicalHeap->Contains(address))
            g_physicalHeap->Free(address);
        else if (g_virtualHeap->Contains(address))
            g_virtualHeap->Free(address);
    }
}

// Used by xpointer<T> in XenonUtils/xbox.h.
extern "C" void* MmGetHostAddress(uint32_t ptr)
{
    return guest::g_base + ptr;
}

using namespace guest;

// NTSTATUS NtAllocateVirtualMemory(u32* baseAddress, u32* regionSize, u32 allocType, u32 protect, u32 debug)
GUEST_IMPORT(NtAllocateVirtualMemory)
{
    const uint32_t pBase = ctx.r3.u32;
    const uint32_t pSize = ctx.r4.u32;
    const uint32_t requested = Load<uint32_t>(pBase);
    const uint32_t size = Load<uint32_t>(pSize);

    if (requested != 0)
    {
        // MEM_COMMIT on an existing reservation: memory is always committed here.
        ctx.r3.u64 = STATUS_SUCCESS;
        return;
    }

    const uint32_t address = AllocVirtual(size, 0x10000);
    if (!address)
    {
        ctx.r3.u64 = STATUS_NO_MEMORY;
        return;
    }
    Store<uint32_t>(pBase, address);
    Store<uint32_t>(pSize, (size + 0xFFF) & ~0xFFFu);
    ctx.r3.u64 = STATUS_SUCCESS;
}

// NTSTATUS NtFreeVirtualMemory(u32* baseAddress, u32* regionSize, u32 freeType, u32 debug)
GUEST_IMPORT(NtFreeVirtualMemory)
{
    constexpr uint32_t MEM_RELEASE = 0x8000;
    if (ctx.r5.u32 & MEM_RELEASE)
        Free(Load<uint32_t>(ctx.r3.u32));
    ctx.r3.u64 = STATUS_SUCCESS;
}

// u32 MmAllocatePhysicalMemoryEx(u32 flags, u32 size, u32 protect, u32 minAddr, u32 maxAddr, u32 alignment)
GUEST_IMPORT(MmAllocatePhysicalMemoryEx)
{
    const uint32_t size = ctx.r4.u32;
    const uint32_t alignment = ctx.r8.u32 ? ctx.r8.u32 : 0x10000;
    ctx.r3.u64 = AllocPhysical(size, alignment);
    if (!ctx.r3.u32)
        TTT2_LOG("MmAllocatePhysicalMemoryEx: out of memory (size 0x%X)", size);
}

// void MmFreePhysicalMemory(u32 type, u32 address)
GUEST_IMPORT(MmFreePhysicalMemory)
{
    Free(ctx.r4.u32);
}

// u32 MmGetPhysicalAddress(u32 address) - GPU-visible address
GUEST_IMPORT(MmGetPhysicalAddress)
{
    ctx.r3.u64 = ctx.r3.u32 & 0x1FFFFFFF;
}

// u32 MmQueryAddressProtect(u32 address)
GUEST_IMPORT(MmQueryAddressProtect)
{
    constexpr uint32_t PAGE_READWRITE = 0x04;
    ctx.r3.u64 = PAGE_READWRITE;
}
