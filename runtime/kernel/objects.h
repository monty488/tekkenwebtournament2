#pragma once

// Kernel object handles. Guest code sees opaque 32-bit handles; the runtime maps
// them to host objects (threads, files, events, ...).

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace guest
{
    struct KernelObject
    {
        virtual ~KernelObject() = default;
    };

    class HandleTable
    {
    public:
        uint32_t Insert(std::shared_ptr<KernelObject> object)
        {
            std::lock_guard lock(m_mutex);
            const uint32_t handle = m_next;
            m_next += 4;
            m_objects.emplace(handle, std::move(object));
            return handle;
        }

        template<typename T>
        std::shared_ptr<T> Get(uint32_t handle)
        {
            std::lock_guard lock(m_mutex);
            auto it = m_objects.find(handle);
            return it == m_objects.end() ? nullptr : std::dynamic_pointer_cast<T>(it->second);
        }

        bool Close(uint32_t handle)
        {
            std::lock_guard lock(m_mutex);
            return m_objects.erase(handle) != 0;
        }

    private:
        std::mutex m_mutex;
        std::unordered_map<uint32_t, std::shared_ptr<KernelObject>> m_objects;
        uint32_t m_next = 0xF8000004; // Never 0, never INVALID_HANDLE_VALUE (0xFFFFFFFF)
    };

    HandleTable& Handles();
}
