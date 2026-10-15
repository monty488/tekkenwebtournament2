#include "vfs.h"

#include <algorithm>
#include <cctype>
#include <string>

namespace vfs
{
    static std::filesystem::path g_root;

    void SetGameRoot(const std::filesystem::path& root) { g_root = root; }
    const std::filesystem::path& GameRoot() { return g_root; }

    static bool IEquals(std::string_view a, std::string_view b)
    {
        return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(),
            [](char x, char y) { return std::tolower((unsigned char)x) == std::tolower((unsigned char)y); });
    }

    static std::string_view StripDevice(std::string_view path)
    {
        static constexpr std::string_view kPrefixes[] =
        {
            "game:", "d:", "dvd:", "\\Device\\Cdrom0", "\\Device\\Harddisk0\\Partition1",
        };
        for (auto prefix : kPrefixes)
        {
            if (path.size() >= prefix.size() && IEquals(path.substr(0, prefix.size()), prefix))
                return path.substr(prefix.size());
        }
        return path;
    }

    std::filesystem::path Resolve(std::string_view guestPath)
    {
        std::filesystem::path result = g_root;
        std::string_view rest = StripDevice(guestPath);

        // Walk component by component so we can fix up case on case-sensitive hosts (OPFS, Linux).
        while (!rest.empty())
        {
            const size_t sep = rest.find_first_of("\\/");
            std::string_view part = rest.substr(0, sep);
            rest = sep == std::string_view::npos ? std::string_view{} : rest.substr(sep + 1);
            if (part.empty() || part == ".")
                continue;

            std::filesystem::path next = result / std::string(part);
            std::error_code ec;
            if (!std::filesystem::exists(next, ec) && std::filesystem::is_directory(result, ec))
            {
                for (const auto& entry : std::filesystem::directory_iterator(result, ec))
                {
                    if (IEquals(entry.path().filename().string(), part))
                    {
                        next = entry.path();
                        break;
                    }
                }
            }
            result = std::move(next);
        }
        return result;
    }
}
