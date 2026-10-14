#pragma once

#include <filesystem>
#include <string_view>

namespace vfs
{
    // Sets the host directory that holds the extracted disc contents.
    //   native: game/files (or a path passed on the command line)
    //   web:    /game, an OPFS-backed WASMFS mount populated by web/loader.js
    void SetGameRoot(const std::filesystem::path& root);
    const std::filesystem::path& GameRoot();

    // Maps an Xbox path ("game:\data\foo.bin", "d:\data\foo.bin",
    // "\Device\Cdrom0\data\foo.bin") to a host path. Matching is
    // case-insensitive, as on the console.
    std::filesystem::path Resolve(std::string_view guestPath);
}
