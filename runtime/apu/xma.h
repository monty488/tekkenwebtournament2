#pragma once

namespace apu
{
    // Allocates the XMA context array, publishes it in the decoder's register file and
    // starts the decoder thread. Call after guest memory is set up, before guest code runs.
    void InitXma();
}
