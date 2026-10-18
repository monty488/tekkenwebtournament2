#pragma once

// One-frame-at-a-time XMA decoding on top of FFmpeg's WMA Pro decoder (xma_decoder.c).
// The console's XMA decoder turns each compressed frame into 512 samples as soon as the
// frame is consumed; FFmpeg's own xma2 decoder buffers whole packets and thousands of
// samples, so this calls its frame decoder directly instead.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TTT2XmaDecoder TTT2XmaDecoder;

enum { TTT2_XMA_FRAME_SAMPLES = 512, TTT2_XMA_INPUT_PADDING = 64 };

// channels: 1 or 2. Returns NULL on failure.
TTT2XmaDecoder* ttt2_xma_create(int channels, int sampleRate);
void ttt2_xma_destroy(TTT2XmaDecoder* decoder);

// Forgets the overlap state (a new stream starts).
void ttt2_xma_reset(TTT2XmaDecoder* decoder);

// Decodes one frame. `data` holds the frame from its first bit (the 15-bit length field)
// and must be followed by TTT2_XMA_INPUT_PADDING readable bytes. Writes 512 samples per
// channel to left (and right, for stereo). Returns 1, or 0 if the frame was corrupt
// (the outputs then hold silence).
int ttt2_xma_decode(TTT2XmaDecoder* decoder, const uint8_t* data, int bits, float* left, float* right);

#ifdef __cplusplus
}
#endif
