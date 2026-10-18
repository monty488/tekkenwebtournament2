// Compiled as part of the FFmpeg library target (runtime/CMakeLists.txt), in place of
// libavcodec/wmaprodec.c, which it includes to reach the WMA Pro frame decoder
// (decode_frame) and the XMA decoder's context. FFmpeg itself stays unmodified.

#include "libavcodec/wmaprodec.c"

#include "xma_decoder.h"

struct TTT2XmaDecoder
{
    AVCodecContext* avctx;
    int channels;
};

TTT2XmaDecoder* ttt2_xma_create(int channels, int sampleRate)
{
    static int quiet = 0;
    if (!quiet)
    {
        av_log_set_level(AV_LOG_QUIET); // corrupt frames are reported by the caller
        quiet = 1;
    }

    const AVCodec* codec = &ff_xma2_decoder.p;
    AVCodecContext* avctx = avcodec_alloc_context3(codec);
    if (!avctx)
        return NULL;
    av_channel_layout_default(&avctx->ch_layout, channels);
    avctx->sample_rate = sampleRate;

    // XMA2WAVEFORMATEX: one stream of 1 or 2 channels.
    avctx->extradata = av_mallocz(34 + AV_INPUT_BUFFER_PADDING_SIZE);
    if (!avctx->extradata)
    {
        avcodec_free_context(&avctx);
        return NULL;
    }
    avctx->extradata_size = 34;
    AV_WL16(avctx->extradata, 1);
    AV_WL32(avctx->extradata + 2, channels == 2 ? (AV_CH_FRONT_LEFT | AV_CH_FRONT_RIGHT) : AV_CH_FRONT_CENTER);

    if (avcodec_open2(avctx, codec, NULL) < 0)
    {
        avcodec_free_context(&avctx);
        return NULL;
    }
    TTT2XmaDecoder* decoder = av_mallocz(sizeof(TTT2XmaDecoder));
    if (!decoder)
    {
        avcodec_free_context(&avctx);
        return NULL;
    }
    decoder->avctx = avctx;
    decoder->channels = channels;
    ttt2_xma_reset(decoder);
    return decoder;
}

void ttt2_xma_destroy(TTT2XmaDecoder* decoder)
{
    if (!decoder)
        return;
    avcodec_free_context(&decoder->avctx);
    av_free(decoder);
}

void ttt2_xma_reset(TTT2XmaDecoder* decoder)
{
    WMAProDecodeCtx* s = &((XMADecodeCtx*)decoder->avctx->priv_data)->xma[0];
    for (int i = 0; i < s->nb_channels; i++)
        memset(s->channel[i].out, 0, sizeof(s->channel[i].out));
    s->packet_loss = 0;
    s->skip_frame = 0;
}

int ttt2_xma_decode(TTT2XmaDecoder* decoder, const uint8_t* data, int bits, float* left, float* right)
{
    WMAProDecodeCtx* s = &((XMADecodeCtx*)decoder->avctx->priv_data)->xma[0];

    // decode_frame only copies the samples into frame->extended_data.
    uint8_t* planes[2] = { (uint8_t*)left, (uint8_t*)(right ? right : left) };
    AVFrame frame = { 0 };
    frame.extended_data = planes;
    frame.nb_samples = TTT2_XMA_FRAME_SAMPLES;

    init_get_bits(&s->gb, data, bits);
    s->frame_offset = 0;
    s->num_saved_bits = bits;
    s->buf_bit_size = bits;
    s->packet_loss = 0;
    s->skip_frame = 0; // every frame is output, 1:1 with the input, as on the console

    int got = 0;
    decode_frame(s, &frame, &got);
    if (s->packet_loss || !got || s->samples_per_frame != TTT2_XMA_FRAME_SAMPLES)
    {
        s->packet_loss = 0;
        memset(left, 0, sizeof(float) * TTT2_XMA_FRAME_SAMPLES);
        if (right)
            memset(right, 0, sizeof(float) * TTT2_XMA_FRAME_SAMPLES);
        return 0;
    }
    return 1;
}
