// WebGPU backend: see webgpu_backend.h.
//
// Render targets: one GPU texture per (EDRAM base tile, pitch, MSAA scale, format), in
// pixel coordinates (MSAA is not emulated: one sample per pixel). Colour targets are
// rgba16float (rgba32float for the 32-bit float formats), depth is depth32float-stencil8.
// A clear through a resolve applies to every target sharing the EDRAM base, which covers
// the common aliasing pattern (a target cleared through one view, drawn through another).
//
// Per draw: the pixel shader is translated to WGSL (cached by its text), a pipeline is
// built for the render state (cached by key), the 256 pixel shader constants go into a
// ring buffer bound with a dynamic offset, and the captured triangles into a vertex ring.
// Work is submitted at resolves (which read targets back) and at the end of each frame.

#include "webgpu_backend.h"
#include "xenos_wgsl.h"

#include <webgpu/webgpu.h>
#ifndef __EMSCRIPTEN__
#include <webgpu/wgpu.h> // wgpuDevicePoll (wgpu-native)
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace gpu
{
    void SoftParallelFor(uint32_t count, const std::function<void(uint32_t)>& fn);

    namespace
    {
        WGPUStringView Str(const char* s) { return { s, WGPU_STRLEN }; }
        WGPUStringView Str(const std::string& s) { return { s.data(), s.size() }; }
        std::string FromView(WGPUStringView v)
        {
            if (!v.data)
                return {};
            return v.length == WGPU_STRLEN ? std::string(v.data) : std::string(v.data, v.length);
        }

        float HalfToFloat(uint16_t h)
        {
            const uint32_t sign = uint32_t(h >> 15) << 31;
            const uint32_t exp = (h >> 10) & 0x1F, mant = h & 0x3FF;
            uint32_t bits;
            if (exp == 0)
            {
                const float v = std::ldexp(float(mant), -24);
                return (h >> 15) ? -v : v;
            }
            if (exp == 31)
                bits = sign | 0x7F800000u | (mant << 13);
            else
                bits = sign | ((exp + 112) << 23) | (mant << 13);
            float f;
            std::memcpy(&f, &bits, 4);
            return f;
        }

        uint16_t FloatToHalf(float f)
        {
            uint32_t bits;
            std::memcpy(&bits, &f, 4);
            const uint16_t sign = uint16_t((bits >> 16) & 0x8000);
            const uint32_t absBits = bits & 0x7FFFFFFF;
            if (absBits >= 0x7F800000)
                return uint16_t(sign | 0x7C00 | (absBits > 0x7F800000 ? 0x200 : 0));
            const int exp = int(absBits >> 23) - 127 + 15;
            if (exp >= 31)
                return uint16_t(sign | 0x7BFF); // clamp to the largest finite half
            if (exp <= 0)
            {
                if (exp < -10)
                    return sign;
                const uint32_t mant = (absBits & 0x7FFFFF) | 0x800000;
                return uint16_t(sign | (mant >> (14 - exp)));
            }
            return uint16_t(sign | (exp << 10) | ((absBits >> 13) & 0x3FF));
        }

        uint64_t Fnv(const void* data, size_t bytes, uint64_t h = 1469598103934665603ull)
        {
            const uint8_t* p = static_cast<const uint8_t*>(data);
            for (size_t i = 0; i < bytes; i++)
                h = (h ^ p[i]) * 1099511628211ull;
            return h;
        }

        WGPUCompareFunction Compare(uint32_t func)
        {
            static constexpr WGPUCompareFunction k[8] = { WGPUCompareFunction_Never, WGPUCompareFunction_Less, WGPUCompareFunction_Equal,
                WGPUCompareFunction_LessEqual, WGPUCompareFunction_Greater, WGPUCompareFunction_NotEqual,
                WGPUCompareFunction_GreaterEqual, WGPUCompareFunction_Always };
            return k[func & 7];
        }

        WGPUStencilOperation StencilOp(uint32_t op)
        {
            static constexpr WGPUStencilOperation k[8] = { WGPUStencilOperation_Keep, WGPUStencilOperation_Zero, WGPUStencilOperation_Replace,
                WGPUStencilOperation_IncrementClamp, WGPUStencilOperation_DecrementClamp, WGPUStencilOperation_Invert,
                WGPUStencilOperation_IncrementWrap, WGPUStencilOperation_DecrementWrap };
            return k[op & 7];
        }

        WGPUBlendFactor BlendFactor(uint32_t f)
        {
            switch (f)
            {
            case 0: return WGPUBlendFactor_Zero;
            case 1: return WGPUBlendFactor_One;
            case 4: return WGPUBlendFactor_Src;
            case 5: return WGPUBlendFactor_OneMinusSrc;
            case 6: return WGPUBlendFactor_SrcAlpha;
            case 7: return WGPUBlendFactor_OneMinusSrcAlpha;
            case 8: return WGPUBlendFactor_Dst;
            case 9: return WGPUBlendFactor_OneMinusDst;
            case 10: return WGPUBlendFactor_DstAlpha;
            case 11: return WGPUBlendFactor_OneMinusDstAlpha;
            case 12: case 14: return WGPUBlendFactor_Constant;          // constant alpha approximated
            case 13: case 15: return WGPUBlendFactor_OneMinusConstant;
            case 16: return WGPUBlendFactor_SrcAlphaSaturated;
            default: return WGPUBlendFactor_One;
            }
        }

        WGPUBlendOperation BlendOp(uint32_t op)
        {
            switch (op)
            {
            case 1: return WGPUBlendOperation_Subtract;
            case 2: return WGPUBlendOperation_Min;
            case 3: return WGPUBlendOperation_Max;
            case 4: return WGPUBlendOperation_ReverseSubtract;
            default: return WGPUBlendOperation_Add;
            }
        }

        bool BlendIsReplace(uint32_t control)
        {
            return (control & 0x1FFF) == 0x0001 && ((control >> 16) & 0x1FFF) == 0x0001;
        }

        WGPUTextureFormat ColorFormat(uint32_t xenosFormat)
        {
            return (xenosFormat == 14 || xenosFormat == 15) ? WGPUTextureFormat_RGBA32Float : WGPUTextureFormat_RGBA16Float;
        }

        bool IsUnormColor(uint32_t format) { return format == 0 || format == 1 || format == 2 || format == 10; }

        constexpr uint32_t kDrawDataStride = 4352; // misc vec4 + 256 vec4 = 4112, aligned to 256
        constexpr uint64_t kDrawRingBytes = 16ull << 20;
        constexpr uint64_t kVertexRingBytes = 64ull << 20;

        struct RenderTarget
        {
            WGPUTexture texture = nullptr;
            WGPUTextureView view = nullptr;
            WGPUTextureFormat format = WGPUTextureFormat_Undefined;
            uint32_t width = 0, height = 0;
            uint32_t baseTile = 0;
            bool depth = false;
        };

        struct GpuTexture
        {
            uint64_t version = 0;
            WGPUTexture texture = nullptr;
            WGPUTextureView view = nullptr;
            uint32_t width = 0, height = 0;
            WGPUTextureFormat format = WGPUTextureFormat_Undefined;
        };

        struct Stats
        {
            uint64_t draws = 0, triangles = 0, pipelinesCreated = 0, shadersCreated = 0, readbacks = 0, clears = 0,
                     textureUploads = 0, submits = 0, readbackNs = 0, encodeNs = 0;
        };

        class WebGpuBackend final : public CaptureSink
        {
        public:
            bool Init();

            void BeginDraw(const CaptureDraw& draw) override;
            void Triangle(const CaptureVertex& a, const CaptureVertex& b, const CaptureVertex& c, bool front) override;
            void EndDraw() override;
            void ReadColor(const RasterState& rs, uint32_t rt, int x0, int y0, int x1, int y1, std::vector<F4>& out) override;
            void ReadDepth(const RasterState& rs, int x0, int y0, int x1, int y1, std::vector<float>& depth, std::vector<uint8_t>& stencil) override;
            void ClearColor(const RasterState& rs, uint32_t rt, int x0, int y0, int x1, int y1, const F4& color) override;
            void ClearDepth(const RasterState& rs, int x0, int y0, int x1, int y1, float depth, uint8_t stencil) override;

            void EndFrame();
            void PrintStats();

        private:
            // ---- device
            WGPUInstance instance_ = nullptr;
            WGPUAdapter adapter_ = nullptr;
            WGPUDevice device_ = nullptr;
            WGPUQueue queue_ = nullptr;
            bool depthClipControl_ = false;

            // ---- recording
            WGPUCommandEncoder encoder_ = nullptr;
            WGPURenderPassEncoder pass_ = nullptr;
            RenderTarget* passColor_[4] = {};
            RenderTarget* passDepth_ = nullptr;

            // ---- rings (offsets reset after every submit)
            WGPUBuffer drawRing_ = nullptr, vertexRing_ = nullptr;
            uint64_t drawOffset_ = 0, vertexOffset_ = 0;

            // ---- caches
            std::map<std::tuple<bool, uint32_t, uint32_t, uint32_t, uint32_t, int>, std::unique_ptr<RenderTarget>> targets_;
            std::unordered_map<uint64_t, WGPUShaderModule> modules_;
            std::unordered_map<std::string, WGPURenderPipeline> pipelines_;
            std::unordered_map<uint32_t, WGPUBindGroupLayout> textureLayouts_;
            std::unordered_map<uint32_t, WGPUPipelineLayout> pipelineLayouts_;
            std::map<std::tuple<int, int, int>, WGPUSampler> samplers_;
            std::unordered_map<const Texture*, GpuTexture> textures_;
            WGPUBindGroupLayout drawLayout_ = nullptr;
            WGPUBindGroup drawBindGroup_ = nullptr;
            GpuTexture dummy_;
            WGPUBuffer readback_ = nullptr;
            uint64_t readbackSize_ = 0;
            WGPUShaderModule clearModule_ = nullptr;

            // ---- current draw
            const RasterState* rs_ = nullptr;
            RenderTarget* drawColor_[4] = {};
            RenderTarget* drawDepth_ = nullptr;
            WGPURenderPipeline drawPipeline_ = nullptr;
            WGPUBindGroup drawTextures_ = nullptr;
            uint32_t drawDataOffset_ = 0;
            uint32_t interpolatorMask_ = 0;
            uint32_t floatsPerVertex_ = 4;
            std::vector<float> vertices_;
            bool skip_ = false;

            Stats stats_;

            // ---- helpers
            RenderTarget* Target(bool depth, const Surface& s, WGPUTextureFormat format);
            void EnsureEncoder();
            void EndPass();
            void Submit();
            void BeginPass(RenderTarget* const color[4], RenderTarget* depth);
            uint32_t WriteDrawData(const float misc[4], const F4* constants, uint32_t count);
            WGPUShaderModule Module(const std::string& code);
            WGPUBindGroupLayout TextureLayout(uint32_t slots);
            WGPUPipelineLayout PipelineLayout(uint32_t slots);
            WGPUSampler Sampler(const Texture& tex);
            const GpuTexture& Upload(const Texture& tex);
            void ReadTexture(WGPUTexture texture, WGPUTextureAspect aspect, uint32_t bytesPerPixel, int x0, int y0, int w, int h,
                std::vector<uint8_t>& out, uint32_t& rowBytes);
            void ClearRect(RenderTarget* target, int x0, int y0, int x1, int y1, const F4& color, float depth, uint8_t stencil);
        };

        WebGpuBackend* g_backend = nullptr;

        bool WebGpuBackend::Init()
        {
            instance_ = wgpuCreateInstance(nullptr);
            if (!instance_)
                return false;

            struct AdapterResult { WGPUAdapter adapter = nullptr; bool done = false; } ar;
            WGPURequestAdapterOptions options = WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
            options.powerPreference = WGPUPowerPreference_HighPerformance;
            WGPURequestAdapterCallbackInfo acb = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
            acb.mode = WGPUCallbackMode_AllowProcessEvents;
            acb.callback = [](WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView message, void* u1, void*) {
                auto* r = static_cast<AdapterResult*>(u1);
                r->adapter = status == WGPURequestAdapterStatus_Success ? adapter : nullptr;
                r->done = true;
                if (status != WGPURequestAdapterStatus_Success)
                    std::fprintf(stderr, "[gpu] webgpu: no adapter: %s\n", FromView(message).c_str());
            };
            acb.userdata1 = &ar;
            wgpuInstanceRequestAdapter(instance_, &options, acb);
            while (!ar.done)
                wgpuInstanceProcessEvents(instance_);
            adapter_ = ar.adapter;
            if (!adapter_)
                return false;

            WGPUAdapterInfo info = WGPU_ADAPTER_INFO_INIT;
            if (wgpuAdapterGetInfo(adapter_, &info) == WGPUStatus_Success)
            {
                std::fprintf(stderr, "[gpu] webgpu: %s (%s)\n", FromView(info.device).c_str(), FromView(info.description).c_str());
                wgpuAdapterInfoFreeMembers(info);
            }

            std::vector<WGPUFeatureName> features;
            if (!wgpuAdapterHasFeature(adapter_, WGPUFeatureName_Depth32FloatStencil8))
            {
                std::fprintf(stderr, "[gpu] webgpu: adapter lacks depth32float-stencil8\n");
                return false;
            }
            features.push_back(WGPUFeatureName_Depth32FloatStencil8);
            if (wgpuAdapterHasFeature(adapter_, WGPUFeatureName_DepthClipControl))
            {
                features.push_back(WGPUFeatureName_DepthClipControl);
                depthClipControl_ = true;
            }
            if (wgpuAdapterHasFeature(adapter_, WGPUFeatureName_Float32Filterable))
                features.push_back(WGPUFeatureName_Float32Filterable);

            struct DeviceResult { WGPUDevice device = nullptr; bool done = false; } dr;
            WGPUDeviceDescriptor dd = WGPU_DEVICE_DESCRIPTOR_INIT;
            dd.label = Str("ttt2");
            dd.requiredFeatureCount = features.size();
            dd.requiredFeatures = features.data();
            dd.uncapturedErrorCallbackInfo.callback = [](WGPUDevice const*, WGPUErrorType type, WGPUStringView message, void*, void*) {
                static std::atomic<int> reported{ 0 };
                if (reported++ < 20)
                    std::fprintf(stderr, "[gpu] webgpu error %d: %s\n", int(type), FromView(message).c_str());
            };
            dd.deviceLostCallbackInfo.callback = [](WGPUDevice const*, WGPUDeviceLostReason reason, WGPUStringView message, void*, void*) {
                if (reason != WGPUDeviceLostReason_Destroyed)
                    std::fprintf(stderr, "[gpu] webgpu device lost (%d): %s\n", int(reason), FromView(message).c_str());
            };
            WGPURequestDeviceCallbackInfo dcb = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
            dcb.mode = WGPUCallbackMode_AllowProcessEvents;
            dcb.callback = [](WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView message, void* u1, void*) {
                auto* r = static_cast<DeviceResult*>(u1);
                r->device = status == WGPURequestDeviceStatus_Success ? device : nullptr;
                r->done = true;
                if (status != WGPURequestDeviceStatus_Success)
                    std::fprintf(stderr, "[gpu] webgpu: no device: %s\n", FromView(message).c_str());
            };
            dcb.userdata1 = &dr;
            wgpuAdapterRequestDevice(adapter_, &dd, dcb);
            while (!dr.done)
                wgpuInstanceProcessEvents(instance_);
            device_ = dr.device;
            if (!device_)
                return false;
            queue_ = wgpuDeviceGetQueue(device_);

            WGPUBufferDescriptor bd = WGPU_BUFFER_DESCRIPTOR_INIT;
            bd.label = Str("draw data");
            bd.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
            bd.size = kDrawRingBytes;
            drawRing_ = wgpuDeviceCreateBuffer(device_, &bd);
            bd.label = Str("vertices");
            bd.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst;
            bd.size = kVertexRingBytes;
            vertexRing_ = wgpuDeviceCreateBuffer(device_, &bd);

            WGPUBindGroupLayoutEntry entry = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
            entry.binding = 0;
            entry.visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
            entry.buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
            entry.buffer.hasDynamicOffset = true;
            entry.buffer.minBindingSize = 4112;
            WGPUBindGroupLayoutDescriptor bld = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
            bld.entryCount = 1;
            bld.entries = &entry;
            drawLayout_ = wgpuDeviceCreateBindGroupLayout(device_, &bld);
            WGPUBindGroupEntry bge = WGPU_BIND_GROUP_ENTRY_INIT;
            bge.binding = 0;
            bge.buffer = drawRing_;
            bge.offset = 0;
            bge.size = 4112;
            WGPUBindGroupDescriptor bgd = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
            bgd.layout = drawLayout_;
            bgd.entryCount = 1;
            bgd.entries = &bge;
            drawBindGroup_ = wgpuDeviceCreateBindGroup(device_, &bgd);

            // 1x1 transparent black for slots whose texture could not be decoded.
            WGPUTextureDescriptor td = WGPU_TEXTURE_DESCRIPTOR_INIT;
            td.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
            td.size = { 1, 1, 1 };
            td.format = WGPUTextureFormat_RGBA8Unorm;
            dummy_.texture = wgpuDeviceCreateTexture(device_, &td);
            dummy_.view = wgpuTextureCreateView(dummy_.texture, nullptr);
            dummy_.width = dummy_.height = 1;
            const uint32_t zero = 0;
            WGPUTexelCopyTextureInfo dst = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
            dst.texture = dummy_.texture;
            WGPUTexelCopyBufferLayout layout = WGPU_TEXEL_COPY_BUFFER_LAYOUT_INIT;
            layout.bytesPerRow = 4;
            layout.rowsPerImage = 1;
            const WGPUExtent3D one = { 1, 1, 1 };
            wgpuQueueWriteTexture(queue_, &dst, &zero, 4, &layout, &one);

            // Clears: a full-screen triangle at depth D.c[0].x writing colour D.misc.
            clearModule_ = Module(R"(
struct DrawData { misc: vec4f, c: array<vec4f, 256>, }
@group(0) @binding(0) var<storage, read> D: DrawData;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
    return vec4f(p, D.c[0].x, 1.0);
}
@fragment fn fs() -> @location(0) vec4f { return D.misc; }
)");
            return true;
        }

        WGPUShaderModule WebGpuBackend::Module(const std::string& code)
        {
            const uint64_t key = Fnv(code.data(), code.size());
            auto it = modules_.find(key);
            if (it != modules_.end())
                return it->second;
            WGPUShaderSourceWGSL source = WGPU_SHADER_SOURCE_WGSL_INIT;
            source.code = Str(code);
            WGPUShaderModuleDescriptor md = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
            md.nextInChain = &source.chain;
            WGPUShaderModule module = wgpuDeviceCreateShaderModule(device_, &md);
            ++stats_.shadersCreated;
            static const bool dumpShaders = [] { const char* v = std::getenv("TTT2_WGSL_DUMP"); return v && *v == '1'; }();
            if (dumpShaders)
            {
                char name[64];
                std::snprintf(name, sizeof(name), "shader_%016llx.wgsl", (unsigned long long)key);
                if (std::FILE* f = std::fopen(name, "wb"))
                {
                    std::fwrite(code.data(), 1, code.size(), f);
                    std::fclose(f);
                }
            }
            modules_[key] = module;
            return module;
        }

        WGPUBindGroupLayout WebGpuBackend::TextureLayout(uint32_t slots)
        {
            auto it = textureLayouts_.find(slots);
            if (it != textureLayouts_.end())
                return it->second;
            std::vector<WGPUBindGroupLayoutEntry> entries;
            for (uint32_t slot = 0; slot < 32; slot++)
            {
                if (!(slots & (1u << slot)))
                    continue;
                WGPUBindGroupLayoutEntry t = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
                t.binding = slot * 2;
                t.visibility = WGPUShaderStage_Fragment;
                t.texture.sampleType = WGPUTextureSampleType_Float;
                t.texture.viewDimension = WGPUTextureViewDimension_2D;
                entries.push_back(t);
                WGPUBindGroupLayoutEntry s = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
                s.binding = slot * 2 + 1;
                s.visibility = WGPUShaderStage_Fragment;
                s.sampler.type = WGPUSamplerBindingType_Filtering;
                entries.push_back(s);
            }
            WGPUBindGroupLayoutDescriptor d = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
            d.entryCount = entries.size();
            d.entries = entries.data();
            WGPUBindGroupLayout layout = wgpuDeviceCreateBindGroupLayout(device_, &d);
            textureLayouts_[slots] = layout;
            return layout;
        }

        WGPUPipelineLayout WebGpuBackend::PipelineLayout(uint32_t slots)
        {
            auto it = pipelineLayouts_.find(slots);
            if (it != pipelineLayouts_.end())
                return it->second;
            WGPUBindGroupLayout layouts[2] = { drawLayout_, TextureLayout(slots) };
            WGPUPipelineLayoutDescriptor d = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
            d.bindGroupLayoutCount = 2;
            d.bindGroupLayouts = layouts;
            WGPUPipelineLayout layout = wgpuDeviceCreatePipelineLayout(device_, &d);
            pipelineLayouts_[slots] = layout;
            return layout;
        }

        WGPUSampler WebGpuBackend::Sampler(const Texture& tex)
        {
            auto address = [](uint32_t mode) {
                return mode == 0 ? WGPUAddressMode_Repeat : mode == 1 ? WGPUAddressMode_MirrorRepeat : WGPUAddressMode_ClampToEdge;
            };
            const int u = int(tex.clampX), v = int(tex.clampY), linear = tex.magFilter == 1 ? 1 : 0;
            const auto key = std::make_tuple(u, v, linear);
            auto it = samplers_.find(key);
            if (it != samplers_.end())
                return it->second;
            WGPUSamplerDescriptor d = WGPU_SAMPLER_DESCRIPTOR_INIT;
            d.addressModeU = address(tex.clampX);
            d.addressModeV = address(tex.clampY);
            d.addressModeW = WGPUAddressMode_ClampToEdge;
            d.magFilter = d.minFilter = linear ? WGPUFilterMode_Linear : WGPUFilterMode_Nearest;
            d.mipmapFilter = WGPUMipmapFilterMode_Nearest;
            WGPUSampler sampler = wgpuDeviceCreateSampler(device_, &d);
            samplers_[key] = sampler;
            return sampler;
        }

        const GpuTexture& WebGpuBackend::Upload(const Texture& tex)
        {
            if (!tex.ok || !tex.width || !tex.height || tex.texels.size() < size_t(tex.width) * tex.height)
                return dummy_;
            GpuTexture& g = textures_[&tex];
            if (g.texture && g.version == tex.version)
                return g;

            // Texels within [0, 1] (most textures) upload as rgba8; others keep half floats.
            const size_t count = size_t(tex.width) * tex.height;
            std::atomic<bool> unorm{ true };
            const uint32_t chunks = uint32_t((count + 65535) / 65536);
            SoftParallelFor(chunks, [&](uint32_t c) {
                const size_t end = std::min(count, size_t(c + 1) * 65536);
                for (size_t i = size_t(c) * 65536; i < end && unorm.load(std::memory_order_relaxed); i++)
                    for (float v : tex.texels[i])
                        if (!(v >= 0.0f && v <= 1.0f))
                        {
                            unorm = false;
                            break;
                        }
            });
            const WGPUTextureFormat format = unorm ? WGPUTextureFormat_RGBA8Unorm : WGPUTextureFormat_RGBA16Float;
            if (!g.texture || g.width != tex.width || g.height != tex.height || g.format != format)
            {
                if (g.view)
                    wgpuTextureViewRelease(g.view);
                if (g.texture)
                    wgpuTextureRelease(g.texture);
                WGPUTextureDescriptor td = WGPU_TEXTURE_DESCRIPTOR_INIT;
                td.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
                td.size = { tex.width, tex.height, 1 };
                td.format = format;
                g.texture = wgpuDeviceCreateTexture(device_, &td);
                g.view = wgpuTextureCreateView(g.texture, nullptr);
                g.width = tex.width;
                g.height = tex.height;
                g.format = format;
            }
            g.version = tex.version;

            const uint32_t bpp = unorm ? 4 : 8;
            static std::vector<uint8_t> staging;
            staging.resize(count * bpp);
            SoftParallelFor(chunks, [&](uint32_t c) {
                const size_t end = std::min(count, size_t(c + 1) * 65536);
                for (size_t i = size_t(c) * 65536; i < end; i++)
                {
                    const F4& t = tex.texels[i];
                    if (unorm)
                        for (int k = 0; k < 4; k++)
                            staging[i * 4 + k] = uint8_t(t[k] * 255.0f + 0.5f);
                    else
                        for (int k = 0; k < 4; k++)
                        {
                            const uint16_t h = FloatToHalf(t[k]);
                            std::memcpy(&staging[i * 8 + k * 2], &h, 2);
                        }
                }
            });
            WGPUTexelCopyTextureInfo dst = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
            dst.texture = g.texture;
            WGPUTexelCopyBufferLayout layout = WGPU_TEXEL_COPY_BUFFER_LAYOUT_INIT;
            layout.bytesPerRow = tex.width * bpp;
            layout.rowsPerImage = tex.height;
            const WGPUExtent3D size = { tex.width, tex.height, 1 };
            wgpuQueueWriteTexture(queue_, &dst, staging.data(), staging.size(), &layout, &size);
            ++stats_.textureUploads;
            return g;
        }

        RenderTarget* WebGpuBackend::Target(bool depth, const Surface& s, WGPUTextureFormat format)
        {
            if (!s.pitch)
                return nullptr;
            const auto key = std::make_tuple(depth, s.baseTile, s.pitch, s.scaleX, s.scaleY, int(format));
            auto& slot = targets_[key];
            if (slot)
                return slot.get();
            auto rt = std::make_unique<RenderTarget>();
            rt->depth = depth;
            rt->format = format;
            rt->baseTile = s.baseTile;
            rt->width = s.pitch;
            // As many rows as EDRAM could hold at this pitch (2048 at 1280 wide, 1x).
            const uint64_t rows = uint64_t(kEdramTiles) * kTileSamples / (uint64_t(s.pitch) * s.scaleX) / s.scaleY;
            rt->height = uint32_t(std::clamp<uint64_t>(rows, 16, 2048));
            WGPUTextureDescriptor td = WGPU_TEXTURE_DESCRIPTOR_INIT;
            td.label = Str(depth ? "depth target" : "colour target");
            td.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc | WGPUTextureUsage_TextureBinding;
            td.size = { rt->width, rt->height, 1 };
            td.format = format;
            rt->texture = wgpuDeviceCreateTexture(device_, &td);
            rt->view = wgpuTextureCreateView(rt->texture, nullptr);
            slot = std::move(rt);
            return slot.get();
        }

        void WebGpuBackend::EnsureEncoder()
        {
            if (!encoder_)
            {
                WGPUCommandEncoderDescriptor d = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
                encoder_ = wgpuDeviceCreateCommandEncoder(device_, &d);
            }
        }

        void WebGpuBackend::EndPass()
        {
            if (pass_)
            {
                wgpuRenderPassEncoderEnd(pass_);
                wgpuRenderPassEncoderRelease(pass_);
                pass_ = nullptr;
            }
            std::fill(std::begin(passColor_), std::end(passColor_), nullptr);
            passDepth_ = nullptr;
        }

        void WebGpuBackend::Submit()
        {
            EndPass();
            if (encoder_)
            {
                WGPUCommandBufferDescriptor d = WGPU_COMMAND_BUFFER_DESCRIPTOR_INIT;
                WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder_, &d);
                wgpuQueueSubmit(queue_, 1, &commands);
                wgpuCommandBufferRelease(commands);
                wgpuCommandEncoderRelease(encoder_);
                encoder_ = nullptr;
                ++stats_.submits;
            }
            drawOffset_ = 0;
            vertexOffset_ = 0;
        }

        void WebGpuBackend::BeginPass(RenderTarget* const color[4], RenderTarget* depth)
        {
            if (pass_ && std::equal(color, color + 4, passColor_) && depth == passDepth_)
                return;
            EndPass();
            EnsureEncoder();
            WGPURenderPassColorAttachment attachments[4];
            uint32_t count = 0;
            for (int k = 0; k < 4; k++)
            {
                attachments[k] = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
                if (color[k])
                {
                    attachments[k].view = color[k]->view;
                    attachments[k].loadOp = WGPULoadOp_Load;
                    attachments[k].storeOp = WGPUStoreOp_Store;
                    count = uint32_t(k + 1);
                }
            }
            WGPURenderPassDepthStencilAttachment ds = WGPU_RENDER_PASS_DEPTH_STENCIL_ATTACHMENT_INIT;
            if (depth)
            {
                ds.view = depth->view;
                ds.depthLoadOp = WGPULoadOp_Load;
                ds.depthStoreOp = WGPUStoreOp_Store;
                ds.stencilLoadOp = WGPULoadOp_Load;
                ds.stencilStoreOp = WGPUStoreOp_Store;
            }
            WGPURenderPassDescriptor d = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
            d.colorAttachmentCount = count;
            d.colorAttachments = attachments;
            d.depthStencilAttachment = depth ? &ds : nullptr;
            pass_ = wgpuCommandEncoderBeginRenderPass(encoder_, &d);
            std::copy(color, color + 4, passColor_);
            passDepth_ = depth;
        }

        uint32_t WebGpuBackend::WriteDrawData(const float misc[4], const F4* constants, uint32_t count)
        {
            if (drawOffset_ + kDrawDataStride > kDrawRingBytes)
                Submit();
            const uint32_t offset = uint32_t(drawOffset_);
            float data[4 + 256 * 4] = {};
            std::memcpy(data, misc, 16);
            if (constants)
                std::memcpy(data + 4, constants, size_t(count) * 16);
            wgpuQueueWriteBuffer(queue_, drawRing_, offset, data, 16 + size_t(count) * 16);
            drawOffset_ += kDrawDataStride;
            return offset;
        }

        // ---- draws ------------------------------------------------------------------------

        void WebGpuBackend::BeginDraw(const CaptureDraw& draw)
        {
            const auto start = std::chrono::steady_clock::now();
            const RasterState& rs = *draw.rs;
            rs_ = &rs;
            skip_ = false;
            vertices_.clear();
            interpolatorMask_ = rs.interpolatorMask;

            // Targets: colour for mode 4 (where a write mask is set), depth when tested,
            // written or stencilled.
            std::fill(std::begin(drawColor_), std::end(drawColor_), nullptr);
            drawDepth_ = nullptr;
            if (rs.mode == 4)
                for (int k = 0; k < 4; k++)
                    if ((rs.colorMask >> (k * 4)) & 0xF)
                        drawColor_[k] = Target(false, rs.color[k], ColorFormat(rs.color[k].format));
            if (rs.depthTest || rs.stencil)
                drawDepth_ = Target(true, rs.depth, WGPUTextureFormat_Depth32FloatStencil8);
            const RenderTarget* any = drawDepth_;
            for (auto* c : drawColor_)
                if (c && !any)
                    any = c;
            if (!any)
            {
                skip_ = true;
                return;
            }
            // All attachments of a pass must have the same size, and a target may be bound
            // once (unused colour slots often still point at RT0's base): drop the rest.
            for (int k = 0; k < 4; k++)
            {
                auto*& c = drawColor_[k];
                if (c && (c->width != any->width || c->height != any->height))
                    c = nullptr;
                for (int j = 0; j < k && c; j++)
                    if (drawColor_[j] == c)
                        c = nullptr;
            }

            // ---- shader
            WgslOptions options;
            options.interpolatorMask = rs.interpolatorMask;
            options.paramGen = rs.paramGen;
            options.paramGenRegister = rs.paramGenRegister;
            options.alphaFunc = rs.alphaTest ? rs.alphaFunc : 7;
            for (int k = 0; k < 4; k++)
                if (drawColor_[k])
                    options.colorTargets |= 1u << k;
            if (rs.mode == 4 && !draw.ps)
            {
                static bool warned = false;
                if (!warned)
                {
                    warned = true;
                    std::fprintf(stderr, "[gpu] webgpu: pixel shader with predicate-driven flow not translated (colour skipped)\n");
                }
            }
            const bool hasFragment = rs.mode == 4 && options.colorTargets && draw.ps;
            const WgslShader shader = TranslateToWgsl(hasFragment ? draw.ps : nullptr, options);
            const WGPUShaderModule module = Module(shader.code);
            const uint32_t slots = hasFragment ? shader.textureSlots : 0;

            // ---- pipeline
            floatsPerVertex_ = 4;
            std::vector<WGPUVertexAttribute> attributes;
            {
                WGPUVertexAttribute a = WGPU_VERTEX_ATTRIBUTE_INIT;
                a.format = WGPUVertexFormat_Float32x4;
                a.offset = 0;
                a.shaderLocation = 0;
                attributes.push_back(a);
                uint32_t location = 1;
                for (uint32_t i = 0; i < 16; i++)
                    if (rs.interpolatorMask & (1u << i))
                    {
                        a.offset = floatsPerVertex_ * 4;
                        a.shaderLocation = location++;
                        attributes.push_back(a);
                        floatsPerVertex_ += 4;
                    }
            }

            std::string key;
            auto add = [&](uint64_t v) { key.append(reinterpret_cast<const char*>(&v), sizeof(v)); };
            add(Fnv(shader.code.data(), shader.code.size()));
            add(rs.interpolatorMask);
            add(hasFragment);
            for (int k = 0; k < 4; k++)
            {
                add(drawColor_[k] ? uint64_t(drawColor_[k]->format) : 0);
                add(drawColor_[k] && (shader.colorExports & (1u << k)) ? (rs.colorMask >> (k * 4)) & 0xF : 0);
                add(drawColor_[k] ? rs.blend[k] : 0);
                add(drawColor_[k] ? rs.color[k].format : 0);
            }
            add(drawDepth_ != nullptr);
            add(rs.depthTest);
            add(rs.depthWrite);
            add(rs.depthFunc);
            add(rs.stencil);
            for (const auto& f : rs.stencilFace)
            {
                add(f.func);
                add(f.fail);
                add(f.zpass);
                add(f.zfail);
            }
            add(rs.stencilFace[0].mask);
            add(rs.stencilFace[0].writeMask);

            WGPURenderPipeline& pipeline = pipelines_[key];
            if (!pipeline)
            {
                WGPUVertexBufferLayout vbl = WGPU_VERTEX_BUFFER_LAYOUT_INIT;
                vbl.stepMode = WGPUVertexStepMode_Vertex;
                vbl.arrayStride = floatsPerVertex_ * 4;
                vbl.attributeCount = attributes.size();
                vbl.attributes = attributes.data();

                WGPURenderPipelineDescriptor pd = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
                pd.layout = PipelineLayout(slots);
                pd.vertex.module = module;
                pd.vertex.entryPoint = Str("vs");
                pd.vertex.bufferCount = 1;
                pd.vertex.buffers = &vbl;
                pd.primitive.topology = WGPUPrimitiveTopology_TriangleList;
                pd.primitive.frontFace = WGPUFrontFace_CCW;
                pd.primitive.cullMode = WGPUCullMode_None; // the front end culls
                pd.primitive.unclippedDepth = depthClipControl_;
                pd.multisample.count = 1;
                pd.multisample.mask = 0xFFFFFFFF;

                WGPUDepthStencilState dss = WGPU_DEPTH_STENCIL_STATE_INIT;
                if (drawDepth_)
                {
                    dss.format = drawDepth_->format;
                    dss.depthWriteEnabled = rs.depthWrite ? WGPUOptionalBool_True : WGPUOptionalBool_False;
                    dss.depthCompare = rs.depthTest ? Compare(rs.depthFunc) : WGPUCompareFunction_Always;
                    auto face = [&](const RasterState::StencilFace& f) {
                        WGPUStencilFaceState s = WGPU_STENCIL_FACE_STATE_INIT;
                        if (rs.stencil)
                        {
                            s.compare = Compare(f.func);
                            s.failOp = StencilOp(f.fail);
                            s.depthFailOp = StencilOp(f.zfail);
                            s.passOp = StencilOp(f.zpass);
                        }
                        else
                        {
                            s.compare = WGPUCompareFunction_Always;
                            s.failOp = s.depthFailOp = s.passOp = WGPUStencilOperation_Keep;
                        }
                        return s;
                    };
                    dss.stencilFront = face(rs.stencilFace[0]);
                    dss.stencilBack = face(rs.stencilFace[1]);
                    dss.stencilReadMask = rs.stencil ? rs.stencilFace[0].mask : 0xFF;
                    dss.stencilWriteMask = rs.stencil ? rs.stencilFace[0].writeMask : 0;
                    pd.depthStencil = &dss;
                }

                WGPUColorTargetState targets[4];
                WGPUBlendState blends[4];
                uint32_t targetCount = 0;
                for (int k = 0; k < 4; k++)
                {
                    targets[k] = WGPU_COLOR_TARGET_STATE_INIT;
                    if (!drawColor_[k])
                    {
                        targets[k].format = WGPUTextureFormat_Undefined;
                        continue;
                    }
                    targetCount = uint32_t(k + 1);
                    targets[k].format = drawColor_[k]->format;
                    targets[k].writeMask = (shader.colorExports & (1u << k)) ? WGPUColorWriteMask((rs.colorMask >> (k * 4)) & 0xF) : WGPUColorWriteMask_None;
                    const bool floatTarget = drawColor_[k]->format == WGPUTextureFormat_RGBA32Float;
                    if (!BlendIsReplace(rs.blend[k]) && !floatTarget)
                    {
                        const uint32_t c = rs.blend[k];
                        blends[k].color = { BlendOp((c >> 5) & 7), BlendFactor(c & 0x1F), BlendFactor((c >> 8) & 0x1F) };
                        blends[k].alpha = { BlendOp((c >> 21) & 7), BlendFactor((c >> 16) & 0x1F), BlendFactor((c >> 24) & 0x1F) };
                        // min/max ignore their factors on Xenos; WebGPU requires One.
                        for (WGPUBlendComponent* bc : { &blends[k].color, &blends[k].alpha })
                            if (bc->operation == WGPUBlendOperation_Min || bc->operation == WGPUBlendOperation_Max)
                                bc->srcFactor = bc->dstFactor = WGPUBlendFactor_One;
                        targets[k].blend = &blends[k];
                    }
                }
                WGPUFragmentState fs = WGPU_FRAGMENT_STATE_INIT;
                fs.module = module;
                fs.entryPoint = Str("fs");
                fs.targetCount = targetCount;
                fs.targets = targets;
                pd.fragment = (hasFragment || shader.depthExport) ? &fs : nullptr;
                if (!pd.fragment && targetCount)
                {
                    // Colour targets bound without a pixel shader: write nothing to them.
                    for (uint32_t k = 0; k < targetCount; k++)
                        targets[k].writeMask = WGPUColorWriteMask_None;
                    pd.fragment = &fs;
                }
                pipeline = wgpuDeviceCreateRenderPipeline(device_, &pd);
                ++stats_.pipelinesCreated;
            }
            drawPipeline_ = pipeline;

            // ---- constants and textures
            const float misc[4] = { rs.alphaRef, 0, 0, 0 };
            drawDataOffset_ = WriteDrawData(misc, hasFragment ? draw.psConstants : nullptr, hasFragment ? 256 : 0);

            if (drawTextures_)
            {
                wgpuBindGroupRelease(drawTextures_);
                drawTextures_ = nullptr;
            }
            std::vector<WGPUBindGroupEntry> entries;
            for (uint32_t slot = 0; slot < 32; slot++)
            {
                if (!(slots & (1u << slot)))
                    continue;
                const Texture& tex = draw.texture(slot);
                const GpuTexture& g = Upload(tex);
                WGPUBindGroupEntry t = WGPU_BIND_GROUP_ENTRY_INIT;
                t.binding = slot * 2;
                t.textureView = g.view;
                entries.push_back(t);
                WGPUBindGroupEntry s = WGPU_BIND_GROUP_ENTRY_INIT;
                s.binding = slot * 2 + 1;
                s.sampler = Sampler(tex);
                entries.push_back(s);
            }
            WGPUBindGroupDescriptor bgd = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
            bgd.layout = TextureLayout(slots);
            bgd.entryCount = entries.size();
            bgd.entries = entries.data();
            drawTextures_ = wgpuDeviceCreateBindGroup(device_, &bgd);
            stats_.encodeNs += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
        }

        void WebGpuBackend::Triangle(const CaptureVertex& a, const CaptureVertex& b, const CaptureVertex& c, bool front)
        {
            if (skip_)
                return;
            const RasterState& rs = *rs_;
            const RenderTarget* rt = drawDepth_;
            for (auto* t : drawColor_)
                if (t && !rt)
                    rt = t;
            const float width = float(rt->width), height = float(rt->height);
            // Xenos samples pixel p at p + pixelCenter, WebGPU at p + 0.5.
            const float shift = 0.5f - rs.pixelCenter;

            const CaptureVertex* v[3] = { &a, &b, &c };
            float ndc[3][2];
            for (int i = 0; i < 3; i++)
            {
                ndc[i][0] = (v[i]->x + shift) / width * 2.0f - 1.0f;
                ndc[i][1] = 1.0f - (v[i]->y + shift) / height * 2.0f;
            }
            // Front faces wind counter-clockwise in NDC for the pipeline (two-sided stencil).
            const float area = (ndc[1][0] - ndc[0][0]) * (ndc[2][1] - ndc[0][1]) - (ndc[2][0] - ndc[0][0]) * (ndc[1][1] - ndc[0][1]);
            int order[3] = { 0, 1, 2 };
            if ((area > 0.0f) != front)
                std::swap(order[1], order[2]);

            for (int o = 0; o < 3; o++)
            {
                const int i = order[o];
                const CaptureVertex& cv = *v[i];
                const float w = (cv.invW != 0.0f && std::isfinite(cv.invW)) ? 1.0f / cv.invW : 1.0f;
                vertices_.push_back(ndc[i][0] * w);
                vertices_.push_back(ndc[i][1] * w);
                vertices_.push_back(cv.z * w);
                vertices_.push_back(w);
                for (uint32_t k = 0; k < 16; k++)
                    if (interpolatorMask_ & (1u << k))
                        vertices_.insert(vertices_.end(), cv.interpolators[k].begin(), cv.interpolators[k].end());
            }
            ++stats_.triangles;
        }

        void WebGpuBackend::EndDraw()
        {
            if (skip_ || vertices_.empty() || !drawPipeline_)
                return;
            const auto start = std::chrono::steady_clock::now();
            const RasterState& rs = *rs_;
            const uint64_t bytes = vertices_.size() * sizeof(float);
            if (bytes > kVertexRingBytes)
                return;
            if (vertexOffset_ + bytes > kVertexRingBytes)
                Submit();
            const uint64_t offset = vertexOffset_;
            wgpuQueueWriteBuffer(queue_, vertexRing_, offset, vertices_.data(), bytes);
            vertexOffset_ = (vertexOffset_ + bytes + 255) & ~255ull;

            BeginPass(drawColor_, drawDepth_);
            const RenderTarget* rt = drawDepth_;
            for (auto* t : drawColor_)
                if (t && !rt)
                    rt = t;
            wgpuRenderPassEncoderSetPipeline(pass_, drawPipeline_);
            wgpuRenderPassEncoderSetBindGroup(pass_, 0, drawBindGroup_, 1, &drawDataOffset_);
            wgpuRenderPassEncoderSetBindGroup(pass_, 1, drawTextures_, 0, nullptr);
            wgpuRenderPassEncoderSetVertexBuffer(pass_, 0, vertexRing_, offset, bytes);
            wgpuRenderPassEncoderSetViewport(pass_, 0, 0, float(rt->width), float(rt->height), 0.0f, 1.0f);
            const int x0 = std::clamp(rs.sx0, 0, int(rt->width)), y0 = std::clamp(rs.sy0, 0, int(rt->height));
            const int x1 = std::clamp(rs.sx1, x0, int(rt->width)), y1 = std::clamp(rs.sy1, y0, int(rt->height));
            if (x1 <= x0 || y1 <= y0)
                return;
            wgpuRenderPassEncoderSetScissorRect(pass_, uint32_t(x0), uint32_t(y0), uint32_t(x1 - x0), uint32_t(y1 - y0));
            const WGPUColor blend = { rs.blendConstant[0], rs.blendConstant[1], rs.blendConstant[2], rs.blendConstant[3] };
            wgpuRenderPassEncoderSetBlendConstant(pass_, &blend);
            wgpuRenderPassEncoderSetStencilReference(pass_, rs.stencilFace[0].ref);
            wgpuRenderPassEncoderDraw(pass_, uint32_t(vertices_.size() / floatsPerVertex_), 1, 0, 0);
            ++stats_.draws;
            stats_.encodeNs += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
        }

        // ---- resolves -------------------------------------------------------------------------

        void WebGpuBackend::ReadTexture(WGPUTexture texture, WGPUTextureAspect aspect, uint32_t bytesPerPixel, int x0, int y0, int w, int h,
            std::vector<uint8_t>& out, uint32_t& rowBytes)
        {
            const auto start = std::chrono::steady_clock::now();
            rowBytes = (uint32_t(w) * bytesPerPixel + 255) & ~255u;
            const uint64_t size = uint64_t(rowBytes) * uint32_t(h);
            if (size > readbackSize_)
            {
                if (readback_)
                    wgpuBufferRelease(readback_);
                WGPUBufferDescriptor bd = WGPU_BUFFER_DESCRIPTOR_INIT;
                bd.label = Str("readback");
                bd.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
                bd.size = (size + 0xFFFF) & ~0xFFFFull;
                readback_ = wgpuDeviceCreateBuffer(device_, &bd);
                readbackSize_ = bd.size;
            }
            EndPass();
            EnsureEncoder();
            WGPUTexelCopyTextureInfo src = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
            src.texture = texture;
            src.origin = { uint32_t(x0), uint32_t(y0), 0 };
            src.aspect = aspect;
            WGPUTexelCopyBufferInfo dst = WGPU_TEXEL_COPY_BUFFER_INFO_INIT;
            dst.buffer = readback_;
            dst.layout.bytesPerRow = rowBytes;
            dst.layout.rowsPerImage = uint32_t(h);
            const WGPUExtent3D extent = { uint32_t(w), uint32_t(h), 1 };
            wgpuCommandEncoderCopyTextureToBuffer(encoder_, &src, &dst, &extent);
            Submit();

            struct MapResult { bool done = false; bool ok = false; } mr;
            WGPUBufferMapCallbackInfo cb = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
            cb.mode = WGPUCallbackMode_AllowProcessEvents;
            cb.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* u1, void*) {
                auto* r = static_cast<MapResult*>(u1);
                r->ok = status == WGPUMapAsyncStatus_Success;
                r->done = true;
            };
            cb.userdata1 = &mr;
            wgpuBufferMapAsync(readback_, WGPUMapMode_Read, 0, size_t(size), cb);
            while (!mr.done)
            {
#ifndef __EMSCRIPTEN__
                wgpuDevicePoll(device_, true, nullptr);
#endif
                wgpuInstanceProcessEvents(instance_);
            }
            out.resize(size_t(size));
            if (mr.ok)
            {
                std::memcpy(out.data(), wgpuBufferGetConstMappedRange(readback_, 0, size_t(size)), size_t(size));
                wgpuBufferUnmap(readback_);
            }
            ++stats_.readbacks;
            stats_.readbackNs += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
        }

        void WebGpuBackend::ReadColor(const RasterState& rs, uint32_t rtIndex, int x0, int y0, int x1, int y1, std::vector<F4>& out)
        {
            const int w = x1 - x0, h = y1 - y0;
            out.assign(size_t(w) * h, F4{ 0, 0, 0, 0 });
            const Surface& s = rs.color[rtIndex & 3];
            RenderTarget* rt = Target(false, s, ColorFormat(s.format));
            if (!rt)
                return;
            const int cx1 = std::min(x1, int(rt->width)), cy1 = std::min(y1, int(rt->height));
            if (cx1 <= x0 || cy1 <= y0)
                return;
            const bool wide = rt->format == WGPUTextureFormat_RGBA32Float;
            const uint32_t bpp = wide ? 16 : 8;
            std::vector<uint8_t> data;
            uint32_t rowBytes = 0;
            ReadTexture(rt->texture, WGPUTextureAspect_All, bpp, x0, y0, cx1 - x0, cy1 - y0, data, rowBytes);
            SoftParallelFor(uint32_t(cy1 - y0), [&](uint32_t row) {
                const uint8_t* line = data.data() + size_t(row) * rowBytes;
                for (int x = 0; x < cx1 - x0; x++)
                {
                    F4& c = out[size_t(row) * w + x];
                    if (wide)
                        std::memcpy(c.data(), line + size_t(x) * 16, 16);
                    else
                        for (int k = 0; k < 4; k++)
                        {
                            uint16_t hbits;
                            std::memcpy(&hbits, line + size_t(x) * 8 + k * 2, 2);
                            c[k] = HalfToFloat(hbits);
                        }
                    if (IsUnormColor(s.format))
                        for (float& v : c)
                            v = std::clamp(v, 0.0f, 1.0f);
                }
            });
        }

        void WebGpuBackend::ReadDepth(const RasterState& rs, int x0, int y0, int x1, int y1, std::vector<float>& depth, std::vector<uint8_t>& stencil)
        {
            const int w = x1 - x0, h = y1 - y0;
            depth.assign(size_t(w) * h, 0.0f);
            stencil.assign(size_t(w) * h, 0);
            RenderTarget* rt = Target(true, rs.depth, WGPUTextureFormat_Depth32FloatStencil8);
            if (!rt)
                return;
            const int cx1 = std::min(x1, int(rt->width)), cy1 = std::min(y1, int(rt->height));
            if (cx1 <= x0 || cy1 <= y0)
                return;
            // Depth/stencil formats only copy whole subresources: read it all, then crop.
            std::vector<uint8_t> data;
            uint32_t rowBytes = 0;
            ReadTexture(rt->texture, WGPUTextureAspect_DepthOnly, 4, 0, 0, int(rt->width), int(rt->height), data, rowBytes);
            for (int y = 0; y < cy1 - y0; y++)
                std::memcpy(&depth[size_t(y) * w], data.data() + size_t(y + y0) * rowBytes + size_t(x0) * 4, size_t(cx1 - x0) * 4);
            ReadTexture(rt->texture, WGPUTextureAspect_StencilOnly, 1, 0, 0, int(rt->width), int(rt->height), data, rowBytes);
            for (int y = 0; y < cy1 - y0; y++)
                std::memcpy(&stencil[size_t(y) * w], data.data() + size_t(y + y0) * rowBytes + size_t(x0), size_t(cx1 - x0));
        }

        void WebGpuBackend::ClearRect(RenderTarget* rt, int x0, int y0, int x1, int y1, const F4& color, float depth, uint8_t stencil)
        {
            x0 = std::clamp(x0, 0, int(rt->width));
            x1 = std::clamp(x1, x0, int(rt->width));
            y0 = std::clamp(y0, 0, int(rt->height));
            y1 = std::clamp(y1, y0, int(rt->height));
            if (x1 <= x0 || y1 <= y0)
                return;

            std::string key = rt->depth ? "clear depth" : "clear colour";
            key += std::to_string(int(rt->format));
            WGPURenderPipeline& pipeline = pipelines_[key];
            if (!pipeline)
            {
                WGPURenderPipelineDescriptor pd = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
                pd.layout = PipelineLayout(0);
                pd.vertex.module = clearModule_;
                pd.vertex.entryPoint = Str("vs");
                pd.primitive.topology = WGPUPrimitiveTopology_TriangleList;
                pd.multisample.count = 1;
                pd.multisample.mask = 0xFFFFFFFF;
                WGPUDepthStencilState dss = WGPU_DEPTH_STENCIL_STATE_INIT;
                WGPUColorTargetState target = WGPU_COLOR_TARGET_STATE_INIT;
                WGPUFragmentState fs = WGPU_FRAGMENT_STATE_INIT;
                if (rt->depth)
                {
                    dss.format = rt->format;
                    dss.depthWriteEnabled = WGPUOptionalBool_True;
                    dss.depthCompare = WGPUCompareFunction_Always;
                    dss.stencilFront.compare = dss.stencilBack.compare = WGPUCompareFunction_Always;
                    dss.stencilFront.passOp = dss.stencilBack.passOp = WGPUStencilOperation_Replace;
                    dss.stencilReadMask = 0xFF;
                    dss.stencilWriteMask = 0xFF;
                    pd.depthStencil = &dss;
                }
                else
                {
                    target.format = rt->format;
                    fs.module = clearModule_;
                    fs.entryPoint = Str("fs");
                    fs.targetCount = 1;
                    fs.targets = &target;
                    pd.fragment = &fs;
                }
                pipeline = wgpuDeviceCreateRenderPipeline(device_, &pd);
                ++stats_.pipelinesCreated;
            }

            F4 constant{ depth, 0, 0, 0 };
            const uint32_t offset = WriteDrawData(color.data(), &constant, 1);
            RenderTarget* color4[4] = { rt->depth ? nullptr : rt, nullptr, nullptr, nullptr };
            BeginPass(color4, rt->depth ? rt : nullptr);
            wgpuRenderPassEncoderSetPipeline(pass_, pipeline);
            wgpuRenderPassEncoderSetBindGroup(pass_, 0, drawBindGroup_, 1, &offset);
            WGPUBindGroupDescriptor bgd = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
            bgd.layout = TextureLayout(0);
            WGPUBindGroup empty = wgpuDeviceCreateBindGroup(device_, &bgd);
            wgpuRenderPassEncoderSetBindGroup(pass_, 1, empty, 0, nullptr);
            wgpuRenderPassEncoderSetViewport(pass_, 0, 0, float(rt->width), float(rt->height), 0.0f, 1.0f);
            wgpuRenderPassEncoderSetScissorRect(pass_, uint32_t(x0), uint32_t(y0), uint32_t(x1 - x0), uint32_t(y1 - y0));
            wgpuRenderPassEncoderSetStencilReference(pass_, stencil);
            wgpuRenderPassEncoderDraw(pass_, 3, 1, 0, 0);
            wgpuBindGroupRelease(empty);
            ++stats_.clears;
        }

        void WebGpuBackend::ClearColor(const RasterState& rs, uint32_t rtIndex, int x0, int y0, int x1, int y1, const F4& color)
        {
            const Surface& s = rs.color[rtIndex & 3];
            Target(false, s, ColorFormat(s.format));
            // Clear every colour view of this EDRAM base (aliasing through other pitches).
            for (auto& [key, rt] : targets_)
                if (!rt->depth && rt->baseTile == s.baseTile)
                    ClearRect(rt.get(), x0, y0, x1, y1, color, 0.0f, 0);
        }

        void WebGpuBackend::ClearDepth(const RasterState& rs, int x0, int y0, int x1, int y1, float depth, uint8_t stencil)
        {
            Target(true, rs.depth, WGPUTextureFormat_Depth32FloatStencil8);
            for (auto& [key, rt] : targets_)
                if (rt->depth && rt->baseTile == rs.depth.baseTile)
                    ClearRect(rt.get(), x0, y0, x1, y1, F4{ 0, 0, 0, 0 }, depth, stencil);
        }

        void WebGpuBackend::EndFrame()
        {
            Submit();
#ifndef __EMSCRIPTEN__
            wgpuDevicePoll(device_, false, nullptr);
#endif
            if (textures_.size() > 512)
            {
                for (auto& [tex, g] : textures_)
                {
                    wgpuTextureViewRelease(g.view);
                    wgpuTextureRelease(g.texture);
                }
                textures_.clear();
            }
        }

        void WebGpuBackend::PrintStats()
        {
            std::fprintf(stderr,
                "[gpu]   webgpu: %llu draws, %llu triangles, %llu clears, %llu readbacks (%.1f ms), encode %.1f ms, %llu uploads, %llu submits, %llu pipelines / %llu shaders created\n",
                (unsigned long long)stats_.draws, (unsigned long long)stats_.triangles, (unsigned long long)stats_.clears,
                (unsigned long long)stats_.readbacks, stats_.readbackNs / 1e6, stats_.encodeNs / 1e6, (unsigned long long)stats_.textureUploads,
                (unsigned long long)stats_.submits, (unsigned long long)stats_.pipelinesCreated, (unsigned long long)stats_.shadersCreated);
            stats_ = {};
        }
    }

    CaptureSink* CreateWebGpuBackend()
    {
        auto backend = std::make_unique<WebGpuBackend>();
        if (!backend->Init())
        {
            std::fprintf(stderr, "[gpu] webgpu: initialisation failed, keeping the software renderer\n");
            return nullptr;
        }
        g_backend = backend.release();
        return g_backend;
    }

    void WebGpuEndFrame()
    {
        if (g_backend)
            g_backend->EndFrame();
    }

    void WebGpuPrintStats()
    {
        if (g_backend)
            g_backend->PrintStats();
    }
}
