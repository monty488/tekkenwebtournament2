# Shaders

Xbox 360 shaders are compiled Xenos microcode, so they have to be translated
ahead of time. The browser cannot run DXC at load time.

```
game shader containers ──XenosRecomp──▶ HLSL ──DXC (-spirv)──▶ SPIR-V ──Tint──▶ WGSL
                                                                                  │
                                                         shaders/generated/ (ignored)
```

## Work needed

1. **Find the shaders.** Locate where TTT2 stores its compiled shaders: standalone
   files in the disc data, or embedded in archives or the XEX. XenosRecomp scans an
   input directory for shader containers, so extract them to a folder first.
2. **WebGPU `shader_common.h`.** XenosRecomp's HLSL expects Unleashed Recompiled's
   resource model: bindless descriptor heaps, and `vk::RawBufferLoad` through buffer
   device addresses. WebGPU supports neither. Write a variant that:
   - puts vertex constants (256 × float4), pixel constants (224 × float4) and shared
     constants in three uniform buffers in bind group 0;
   - binds textures and samplers to fixed slots (16 per stage) in bind group 1,
     instead of indexing descriptor heaps.
3. **Tooling.** Add DXC (ships with XenosRecomp under `thirdparty/dxc-bin`) and
   Tint (or naga) to `scripts/`. Then a `scripts/build-shaders.ps1` produces
   `shaders/generated/*.wgsl` plus a lookup table keyed by the shader's hash.
4. **Vertex fetch.** Unleashed-specific assumptions in XenosRecomp, such as
   instancing via `POSITION1` and texcoord swizzles, must be checked against TTT2's
   vertex declarations.
