#include "xenos_wgsl.h"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <set>
#include <string>

namespace gpu
{
    namespace
    {
        constexpr const char* kFloatMax = "3.40282347e38";

        std::string Num(uint32_t v) { return std::to_string(v); }

        std::string Reg(uint32_t index) { return "r" + Num(index & 127); }

        // Operand expression with swizzle, |abs| and negation applied.
        std::string Source(const AluOp::Source& s, uint32_t constBase)
        {
            std::string raw;
            if (s.isRegister)
                raw = Reg(s.index);
            else if (s.relativeA0)
                raw = "D.c[" + Num(constBase) + "u + ((" + Num(s.index) + "u + u32(a0)) & 255u)]";
            else
                raw = "D.c[" + Num(constBase + (s.index & 255)) + "]";
            static const char kComp[] = "xyzw";
            if (!(s.comp[0] == 0 && s.comp[1] == 1 && s.comp[2] == 2 && s.comp[3] == 3))
            {
                raw += '.';
                for (int i = 0; i < 4; i++)
                    raw += kComp[s.comp[i] & 3];
            }
            if (s.abs)
                raw = "abs(" + raw + ")";
            if (s.negate)
                raw = "(-" + raw + ")";
            return raw;
        }

        const char* CompareOp(uint32_t func)
        {
            switch (func & 7)
            {
            case 1: return "<";
            case 2: return "==";
            case 3: return "<=";
            case 4: return ">";
            case 5: return "!=";
            case 6: return ">=";
            default: return nullptr; // 0 never, 7 always
            }
        }

        struct Emitter
        {
            std::string body;
            std::set<uint32_t> regs;
            uint32_t colorExports = 0;
            bool depthExport = false;
            uint32_t textureSlots = 0;
            int indent = 1;
            const WgslOptions* options = nullptr;
            bool pixel = true;
            uint32_t constBase = 256;
            uint32_t vsExports = 0;      // vertex stage: interpolators written
            uint64_t vertexSlots[2] = {};
            // Mini vertex fetches reuse the previous full fetch's slot, index and stride.
            uint32_t fetchSlot = 0, fetchStride = 0;
            int fetchCounter = 0;
            WgslShader::SlotRange slotRange[96];

            void Line(const std::string& text)
            {
                body.append(size_t(indent) * 4, ' ');
                body += text;
                body += '\n';
            }

            void Use(const AluOp::Source& s)
            {
                if (s.isRegister)
                    regs.insert(s.index & 127);
            }

            void Alu(const AluOp& op)
            {
                const AluInstruction& a = op.a;
                const auto vop = a.vectorOpcode;
                const auto sop = a.scalarOpcode;
                const uint32_t sopIndex = uint32_t(sop);

                Line(a.isPredicated ? std::string("if (p0 == ") + (a.predicateCondition ? "true" : "false") + ") {" : "{");
                ++indent;
                Use(op.src[0]);
                Use(op.src[1]);
                Line("let s1 = " + Source(op.src[0], constBase) + ";");
                Line("let s2 = " + Source(op.src[1], constBase) + ";");
                if (op.scalarConst)
                {
                    Use(op.src[2]);
                    regs.insert(op.scalarGpr & 127);
                    static const char kComp[] = "xyzw";
                    Line("let s3 = vec4f(0.0);");
                    Line(std::string("let sa = ") + Source(op.src[2], constBase) + "." + kComp[op.scalarConstComp & 3] + ";");
                    std::string gpr = Reg(op.scalarGpr) + "." + kComp[op.scalarGprComp & 3];
                    if (a.absConstants)
                        gpr = "abs(" + gpr + ")";
                    Line("let sb = " + std::string(a.src3Negate ? "-" : "") + gpr + ";");
                }
                else
                {
                    Use(op.src[2]);
                    Line("let s3 = " + Source(op.src[2], constBase) + ";");
                    Line("let sa = s3.w;");
                    Line("let sb = s3.x;");
                }
                Line("var np0 = p0;");

                // ---- vector op
                std::string vr = "vec4f(0.0)";
                bool push = false;
                switch (vop)
                {
                case AluVectorOpcode::Add: vr = "s1 + s2"; break;
                case AluVectorOpcode::Mul: vr = "mulx(s1, s2)"; break;
                case AluVectorOpcode::Max:
                case AluVectorOpcode::MaxA: vr = "select(s2, s1, s1 >= s2)"; break;
                case AluVectorOpcode::Min: vr = "select(s2, s1, s1 < s2)"; break;
                case AluVectorOpcode::Seq: vr = "select(vec4f(0.0), vec4f(1.0), s1 == s2)"; break;
                case AluVectorOpcode::Sgt: vr = "select(vec4f(0.0), vec4f(1.0), s1 > s2)"; break;
                case AluVectorOpcode::Sge: vr = "select(vec4f(0.0), vec4f(1.0), s1 >= s2)"; break;
                case AluVectorOpcode::Sne: vr = "select(vec4f(0.0), vec4f(1.0), s1 != s2)"; break;
                case AluVectorOpcode::Frc: vr = "s1 - floor(s1)"; break;
                case AluVectorOpcode::Trunc: vr = "trunc(s1)"; break;
                case AluVectorOpcode::Floor: vr = "floor(s1)"; break;
                case AluVectorOpcode::Mad: vr = "mulx(s1, s2) + s3"; break;
                case AluVectorOpcode::CndEq: vr = "select(s3, s2, s1 == vec4f(0.0))"; break;
                case AluVectorOpcode::CndGe: vr = "select(s3, s2, s1 >= vec4f(0.0))"; break;
                case AluVectorOpcode::CndGt: vr = "select(s3, s2, s1 > vec4f(0.0))"; break;
                case AluVectorOpcode::Dp4: vr = "vec4f(dot4x(s1, s2))"; break;
                case AluVectorOpcode::Dp3: vr = "vec4f(dot3x(s1, s2))"; break;
                case AluVectorOpcode::Dp2Add: vr = "vec4f(mulxs(s1.x, s2.x) + mulxs(s1.y, s2.y) + s3.x)"; break;
                case AluVectorOpcode::Cube: vr = "cube(s1, s2)"; break;
                case AluVectorOpcode::Max4: vr = "vec4f(max(max(s1.x, s1.y), max(s1.z, s1.w)))"; break;
                case AluVectorOpcode::SetpEqPush:
                case AluVectorOpcode::SetpNePush:
                case AluVectorOpcode::SetpGtPush:
                case AluVectorOpcode::SetpGePush:
                {
                    const char* cmp = vop == AluVectorOpcode::SetpEqPush ? "==" : vop == AluVectorOpcode::SetpNePush ? "!=" : vop == AluVectorOpcode::SetpGtPush ? ">" : ">=";
                    Line(std::string("np0 = s1.w == 0.0 && s2.w ") + cmp + " 0.0;");
                    vr = "vec4f(select(s1.x + 1.0, 0.0, np0))";
                    push = true;
                    break;
                }
                case AluVectorOpcode::KillEq:
                case AluVectorOpcode::KillGt:
                case AluVectorOpcode::KillGe:
                case AluVectorOpcode::KillNe:
                {
                    const char* cmp = vop == AluVectorOpcode::KillEq ? "==" : vop == AluVectorOpcode::KillGt ? ">" : vop == AluVectorOpcode::KillGe ? ">=" : "!=";
                    Line(std::string("let vk = any(s1 ") + cmp + " s2);");
                    Line("killed = killed || vk;");
                    vr = "vec4f(select(0.0, 1.0, vk))";
                    break;
                }
                case AluVectorOpcode::Dst: vr = "vec4f(1.0, mulxs(s1.y, s2.y), s1.z, s2.w)"; break;
                default: break;
                }
                Line("var vr = " + vr + ";");
                if (vop == AluVectorOpcode::MaxA)
                    Line("a0 = i32(clamp(floor(s1.w + 0.5), -256.0, 255.0));");
                if (a.vectorSaturate)
                    Line("vr = clamp(vr, vec4f(0.0), vec4f(1.0));");

                // ---- scalar op
                bool writesPs = true;
                std::string sr;
                switch (sop)
                {
                case AluScalarOpcode::Adds: sr = "sa + sb"; break;
                case AluScalarOpcode::AddsPrev: sr = "sa + ps"; break;
                case AluScalarOpcode::Muls: sr = "mulxs(sa, sb)"; break;
                case AluScalarOpcode::MulsPrev:
                case AluScalarOpcode::MulsPrev2: sr = "mulxs(sa, ps)"; break;
                case AluScalarOpcode::Maxs: sr = "select(sb, sa, sa >= sb)"; break;
                case AluScalarOpcode::Mins: sr = "select(sb, sa, sa < sb)"; break;
                case AluScalarOpcode::Seqs: sr = "select(0.0, 1.0, sa == 0.0)"; break;
                case AluScalarOpcode::Sgts: sr = "select(0.0, 1.0, sa > 0.0)"; break;
                case AluScalarOpcode::Sges: sr = "select(0.0, 1.0, sa >= 0.0)"; break;
                case AluScalarOpcode::Snes: sr = "select(0.0, 1.0, sa != 0.0)"; break;
                case AluScalarOpcode::Frcs: sr = "sa - floor(sa)"; break;
                case AluScalarOpcode::Truncs: sr = "trunc(sa)"; break;
                case AluScalarOpcode::Floors: sr = "floor(sa)"; break;
                case AluScalarOpcode::Exp: sr = "exp2(sa)"; break;
                case AluScalarOpcode::Logc: sr = std::string("max(log2(sa), -") + kFloatMax + ")"; break;
                case AluScalarOpcode::Log: sr = "log2(sa)"; break;
                case AluScalarOpcode::Rcpc: sr = std::string("clamp(1.0 / sa, -") + kFloatMax + ", " + kFloatMax + ")"; break;
                case AluScalarOpcode::Rcpf: sr = std::string("select(1.0 / sa, 0.0, abs(1.0 / sa) > ") + kFloatMax + ")"; break;
                case AluScalarOpcode::Rcp: sr = "1.0 / sa"; break;
                case AluScalarOpcode::Rsqc: sr = std::string("clamp(inverseSqrt(sa), -") + kFloatMax + ", " + kFloatMax + ")"; break;
                case AluScalarOpcode::Rsqf: sr = std::string("select(inverseSqrt(sa), 0.0, abs(inverseSqrt(sa)) > ") + kFloatMax + ")"; break;
                case AluScalarOpcode::Rsq: sr = "inverseSqrt(sa)"; break;
                case AluScalarOpcode::MaxAs:
                    Line("a0 = i32(clamp(floor(sa + 0.5), -256.0, 255.0));");
                    sr = "select(sb, sa, sa >= sb)";
                    break;
                case AluScalarOpcode::MaxAsf:
                    Line("a0 = i32(clamp(floor(sa), -256.0, 255.0));");
                    sr = "select(sb, sa, sa >= sb)";
                    break;
                case AluScalarOpcode::Subs: sr = "sa - sb"; break;
                case AluScalarOpcode::SubsPrev: sr = "sa - ps"; break;
                case AluScalarOpcode::SetpEq: Line("np0 = sa == 0.0;"); sr = "select(1.0, 0.0, np0)"; break;
                case AluScalarOpcode::SetpNe: Line("np0 = sa != 0.0;"); sr = "select(1.0, 0.0, np0)"; break;
                case AluScalarOpcode::SetpGt: Line("np0 = sa > 0.0;"); sr = "select(1.0, 0.0, np0)"; break;
                case AluScalarOpcode::SetpGe: Line("np0 = sa >= 0.0;"); sr = "select(1.0, 0.0, np0)"; break;
                case AluScalarOpcode::SetpInv:
                    Line("np0 = sa == 1.0;");
                    sr = "select(select(sa, 1.0, sa == 0.0), 0.0, np0)";
                    break;
                case AluScalarOpcode::SetpPop:
                    Line("np0 = sa - 1.0 <= 0.0;");
                    sr = "select(sa - 1.0, 0.0, np0)";
                    break;
                case AluScalarOpcode::SetpClr: Line("np0 = false;"); sr = kFloatMax; break;
                case AluScalarOpcode::SetpRstr: Line("np0 = sa == 0.0;"); sr = "sa"; break;
                case AluScalarOpcode::KillsEq:
                case AluScalarOpcode::KillsGt:
                case AluScalarOpcode::KillsGe:
                case AluScalarOpcode::KillsNe:
                case AluScalarOpcode::KillsOne:
                {
                    const char* cond = sop == AluScalarOpcode::KillsEq ? "sa == 0.0" : sop == AluScalarOpcode::KillsGt ? "sa > 0.0"
                        : sop == AluScalarOpcode::KillsGe ? "sa >= 0.0" : sop == AluScalarOpcode::KillsNe ? "sa != 0.0" : "sa == 1.0";
                    Line(std::string("let sk = ") + cond + ";");
                    Line("killed = killed || sk;");
                    sr = "select(0.0, 1.0, sk)";
                    break;
                }
                case AluScalarOpcode::Sqrt: sr = "sqrt(sa)"; break;
                case AluScalarOpcode::Mulsc0:
                case AluScalarOpcode::Mulsc1: sr = "mulxs(sa, sb)"; break;
                case AluScalarOpcode::Addsc0:
                case AluScalarOpcode::Addsc1: sr = "sa + sb"; break;
                case AluScalarOpcode::Subsc0:
                case AluScalarOpcode::Subsc1: sr = "sa - sb"; break;
                case AluScalarOpcode::Sin: sr = "sin(sa)"; break;
                case AluScalarOpcode::Cos: sr = "cos(sa)"; break;
                default: writesPs = false; break; // RetainPrev and unknown opcodes
                }
                if (writesPs)
                {
                    Line("var sr = " + sr + ";");
                    if (a.scalarSaturate)
                        Line("sr = clamp(sr, 0.0, 1.0);");
                }
                if (push || (sopIndex >= 27 && sopIndex <= 34))
                    Line("p0 = np0;");
                if (writesPs)
                    Line("ps = sr;");

                // ---- writes
                static const char kComp[] = "xyzw";
                const uint32_t vmask = a.vectorWriteMask, smask = a.scalarWriteMask;
                if (a.exportData)
                {
                    const uint32_t dest = a.vectorDest & 63;
                    std::string target;
                    if (!pixel)
                    {
                        if (dest == 62)
                            target = "oPos";
                        else if (dest < 16)
                        {
                            target = "oV" + Num(dest);
                            vsExports |= 1u << dest;
                        }
                    }
                    else if (dest < 4)
                    {
                        target = "o" + Num(dest);
                        colorExports |= 1u << dest;
                    }
                    else if (dest == 61)
                    {
                        target = "oD";
                        depthExport = true;
                    }
                    const uint32_t ones = vmask & smask;
                    const uint32_t zeros = a.scalarDestRelative ? (0xFu & ~(vmask | smask)) : 0u;
                    if (!target.empty())
                    {
                        for (int i = 0; i < 4; i++)
                        {
                            const uint32_t bit = 1u << i;
                            const std::string lhs = target + "." + kComp[i];
                            if (ones & bit) Line(lhs + " = 1.0;");
                            else if (vmask & bit) Line(lhs + " = vr." + kComp[i] + ";");
                            else if (smask & bit) Line(lhs + " = ps;");
                            else if (zeros & bit) Line(lhs + " = 0.0;");
                        }
                    }
                }
                else
                {
                    if (vmask)
                    {
                        regs.insert(op.vectorDest & 127);
                        for (int i = 0; i < 4; i++)
                            if (vmask & (1u << i))
                                Line(Reg(op.vectorDest) + "." + kComp[i] + " = vr." + kComp[i] + ";");
                    }
                    if (smask)
                    {
                        regs.insert(op.scalarDest & 127);
                        for (int i = 0; i < 4; i++)
                            if (smask & (1u << i))
                                Line(Reg(op.scalarDest) + "." + kComp[i] + " = ps;");
                    }
                }
                --indent;
                Line("}");
            }

            void DestSwizzle(uint32_t dstRegister, uint32_t dstSwizzle, const std::string& value)
            {
                static const char kComp[] = "xyzw";
                regs.insert(dstRegister & 63);
                for (int i = 0; i < 4; i++)
                {
                    const uint32_t sel = (dstSwizzle >> (i * 3)) & 7;
                    const std::string lhs = Reg(dstRegister & 63) + "." + kComp[i];
                    if (sel < 4) Line(lhs + " = " + value + "." + kComp[sel] + ";");
                    else if (sel == 4) Line(lhs + " = 0.0;");
                    else if (sel == 5) Line(lhs + " = 1.0;");
                }
            }

            void VertexFetch(const VertexFetchInstruction& f)
            {
                if (!f.isMiniFetch)
                {
                    fetchSlot = (f.constIndex * 3 + f.constIndexSelect) % 96;
                    fetchStride = f.stride;
                    vertexSlots[fetchSlot / 64] |= 1ull << (fetchSlot % 64);
                    regs.insert(f.srcRegister & 63);
                    static const char kComp[] = "xyzw";
                    // The index is taken when the full fetch runs (later ones reuse it).
                    const std::string idx = "fi" + Num(uint32_t(++fetchCounter));
                    Line(idx + " = i32(floor(" + Reg(f.srcRegister & 63) + "." + kComp[f.srcSwizzle & 3] +
                        (f.isIndexRounded ? " + 0.5" : "") + "));");
                    lastIndex = idx;
                }
                if (lastIndex.empty())
                    return;
                Line(f.isPredicated ? std::string("if (p0 == ") + (f.predicateCondition ? "true" : "false") + ") {" : "{");
                ++indent;
                Line("let vf = D.vf[" + Num(fetchSlot) + "];");
                Line("let e = vf.y;");
                Line("let w0 = i32(vf.x) + " + lastIndex + " * " + Num(fetchStride) + " + (" + std::to_string(int(f.offset)) + ");");
                const std::string sgn = f.formatCompAll ? "true" : "false";
                const std::string integer = f.numFormatAll ? "true" : "false";
                const std::string rf = f.signedRfModeAll ? "true" : "false";
                auto n = [&](const std::string& raw, uint32_t bits) {
                    return "nrm(" + raw + ", " + Num(bits) + "u, " + sgn + ", " + integer + ", " + rf + ")";
                };
                auto field = [&](const char* d, uint32_t off, uint32_t bits) {
                    return n(std::string("bitsAt(") + d + ", " + Num(off) + "u, " + Num(bits) + "u)", bits);
                };
                bool isFloat = false;
                std::string v = "vec4f(0.0, 0.0, 0.0, 1.0)";
                switch (f.format)
                {
                case 6:
                    Line("let d0 = ld(w0, e);");
                    v = "vec4f(" + field("d0", 0, 8) + ", " + field("d0", 8, 8) + ", " + field("d0", 16, 8) + ", " + field("d0", 24, 8) + ")";
                    break;
                case 7:
                    Line("let d0 = ld(w0, e);");
                    v = "vec4f(" + field("d0", 0, 10) + ", " + field("d0", 10, 10) + ", " + field("d0", 20, 10) + ", " + field("d0", 30, 2) + ")";
                    break;
                case 16:
                    Line("let d0 = ld(w0, e);");
                    v = "vec4f(" + field("d0", 0, 10) + ", " + field("d0", 10, 11) + ", " + field("d0", 21, 11) + ", 1.0)";
                    break;
                case 17:
                    Line("let d0 = ld(w0, e);");
                    v = "vec4f(" + field("d0", 0, 11) + ", " + field("d0", 11, 11) + ", " + field("d0", 22, 10) + ", 1.0)";
                    break;
                case 25:
                    Line("let d0 = ld(w0, e);");
                    v = "vec4f(" + field("d0", 0, 16) + ", " + field("d0", 16, 16) + ", 0.0, 1.0)";
                    break;
                case 26:
                    Line("let d0 = ld(w0, e);");
                    Line("let d1 = ld(w0 + 1, e);");
                    v = "vec4f(" + field("d0", 0, 16) + ", " + field("d0", 16, 16) + ", " + field("d1", 0, 16) + ", " + field("d1", 16, 16) + ")";
                    break;
                case 31:
                    isFloat = true;
                    Line("let d0 = ld(w0, e);");
                    v = "vec4f(h2f(d0 & 0xFFFFu), h2f(d0 >> 16u), 0.0, 1.0)";
                    break;
                case 32:
                    isFloat = true;
                    Line("let d0 = ld(w0, e);");
                    Line("let d1 = ld(w0 + 1, e);");
                    v = "vec4f(h2f(d0 & 0xFFFFu), h2f(d0 >> 16u), h2f(d1 & 0xFFFFu), h2f(d1 >> 16u))";
                    break;
                case 33: case 34: case 35:
                {
                    const uint32_t count = f.format - 32;
                    std::string c[4] = { "0.0", "0.0", "0.0", "1.0" };
                    for (uint32_t i = 0; i < count; i++)
                        c[i] = n("ld(w0 + " + Num(i) + ", e)", 32);
                    v = "vec4f(" + c[0] + ", " + c[1] + ", " + c[2] + ", " + c[3] + ")";
                    break;
                }
                case 36: case 37: case 38: case 57:
                {
                    isFloat = true;
                    const uint32_t count = f.format == 57 ? 3 : f.format - 35;
                    std::string c[4] = { "0.0", "0.0", "0.0", "1.0" };
                    for (uint32_t i = 0; i < count; i++)
                        c[i] = "bitcast<f32>(ld(w0 + " + Num(i) + ", e))";
                    v = "vec4f(" + c[0] + ", " + c[1] + ", " + c[2] + ", " + c[3] + ")";
                    break;
                }
                default:
                    unsupportedVertexFormat = true;
                    break;
                }
                if (!isFloat && f.expAdjust)
                    v = "(" + v + " * exp2(" + std::to_string(int(f.expAdjust)) + ".0))";
                {
                    uint32_t words = 1;
                    switch (f.format)
                    {
                    case 26: case 32: case 34: case 37: words = 2; break;
                    case 57: words = 3; break;
                    case 35: case 38: words = 4; break;
                    default: break;
                    }
                    auto& range = slotRange[fetchSlot];
                    range.stride = fetchStride;
                    range.first = std::min(range.first, int32_t(f.offset));
                    range.end = std::max(range.end, int32_t(f.offset) + int32_t(words));
                }
                Line("let fv = " + v + ";");
                DestSwizzle(f.dstRegister, f.dstSwizzle, "fv");
                --indent;
                Line("}");
            }

            std::string lastIndex;
            bool unsupportedVertexFormat = false;

            void TextureFetch(const TextureFetchInstruction& f)
            {
                Line(f.isPredicated ? std::string("if (p0 == ") + (f.predCondition ? "true" : "false") + ") {" : "{");
                ++indent;
                if (f.opcode == FetchOpcode::TextureFetch)
                {
                    const uint32_t slot = f.constIndex & 31;
                    textureSlots |= 1u << slot;
                    static const char kComp[] = "xyzw";
                    regs.insert(f.srcRegister & 63);
                    const std::string src = Reg(f.srcRegister & 63);
                    const std::string t = "t" + Num(slot), s = "s" + Num(slot);
                    Line("let size = vec2f(textureDimensions(" + t + "));");
                    std::string uv = std::string("vec2f(") + src + "." + kComp[f.srcSwizzle & 3] + ", " + src + "." + kComp[(f.srcSwizzle >> 2) & 3] + ")";
                    if (f.texCoordDenorm)
                        uv += " / size";
                    if (f.offsetX || f.offsetY)
                    {
                        char buf[96];
                        std::snprintf(buf, sizeof(buf), " + vec2f(%.1f, %.1f) / size", f.offsetX * 0.5, f.offsetY * 0.5);
                        uv += buf;
                    }
                    Line("let tv = textureSampleLevel(" + t + ", " + s + ", " + uv + ", 0.0);");
                    const uint32_t swizzle = options ? options->slotSwizzle[slot] : 0x688u;
                    if (swizzle != 0x688u)
                    {
                        std::string remap = "vec4f(";
                        for (int i = 0; i < 4; i++)
                        {
                            const uint32_t sel = (swizzle >> (i * 3)) & 7;
                            remap += sel < 4 ? std::string("tv.") + kComp[sel] : (sel == 5 ? "1.0" : "0.0");
                            remap += i < 3 ? ", " : ")";
                        }
                        Line("let tvs = " + remap + ";");
                        DestSwizzle(f.dstRegister, f.dstSwizzle, "tvs");
                    }
                    else
                    {
                        DestSwizzle(f.dstRegister, f.dstSwizzle, "tv");
                    }
                }
                else if (f.opcode == FetchOpcode::GetTextureWeights || f.opcode == FetchOpcode::GetTextureComputedLod ||
                    f.opcode == FetchOpcode::GetTextureGradients || f.opcode == FetchOpcode::GetTextureBorderColorFrac)
                {
                    DestSwizzle(f.dstRegister, f.dstSwizzle, "vec4f(0.0)");
                }
                --indent;
                Line("}");
            }
        };

        const char* kHelpers = R"(
fn ld(w: i32, e: u32) -> u32 {
    let v = M[u32(max(w, 0))];
    switch (e) {
        case 1u: { return ((v & 0x00FF00FFu) << 8u) | ((v >> 8u) & 0x00FF00FFu); }
        case 2u: { return (v >> 24u) | ((v >> 8u) & 0xFF00u) | ((v << 8u) & 0xFF0000u) | (v << 24u); }
        case 3u: { return (v << 16u) | (v >> 16u); }
        default: { return v; }
    }
}
fn bitsAt(v: u32, off: u32, bits: u32) -> u32 {
    if (bits == 32u) { return v; }
    return (v >> off) & ((1u << bits) - 1u);
}
fn nrm(raw: u32, bits: u32, sgn: bool, integer: bool, rf: bool) -> f32 {
    if (sgn) {
        let sv = i32(raw << (32u - bits)) >> (32u - bits);
        if (integer) { return f32(sv); }
        let maxv = f32((1u << (bits - 1u)) - 1u);
        if (rf) { return (f32(sv) * 2.0 + 1.0) / (maxv * 2.0 + 1.0); }
        return max(f32(sv) / maxv, -1.0);
    }
    if (integer) { return f32(raw); }
    if (bits == 32u) { return f32(raw) / 4294967295.0; }
    return f32(raw) / f32((1u << bits) - 1u);
}
// Xenos half floats have no inf/NaN: exponent 31 is an ordinary value.
fn h2f(h: u32) -> f32 {
    let e = (h >> 10u) & 31u;
    let m = f32(h & 0x3FFu);
    var v = select((1.0 + m / 1024.0) * exp2(f32(e) - 15.0), m * exp2(-24.0), e == 0u);
    return select(v, -v, ((h >> 15u) & 1u) == 1u);
}
)";

        const char* kPrelude = R"(
fn mulxs(a: f32, b: f32) -> f32 { return select(a * b, 0.0, a == 0.0 || b == 0.0); }
fn mulx(a: vec4f, b: vec4f) -> vec4f { return select(a * b, vec4f(0.0), (a == vec4f(0.0)) | (b == vec4f(0.0))); }
fn dot4x(a: vec4f, b: vec4f) -> f32 { let m = mulx(a, b); return m.x + m.y + m.z + m.w; }
fn dot3x(a: vec4f, b: vec4f) -> f32 { let m = mulx(a, b); return m.x + m.y + m.z; }
fn cube(s1: vec4f, s2: vec4f) -> vec4f {
    let x = s1.z; let y = s2.x; let z = s1.x;
    let ax = abs(x); let ay = abs(y); let az = abs(z);
    if (az >= ax && az >= ay) { return vec4f(-y, select(x, -x, z < 0.0), 2.0 * az, select(4.0, 5.0, z < 0.0)); }
    if (ay >= ax) { return vec4f(select(z, -z, y < 0.0), x, 2.0 * ay, select(2.0, 3.0, y < 0.0)); }
    return vec4f(-y, select(-z, z, x < 0.0), 2.0 * ax, select(0.0, 1.0, x < 0.0));
}

)";
    }

    const char* const kWgslDrawData = R"(
struct DrawData {
    misc: vec4f,
    xf: array<vec4f, 3>,
    c: array<vec4f, 512>,
    vf: array<vec4u, 96>,
}
@group(0) @binding(0) var<storage, read> D: DrawData;
@group(0) @binding(1) var<storage, read> M: array<u32>;
)";

    namespace
    {
        void EmitProgram(Emitter& e, const CompiledShader& program)
        {
            for (const CompiledOp& op : program.ops)
            {
                switch (op.kind)
                {
                case CompiledOp::Kind::Clause:
                    // A clause ends where the next begins; predicated clauses run where p0
                    // matched at their start.
                    if (e.indent > 1)
                    {
                        --e.indent;
                        e.Line("}");
                    }
                    if (op.clausePredicate >= 0)
                        e.Line(std::string("if (p0 == ") + (op.clausePredicate ? "true" : "false") + ") {");
                    else
                        e.Line("{");
                    ++e.indent;
                    break;
                case CompiledOp::Kind::Alu:
                    e.Alu(op.alu);
                    break;
                case CompiledOp::Kind::TextureFetch:
                    if (e.pixel)
                        e.TextureFetch(op.textureFetch);
                    break;
                case CompiledOp::Kind::VertexFetch:
                    if (!e.pixel)
                        e.VertexFetch(op.vertexFetch);
                    break;
                }
            }
            while (e.indent > 1)
            {
                --e.indent;
                e.Line("}");
            }
        }

        std::string Declarations(const Emitter& e)
        {
            std::string w;
            for (uint32_t reg : e.regs)
                w += "    var " + Reg(reg) + " = vec4f(0.0);\n";
            // Vertex fetch indices outlive the clause of their full fetch (mini fetches).
            for (int i = 1; i <= e.fetchCounter; i++)
                w += "    var fi" + Num(uint32_t(i)) + " = 0i;\n";
            w += "    var p0 = false;\n    var ps = 0.0;\n    var a0 = 0i;\n    var killed = false;\n";
            return w;
        }
    }

    WgslShader TranslateToWgsl(const CompiledShader* vs, const CompiledShader* ps, const WgslOptions& options)
    {
        WgslShader out;
        std::string& w = out.code;
        w = kWgslDrawData;
        w += kHelpers;
        w += kPrelude;

        // ---- vertex stage
        Emitter ve;
        ve.pixel = false;
        ve.constBase = 0;
        ve.options = &options;
        if (vs)
        {
            ve.regs.insert(0);
            EmitProgram(ve, *vs);
            out.interpolators = ve.vsExports;
            out.vertexSlots[0] = ve.vertexSlots[0];
            out.vertexSlots[1] = ve.vertexSlots[1];
            std::copy(std::begin(ve.slotRange), std::end(ve.slotRange), std::begin(out.slotRange));
        }
        else
        {
            out.interpolators = options.interpolatorMask;
        }
        const uint32_t interpolators = out.interpolators;

        w += "\nstruct VsOut {\n    @builtin(position) pos: vec4f,\n";
        uint32_t location = 0;
        for (uint32_t i = 0; i < 16; i++)
            if (interpolators & (1u << i))
                w += "    @location(" + Num(location++) + ") v" + Num(i) + ": vec4f,\n";
        w += "}\n";

        if (vs)
        {
            w += "@vertex fn vs(@builtin(vertex_index) vertexIndex: u32) -> VsOut {\n";
            w += Declarations(ve);
            w += "    r0 = vec4f(f32(vertexIndex), 0.0, 0.0, 0.0);\n";
            w += "    var oPos = vec4f(0.0, 0.0, 0.0, 1.0);\n";
            for (uint32_t i = 0; i < 16; i++)
                if (interpolators & (1u << i))
                    w += "    var oV" + Num(i) + " = vec4f(0.0);\n";
            w += ve.body;
            // Xenos viewport transform (PA_CL_VTE_CNTL, viewport, window offset) into clip
            // space for the render target. With the usual perspective divide the result
            // stays linear in (x, w), so the GPU clips and interpolates it correctly.
            w += R"(
    let xf0 = D.xf[0];
    let xf1 = D.xf[1];
    let fl = D.xf[2];
    let p = oPos;
    var o: VsOut;
    if (fl.x < 0.5 && fl.y < 0.5 && fl.z < 0.5) {
        o.pos = vec4f((p.x * xf0.x + xf0.y * p.w) * 2.0 / xf1.z - p.w,
                      p.w - (p.y * xf0.z + xf0.w * p.w) * 2.0 / xf1.w,
                      p.z * xf1.x + xf1.y * p.w,
                      p.w);
    } else {
        let invW = select(1.0 / p.w, p.w, fl.z > 0.5);
        let xs = select(p.x * invW, p.x, fl.x > 0.5);
        let ys = select(p.y * invW, p.y, fl.x > 0.5);
        let zs = select(p.z * invW, p.z, fl.y > 0.5);
        let cw = select(1.0 / invW, 1.0, fl.x > 0.5);
        o.pos = vec4f(((xs * xf0.x + xf0.y) * 2.0 / xf1.z - 1.0) * cw,
                      (1.0 - (ys * xf0.z + xf0.w) * 2.0 / xf1.w) * cw,
                      (zs * xf1.x + xf1.y) * cw,
                      cw);
    }
)";
            for (uint32_t i = 0; i < 16; i++)
                if (interpolators & (1u << i))
                    w += "    o.v" + Num(i) + " = oV" + Num(i) + ";\n";
            w += "    return o;\n}\n\n";
        }
        else
        {
            w += "struct VsIn {\n    @location(0) pos: vec4f,\n";
            location = 1;
            for (uint32_t i = 0; i < 16; i++)
                if (interpolators & (1u << i))
                    w += "    @location(" + Num(location++) + ") v" + Num(i) + ": vec4f,\n";
            w += "}\n@vertex fn vs(input: VsIn) -> VsOut {\n    var o: VsOut;\n    o.pos = input.pos;\n";
            for (uint32_t i = 0; i < 16; i++)
                if (interpolators & (1u << i))
                    w += "    o.v" + Num(i) + " = input.v" + Num(i) + ";\n";
            w += "    return o;\n}\n\n";
        }

        // ---- pixel stage
        Emitter e;
        e.options = &options;
        if (ps)
            EmitProgram(e, *ps);
        out.textureSlots = e.textureSlots;
        out.colorExports = e.colorExports;
        out.depthExport = e.depthExport;

        for (uint32_t slot = 0; slot < 32; slot++)
            if (e.textureSlots & (1u << slot))
                w += "@group(1) @binding(" + Num(slot * 2) + ") var t" + Num(slot) + ": texture_2d<f32>;\n@group(1) @binding(" +
                    Num(slot * 2 + 1) + ") var s" + Num(slot) + ": sampler;\n";

        w += "struct FsOut {\n";
        bool anyOutput = false;
        for (uint32_t k = 0; k < 4; k++)
            if (options.colorTargets & (1u << k))
            {
                w += "    @location(" + Num(k) + ") o" + Num(k) + ": vec4f,\n";
                anyOutput = true;
            }
        if (e.depthExport)
        {
            w += "    @builtin(frag_depth) depth: f32,\n";
            anyOutput = true;
        }
        if (!anyOutput)
            w += "    @location(0) unused: vec4f,\n"; // never bound (no colour targets)
        w += "}\n@fragment fn fs(input: VsOut) -> FsOut {\n";
        w += Declarations(e);
        for (uint32_t i = 0; i < 16; i++)
            if ((interpolators & (1u << i)) && e.regs.count(i))
                w += "    " + Reg(i) + " = input.v" + Num(i) + ";\n";
        if (options.paramGen && e.regs.count(options.paramGenRegister & 127))
            w += "    " + Reg(options.paramGenRegister) + " = vec4f(floor(input.pos.x), floor(input.pos.y), 0.0, 0.0);\n";
        w += "    var o0 = vec4f(0.0);\n    var o1 = vec4f(0.0);\n    var o2 = vec4f(0.0);\n    var o3 = vec4f(0.0);\n    var oD = vec4f(input.pos.z);\n";
        w += e.body;
        w += "    if (killed) { discard; }\n";
        if (const char* cmp = CompareOp(options.alphaFunc))
            w += std::string("    if (!(o0.w ") + cmp + " D.misc.x)) { discard; }\n";
        else if ((options.alphaFunc & 7) == 0)
            w += "    discard;\n";
        w += "    var result: FsOut;\n";
        for (uint32_t k = 0; k < 4; k++)
            if (options.colorTargets & (1u << k))
                w += "    result.o" + Num(k) + " = o" + Num(k) + ";\n";
        if (e.depthExport)
            w += "    result.depth = clamp(oD.x, 0.0, 1.0);\n";
        if (!anyOutput)
            w += "    result.unused = vec4f(0.0);\n";
        w += "    return result;\n}\n";
        if (ve.unsupportedVertexFormat)
            out.code.clear(); // the caller falls back to the software front end
        return out;
    }
}
