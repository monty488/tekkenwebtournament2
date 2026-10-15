"""Jump table detector for XenonRecomp that tolerates instruction scheduling.

XenonAnalyse matches fixed instruction sequences (lis, addi, rlwinm, lwzx, ...).
TTT2's compiler interleaves those differently (e.g. lis, rlwinm, addi, lwzx),
so XenonAnalyse finds nothing. This script instead symbolically executes the
few instructions between the bounds check (cmplwi + bgt/ble) and each bctr,
and recognises any ordering of:

  absolute : target = load32(table + idx*4)
  offset   : target = base + (load8/16(table + idx*n) << shift)

It also fixes function boundaries. Leaf functions have no .pdata entry, and
XenonRecomp's analyser guesses their size, stopping too early around jump
tables and bdzf chains. For every function start (pdata entries and bl targets),
the reachable code is found by following branches and switch labels without
entering another function. Every leaf function, and every .pdata function the
walk outgrows, gets an explicit { address, size } in the generated block of
the main TOML (between the BEGIN/END GENERATED FUNCTIONS markers).

Usage: python -I tools/find_switch_tables.py <game/scan> <switch_tables.toml> [main.toml]
Reads image.bin + image_info.toml (written by ttt2-scan).
"""

import re
import struct
import sys
from pathlib import Path

BCTR = 0x4E800420
WINDOW = 24  # instructions searched back from each bctr for the bounds check


def load_sections(scan_dir: Path):
    info = (scan_dir / "image_info.toml").read_text()
    return {
        n: (int(b, 16), int(sz, 16))
        for n, b, sz in re.findall(r'name = "([^"]+)"\nbase = (0x[0-9A-Fa-f]+)\nsize = (0x[0-9A-Fa-f]+)', info)
    }


def load_image(scan_dir: Path):
    info = (scan_dir / "image_info.toml").read_text()
    base = int(re.search(r"^base = (0x[0-9A-Fa-f]+)", info, re.M).group(1), 16)
    code = [
        (int(b, 16), int(s, 16))
        for b, s, c in re.findall(
            r"base = (0x[0-9A-Fa-f]+)\nsize = (0x[0-9A-Fa-f]+)\ncode = (true|false)", info)
        if c == "true"
    ]
    return base, (scan_dir / "image.bin").read_bytes(), code


def fields(i):
    return {
        "op": i >> 26, "rd": (i >> 21) & 31, "ra": (i >> 16) & 31, "rb": (i >> 11) & 31,
        "simm": struct.unpack(">h", struct.pack(">H", i & 0xFFFF))[0], "uimm": i & 0xFFFF,
        "xo": (i >> 1) & 0x3FF, "sh": (i >> 11) & 31, "mb": (i >> 6) & 31, "me": (i >> 1) & 31,
    }


def is_mtctr(i):
    f = fields(i)
    spr = ((i >> 16) & 31) | (((i >> 11) & 31) << 5)
    return f["op"] == 31 and f["xo"] == 467 and spr == 9


def dest_reg(i):
    """Integer register written by a (non-branch) instruction, or None if unknown/none."""
    f = fields(i)
    op = f["op"]
    if op in (14, 15, 7, 12, 13, 32, 33, 34, 35, 40, 41, 42, 43, 58):  # addi addis mulli addic lwz lbz lhz lha ld
        return f["rd"]
    if op in (20, 21, 23, 24, 25, 26, 27, 28, 29, 30):  # rotates, ori/oris, xori, andi.
        return f["ra"]
    if op == 31:
        if f["xo"] in (28, 60, 124, 284, 316, 412, 444, 476, 24, 536, 792, 824, 26, 922, 954, 986):
            return f["ra"]  # logical / shift ops write rA
        if f["xo"] in (467, 0, 32, 150, 151, 215, 183, 407, 439, 247, 54, 86, 278, 470, 982):
            return None  # mtspr, compares, stores, cache ops
        return f["rd"]
    return None


def analyse(addr, word, base_addr):
    """Returns (index_reg, count, labels) for a switch bctr at `addr`, else None."""
    if not is_mtctr(word(addr - 4)):
        return None

    # Find the bounds check: cmplwi crN, rI, max followed by a conditional branch on crN.
    # Scanning backwards, the conditional branch is met first, with the cmplwi just above it.
    # The bounds branch is either bc (bgt default) or a conditional bclr (bgtlr).
    # The compiler may schedule unrelated instructions between it and the cmplwi,
    # and may copy the index to another register (mr) before the compare.
    start = None
    aliases = set()
    for k in range(2, WINDOW):
        a = addr - 4 * k
        i = word(a)
        is_bc = (i >> 26) == 16
        is_bclr_cond = (i >> 26) == 19 and ((i >> 1) & 0x3FF) == 16 and i != 0x4E800020
        if is_bc or is_bclr_cond:
            crf = (i >> 18) & 7
            for j in range(1, 33):
                c = a - 4 * j
                f = fields(word(c))
                if f["op"] == 10 and (f["rd"] & 1) == 0 and (f["rd"] >> 2) == crf:  # cmplwi (L=0)
                    # The index must survive unchanged from the compare to the branch.
                    if any(dest_reg(word(x)) == f["ra"] for x in range(c + 4, a, 4)):
                        return None
                    start, idx_reg, count = a + 4, f["ra"], f["uimm"] + 1
                    # mr rX, rIdx shortly before the compare
                    for m in range(1, 7):
                        g = word(c - 4 * m)
                        gf = fields(g)
                        if gf["op"] == 31 and gf["xo"] == 444 and gf["rd"] == gf["rb"] == idx_reg:
                            aliases.add(gf["ra"])
                    break
                if (word(c) >> 26) in (16, 18, 19):
                    break
                # Another compare into the same CR field means the branch tests that one.
                if f["op"] in (10, 11) and (f["rd"] >> 2) == crf:
                    break
                if f["op"] == 31 and f["xo"] in (0, 32) and (f["rd"] >> 2) == crf:
                    break
            break
        if (i >> 26) in (18, 19) or i == BCTR:
            break  # other control flow first: no bounds check in this block

    if start is None:
        # Unbounded byte switch: `lbz rI` feeds the dispatch and the table has all 256
        # entries, so the compiler emits no bounds check.
        for k in range(2, WINDOW):
            a = addr - 4 * k
            i = word(a)
            if (i >> 26) in (16, 18, 19) or i == BCTR:
                break
            if (i >> 26) == 34:  # lbz
                rd = (i >> 21) & 31
                if any(dest_reg(word(x)) == rd for x in range(a + 4, addr, 4)):
                    break  # index clobbered before the bctr
                start, idx_reg, count = a + 4, rd, 256
                break

    if start is None or count > 4096:
        return None

    regs = {r: ("idx", 0) for r in aliases | {idx_reg}}
    for a in range(start, addr - 4, 4):
        i = word(a)
        f = fields(i)
        op = f["op"]
        val = None
        if op == 15 and f["ra"] == 0:                                   # lis
            val = ("const", (f["uimm"] << 16) & 0xFFFFFFFF)
        elif op == 14:                                                  # addi / li
            if f["ra"] == 0:
                val = ("const", f["simm"] & 0xFFFFFFFF)
            elif regs.get(f["ra"], ("?",))[0] == "const":
                val = ("const", (regs[f["ra"]][1] + f["simm"]) & 0xFFFFFFFF)
        elif op == 24 and f["uimm"] == 0:                              # ori rA,rS,0: copy (nop when rA == rS)
            val = regs.get(f["rd"])
        elif op == 24 and regs.get(f["rd"], ("?",))[0] == "const":     # ori
            val = ("const", regs[f["rd"]][1] | f["uimm"])
        elif op == 31 and f["xo"] == 444 and f["rd"] == f["rb"]:     # mr rA,rS (or rA,rS,rS)
            val = regs.get(f["rd"])
        elif op == 21:                                                  # rlwinm as slwi
            src = regs.get(f["rd"])
            if src and f["mb"] == 0 and f["me"] == 31 - f["sh"]:
                if src[0] == "idx":
                    val = ("idx", src[1] + f["sh"])
                elif src[0] == "load":
                    val = ("load", src[1], src[2], src[3], src[4] + f["sh"])
        elif op == 31 and f["xo"] in (23, 87, 279):                   # lwzx / lbzx / lhzx
            size = {23: 4, 87: 1, 279: 2}[f["xo"]]
            ra, rb = regs.get(f["ra"]), regs.get(f["rb"])
            for tbl, ix in ((ra, rb), (rb, ra)):
                if tbl and ix and tbl[0] == "const" and ix[0] == "idx" and (1 << ix[1]) == size:
                    val = ("load", size, tbl[1], None, 0)
        elif op == 31 and f["xo"] == 266:                             # add
            ra, rb = regs.get(f["ra"]), regs.get(f["rb"])
            for c, l in ((ra, rb), (rb, ra)):
                if c and l and c[0] == "const" and l[0] == "load" and l[3] is None:
                    val = ("load", l[1], l[2], c[1], l[4])

        d = dest_reg(i)
        if (i >> 26) in (16, 17, 18, 19):
            return None  # another branch inside the dispatch sequence
        if d is not None:
            if d == idx_reg and val is None:
                return None  # index register clobbered: XenonRecomp switches on it at the bctr
            if val is None:
                regs.pop(d, None)
            else:
                regs[d] = val

    target = regs.get(fields(word(addr - 4))["rd"])
    if not target or target[0] != "load":
        return None
    _, size, table, add_base, shift = target

    labels = []
    for n in range(count):
        p = table + n * size
        if size == 4:
            labels.append(word(p))
        else:
            raw = int.from_bytes(bytes(word_bytes(p, size)), "big")
            if add_base is None:
                return None
            labels.append((add_base + (raw << shift)) & 0xFFFFFFFF)
    if size == 4 and add_base is not None:
        return None
    return idx_reg, labels


_image = None

BEGIN = "# BEGIN GENERATED FUNCTIONS (tools/find_switch_tables.py)"
END = "# END GENERATED FUNCTIONS"


# XenonRecomp generates the register save/restore helpers itself: (name, size of the _14 entry).
HELPERS = {
    "restgprlr_14_address": 18 * 4 + 12, "savegprlr_14_address": 18 * 4 + 8,
    "restfpr_14_address": 18 * 4 + 4, "savefpr_14_address": 18 * 4 + 4,
    "restvmx_14_address": 18 * 8 + 4, "savevmx_14_address": 18 * 8 + 4,
    "restvmx_64_address": 64 * 8 + 4, "savevmx_64_address": 64 * 8 + 4,
}


def helper_regions(main_toml_text):
    regions = []
    for key, size in HELPERS.items():
        m = re.search(rf"^{key}\s*=\s*(0x[0-9A-Fa-f]+)", main_toml_text, re.M)
        if m and int(m.group(1), 16):
            regions.append((int(m.group(1), 16), int(m.group(1), 16) + size))
    return regions


def fix_function_bounds(found, word, sections, code, helpers=()):
    import bisect

    pbase, psize = sections[".pdata"]
    pdata = []
    for o in range(0, psize, 8):
        begin, info = word(pbase + o), word(pbase + o + 4)
        if begin:
            pdata.append((begin, ((info >> 8) & 0x3FFFFF) * 4))
    pdata.sort()
    pdata_size = dict(pdata)

    switches = {addr: labels for addr, _, labels in found}
    code_ranges = [(b, b + s) for b, s in code]

    def in_code(a):
        return any(lo <= a < hi for lo, hi in code_ranges)

    # Import thunks (nop, nop, nop, blr after XenonUtils patches them) are XenonRecomp's
    # __imp__ symbols. Giving them explicit bounds would turn them into empty sub_
    # functions and silently break every kernel call, so treat them like the helpers.
    thunk = (0x60000000, 0x60000000, 0x60000000, 0x4E800020)
    regions = list(helpers)
    for sec_base, sec_size in code:
        for a in range(sec_base, sec_base + sec_size - 12, 4):
            if (word(a), word(a + 4), word(a + 8), word(a + 12)) == thunk:
                regions.append((a, a + 16))
    covered = {}
    for lo, hi in regions:
        for a in range(lo, hi, 4):
            covered[a] = hi

    def helper_end(a):
        return covered.get(a)

    def branch_target(a, i):
        disp = (i & 0x03FFFFFC) if (i >> 26) == 18 else (i & 0xFFFC)
        if (i >> 26) == 18 and disp & 0x02000000:
            disp -= 0x04000000
        if (i >> 26) == 16 and disp & 0x8000:
            disp -= 0x10000
        return disp if (i & 2) else (a + disp) & 0xFFFFFFFF  # AA bit

    # Every branch in the image, as (source, target, is_bl).
    branches = []
    for sec_base, sec_size in code:
        for a in range(sec_base, sec_base + sec_size, 4):
            i = word(a)
            if (i >> 26) in (16, 18):
                branches.append((a, branch_target(a, i), bool(i & 1)))

    # Function entry points: .pdata, bl targets, code addresses stored in data (vtables,
    # callbacks), then (iteratively) any address another function branches into.
    starts = set(pdata_size) | {t for _, t, link in branches if link}

    def no_fallthrough_into(t):
        # Only split where the previous instruction cannot fall through into `t`.
        p = word(t - 4)
        return (p in (0, 0x4E800020, BCTR) or (p >> 26) == 3          # padding, blr, bctr, twi
                or ((p >> 26) == 18 and not (p & 1))                    # b
                or ((p >> 26) == 31 and ((p >> 1) & 0x3FF) == 4))       # tw

    # Data also points *into* functions: SEH scope tables reference __except handler
    # labels inside framed functions. Every function with a frame has a .pdata entry, so
    # only accept references that fall outside .pdata ranges (leaf code, vtable thunks).
    pdata_starts = [b for b, _ in pdata]

    def inside_pdata_function(t):
        k = bisect.bisect_right(pdata_starts, t) - 1
        return k >= 0 and pdata[k][0] < t < pdata[k][0] + pdata[k][1]

    switch_labels = {l for labels in switches.values() for l in labels}
    for name, (sec_base, sec_size) in sections.items():
        if name in (".pdata", ".reloc") or any(lo == sec_base for lo, _ in code_ranges):
            continue
        for a in range(sec_base, sec_base + sec_size - 3, 4):
            t = word(a)
            if t & 3 or not in_code(t) or t in switch_labels or inside_pdata_function(t):
                continue
            if no_fallthrough_into(t):
                starts.add(t)

    starts = {s for s in starts if in_code(s) and helper_end(s) is None}

    def compute(starts):
        ordered = sorted(starts)

        def extent(start):
            """End of the code reachable from `start` without entering another function."""
            end = start + 4
            seen = set()
            work = [start]
            while work:
                a = work.pop()
                while a not in seen and in_code(a) and helper_end(a) is None and (a == start or a not in starts):
                    seen.add(a)
                    end = max(end, a + 4)
                    i = word(a)
                    op = i >> 26
                    if i == 0x4E800020:                               # blr
                        break
                    if op == 18 and not (i & 1):                      # b (tail call if it leaves)
                        t = branch_target(a, i)
                        if t not in starts:
                            work.append(t)
                        break
                    if op == 16 and not (i & 1):                      # bc / bdz / bdzf / ...
                        t = branch_target(a, i)
                        if t not in starts:
                            work.append(t)
                        if ((i >> 21) & 0x14) == 0x14:                # branch always
                            break
                    elif i == BCTR:
                        work.extend(switches.get(a, []))
                        break
                    a += 4
            return end

        def capped_extent(s):
            # Never overlap the next known function: XenonRecomp's symbol table matches by
            # containment, and overlapping ranges make its analysis sweep loop forever.
            e = extent(s)
            j = bisect.bisect_right(ordered, s)
            return min(e, ordered[j]) if j < len(ordered) else e

        bounds = {}
        for s in ordered:
            e = capped_extent(s)
            if s in pdata_size:
                # Trust .pdata unless the walk outgrows it, but still stop at the next
                # entry point so no two functions overlap.
                j = bisect.bisect_right(ordered, s)
                limit = ordered[j] if j < len(ordered) else s + pdata_size[s]
                e = min(max(e, s + pdata_size[s]), limit)
                if e == s + pdata_size[s]:
                    continue  # .pdata size is already right
            bounds[s] = e - s

        # XenonRecomp then sweeps each code section linearly and starts a new function at
        # every address nothing covers. Mirror that sweep so those functions get
        # control-flow-based sizes too instead of XenonRecomp's guess.
        layout = {}
        for sec_base, sec_size in code:
            a, sec_end = sec_base, sec_base + sec_size
            while a < sec_end:
                if word(a) == 0:                   # padding (skipped via invalid_instructions)
                    a += 4
                    continue
                h = helper_end(a)
                if h is not None:
                    a = h
                    continue
                if a in bounds:
                    size = bounds[a]
                elif a in pdata_size:
                    size = pdata_size[a]
                else:
                    size = max(capped_extent(a) - a, 4)
                    bounds[a] = size
                layout[a] = size
                a += size
        return bounds, layout

    for _ in range(10):
        bounds, layout = compute(starts)
        fstarts = sorted(layout)

        def owner(a):
            k = bisect.bisect_right(fstarts, a) - 1
            return fstarts[k] if k >= 0 and a < fstarts[k] + layout[fstarts[k]] else None

        new = set()
        for src, dst, link in branches:
            if link or dst in starts or not in_code(dst) or helper_end(dst) is not None:
                continue
            od = owner(dst)
            if od is not None and od != dst and owner(src) != od:
                new.add(dst)  # another function jumps into the middle: it is an entry point
        if not new:
            break
        starts |= new

    return sorted(bounds.items())


def write_functions(main_toml: Path, functions):
    text = main_toml.read_text()
    block = [BEGIN, "functions = ["]
    block += [f"    {{ address = 0x{a:X}, size = 0x{sz:X} }}," for a, sz in functions]
    block += ["]", END]
    if BEGIN in text:
        pre, rest = text.split(BEGIN, 1)
        post = rest.split(END, 1)[1]
        text = pre + "\n".join(block) + post
    else:
        text = text.rstrip() + "\n\n" + "\n".join(block) + "\n"
    main_toml.write_text(text)


def word_bytes(addr, size):
    base, data = _image
    return data[addr - base: addr - base + size]


def main():
    global _image
    scan_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    base, data, code = load_image(scan_dir)
    _image = (base, data)

    def word(a):
        return struct.unpack_from(">I", data, a - base)[0]

    import bisect
    sections = load_sections(scan_dir)
    pbase, psize = sections[".pdata"]
    pdata = sorted((word(pbase + o), ((word(pbase + o + 4) >> 8) & 0x3FFFFF) * 4)
                   for o in range(0, psize, 8) if word(pbase + o))
    pdata_starts = [b for b, _ in pdata]

    found, unresolved = [], []
    for sec_base, sec_size in code:
        for a in range(sec_base, sec_base + sec_size, 4):
            if word(a) != BCTR:
                continue
            res = analyse(a, word, base)
            if res and len(res[1]) == 256:
                # Unbounded byte switch: the real table is usually shorter. Keep entries
                # up to the first label outside the enclosing .pdata function.
                k = bisect.bisect_right(pdata_starts, a) - 1
                lo, hi = (pdata[k][0], pdata[k][0] + pdata[k][1]) if k >= 0 and a < pdata[k][0] + pdata[k][1] else (a - 0x8000, a + 0x8000)
                labels = []
                for label in res[1]:
                    if not (lo <= label < hi):
                        break
                    labels.append(label)
                res = (res[0], labels) if labels else None
            if res:
                found.append((a, *res))
            elif any(fields(word(a - 4 * k))["op"] == 10 for k in range(2, 10)):
                unresolved.append(a)  # has a bounds check nearby but no recognised dispatch

    helpers = helper_regions(Path(sys.argv[3]).read_text()) if len(sys.argv) > 3 else []
    functions = fix_function_bounds(found, word, load_sections(scan_dir), code, helpers)

    lines = ["# Generated by tools/find_switch_tables.py", ""]
    for a, r, labels in found:
        lines += ["[[switch]]", f"base = 0x{a:X}", f"r = {r}", "default = 0x0", "labels = ["]
        lines += [f"    0x{l:X}," for l in labels]
        lines += ["]", ""]
    out_path.write_text("\n".join(lines))

    if len(sys.argv) > 3:
        write_functions(Path(sys.argv[3]), functions)

    print(f"switch tables: {len(found)}")
    print(f"explicit function bounds: {len(functions)}")
    print(f"bctr with a bounds check but unrecognised dispatch: {len(unresolved)}")
    for a in unresolved[:20]:
        print(f"  0x{a:X}")


if __name__ == "__main__":
    main()
