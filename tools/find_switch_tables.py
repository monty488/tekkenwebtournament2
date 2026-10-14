"""Jump table detector for XenonRecomp that tolerates instruction scheduling.

XenonAnalyse matches fixed instruction sequences (lis, addi, rlwinm, lwzx, ...).
TTT2's compiler interleaves those differently (e.g. lis, rlwinm, addi, lwzx),
so XenonAnalyse finds nothing. This script instead symbolically executes the
few instructions between the bounds check (cmplwi + bgt/ble) and each bctr,
and recognises any ordering of:

  absolute : target = load32(table + idx*4)
  offset   : target = base + (load8/16(table + idx*n) << shift)

It also fixes function boundaries. Leaf functions have no .pdata entry, and
XenonRecomp's analyser sizes them by scanning for a return, which stops before
the cases of a jump table. For every switch whose labels fall outside its
.pdata function, an explicit { address, size } is written into the generated
block of the main TOML (between the BEGIN/END GENERATED FUNCTIONS markers).

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
            for j in range(1, 8):
                c = a - 4 * j
                f = fields(word(c))
                if f["op"] == 10 and (f["rd"] & 1) == 0 and (f["rd"] >> 2) == crf:  # cmplwi (L=0)
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
            if start is None:
                return None
            break
        if (i >> 26) in (18, 19) or i == BCTR:
            return None  # other control flow first: not a simple switch
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
        elif op == 24 and regs.get(f["rd"], ("?",))[0] == "const":     # ori
            val = ("const", regs[f["rd"]][1] | f["uimm"])
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


def fix_function_bounds(found, word, sections, code):
    import bisect

    pbase, psize = sections[".pdata"]
    pdata = []
    for o in range(0, psize, 8):
        begin, info = word(pbase + o), word(pbase + o + 4)
        if begin:
            pdata.append((begin, ((info >> 8) & 0x3FFFFF) * 4))
    pdata.sort()
    pstarts = [b for b, _ in pdata]

    # Every bl target is a function start as far as XenonRecomp is concerned.
    starts = set(pstarts)
    for sec_base, sec_size in code:
        for a in range(sec_base, sec_base + sec_size, 4):
            i = word(a)
            if (i >> 26) == 18 and (i & 3) == 1:
                disp = i & 0x03FFFFFC
                if disp & 0x02000000:
                    disp -= 0x04000000
                starts.add((a + disp) & 0xFFFFFFFF)
    starts = sorted(starts)

    bounds = {}
    for addr, _, labels in found:
        i = bisect.bisect_right(pstarts, addr) - 1
        if i >= 0 and pdata[i][0] <= addr < pdata[i][0] + pdata[i][1] and                 all(pdata[i][0] <= l < pdata[i][0] + pdata[i][1] for l in labels):
            continue  # .pdata already covers it

        # Start: nearest known function start at or before the switch.
        start = starts[bisect.bisect_right(starts, addr) - 1]
        if i >= 0 and pdata[i][0] + pdata[i][1] > start and pdata[i][0] <= addr < pdata[i][0] + pdata[i][1]:
            start = pdata[i][0]
        # End: next known function start after the last case, minus zero padding.
        last = max(labels + [addr])
        j = bisect.bisect_right(starts, last)
        end = starts[j] if j < len(starts) else last + 4
        while end - 4 > last and word(end - 4) == 0:
            end -= 4
        if start in bounds:
            end = max(end, start + bounds[start])
        bounds[start] = end - start

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

    found, unresolved = [], []
    for sec_base, sec_size in code:
        for a in range(sec_base, sec_base + sec_size, 4):
            if word(a) != BCTR:
                continue
            res = analyse(a, word, base)
            if res:
                found.append((a, *res))
            elif any(fields(word(a - 4 * k))["op"] == 10 for k in range(2, 10)):
                unresolved.append(a)  # has a bounds check nearby but no recognised dispatch

    functions = fix_function_bounds(found, word, load_sections(scan_dir), code)

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
