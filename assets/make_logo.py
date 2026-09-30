#!/usr/bin/env python3
"""Regenerate the MICA flame logo SVGs from the cell grids below.

    python assets/make_logo.py

The flame is drawn as cells of a grid, like MICA's cellular automaton.
Levels: 1 outer red-orange, 2 orange, 3 amber, 4 pale-yellow core.
"""
from pathlib import Path

BIG = """
.......1......
.......11.....
.......111....
......1211....
......12211...
.1....122211..
.11..1122221..
.11..1223221..
.111.12233221.
.111122333221.
.112233433221.
.112334443321.
.122344443321.
.122344443321.
.112334443211.
..11233332211.
...112222111..
....1111111...
"""
SMALL = """
....1...
....11..
...121..
.1.1221.
.1122221
11233221
12344321
12344321
.123321.
..1111..
"""
COLORS = {1: "#e0421b", 2: "#f97316", 3: "#fbbf24", 4: "#fff3c4"}


def parse(s: str) -> list[list[int]]:
    return [[int(c) if c != "." else 0 for c in row] for row in s.strip().splitlines()]


def svg(grid, cell=10, gap=1.6, rx=2, bg=None, pad=0) -> str:
    h, w = len(grid), len(grid[0])
    W, H = w * cell + 2 * pad, h * cell + 2 * pad
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:g} {H:g}">']
    if bg:
        out.append(f'<rect width="{W:g}" height="{H:g}" rx="{W * 0.22:g}" fill="{bg}"/>')
    for y, row in enumerate(grid):
        for x, v in enumerate(row):
            if v:
                out.append(f'<rect x="{pad + x * cell + gap / 2:g}" y="{pad + y * cell + gap / 2:g}" '
                           f'width="{cell - gap:g}" height="{cell - gap:g}" rx="{rx:g}" fill="{COLORS[v]}"/>')
    out.append("</svg>")
    return "".join(out)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    big, small = parse(BIG), parse(SMALL)
    files = {
        "assets/mica-flame.svg": svg(big),
        "assets/mica-flame-small.svg": svg(small, gap=1.2, rx=1.5),
        "assets/mica-icon.svg": svg(small, gap=1.2, rx=1.5, bg="#1c1917", pad=14),
        "public/logo.svg": svg(big),
        "public/favicon.svg": svg(small, gap=1.2, rx=1.5, bg="#1c1917", pad=14),
    }
    for path, text in files.items():
        (root / path).write_text(text)
        print("wrote", path)
