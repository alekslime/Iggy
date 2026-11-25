"""SeaCode's ASCII mascot (provided by the project's human) plus the
ordered-dither backdrop generator for the TUI splash screen."""

import math
from textwrap import dedent

MASCOT = dedent(r"""
                     ,  ,
                     \:.|`._
                  /\/;.:':::;;;._
                 <  .'     ':::;(
                  < ' _      '::;>
                   \ (9)  _  :::;(
                   |     / \  \\:;`>
                   |    /  |  //:;(
                   |   (  <=-  .::;>
                   (  a) )=-  .::;(
                    '-' <=- _.::;>
                       )==.'_(::(  ,
                      <==- ~_ >(,-'(
                      )=- '._(:  _.~>
                     <==-    ':.' _(
                      <==-    .:'_ (
                       )==- .::'  '~>
                        <=- .:;(`'.(
                         `)  ':;>  `
                    .-.  <    :;(
                  <`.':\  )    :;>
                 < :/<_/  <  .:;>
                 <  `---'`  .::(`
                  <       .:;>'
                   `-..:::-'`
""").strip("\n")


# Shade ramp for ordered dithering: quantize a 0..1 density into
# space/light/medium/full block, dithering the fractional remainder
# with a periodic threshold so transitions scatter instead of banding.
SHADES = " ░▒▓"


def dither_rule(width: int, phase: int = 0) -> str:
    """One symmetric dither-fade row: dense ▓ at the edges fading
    through ▒░ to clear space in the middle. Pure function of x;
    phase shifts the scatter so frames shimmer (period 8)."""

    width = max(int(width), 1)

    if width == 1:
        return " "

    center = (width - 1) / 2
    chars = []

    for x in range(width):
        # Distance-quantized scatter keeps the row mirrored.
        step = int(abs(x - center) + 0.5)
        edge = abs(x - center) / center
        level = edge ** 1.6 * 3.0
        base = int(level)
        frac = level - base
        scatter = (((step + phase) * 5) % 8) / 8

        if scatter < frac:
            base += 1

        chars.append(SHADES[min(base, 3)])

    return "".join(chars)


def wave_rule(width: int, phase: int = 0) -> str:
    """One row of sea waves (∼≈~) fading out at both edges.

    Replaces the blocky dither strip under the splash input: same
    symmetric-fade idea, but unmistakably water. Glyph choice is a
    pure function of position + phase so frames drift (period 12).
    Edge characters fade to spaces so the row melts into the
    background instead of ending in hard blocks.
    """

    width = max(int(width), 1)
    glyphs = "∼≈~"
    chars = []

    for x in range(width):
        # Fade in/out over the outer quarter on each side.
        edge_distance = min(x, width - 1 - x)
        fade_span = max(width // 4, 1)

        if edge_distance == 0:
            chars.append(" ")
            continue

        if edge_distance < fade_span and (edge_distance + phase) % 3 == 0:
            chars.append(" ")
            continue

        chars.append(glyphs[(x + phase) % len(glyphs)])

    return "".join(chars)


FISH_RIGHT = ("><>", "><")
FISH_LEFT = ("<><", "<>")


def side_column(width: int, height: int, phase: int = 0, seed: int = 0,
                direction: int = 1) -> str:
    """One animated sea strip: bubbles rising through lanes plus fish
    drifting across. Pure function of (x, y, phase): same inputs give
    the same frame, so animation is just repaints over time."""

    width = max(int(width), 1)
    height = max(int(height), 1)

    grid = [[" "] * width for _ in range(height)]

    lanes = max(2, width // 6)

    for i in range(lanes):
        x = (seed * 11 + i * 17) % width
        speed = 1 + (i % 2)
        y = (height - 1 - (phase * speed + i * 7 + seed * 3)) % height
        grid[y][x] = (".", "o", "°")[i % 3]

    fishes = FISH_RIGHT if direction > 0 else FISH_LEFT

    for i, glyph in enumerate(fishes):
        y = (height * (i + 1)) // (len(fishes) + 1)
        span = width + len(glyph) + 4
        pos = (phase * (1 + i) + seed * 5 + i * 11) % span

        if direction > 0:
            x = pos - len(glyph) - 2
        else:
            x = width + 2 - pos

        for j, char in enumerate(glyph):
            xx = x + j

            if 0 <= xx < width:
                grid[y][xx] = char

    return "\n".join("".join(row).rstrip() for row in grid)


# -----------------------------------------------------------------------
# Mosaic: generative block-tile animation (p5.js-style noise fields)
# -----------------------------------------------------------------------
#
# The splash side strips are a grid of 2x1 tiles ("pixels" that look
# square in a terminal cell). Each tile samples a scalar field that
# moves over time -- drifting value noise, a diagonal ripple, or a
# radial bloom -- and is quantized onto the shade ramp " ░▒▓█". A
# static per-tile grain nudges the quantization threshold so
# neighbouring tiles flip at different moments, which is what makes it
# read as shifting mosaic tiles rather than a smooth gradient.
#
# Everything is a pure function of (x, y, phase, seed): same inputs
# give the same frame, so animation is just repainting over time.

MOSAIC_GLYPHS = " ░▒▓█"
MOSAIC_LEVELS = len(MOSAIC_GLYPHS)
MOSAIC_TILE_W = 2

# Frames each pattern runs before melting into the next, and how many
# of those frames are spent crossfading.
MOSAIC_MODE_FRAMES = 80
MOSAIC_FADE_FRAMES = 16
MOSAIC_MODES = 3

# Kind index used for fish/bubble glyphs overlaid on the mosaic.
CREATURE = MOSAIC_LEVELS


def _hash01(ix: int, iy: int, seed: int = 0) -> float:
    """Deterministic lattice hash in [0, 1)."""

    n = (
        ix * 374761393 + iy * 668265263 + seed * 2246822519
    ) & 0xFFFFFFFF
    n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
    n ^= n >> 16

    return (n & 0xFFFF) / 65536.0


def _smooth(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def value_noise(x: float, y: float, seed: int = 0) -> float:
    """Smooth 2D value noise in [0, 1] (p5's noise(), minus Perlin)."""

    x0 = int(x // 1)
    y0 = int(y // 1)
    fx = _smooth(x - x0)
    fy = _smooth(y - y0)

    a = _hash01(x0, y0, seed)
    b = _hash01(x0 + 1, y0, seed)
    c = _hash01(x0, y0 + 1, seed)
    d = _hash01(x0 + 1, y0 + 1, seed)

    top = a + (b - a) * fx
    bottom = c + (d - c) * fx

    return top + (bottom - top) * fy


def _field_drift(tx, ty, t, w, h, seed):
    """Two noise octaves sliding in different directions."""

    base = value_noise(tx * 0.17 + t * 0.060, ty * 0.26 - t * 0.040, seed)
    fine = value_noise(tx * 0.45 - t * 0.090, ty * 0.60 + t * 0.070,
                       seed + 1)

    return base * 0.68 + fine * 0.32


def _field_ripple(tx, ty, t, w, h, seed):
    """Diagonal bands sweeping across, bent by a little noise."""

    warp = value_noise(tx * 0.20, ty * 0.30 + t * 0.05, seed + 2)
    angle = (tx * 0.55 + ty * 0.85) - t * 0.18 + warp * 3.0

    return 0.5 + 0.5 * math.sin(angle)


def _field_bloom(tx, ty, t, w, h, seed):
    """Concentric rings pulsing out from a wandering centre."""

    cx = w / 2.0 + (w / 3.0) * math.sin(t * 0.035 + seed)
    cy = h / 2.0 + (h / 3.0) * math.cos(t * 0.027 + seed)
    dist = math.hypot(tx - cx, ty - cy)

    return 0.5 + 0.5 * math.sin(dist * 0.85 - t * 0.21)


_FIELDS = (_field_drift, _field_ripple, _field_bloom)


def _mosaic_value(tx, ty, phase, w, h, seed):
    """The active field value, crossfading into the next pattern over
    the last MOSAIC_FADE_FRAMES frames of each pattern's run."""

    slot, within = divmod(phase, MOSAIC_MODE_FRAMES)
    current = _FIELDS[slot % MOSAIC_MODES]
    value = current(tx, ty, phase, w, h, seed)

    remaining = MOSAIC_MODE_FRAMES - within

    if remaining <= MOSAIC_FADE_FRAMES:
        following = _FIELDS[(slot + 1) % MOSAIC_MODES]
        mix = _smooth(1.0 - remaining / MOSAIC_FADE_FRAMES)
        value += (
            following(tx, ty, phase, w, h, seed) - value
        ) * mix

    return value


def mosaic_grid(width: int, height: int, phase: int = 0, seed: int = 0,
                fade: int = 0, origin: tuple = (0, 0),
                canvas: tuple = None) -> list:
    """Shade levels (0..4) for a width x height cell grid.

    Cells pair up into 2-wide tiles so each tile is one square-ish
    block. `fade` dims the mosaic toward one side so it melts into the
    mascot instead of crowding it: +1 fades toward the right edge, -1
    toward the left, 0 leaves it uniform.

    `origin` is the (column, row) of this grid's top-left cell inside a
    larger shared canvas of `canvas` = (columns, rows). Strips that
    sample the same canvas and seed continue one pattern across their
    seams instead of each starting a fresh one."""

    width = max(int(width), 1)
    height = max(int(height), 1)
    ox, oy = int(origin[0]), int(origin[1])
    tiles_w = (width + MOSAIC_TILE_W - 1) // MOSAIC_TILE_W

    if canvas is None:
        field_w, field_h = tiles_w, height
    else:
        field_w = max((int(canvas[0]) + MOSAIC_TILE_W - 1)
                      // MOSAIC_TILE_W, 1)
        field_h = max(int(canvas[1]), 1)

    grid = []

    for y in range(height):
        row = []
        cache = {}

        for x in range(width):
            tx = (ox + x) // MOSAIC_TILE_W

            if tx not in cache:
                ty = oy + y
                value = _mosaic_value(tx, ty, phase, field_w, field_h, seed)

                # Static per-tile grain staggers when each tile flips.
                grain = (_hash01(tx, ty, seed + 7) - 0.5) * 0.30
                # Push mid values down so level 0 (clear) stays common.
                value = max(value + grain, 0.0) ** 1.35

                if fade and tiles_w > 1:
                    local = x // MOSAIC_TILE_W
                    pos = local / (tiles_w - 1)
                    edge = pos if fade > 0 else 1.0 - pos
                    # Full strength for the outer 40%, then ramp to clear.
                    value *= 1.0 - _smooth(max(edge - 0.4, 0.0) / 0.6)

                cache[tx] = min(int(value * MOSAIC_LEVELS),
                                MOSAIC_LEVELS - 1)

            row.append(cache[tx])

        grid.append(row)

    return grid


def mosaic_frame(width: int, height: int, phase: int = 0, seed: int = 0,
                 fade: int = 0) -> str:
    """Plain-text mosaic frame (no color)."""

    return "\n".join(
        "".join(MOSAIC_GLYPHS[level] for level in row).rstrip()
        for row in mosaic_grid(width, height, phase, seed, fade)
    )


def sea_runs(width: int, height: int, phase: int = 0, seed: int = 0,
             direction: int = 1, origin: tuple = (0, 0),
             canvas: tuple = None, field_seed: int = None) -> list:
    """Side strip = mosaic backdrop with the existing bubbles and fish
    drawn on top. Returns, per row, a list of (text, kind) runs where
    kind is a shade level 0..4 or CREATURE, so the caller can color
    each run without this module knowing about any UI library.

    `origin`, `canvas` and `field_seed` place the backdrop inside a
    mosaic shared with neighbouring strips; the fish and bubbles keep
    using `seed`, so each strip still has its own creatures."""

    width = max(int(width), 1)
    height = max(int(height), 1)

    # Left strip (direction +1) sits left of the mascot: fade right.
    fade = 1 if direction > 0 else -1
    backdrop = mosaic_grid(
        width, height, phase,
        seed if field_seed is None else field_seed,
        fade, origin, canvas,
    )
    overlay = side_column(width, height, phase, seed, direction)
    overlay_rows = overlay.split("\n")

    rows = []

    for y in range(height):
        over = overlay_rows[y] if y < len(overlay_rows) else ""
        runs = []

        for x in range(width):
            char = over[x] if x < len(over) and over[x] != " " else None

            if char is not None:
                kind, glyph = CREATURE, char
            else:
                kind = backdrop[y][x]
                glyph = MOSAIC_GLYPHS[kind]

            if runs and runs[-1][1] == kind:
                runs[-1] = (runs[-1][0] + glyph, kind)
            else:
                runs.append((glyph, kind))

        rows.append(runs)

    return rows


def blend_hex(base: str, tint: str, amount: float) -> str:
    """Mix two #rrggbb colors: amount 0 -> base, 1 -> tint."""

    amount = min(max(amount, 0.0), 1.0)
    a = [int(base.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    b = [int(tint.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]

    return "#" + "".join(
        f"{round(x + (y - x) * amount):02x}" for x, y in zip(a, b)
    )


# How much of the accent each shade level mixes into the background:
# level 0 is clear (never drawn), 1..4 climb toward the accent.
MOSAIC_TINTS = (0.0, 0.16, 0.30, 0.48, 0.70)


def mosaic_palette(bg: str, accent: str) -> tuple:
    """One color per kind: levels 0..4 blended bg->accent, plus a
    bright color for the fish/bubbles on top."""

    levels = tuple(blend_hex(bg, accent, t) for t in MOSAIC_TINTS)

    return levels + (blend_hex(accent, "#ffffff", 0.25),)
