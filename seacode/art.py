"""SeaCode's ASCII mascot (provided by the project's human) plus the
ordered-dither backdrop generator for the TUI splash screen."""

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
