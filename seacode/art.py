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
