"""Publication figure style: font sizes, rcParams, and Illustrator-safe SVG export.

Use :func:`setup_matplotlib_style` (applied automatically on import) for the
rcParams, and :func:`savefig` instead of ``fig.savefig``/``plt.savefig`` for SVG
output. See :func:`expand_font_shorthand` for why the latter is needed.
"""

import re
from pathlib import Path

import matplotlib as mpl

CM = 1 / 2.54

# Default manuscript figure font sizes (panel=10, labels/titles=7, ticks/legends=5)
FONTSIZE_DICT = {
    "panel": 10,
    "title": 7,
    "label": 7,
    "tick": 5,
    "legend": 5,
}

# Presentation / poster font sizes
FONTSIZE_DICT_PRESENTATION = {
    "panel": 16,
    "title": 14,
    "label": 12,
    "tick": 10,
    "legend": 10,
}


def setup_matplotlib_style(font_sans_serif="Arial", font_dict=None):
    """Applies common publication rcParams for fonts, tick padding, and vector export."""
    if font_dict is None:
        font_dict = FONTSIZE_DICT

    # Vector export settings
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"

    # Font family & sizes
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = [
        font_sans_serif,
        "DejaVu Sans",
        "Helvetica",
        "Arial",
    ]
    mpl.rcParams["axes.titlesize"] = font_dict.get("title", 7)
    mpl.rcParams["axes.labelsize"] = font_dict.get("label", 7)
    mpl.rcParams["xtick.labelsize"] = font_dict.get("tick", 5)
    mpl.rcParams["ytick.labelsize"] = font_dict.get("tick", 5)
    mpl.rcParams["legend.fontsize"] = font_dict.get("legend", 5)

    # Tick label padding (Matplotlib defaults are ~3.5; reduced by >2x)
    mpl.rcParams["xtick.major.pad"] = 1.5
    mpl.rcParams["ytick.major.pad"] = 1.5
    mpl.rcParams["xtick.minor.pad"] = 1.5
    mpl.rcParams["ytick.minor.pad"] = 1.5

    # Axis label padding (Matplotlib default is 4.0; reduced by >2x)
    mpl.rcParams["axes.labelpad"] = 1.5


# Apply defaults on import
setup_matplotlib_style()


# ─────────────────────── Illustrator-safe SVG export ────────────────────────
# Matplotlib emits: font: [<style>] [<variant>] [<weight>] <size>px <family list>
# The family list can hold several comma-separated, quoted names, but never a
# ";" or a '"' - the declaration lives inside a double-quoted style attribute.
_FONT_SHORTHAND = re.compile(
    r"font:\s*"
    r"(?:(italic|oblique)\s+)?"  # style
    r"(?:(small-caps)\s+)?"  # variant
    r"(?:(\d{3})\s+)?"  # numeric weight, omitted when 400
    r"([\d.]+)px\s+"  # size, always in px == user units == pt
    r"([^;\"]+)"  # family list
)


def _expand_shorthand(match):
    """Rewrite one CSS ``font`` shorthand match as longhand properties."""
    style, variant, weight, size, family = match.groups()
    parts = [f"font-family: {family.strip()}", f"font-size: {size}px"]
    if weight is not None:
        parts.append(f"font-weight: {weight}")
    if style is not None:
        parts.append(f"font-style: {style}")
    if variant is not None:
        parts.append(f"font-variant: {variant}")
    return "; ".join(parts)


def expand_font_shorthand(svg_text):
    """Rewrite CSS ``font`` shorthand declarations in an SVG as longhand properties.

    Why this is needed
    ------------------
    Since 3.5.0, matplotlib's SVG backend writes font properties using the CSS
    ``font`` *shorthand* rather than individual properties, and prepends a
    numeric weight only when it is not 400 (``RendererSVG._draw_text_as_text``)::

        <text style="font: 5px 'Arial'">...</text>          # normal weight
        <text style="font: 700 10px 'Arial'">...</text>     # fontweight="bold"

    Several vector editors, Adobe Illustrator among them, parse the two-token
    ``<size> <family>`` form but choke on the three-token
    ``<weight> <size> <family>`` form: they discard the whole declaration and
    fall back to their own default character size, 12 pt in Illustrator. The
    symptom is that **bold text opens at 12 pt however the figure was built** -
    typically the panel letters, since they are usually the only bold text -
    while every unweighted label comes in at the correct size. Illustrator's
    Character panel shows 12 pt, not a scaled version of the real size, which is
    how you tell this apart from a genuine unit-conversion bug (a pt/px mix-up
    would give 10 * 96/72 = 13.33 pt).

    The SVG itself is valid: matplotlib declares the canvas in points with a
    matching ``viewBox``, so one user unit is one point and ``10px`` really does
    mean 10 pt. Only the shorthand needs expanding::

        <text style="font-family: 'Arial'; font-size: 10px; font-weight: 700">...

    References
    ----------
    - matplotlib PR #19253, which introduced the shorthand in 3.5.0:
      https://github.com/matplotlib/matplotlib/pull/19253
    - matplotlib issue #22528, an earlier case of a viewer (Chrome) rejecting
      the shorthand: https://github.com/matplotlib/matplotlib/issues/22528
    - Affinity bug report describing the same parsing failure in an editor:
      "Text in imported SVG documents is not rendered in the correct size or
      font family if the font was specified using the shorthand 'font'
      property": https://forum.affinity.serif.com/index.php?/topic/173734-font-sizes-in-imported-svg-documents-are-sometimes-interpreted-incorrectly/

    Args:
        svg_text (str): Contents of an SVG file written by matplotlib with
            ``rcParams["svg.fonttype"] = "none"``.

    Returns:
        tuple[str, int]: The rewritten SVG text and the number of declarations
            that were expanded.
    """
    return _FONT_SHORTHAND.subn(_expand_shorthand, svg_text)


def fix_svg_fonts(path, verbose=False):
    """Expand the CSS ``font`` shorthand of an existing SVG file, in place.

    Safe to run more than once: longhand properties no longer match the
    shorthand pattern, so a second call is a no-op. See
    :func:`expand_font_shorthand` for why this is needed.

    Args:
        path (str or Path): Path to the SVG file to rewrite.
        verbose (bool): Print how many declarations were expanded. Defaults to
            False.

    Returns:
        int: Number of font declarations that were expanded.
    """
    path = Path(path)
    new_text, n = expand_font_shorthand(path.read_text())
    if n:
        path.write_text(new_text)
    if verbose:
        print(f"{path.name}: expanded {n} font declaration(s)")
    return n


def savefig(path, fig=None, verbose=False, **kwargs):
    """Save a figure, keeping SVG text at the right size in Adobe Illustrator.

    Drop-in replacement for ``plt.savefig(path, ...)`` and
    ``fig.savefig(path, ...)``. For ``.svg`` output the CSS ``font`` shorthand is
    expanded into longhand properties (see :func:`expand_font_shorthand`);
    all other formats are passed straight through untouched.

    Args:
        path (str or Path): Output path. Post-processing is applied only when the
            suffix is ``.svg``.
        fig (matplotlib.figure.Figure): Figure to save. Defaults to the current
            figure, matching ``plt.savefig`` behaviour.
        verbose (bool): Print how many declarations were expanded. Defaults to
            False.
        **kwargs: Forwarded to ``fig.savefig`` (e.g. ``bbox_inches="tight"``).

    Returns:
        Path: The path that was written.

    Example::

        from cottage_analysis.plotting import style
        style.savefig(SAVE_ROOT / "fig.svg", fig=fig, bbox_inches="tight", dpi=300)
    """
    import matplotlib.pyplot as plt

    if fig is None:
        fig = plt.gcf()
    path = Path(path)
    fig.savefig(path, **kwargs)
    if path.suffix.lower() == ".svg":
        fix_svg_fonts(path, verbose=verbose)
    return path
