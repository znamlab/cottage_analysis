"""Publication figure style: font sizes, rcParams, and Illustrator-safe SVG export.

Use :func:`setup_matplotlib_style` (applied automatically on import) for the
rcParams, and :func:`savefig` instead of ``fig.savefig``/``plt.savefig`` for SVG
output. See :func:`expand_font_shorthand` for why the latter is needed.
"""

import re
import warnings
from pathlib import Path

import matplotlib as mpl

CM = 1 / 2.54

# Directories searched by :func:`setup_figure_fonts` for the Arial ``.ttf`` faces,
# in order. The first one that exists wins.
FONT_SEARCH_DIRS = (
    "/Volumes/BlackPasspo/v1_depth_map/processed/v1_manuscript_figures/fonts",
    "/nemo/lab/znamenskiyp/home/shared/resources/fonts",
)

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


def _weight_as_int(weight):
    """Normalise a FontEntry weight, which may be an int or a name like 'bold'."""
    import matplotlib.font_manager as fm

    try:
        return int(weight)
    except (TypeError, ValueError):
        return fm.weight_dict.get(str(weight).lower(), 400)


def _registered_faces(family):
    """Return the faces matplotlib knows for ``family``, as (weight, style) pairs."""
    import matplotlib.font_manager as fm

    return [
        (_weight_as_int(entry.weight), entry.style)
        for entry in fm.fontManager.ttflist
        if entry.name == family
    ]


def setup_figure_fonts(font_dir=None, font_dict=None, family="Arial", verbose=False):
    """Register the manuscript font faces and apply the publication rcParams.

    This is the single setup call a figure notebook should make. It replaces the
    hand-rolled ``arial_font_path`` boilerplate that used to be pasted into every
    notebook, and fixes two problems with it.

    **It registers every face, not just the regular one.** The old boilerplate
    called ``addfont`` on ``arial.ttf`` alone, leaving ``arialbd.ttf`` unused in the
    same folder. ``findfont`` does not fail when asked for a weight it cannot
    supply - it silently returns the closest match - so ``fontweight="bold"``
    quietly produced *non-bold* text in PDF and PNG on any machine without a system
    Arial Bold (NEMO and most Linux runners; macOS happens to ship one in
    ``/System/Library/Fonts/Supplemental``, which is why this went unnoticed).
    Panel letters are usually the only bold text in a figure, so the symptom is
    panel letters that are not bold.

    **It never raises on a missing font directory.** The fonts live on an external
    drive, so three of the old variants failed at import time whenever it was not
    mounted. A missing directory is a warning here, and the requested family is
    used as-is in the hope that the system provides it.

    ``font.family`` is set to a single family name rather than ``"sans-serif"``: with
    the generic alias, matplotlib writes the whole ``font.sans-serif`` fallback
    chain into every SVG ``font-family``, and one name is what Illustrator resolves
    most reliably. See :func:`expand_font_shorthand` for the separate SVG-side fix
    that :func:`savefig` applies on top of this.

    Args:
        font_dir (str or Path): Directory holding the ``.ttf``/``.otf`` faces. Every
            font file in it is registered, so Arial Narrow and Arial Black come
            along too. Defaults to the first existing entry of
            :data:`FONT_SEARCH_DIRS`.
        font_dict (dict): Font sizes, forwarded to :func:`setup_matplotlib_style`.
            Defaults to :data:`FONTSIZE_DICT`.
        family (str): Font family to use for all text. Defaults to ``"Arial"``.
        verbose (bool): Print the directory used and the faces registered. Defaults
            to False.

    Returns:
        str: The family name that was applied, so a caller can assert on it.
    """
    import matplotlib.font_manager as fm

    if font_dir is None:
        font_dir = next((d for d in FONT_SEARCH_DIRS if Path(d).is_dir()), None)
        if font_dir is None and verbose:
            print(f"No font directory found in {FONT_SEARCH_DIRS}")

    registered = []
    if font_dir is not None:
        font_dir = Path(font_dir)
        if font_dir.is_dir():
            for path in sorted(font_dir.iterdir()):
                if path.suffix.lower() in (".ttf", ".otf"):
                    fm.fontManager.addfont(str(path))
                    registered.append(path.name)
            if verbose:
                print(f"Registered {len(registered)} font(s) from {font_dir}")
        else:
            warnings.warn(
                f"Font directory {font_dir} does not exist; falling back to any "
                f"system-installed {family!r}."
            )

    setup_matplotlib_style(font_sans_serif=family, font_dict=font_dict)
    # Name the one family directly, rather than leaving the "sans-serif" alias that
    # setup_matplotlib_style sets - see the note on font.family above.
    mpl.rcParams["font.family"] = family
    mpl.rcParams["mathtext.default"] = "regular"  # keep math mode in the same font

    faces = _registered_faces(family)
    if not faces:
        warnings.warn(
            f"No {family!r} face is available to matplotlib; text will fall back to "
            f"another font. Pass font_dir= to point at the manuscript fonts."
        )
    elif not any(weight >= 700 and style == "normal" for weight, style in faces):
        warnings.warn(
            f"No bold {family!r} face is available to matplotlib; text drawn with "
            f'fontweight="bold" (typically the panel letters) will not be bold in '
            f"PDF or PNG output. Pass font_dir= to point at the manuscript fonts."
        )
    if verbose:
        print(f"font.family -> {family} ({len(faces)} face(s) available)")
    return family


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


# ──────────────────────── Centimetre-based panel layout ─────────────────────
def rect_cm(fig, x, y, w, h):
    """Convert centimetre coordinates (bottom-left origin) to figure fractions.

    Args:
        fig (matplotlib.figure.Figure): Figure the rectangle belongs to.
        x, y (float): Position of the bottom-left corner, in cm from the bottom
            left of the figure.
        w, h (float): Width and height, in cm.

    Returns:
        list[float]: ``[left, bottom, width, height]`` in figure fractions, ready
            for ``fig.add_axes`` or a ``GridSpec`` rect.
    """
    fig_w, fig_h = (size / CM for size in fig.get_size_inches())
    return [x / fig_w, y / fig_h, w / fig_w, h / fig_h]


def panel_letter(fig, letter, x, y, fontsize=None, **kwargs):
    """Draw a bold panel letter at a centimetre position on the figure.

    Args:
        fig (matplotlib.figure.Figure): Figure to annotate.
        letter (str): Panel letter, e.g. ``"A"``.
        x, y (float): Position in cm from the bottom left of the figure. The letter
            is top-left aligned on that point, so ``y`` is its top edge.
        fontsize (float): Size in points. Defaults to ``FONTSIZE_DICT["panel"]``,
            which is what keeps panel letters consistent across figures.
        **kwargs: Forwarded to ``fig.text``.

    Returns:
        matplotlib.text.Text: The text artist that was added.
    """
    if fontsize is None:
        fontsize = FONTSIZE_DICT["panel"]
    fig_w, fig_h = (size / CM for size in fig.get_size_inches())
    kwargs.setdefault("va", "top")
    kwargs.setdefault("ha", "left")
    return fig.text(
        x / fig_w,
        y / fig_h,
        letter,
        fontsize=fontsize,
        fontweight="bold",
        **kwargs,
    )
