"""Tests for the Illustrator-safe SVG export helpers in cottage_analysis.plotting.style."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from cottage_analysis.plotting import style


def test_expands_bold_shorthand():
    """The three-token weight form is what Illustrator chokes on."""
    svg = "<text style=\"font: 700 10px 'Arial'\">A</text>"
    out, n = style.expand_font_shorthand(svg)
    assert n == 1
    assert "font-family: 'Arial'" in out
    assert "font-size: 10px" in out
    assert "font-weight: 700" in out
    assert "font:" not in out


def test_expands_plain_shorthand_without_inventing_a_weight():
    """Matplotlib omits the weight at 400, and so must the longhand."""
    svg = "<text style=\"font: 5px 'Arial'\">tick</text>"
    out, n = style.expand_font_shorthand(svg)
    assert n == 1
    assert "font-family: 'Arial'" in out
    assert "font-size: 5px" in out
    assert "font-weight" not in out


def test_expands_multi_family_fallback_list():
    """With font.family="sans-serif" matplotlib writes the whole fallback chain."""
    svg = (
        "<text style=\"font: 700 10px 'Arial', 'DejaVu Sans', 'Helvetica', "
        'sans-serif">A</text>'
    )
    out, n = style.expand_font_shorthand(svg)
    assert n == 1
    assert "font-family: 'Arial', 'DejaVu Sans', 'Helvetica', sans-serif" in out, out
    assert "font-size: 10px" in out
    assert "font-weight: 700" in out


def test_keeps_following_declarations_intact():
    """The family group must stop at the ';' separating the next declaration."""
    svg = "<text style=\"font: 7px 'Arial'; text-anchor: middle\">x</text>"
    out, n = style.expand_font_shorthand(svg)
    assert n == 1
    assert out.endswith('text-anchor: middle">x</text>')
    assert "font-family: 'Arial'" in out


def test_expands_italic_and_fractional_size():
    svg = "<text style=\"font: italic 6.5px 'Arial'\">n</text>"
    out, n = style.expand_font_shorthand(svg)
    assert n == 1
    assert "font-size: 6.5px" in out
    assert "font-style: italic" in out


def test_is_idempotent():
    """Longhand no longer matches, so a second pass is a no-op."""
    svg = "<text style=\"font: 700 10px 'Arial'\">A</text>"
    once, first = style.expand_font_shorthand(svg)
    twice, second = style.expand_font_shorthand(once)
    assert first == 1
    assert second == 0
    assert twice == once


def test_leaves_svg_without_shorthand_untouched():
    svg = "<text style=\"font-family: 'Arial'; font-size: 10px\">A</text>"
    out, n = style.expand_font_shorthand(svg)
    assert n == 0
    assert out == svg


def test_fix_svg_fonts_rewrites_in_place_and_is_idempotent(tmp_path):
    path = tmp_path / "panel.svg"
    path.write_text("<text style=\"font: 700 10px 'Arial'\">A</text>")
    assert style.fix_svg_fonts(path) == 1
    assert "font-weight: 700" in path.read_text()
    assert style.fix_svg_fonts(path) == 0


def test_savefig_expands_shorthand_for_svg(tmp_path):
    """End-to-end: a real matplotlib SVG comes out with no `font:` shorthand."""
    style.setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(4 * style.CM, 3 * style.CM))
    ax.set_xlabel("x")
    style.panel_letter(fig, "A", 0.1, 2.9)
    path = style.savefig(tmp_path / "fig.svg", fig=fig, bbox_inches="tight")
    plt.close(fig)

    text = path.read_text()
    assert 'style="font:' not in text
    assert "font-family:" in text
    # The panel letter is bold, at FONTSIZE_DICT["panel"].
    assert "font-weight: 700" in text
    assert f"font-size: {style.FONTSIZE_DICT['panel']}px" in text


def test_savefig_leaves_non_svg_formats_alone(tmp_path):
    fig, ax = plt.subplots()
    path = style.savefig(tmp_path / "fig.pdf", fig=fig)
    plt.close(fig)
    assert path.exists() and path.suffix == ".pdf"


@pytest.mark.parametrize("figsize_cm", [(19.0, 18.8), (8.5, 6.0)])
def test_rect_cm_maps_the_full_figure_to_the_unit_square(figsize_cm):
    w_cm, h_cm = figsize_cm
    fig = plt.figure(figsize=(w_cm * style.CM, h_cm * style.CM))
    assert style.rect_cm(fig, 0, 0, w_cm, h_cm) == pytest.approx([0, 0, 1, 1])
    assert style.rect_cm(fig, w_cm / 2, h_cm / 2, 0, 0) == pytest.approx(
        [0.5, 0.5, 0, 0]
    )
    plt.close(fig)


def test_panel_letter_defaults_to_the_shared_panel_size():
    fig = plt.figure(figsize=(10 * style.CM, 10 * style.CM))
    text = style.panel_letter(fig, "A", 1.0, 9.0)
    assert text.get_fontsize() == style.FONTSIZE_DICT["panel"]
    assert text.get_fontweight() == "bold"
    assert text.get_position() == pytest.approx((0.1, 0.9))
    plt.close(fig)


def test_setup_figure_fonts_does_not_raise_on_a_missing_dir():
    """The fonts live on an external drive; an unmounted drive must not break a run."""
    with pytest.warns(UserWarning):
        family = style.setup_figure_fonts(font_dir="/nonexistent/fonts")
    assert family == "Arial"
    assert matplotlib.rcParams["font.family"] == ["Arial"]
    assert matplotlib.rcParams["svg.fonttype"] == "none"
    assert matplotlib.rcParams["pdf.fonttype"] == 42
