"""Build the DCC'26 intro motivation deck — visual-first version.

Same narrative as v1 (`build_intro_deck.py`) but rebuilt for less text and
more imagery. Each slide leads with a dominant visual: hero photo placeholder,
icon-led illustration, big-type stat, timeline, spider chart, or shape clipart.

Run:
    python3 workshops/dcc26/slides/build_intro_deck_v2.py

Output:
    workshops/dcc26/slides/introduction-benchmarking-genai-engineering-design-visual.pptx

Image placeholders are clearly labeled `[IMAGE: ...]` so a designer can swap in
real photos later without changing layout.
"""
from __future__ import annotations

import math
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt, Emu
from lxml import etree


# ---------- design system (matches workshop deck) ----------

BG = RGBColor(0xFB, 0xFA, 0xF7)
BLUE = RGBColor(0x22, 0x5E, 0x9B)
BLUE_DEEP = RGBColor(0x17, 0x42, 0x6E)
BLUE_LIGHT = RGBColor(0xD7, 0xE8, 0xF6)
BLUE_TINT = RGBColor(0xEC, 0xF3, 0xFA)
BLUE_TXT_LIGHT = RGBColor(0xEA, 0xF3, 0xFA)
BLUE_TXT_CHIP = RGBColor(0xCF, 0xE3, 0xF3)
NEAR_BLACK = RGBColor(0x17, 0x21, 0x2B)
GREY = RGBColor(0x5A, 0x66, 0x75)
GREY_LIGHT = RGBColor(0xC8, 0xCE, 0xD6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
RED = RGBColor(0xC8, 0x48, 0x37)
RED_TINT = RGBColor(0xF6, 0xE0, 0xDD)
ORANGE = RGBColor(0xC4, 0x7B, 0x20)
ORANGE_TINT = RGBColor(0xF6, 0xE9, 0xD2)
GREEN = RGBColor(0x2F, 0x7D, 0x62)
GREEN_TINT = RGBColor(0xDB, 0xEC, 0xE5)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

FONT_DISPLAY = "Aptos Display"
FONT_BODY = "Aptos"


# ---------- low-level helpers ----------

def add_rect(slide, x, y, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE):
    shp = slide.shapes.add_shape(shape, x, y, w, h)
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
    shp.shadow.inherit = False
    return shp


def add_text(slide, x, y, w, h, text, *, font=FONT_BODY, size=14, bold=False,
             italic=False, color=NEAR_BLACK, align=PP_ALIGN.LEFT,
             anchor=MSO_ANCHOR.TOP, line_spacing=None):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Emu(0)
    tf.margin_right = Emu(0)
    tf.margin_top = Emu(0)
    tf.margin_bottom = Emu(0)
    tf.vertical_anchor = anchor
    lines = text if isinstance(text, list) else [text]
    for i, line in enumerate(lines):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.alignment = align
        if line_spacing is not None:
            p.line_spacing = line_spacing
        r = p.add_run()
        r.text = line
        r.font.name = font
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.color.rgb = color
    return tb


def add_shape_with_text(slide, shape, x, y, w, h, fill, text, *, color=WHITE,
                        size=14, bold=True, font=FONT_BODY,
                        line=None, align=PP_ALIGN.CENTER):
    shp = add_rect(slide, x, y, w, h, fill, line=line, shape=shape)
    tf = shp.text_frame
    tf.margin_left = Emu(0)
    tf.margin_right = Emu(0)
    tf.margin_top = Emu(0)
    tf.margin_bottom = Emu(0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.name = font
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    return shp


def add_chrome(slide, section_label):
    add_rect(slide, Emu(0), Emu(0), SLIDE_W, SLIDE_H, BG)
    add_rect(slide, Emu(0), Emu(0), Inches(0.12), SLIDE_H, BLUE)
    add_text(slide, Inches(0.44), Inches(7.02), Inches(4.38), Inches(0.19),
             section_label, size=8.25, color=GREY)
    add_text(slide, Inches(8.54), Inches(7.02), Inches(4.06), Inches(0.19),
             "DCC 2026 workshop  |  EngiBench + EngiOpt",
             size=8.25, color=GREY, align=PP_ALIGN.RIGHT)


def add_eyebrow(slide, text, *, color=BLUE):
    add_text(slide, Inches(0.65), Inches(0.48), Inches(10.0), Inches(0.25),
             text, size=10, bold=True, color=color)


def add_title(slide, text, *, size=24, top=0.80, height=1.0, color=NEAR_BLACK,
              width=12.0):
    add_text(slide, Inches(0.65), Inches(top), Inches(width), Inches(height),
             text, font=FONT_DISPLAY, size=size, bold=True, color=color)


# ---------- visual / clipart helpers ----------

def image_placeholder(slide, x, y, w, h, label, *, tint=BLUE_TINT,
                      label_color=GREY, dashed=True, icon_color=BLUE):
    """Tinted rectangle that visually says 'image will go here'."""
    shp = add_rect(slide, x, y, w, h, tint)
    if dashed:
        ln = shp.line
        ln.color.rgb = BLUE
        ln.width = Pt(0.75)
        # dashed style
        spPr = shp.fill._xPr  # access OOXML
        # apply prstDash
        nsmap = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
        ln_elem = spPr.find("a:ln", nsmap)
        if ln_elem is not None:
            existing = ln_elem.find("a:prstDash", nsmap)
            if existing is None:
                dash = etree.SubElement(ln_elem, qn("a:prstDash"))
                dash.set("val", "dash")
    # camera-frame icon (mountain triangle + sun)
    ix = x + w / 2 - Inches(0.45)
    iy = y + h / 2 - Inches(0.55)
    frame = add_rect(slide, ix, iy, Inches(0.90), Inches(0.65),
                     None, line=icon_color, shape=MSO_SHAPE.RECTANGLE)
    frame.line.color.rgb = icon_color
    frame.line.width = Pt(1.5)
    # mountain
    tri = slide.shapes.add_shape(MSO_SHAPE.ISOSCELES_TRIANGLE,
                                 ix + Inches(0.18), iy + Inches(0.22),
                                 Inches(0.55), Inches(0.40))
    tri.fill.solid(); tri.fill.fore_color.rgb = icon_color
    tri.line.fill.background(); tri.shadow.inherit = False
    # sun
    sun = slide.shapes.add_shape(MSO_SHAPE.OVAL,
                                 ix + Inches(0.62), iy + Inches(0.10),
                                 Inches(0.16), Inches(0.16))
    sun.fill.solid(); sun.fill.fore_color.rgb = icon_color
    sun.line.fill.background(); sun.shadow.inherit = False
    # caption
    add_text(slide, x + Inches(0.10), y + h - Inches(0.34), w - Inches(0.20), Inches(0.28),
             label, size=9, italic=True, color=label_color, align=PP_ALIGN.CENTER)


def icon_circle(slide, cx, cy, r, fill, *, label=None, label_color=WHITE,
                size=18, line=None):
    """Centered circular badge with optional text label."""
    x = cx - r
    y = cy - r
    shp = slide.shapes.add_shape(MSO_SHAPE.OVAL, x, y, 2 * r, 2 * r)
    shp.fill.solid(); shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(2)
    shp.shadow.inherit = False
    if label is not None:
        tf = shp.text_frame
        tf.margin_left = Emu(0); tf.margin_right = Emu(0)
        tf.margin_top = Emu(0); tf.margin_bottom = Emu(0)
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r2 = p.add_run()
        r2.text = label
        r2.font.name = FONT_DISPLAY
        r2.font.size = Pt(size)
        r2.font.bold = True
        r2.font.color.rgb = label_color
    return shp


def icon_in_panel(slide, x, y, w, h, *, shape, icon_color, label, fill_panel=BLUE_TINT,
                  label_color=NEAR_BLACK, icon_pad=0.30, label_pt=14):
    """Soft panel with a centered MSO_SHAPE icon and a label below."""
    add_rect(slide, x, y, w, h, fill_panel)
    icon_size = min(w, h) - Inches(0.60 + icon_pad)
    icon_x = x + (w - icon_size) / 2
    icon_y = y + Inches(0.30)
    shp = slide.shapes.add_shape(shape, icon_x, icon_y, icon_size, icon_size)
    shp.fill.solid(); shp.fill.fore_color.rgb = icon_color
    shp.line.fill.background(); shp.shadow.inherit = False
    add_text(slide, x, y + h - Inches(0.55), w, Inches(0.40),
             label, size=label_pt, bold=True, color=label_color,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


def eye_icon(slide, cx, cy, w, color):
    """Eye composed of ellipse + filled circle."""
    h = w * 0.55
    add_rect(slide, cx - w / 2, cy - h / 2, w, h, None, shape=MSO_SHAPE.OVAL,
             line=color)
    pupil = slide.shapes.add_shape(MSO_SHAPE.OVAL,
                                   cx - h * 0.35, cy - h * 0.35,
                                   h * 0.70, h * 0.70)
    pupil.fill.solid(); pupil.fill.fore_color.rgb = color
    pupil.line.fill.background(); pupil.shadow.inherit = False


def document_icon(slide, x, y, w, h, *, fill=WHITE, line=BLUE, lines_color=GREY_LIGHT):
    """A page with corner folded + 4 horizontal text-lines inside."""
    shp = add_rect(slide, x, y, w, h, fill, shape=MSO_SHAPE.FOLDED_CORNER)
    shp.line.color.rgb = line
    shp.line.width = Pt(1.5)
    # text lines inside the page
    line_h = Inches(0.06)
    pad_x = Inches(0.18)
    pad_y = Inches(0.30)
    gap = Inches(0.16)
    for i, frac in enumerate([0.70, 0.85, 0.55, 0.78]):
        ly = y + pad_y + i * gap
        add_rect(slide, x + pad_x, ly, (w - 2 * pad_x) * frac, line_h, lines_color)
    return shp


def bar_chart(slide, x, y, w, h, values, labels, *, max_v=None,
              bar_color=BLUE, highlight_idx=None, highlight_color=RED,
              good_color=GREEN, good_threshold=None,
              show_values=True):
    """Simple bar chart drawn from shapes."""
    if max_v is None:
        max_v = max(values) * 1.10
    n = len(values)
    slot_w = w / n
    bar_w = slot_w * 0.55
    label_h = Inches(0.30)
    value_h = Inches(0.30)
    chart_h = h - label_h - value_h
    for i, (v, lab) in enumerate(zip(values, labels)):
        bh = chart_h * (v / max_v)
        bx = x + slot_w * i + (slot_w - bar_w) / 2
        by = y + value_h + (chart_h - bh)
        c = bar_color
        if highlight_idx is not None and i == highlight_idx:
            c = highlight_color
        elif good_threshold is not None and v <= good_threshold:
            c = good_color
        add_rect(slide, bx, by, bar_w, bh, c)
        if show_values:
            add_text(slide, x + slot_w * i, by - Inches(0.28),
                     slot_w, Inches(0.26),
                     f"{v:g}", size=9.5, bold=True, color=NEAR_BLACK,
                     align=PP_ALIGN.CENTER)
        add_text(slide, x + slot_w * i, y + value_h + chart_h + Inches(0.04),
                 slot_w, label_h,
                 str(lab), size=9.5, color=GREY, align=PP_ALIGN.CENTER)


def spider_chart(slide, cx, cy, r, axis_labels, *, n_rings=3, color=BLUE,
                 fill=BLUE_TINT, axis_color=GREY_LIGHT, label_color=NEAR_BLACK):
    """Radar chart skeleton with `len(axis_labels)` axes."""
    n = len(axis_labels)
    # rings
    for k in range(1, n_rings + 1):
        rr = r * k / n_rings
        ring = slide.shapes.add_shape(MSO_SHAPE.OVAL,
                                      cx - rr, cy - rr, 2 * rr, 2 * rr)
        ring.fill.background()
        ring.line.color.rgb = axis_color
        ring.line.width = Pt(0.75)
        ring.shadow.inherit = False
    # axes (lines + labels)
    for i, label in enumerate(axis_labels):
        angle = -math.pi / 2 + i * 2 * math.pi / n
        ax = int(cx + r * math.cos(angle))
        ay = int(cy + r * math.sin(angle))
        ln = slide.shapes.add_connector(1, int(cx), int(cy), ax, ay)
        ln.line.color.rgb = axis_color
        ln.line.width = Pt(0.75)
        # label outside axis
        lx_emu = int(cx + (r + Inches(0.45)) * math.cos(angle))
        ly_emu = int(cy + (r + Inches(0.45)) * math.sin(angle))
        box_w = Inches(2.6)
        box_h = Inches(0.35)
        add_text(slide,
                 lx_emu - box_w / 2, ly_emu - box_h / 2,
                 box_w, box_h,
                 label, size=11.5, bold=True, color=label_color,
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    # one filled exemplar polygon (showing "a balanced-good model")
    radii = [0.85, 0.65, 0.75, 0.55, 0.80][:n]
    pts = []
    for i, rr in enumerate(radii):
        angle = -math.pi / 2 + i * 2 * math.pi / n
        pts.append((int(cx + r * rr * math.cos(angle)),
                    int(cy + r * rr * math.sin(angle))))
    poly = _polygon(slide, pts, fill=fill, line=color, line_w=2.0)
    return poly


def _polygon(slide, pts, *, fill, line, line_w=1.5):
    """Closed polygon from a list of (x,y) Emu points."""
    from pptx.util import Emu as _E
    # bounding box
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = x_max - x_min; h = y_max - y_min
    spTree = slide.shapes._spTree
    nsm = "http://schemas.openxmlformats.org/drawingml/2006/main"
    nsm_a = f"{{{nsm}}}"
    sp = etree.SubElement(spTree, qn("p:sp"))
    nvSpPr = etree.SubElement(sp, qn("p:nvSpPr"))
    cNvPr = etree.SubElement(nvSpPr, qn("p:cNvPr"))
    cNvPr.set("id", "9999")
    cNvPr.set("name", "Polygon")
    etree.SubElement(nvSpPr, qn("p:cNvSpPr"))
    etree.SubElement(nvSpPr, qn("p:nvPr"))
    spPr = etree.SubElement(sp, qn("p:spPr"))
    xfrm = etree.SubElement(spPr, qn("a:xfrm"))
    off = etree.SubElement(xfrm, qn("a:off"))
    off.set("x", str(int(x_min))); off.set("y", str(int(y_min)))
    ext = etree.SubElement(xfrm, qn("a:ext"))
    ext.set("cx", str(int(w))); ext.set("cy", str(int(h)))
    custGeom = etree.SubElement(spPr, qn("a:custGeom"))
    etree.SubElement(custGeom, qn("a:avLst"))
    etree.SubElement(custGeom, qn("a:gdLst"))
    etree.SubElement(custGeom, qn("a:ahLst"))
    etree.SubElement(custGeom, qn("a:cxnLst"))
    rect = etree.SubElement(custGeom, qn("a:rect"))
    rect.set("l", "0"); rect.set("t", "0"); rect.set("r", "r"); rect.set("b", "b")
    pathLst = etree.SubElement(custGeom, qn("a:pathLst"))
    path = etree.SubElement(pathLst, qn("a:path"))
    path.set("w", str(int(w))); path.set("h", str(int(h)))
    for i, (px, py) in enumerate(pts):
        lx = int(px - x_min); ly = int(py - y_min)
        cmd = etree.SubElement(path, qn("a:moveTo") if i == 0 else qn("a:lnTo"))
        pt = etree.SubElement(cmd, qn("a:pt"))
        pt.set("x", str(lx)); pt.set("y", str(ly))
    etree.SubElement(path, qn("a:close"))
    # fill
    fillElem = etree.SubElement(spPr, qn("a:solidFill"))
    clr = etree.SubElement(fillElem, qn("a:srgbClr"))
    clr.set("val", f"{int(fill[0]):02X}{int(fill[1]):02X}{int(fill[2]):02X}")
    # line
    ln = etree.SubElement(spPr, qn("a:ln"))
    ln.set("w", str(int(line_w * 12700)))
    fillL = etree.SubElement(ln, qn("a:solidFill"))
    clrL = etree.SubElement(fillL, qn("a:srgbClr"))
    clrL.set("val", f"{int(line[0]):02X}{int(line[1]):02X}{int(line[2]):02X}")
    return sp


# ---------- slide builders ----------

def slide_blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def s_title(prs):
    s = slide_blank(prs)
    add_rect(s, Emu(0), Emu(0), SLIDE_W, SLIDE_H, BG)
    # left blue hero panel (with a placeholder for hero imagery)
    add_rect(s, Emu(0), Emu(0), Inches(6.65), SLIDE_H, BLUE)
    add_text(s, Inches(0.85), Inches(0.65), Inches(5.0), Inches(0.30),
             "DCC 2026 PARIS  ·  OPENING KEYNOTE",
             size=11, bold=True, color=BLUE_TXT_CHIP)
    # large icon: generative-design abstraction (concentric shapes)
    cx = Inches(3.30); cy = Inches(3.75)
    icon_circle(s, cx, cy, Inches(1.50), BLUE_DEEP)
    icon_circle(s, cx, cy, Inches(1.10), BLUE)
    icon_circle(s, cx, cy, Inches(0.55), BLUE_TXT_CHIP, label="AI",
                label_color=BLUE_DEEP, size=22)
    add_text(s, Inches(0.85), Inches(5.75), Inches(5.0), Inches(0.32),
             "[HERO: replace circle motif with a generative-designed part photo]",
             size=9, italic=True, color=BLUE_TXT_CHIP, align=PP_ALIGN.CENTER)
    # right text panel
    add_text(s, Inches(7.05), Inches(1.60), Inches(5.85), Inches(0.30),
             "WORKSHOP OPENING", size=10, bold=True, color=BLUE)
    add_text(s, Inches(7.05), Inches(2.05), Inches(6.0), Inches(2.6),
             ["Benchmarking",
              "Generative AI",
              "for Engineering",
              "Design."],
             font=FONT_DISPLAY, size=40, bold=True, color=NEAR_BLACK,
             line_spacing=1.0)
    add_rect(s, Inches(7.05), Inches(5.10), Inches(0.40), Inches(0.05), BLUE)
    add_text(s, Inches(7.05), Inches(5.25), Inches(6.0), Inches(0.50),
             "Why shared, executable contracts are the missing infrastructure.",
             size=15, color=GREY)
    add_text(s, Inches(7.05), Inches(6.30), Inches(6.0), Inches(0.30),
             "Matthew Keeler  ·  Soheyl Massoudi  ·  Mark Fuge",
             size=12, bold=True, color=NEAR_BLACK)
    add_text(s, Inches(7.05), Inches(6.62), Inches(6.0), Inches(0.25),
             "D-MAVT, ETH Zürich  ·  EngiBench + EngiOpt",
             size=10.5, color=GREY)


def s_promise(prs):
    s = slide_blank(prs)
    add_chrome(s, "Why this matters")
    add_eyebrow(s, "THE PROMISE")
    add_title(s, "AI is already designing parts we manufacture and fly.",
              size=26, height=0.7)
    # three large hero image placeholders
    y = Inches(2.10)
    h = Inches(4.20)
    pad = Inches(0.25)
    n = 3
    total_w = Inches(13.33 - 1.30)
    w = (total_w - pad * (n - 1)) / n
    x = Inches(0.65)
    items = [
        ("STRUCTURAL", "Airbus A320 cabin partition", "Generatively designed, 45% lighter, certified, flown."),
        ("PROPULSION", "GE jet-engine bracket", "84% mass reduction via topology optimization."),
        ("PHOTONICS", "Inverse-designed metasurface", "Optical devices beyond hand-crafted baselines."),
    ]
    for tag, title, sub in items:
        # tag chip
        add_rect(s, x, y, w, Inches(0.40), BLUE)
        add_text(s, x, y + Inches(0.08), w, Inches(0.28),
                 tag, size=10, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        # photo area
        photo_y = y + Inches(0.40)
        photo_h = Inches(2.90)
        image_placeholder(s, x, photo_y, w, photo_h, f"[PHOTO: {title}]")
        # title + caption
        add_text(s, x, photo_y + photo_h + Inches(0.15), w, Inches(0.32),
                 title, size=14, bold=True, color=NEAR_BLACK,
                 align=PP_ALIGN.CENTER)
        add_text(s, x, photo_y + photo_h + Inches(0.50), w, Inches(0.30),
                 sub, size=10.5, color=GREY, align=PP_ALIGN.CENTER)
        x += w + pad


def s_renaissance(prs):
    s = slide_blank(prs)
    add_chrome(s, "Why this matters")
    add_eyebrow(s, "A RENAISSANCE OF METHODS")
    add_title(s, "Every year, a new generative recipe.", size=28, height=0.7)
    add_text(s, Inches(0.65), Inches(1.70), Inches(12.0), Inches(0.40),
             "And every paper claims a new state of the art.",
             size=15, italic=True, color=GREY)

    # timeline strip
    y_axis = Inches(4.40)
    add_rect(s, Inches(0.95), y_axis, Inches(11.45), Inches(0.04), GREY_LIGHT)

    eras = [
        ("2017", "GAN era", "cGANs for layouts and shapes.", BLUE),
        ("2019", "VAE era", "Smooth latent design manifolds.", GREEN),
        ("2022", "Diffusion", "Score-based generative design.", ORANGE),
        ("2024", "LLMs + agents", "Code-writing design pipelines.", RED),
    ]
    n = len(eras)
    span = Inches(10.50)
    start_x = Inches(1.40)
    for i, (yr, name, blurb, color) in enumerate(eras):
        cx = start_x + span * i / (n - 1)
        # large circular node
        icon_circle(s, cx, y_axis + Inches(0.02), Inches(0.55), color,
                    label=yr, size=15, label_color=WHITE)
        # name above
        add_text(s, cx - Inches(1.50), Inches(2.60), Inches(3.00), Inches(0.40),
                 name, size=16, bold=True, color=color, align=PP_ALIGN.CENTER)
        # one-line blurb below
        add_text(s, cx - Inches(1.50), Inches(5.40), Inches(3.00), Inches(0.40),
                 blurb, size=11.5, color=GREY, align=PP_ALIGN.CENTER)

    # kicker
    add_rect(s, Inches(0.65), Inches(6.20), Inches(12.05), Inches(0.55), BLUE)
    add_text(s, Inches(0.65), Inches(6.32), Inches(12.05), Inches(0.32),
             "The methods change quickly. The way we evaluate them barely changes at all.",
             size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER)


def s_honest_question(prs):
    s = slide_blank(prs)
    add_chrome(s, "Why this matters")
    add_eyebrow(s, "AN HONEST QUESTION")
    # giant question mark in a soft circle
    cx = Inches(6.67); cy = Inches(4.10)
    icon_circle(s, cx, cy, Inches(1.85), BLUE_TINT)
    add_text(s, cx - Inches(1.5), cy - Inches(1.6), Inches(3.0), Inches(3.2),
             "?", font=FONT_DISPLAY, size=180, bold=True, color=BLUE,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    add_title(s, "When a paper claims SOTA on topology optimization,\nwhat actually changed?",
              size=28, height=1.4, top=0.95, width=13.0)
    add_text(s, Inches(0.65), Inches(6.30), Inches(12.05), Inches(0.40),
             "Better model? Easier conditions? Lower bar?  —  today's literature does not let you tell.",
             size=14, italic=True, color=GREY, align=PP_ALIGN.CENTER)


def s_repro_crisis(prs):
    s = slide_blank(prs)
    add_chrome(s, "Why we cannot tell")
    add_eyebrow(s, "ML'S REPRODUCIBILITY CRISIS")
    add_title(s, "We are not the first field to face this.",
              size=26, height=0.7)
    # giant stat on the left
    add_text(s, Inches(0.65), Inches(2.10), Inches(6.5), Inches(3.5),
             "≈30%",
             font=FONT_DISPLAY, size=220, bold=True, color=RED,
             align=PP_ALIGN.LEFT, line_spacing=1.0)
    add_text(s, Inches(0.65), Inches(5.40), Inches(6.5), Inches(0.40),
             "of ML papers failed to reproduce from text alone.",
             size=14, bold=True, color=NEAR_BLACK)
    add_text(s, Inches(0.65), Inches(5.80), Inches(6.5), Inches(0.30),
             "Pineau et al., 2019 — independent replication study.",
             size=10.5, italic=True, color=GREY)
    # right column: response
    rx = Inches(7.80); rw = Inches(5.0)
    add_text(s, rx, Inches(2.10), rw, Inches(0.32),
             "MAINSTREAM ML'S RESPONSE", size=11, bold=True, color=BLUE)
    add_text(s, rx, Inches(2.55), rw, Inches(0.60),
             "Reproducibility checklists.",
             font=FONT_DISPLAY, size=22, bold=True, color=NEAR_BLACK)
    add_text(s, rx, Inches(3.25), rw, Inches(0.60),
             "Open code, weights, harnesses.",
             font=FONT_DISPLAY, size=22, bold=True, color=NEAR_BLACK)
    add_text(s, rx, Inches(3.95), rw, Inches(0.60),
             "Shared benchmarks by default.",
             font=FONT_DISPLAY, size=22, bold=True, color=NEAR_BLACK)
    add_text(s, rx, Inches(5.05), rw, Inches(0.40),
             "Engineering design ML is starting now from where ML stood in 2017.",
             size=13, italic=True, color=GREY)


def s_design_harder(prs):
    s = slide_blank(prs)
    add_chrome(s, "Why we cannot tell")
    add_eyebrow(s, "ENGINEERING DESIGN IS HARDER")
    add_title(s, "Each paper hides many silent choices.", size=26, height=0.7)
    add_text(s, Inches(0.65), Inches(1.55), Inches(12.0), Inches(0.40),
             "Vision papers hide a handful. Design papers hide all of these — and any one can flip a conclusion.",
             size=13, italic=True, color=GREY)

    items = [
        (MSO_SHAPE.GEAR_6, BLUE, "Simulator"),
        (MSO_SHAPE.NO_SYMBOL, RED, "Constraints"),
        (MSO_SHAPE.SUN, ORANGE, "Conditions"),
        (MSO_SHAPE.STAR_5_POINT, GREEN, "Baselines"),
        (MSO_SHAPE.CUBE, BLUE, "Representation"),
        (MSO_SHAPE.CHEVRON, ORANGE, "Units & scaling"),
    ]
    y = Inches(2.45)
    h = Inches(2.20)
    pad_x = Inches(0.20); pad_y = Inches(0.20)
    cols = 3
    total_w = Inches(13.33 - 1.30)
    w = (total_w - pad_x * (cols - 1)) / cols
    for i, (shape, color, label) in enumerate(items):
        col = i % cols; row = i // cols
        x = Inches(0.65) + col * (w + pad_x)
        ypos = y + row * (h + pad_y)
        icon_in_panel(s, x, ypos, w, h, shape=shape,
                      icon_color=color, label=label)


def s_concrete_case(prs):
    s = slide_blank(prs)
    add_chrome(s, "Why we cannot tell")
    add_eyebrow(s, "A CONCRETE CASE")
    add_title(s, "Same task name. Same headline number.", size=26, height=0.7)
    add_text(s, Inches(0.65), Inches(1.55), Inches(12.0), Inches(0.40),
             "Two recent \"Beams2D\" papers report MSE = 0.04. Are they comparable?",
             size=13, italic=True, color=GREY)

    # two document icons + key facts + ≠
    doc_w = Inches(4.80); doc_h = Inches(4.20)
    y = Inches(2.20)
    x_a = Inches(0.85)
    x_b = SLIDE_W - Inches(0.85) - doc_w

    document_icon(s, x_a, y, doc_w, doc_h, line=BLUE)
    add_text(s, x_a, y + Inches(0.32), doc_w, Inches(0.30),
             "PAPER A", size=11, bold=True, color=BLUE, align=PP_ALIGN.CENTER)
    rows_a = [
        ("volume frac", "0.50"),
        ("filter radius", "1.5 px"),
        ("feasibility", "soft penalty"),
        ("solver tol.", "1e-3"),
        ("test conds.", "near training"),
    ]
    ry = y + Inches(1.20)
    for k, v in rows_a:
        add_text(s, x_a + Inches(0.55), ry, Inches(2.10), Inches(0.28),
                 k, size=11, bold=True, color=NEAR_BLACK)
        add_text(s, x_a + Inches(2.65), ry, Inches(1.80), Inches(0.28),
                 v, size=11, color=GREY)
        ry += Inches(0.42)

    document_icon(s, x_b, y, doc_w, doc_h, line=ORANGE)
    add_text(s, x_b, y + Inches(0.32), doc_w, Inches(0.30),
             "PAPER B", size=11, bold=True, color=ORANGE, align=PP_ALIGN.CENTER)
    rows_b = [
        ("volume frac", "0.30 – 0.55"),
        ("filter radius", "2.5 px"),
        ("feasibility", "hard reject"),
        ("solver tol.", "1e-5"),
        ("test conds.", "OOD corners"),
    ]
    ry = y + Inches(1.20)
    for k, v in rows_b:
        add_text(s, x_b + Inches(0.55), ry, Inches(2.10), Inches(0.28),
                 k, size=11, bold=True, color=NEAR_BLACK)
        add_text(s, x_b + Inches(2.65), ry, Inches(1.80), Inches(0.28),
                 v, size=11, color=GREY)
        ry += Inches(0.42)

    # giant ≠ in the middle
    add_text(s, Inches(5.80), Inches(3.70), Inches(1.80), Inches(1.80),
             "≠", font=FONT_DISPLAY, size=100, bold=True, color=RED,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

    # kicker
    add_rect(s, Inches(0.65), Inches(6.60), Inches(12.05), Inches(0.40), BLUE)
    add_text(s, Inches(0.65), Inches(6.68), Inches(12.05), Inches(0.26),
             "Different problem, same number.",
             size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)


def s_failure_modes(prs):
    s = slide_blank(prs)
    add_chrome(s, "Why we cannot tell")
    add_eyebrow(s, "THREE FAILURE MODES")
    add_title(s, "Without a shared contract, three patterns recur.",
              size=26, height=0.7)

    y = Inches(2.30)
    h = Inches(4.20)
    pad = Inches(0.25)
    total_w = Inches(13.33 - 1.30)
    w = (total_w - pad * 2) / 3
    x = Inches(0.65)

    # Panel 1: cherry-picked conditions — bull's-eye with marker far from center
    add_rect(s, x, y, w, h, BG)
    cx = x + w / 2; cy = y + Inches(1.70)
    for k, (radius_in, color) in enumerate([(1.05, RED), (0.75, RED_TINT), (0.45, WHITE), (0.18, RED)]):
        r_emu = Inches(radius_in)
        icon_circle(s, cx, cy, r_emu, color, line=RED)
    # marker (X) far off-center
    mx = cx + Inches(0.85); my = cy - Inches(0.40)
    add_text(s, mx - Inches(0.30), my - Inches(0.30), Inches(0.60), Inches(0.60),
             "✗", font=FONT_DISPLAY, size=32, bold=True, color=NEAR_BLACK,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    add_text(s, x, y + Inches(3.40), w, Inches(0.36),
             "Cherry-picked conditions", size=15, bold=True, color=RED,
             align=PP_ALIGN.CENTER)
    add_text(s, x + Inches(0.20), y + Inches(3.75), w - Inches(0.40), Inches(0.40),
             "Easy near the training mean. Collapses elsewhere.",
             size=11, color=GREY, align=PP_ALIGN.CENTER)

    # Panel 2: visual-only — eye icon
    x2 = x + w + pad
    add_rect(s, x2, y, w, h, BG)
    cx2 = x2 + w / 2; cy2 = y + Inches(1.70)
    eye_icon(s, cx2, cy2, Inches(2.0), ORANGE)
    add_text(s, x2, y + Inches(3.40), w, Inches(0.36),
             "Visual-only evaluation", size=15, bold=True, color=ORANGE,
             align=PP_ALIGN.CENTER)
    add_text(s, x2 + Inches(0.20), y + Inches(3.75), w - Inches(0.40), Inches(0.40),
             "Looks plausible. Fails the simulator.",
             size=11, color=GREY, align=PP_ALIGN.CENTER)

    # Panel 3: one-number — single tall bar
    x3 = x2 + w + pad
    add_rect(s, x3, y, w, h, BG)
    bar_x = x3 + w / 2 - Inches(0.30); bar_y = y + Inches(0.40)
    add_rect(s, bar_x, bar_y, Inches(0.60), Inches(2.60), BLUE)
    # tiny dimmed bars beside it
    add_rect(s, bar_x - Inches(0.80), bar_y + Inches(1.90), Inches(0.30), Inches(0.70), GREY_LIGHT)
    add_rect(s, bar_x + Inches(0.85), bar_y + Inches(1.50), Inches(0.30), Inches(1.10), GREY_LIGHT)
    add_text(s, x3, y + Inches(3.40), w, Inches(0.36),
             "One-number scores", size=15, bold=True, color=BLUE,
             align=PP_ALIGN.CENTER)
    add_text(s, x3 + Inches(0.20), y + Inches(3.75), w - Inches(0.40), Inches(0.40),
             "Hides feasibility, diversity, and warm-start.",
             size=11, color=GREY, align=PP_ALIGN.CENTER)


def s_divider(prs, *, eyebrow, title, subtitle, icon_shape=None, icon_color=BLUE_TXT_CHIP):
    s = slide_blank(prs)
    add_rect(s, Emu(0), Emu(0), SLIDE_W, SLIDE_H, BLUE)
    # left circular icon
    if icon_shape is not None:
        icx = Inches(2.80); icy = Inches(3.75)
        icon_circle(s, icx, icy, Inches(1.45), BLUE_DEEP)
        shp = s.shapes.add_shape(icon_shape,
                                  icx - Inches(0.85), icy - Inches(0.85),
                                  Inches(1.70), Inches(1.70))
        shp.fill.solid(); shp.fill.fore_color.rgb = icon_color
        shp.line.fill.background(); shp.shadow.inherit = False
    add_text(s, Inches(5.20), Inches(2.65), Inches(7.50), Inches(0.36),
             eyebrow, size=12, bold=True, color=BLUE_TXT_CHIP)
    add_text(s, Inches(5.20), Inches(3.10), Inches(7.50), Inches(1.80),
             title, font=FONT_DISPLAY, size=38, bold=True, color=WHITE,
             line_spacing=1.0)
    add_text(s, Inches(5.20), Inches(5.20), Inches(7.50), Inches(0.55),
             subtitle, size=15, color=BLUE_TXT_LIGHT)


def s_imagenet(prs):
    s = slide_blank(prs)
    add_chrome(s, "Lessons from other fields")
    add_eyebrow(s, "LESSON 1  —  IMAGENET")
    add_title(s, "One frozen evaluation made breakthroughs unambiguous.",
              size=24, height=0.7)
    # chart panel
    cx = Inches(0.65); cy = Inches(1.80)
    cw = Inches(12.05); ch = Inches(4.40)
    add_rect(s, cx, cy, cw, ch, WHITE)
    add_text(s, cx + Inches(0.30), cy + Inches(0.20), Inches(8.0), Inches(0.30),
             "ImageNet top-5 error", size=13, bold=True, color=NEAR_BLACK)
    years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
    vals = [28.2, 25.8, 16.4, 11.7, 6.7, 3.6, 3.0, 2.3]
    bar_chart(s, cx + Inches(0.50), cy + Inches(0.65),
              cw - Inches(1.0), ch - Inches(0.85),
              vals, [str(y) for y in years],
              max_v=30, highlight_idx=2, highlight_color=RED,
              good_threshold=5, good_color=GREEN, bar_color=BLUE)
    # annotation
    add_text(s, cx + Inches(2.95), cy + Inches(1.05), Inches(2.5), Inches(0.30),
             "AlexNet, 2012", size=11, bold=True, color=RED)
    # kicker
    add_rect(s, Inches(0.65), Inches(6.40), Inches(12.05), Inches(0.55), BLUE)
    add_text(s, Inches(0.65), Inches(6.52), Inches(12.05), Inches(0.32),
             "A shared task + a shared evaluation = a decade of compounding progress.",
             size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER)


def s_glue(prs):
    s = slide_blank(prs)
    add_chrome(s, "Lessons from other fields")
    add_eyebrow(s, "LESSON 2  —  GLUE / SUPERGLUE")
    add_title(s, "When one task saturates, a suite of tasks sustains the conversation.",
              size=22, height=1.0)

    # 3x3 grid of mini "task tiles" on the left
    grid_x = Inches(0.65); grid_y = Inches(2.45)
    cell = Inches(1.10); pad = Inches(0.15)
    task_names = ["CoLA", "SST-2", "MRPC", "STS-B", "QQP", "MNLI", "QNLI", "RTE", "WNLI"]
    for i, name in enumerate(task_names):
        col = i % 3; row = i // 3
        gx = grid_x + col * (cell + pad)
        gy = grid_y + row * (cell + pad)
        add_rect(s, gx, gy, cell, cell, BLUE_TINT)
        add_text(s, gx, gy, cell, cell, name, size=12, bold=True,
                 color=BLUE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    # arrow → upgraded suite
    arrow = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW,
                               Inches(4.45), Inches(3.65), Inches(1.30), Inches(0.70))
    arrow.fill.solid(); arrow.fill.fore_color.rgb = BLUE
    arrow.line.fill.background(); arrow.shadow.inherit = False
    add_text(s, Inches(4.45), Inches(3.30), Inches(1.30), Inches(0.30),
             "harder", size=10, bold=True, color=BLUE, align=PP_ALIGN.CENTER)
    # SuperGLUE block (compact harder grid)
    sg_x = Inches(5.90); sg_y = grid_y
    sg_w = Inches(3.50); sg_h = Inches(3.50)
    add_rect(s, sg_x, sg_y, sg_w, sg_h, BLUE)
    add_text(s, sg_x, sg_y + Inches(0.30), sg_w, Inches(0.45),
             "SuperGLUE", size=22, bold=True, color=WHITE,
             font=FONT_DISPLAY, align=PP_ALIGN.CENTER)
    add_text(s, sg_x + Inches(0.30), sg_y + Inches(1.05), sg_w - Inches(0.60), Inches(1.80),
             "Eight harder language understanding tasks. Designed when GLUE saturated in a year.",
             size=11, color=BLUE_TXT_LIGHT, align=PP_ALIGN.CENTER)

    # right column lesson
    rx = Inches(9.80); rw = Inches(3.0)
    add_text(s, rx, grid_y, rw, Inches(0.32),
             "THE LESSON", size=11, bold=True, color=ORANGE)
    add_text(s, rx, grid_y + Inches(0.40), rw, Inches(1.60),
             "Benchmarks have a lifecycle.",
             font=FONT_DISPLAY, size=22, bold=True, color=NEAR_BLACK,
             line_spacing=1.05)
    add_text(s, rx, grid_y + Inches(2.30), rw, Inches(1.40),
             "When models clear the bar, the community raises it. Design needs a suite, not one beam problem.",
             size=12, color=GREY)


def s_casp(prs):
    s = slide_blank(prs)
    add_chrome(s, "Lessons from other fields")
    add_eyebrow(s, "LESSON 3  —  CASP / ALPHAFOLD")
    add_title(s, "A 25-year community benchmark turned a breakthrough into a fact.",
              size=22, height=1.0)
    # left: protein illustration placeholder
    image_placeholder(s, Inches(0.65), Inches(2.40), Inches(5.80), Inches(4.20),
                       "[IMAGE: AlphaFold predicted structure / CASP scoring]")
    # right: timeline strip + statement
    rx = Inches(7.00); rw = Inches(5.85); ry = Inches(2.40)
    add_text(s, rx, ry, rw, Inches(0.32),
             "CASP", size=12, bold=True, color=BLUE)
    add_text(s, rx, ry + Inches(0.40), rw, Inches(2.0),
             "Held-out targets.\nBlind submissions.\nOne shared score.",
             font=FONT_DISPLAY, size=24, bold=True, color=NEAR_BLACK,
             line_spacing=1.10)
    # timeline ruler
    rule_y = ry + Inches(3.10)
    add_rect(s, rx, rule_y, rw, Inches(0.04), GREY_LIGHT)
    for i, yr in enumerate(["1994", "2000", "2010", "2020", "CASP14"]):
        cx = rx + rw * i / 4
        icon_circle(s, cx, rule_y + Inches(0.02), Inches(0.12),
                    BLUE if yr != "CASP14" else GREEN)
        add_text(s, cx - Inches(0.50), rule_y + Inches(0.25), Inches(1.0), Inches(0.25),
                 yr, size=9.5, bold=True, color=GREY, align=PP_ALIGN.CENTER)
    add_text(s, rx, ry + Inches(3.95), rw, Inches(0.65),
             "Without CASP, AlphaFold is a press release. With it, a verdict.",
             size=14, italic=True, color=NEAR_BLACK)


def s_shared_dna(prs):
    s = slide_blank(prs)
    add_chrome(s, "Lessons from other fields")
    add_eyebrow(s, "WHAT GOOD BENCHMARKS SHARE")
    add_title(s, "Four ingredients turn a dataset into a benchmark.",
              size=26, height=0.7)

    items = [
        ("Fixed inputs", MSO_SHAPE.HEXAGON, BLUE,
         "Same data, same splits."),
        ("Standard eval", MSO_SHAPE.GEAR_6, GREEN,
         "Scoring is code, not prose."),
        ("Open access", MSO_SHAPE.RIGHT_ARROW, ORANGE,
         "Low cost of entry."),
        ("Community", MSO_SHAPE.STAR_5_POINT, RED,
         "Lives across years."),
    ]
    y = Inches(2.10); h = Inches(4.30)
    pad = Inches(0.25); total = Inches(13.33 - 1.30)
    w = (total - pad * 3) / 4
    x = Inches(0.65)
    for label, shape, color, blurb in items:
        # panel
        add_rect(s, x, y, w, h, WHITE)
        add_rect(s, x, y, w, Inches(0.08), color)
        # icon
        icon_size = Inches(1.50)
        ix = x + (w - icon_size) / 2
        iy = y + Inches(0.70)
        shp = s.shapes.add_shape(shape, ix, iy, icon_size, icon_size)
        shp.fill.solid(); shp.fill.fore_color.rgb = color
        shp.line.fill.background(); shp.shadow.inherit = False
        # label
        add_text(s, x, y + Inches(2.55), w, Inches(0.40),
                 label, size=16, bold=True, color=NEAR_BLACK,
                 align=PP_ALIGN.CENTER, font=FONT_DISPLAY)
        # blurb
        add_text(s, x + Inches(0.30), y + Inches(3.10), w - Inches(0.60), Inches(1.0),
                 blurb, size=11.5, color=GREY, align=PP_ALIGN.CENTER)
        x += w + pad


def s_five_missing(prs):
    s = slide_blank(prs)
    add_chrome(s, "What the field needs")
    add_eyebrow(s, "FIVE MISSING PIECES")
    add_title(s, "Engineering design ML lacks shared infrastructure.",
              size=26, height=0.7)
    add_text(s, Inches(0.65), Inches(1.55), Inches(12.0), Inches(0.40),
             "Each can be rebuilt — but no one should have to. That is the gap.",
             size=13, italic=True, color=GREY)

    items = [
        (MSO_SHAPE.HEXAGON, BLUE, "Standard\nproblems"),
        (MSO_SHAPE.GEAR_6, GREEN, "Shared\nsimulators"),
        (MSO_SHAPE.FOLDED_CORNER, ORANGE, "Curated\ndatasets"),
        (MSO_SHAPE.STAR_5_POINT, RED, "Multi-faceted\nmetrics"),
        (MSO_SHAPE.RIGHT_ARROW, BLUE, "Reproducible\nrunners"),
    ]
    n = len(items)
    y = Inches(2.65); h = Inches(3.80)
    pad = Inches(0.18); total = Inches(13.33 - 1.30)
    w = (total - pad * (n - 1)) / n
    x = Inches(0.65)
    for shape, color, label in items:
        # hexagon backdrop tinted
        add_rect(s, x, y, w, h, BG)
        # icon centered top
        ic = Inches(1.50)
        ix = x + (w - ic) / 2; iy = y + Inches(0.45)
        shp = s.shapes.add_shape(shape, ix, iy, ic, ic)
        shp.fill.solid(); shp.fill.fore_color.rgb = color
        shp.line.fill.background(); shp.shadow.inherit = False
        # underline accent
        add_rect(s, x + (w - Inches(0.50)) / 2, y + Inches(2.30),
                 Inches(0.50), Inches(0.04), color)
        # multi-line label
        for li, line in enumerate(label.split("\n")):
            add_text(s, x, y + Inches(2.55 + li * 0.40), w, Inches(0.40),
                     line, size=14, bold=True, color=NEAR_BLACK,
                     align=PP_ALIGN.CENTER)
        x += w + pad


def s_multifaceted(prs):
    s = slide_blank(prs)
    add_chrome(s, "What the field needs")
    add_eyebrow(s, "EVALUATION, PROPERLY SCOPED")
    add_title(s, "Engineering quality has many axes. One scalar will betray you.",
              size=22, height=1.0)

    # spider chart on the left
    spider_chart(s, Inches(4.20), Inches(4.55), Inches(2.15),
                 axis_labels=["Feasibility", "Performance", "Diversity", "Novelty", "Warm-start"],
                 color=BLUE, fill=BLUE_LIGHT)

    # right column legend
    rx = Inches(8.00); rw = Inches(4.85)
    axes_legend = [
        ("Feasibility", "Constraints actually satisfied.", BLUE),
        ("Performance", "Sim score vs. optimizer baseline.", GREEN),
        ("Diversity", "Modes covered, not collapsed.", ORANGE),
        ("Novelty", "New designs, not paraphrases.", RED),
        ("Warm-start", "Does it help downstream search?", BLUE),
    ]
    ry = Inches(2.40)
    for name, body, color in axes_legend:
        add_rect(s, rx, ry, Inches(0.14), Inches(0.65), color)
        add_text(s, rx + Inches(0.30), ry + Inches(0.04), rw - Inches(0.4), Inches(0.30),
                 name, size=13.5, bold=True, color=NEAR_BLACK)
        add_text(s, rx + Inches(0.30), ry + Inches(0.36), rw - Inches(0.4), Inches(0.30),
                 body, size=10.5, color=GREY)
        ry += Inches(0.85)


def s_dataset_not_enough(prs):
    s = slide_blank(prs)
    add_chrome(s, "What the field needs")
    add_eyebrow(s, "DATASETS ALONE DO NOT BENCHMARK")
    add_title(s, "Releasing the dataset is necessary, but not enough.",
              size=26, height=0.7)

    y = Inches(2.40); h = Inches(3.80)
    panel_w = Inches(5.70)
    x_a = Inches(0.85)
    x_b = SLIDE_W - Inches(0.85) - panel_w

    # Panel A: dataset only
    add_rect(s, x_a, y, panel_w, h, BG)
    add_text(s, x_a, y + Inches(0.10), panel_w, Inches(0.30),
             "DATASET ONLY", size=11, bold=True, color=RED, align=PP_ALIGN.CENTER)
    # one document icon, big
    doc_x = x_a + (panel_w - Inches(1.6)) / 2
    document_icon(s, doc_x, y + Inches(0.65), Inches(1.6), Inches(2.0), line=RED)
    # forbidden symbol over it
    shp = s.shapes.add_shape(MSO_SHAPE.NO_SYMBOL,
                             doc_x + Inches(0.30), y + Inches(0.95),
                             Inches(1.0), Inches(1.40))
    shp.fill.solid(); shp.fill.fore_color.rgb = RED
    shp.line.fill.background(); shp.shadow.inherit = False
    # caption
    add_text(s, x_a + Inches(0.50), y + Inches(2.90), panel_w - Inches(1.0), Inches(0.80),
             "Designs can be displayed.\nThey cannot be scored.",
             size=14, bold=True, color=NEAR_BLACK, align=PP_ALIGN.CENTER)

    # giant arrow / vs
    add_text(s, Inches(6.20), Inches(3.95), Inches(1.0), Inches(0.65),
             "vs", font=FONT_DISPLAY, size=36, bold=True, color=GREY,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

    # Panel B: dataset + simulator + scoring
    add_rect(s, x_b, y, panel_w, h, BG)
    add_text(s, x_b, y + Inches(0.10), panel_w, Inches(0.30),
             "DATASET  +  SIMULATOR  +  SCORING", size=11, bold=True, color=GREEN,
             align=PP_ALIGN.CENTER)
    # three icons in a row inside the panel
    pad = Inches(0.30)
    ic_size = Inches(1.10)
    total_ic = ic_size * 3 + pad * 2
    start_x = x_b + (panel_w - total_ic) / 2
    iy = y + Inches(0.75)
    document_icon(s, start_x, iy, ic_size, Inches(1.40), line=GREEN)
    gear = s.shapes.add_shape(MSO_SHAPE.GEAR_6,
                              start_x + ic_size + pad, iy + Inches(0.10),
                              ic_size, ic_size)
    gear.fill.solid(); gear.fill.fore_color.rgb = GREEN
    gear.line.fill.background(); gear.shadow.inherit = False
    star = s.shapes.add_shape(MSO_SHAPE.STAR_5_POINT,
                              start_x + (ic_size + pad) * 2, iy + Inches(0.10),
                              ic_size, ic_size)
    star.fill.solid(); star.fill.fore_color.rgb = GREEN
    star.line.fill.background(); star.shadow.inherit = False
    # tiny labels under each
    labels = ["data", "simulate", "score"]
    for i, lab in enumerate(labels):
        lx = start_x + i * (ic_size + pad)
        add_text(s, lx, iy + Inches(1.50), ic_size, Inches(0.30),
                 lab, size=10, color=GREY, align=PP_ALIGN.CENTER)
    # caption
    add_text(s, x_b + Inches(0.50), y + Inches(2.90), panel_w - Inches(1.0), Inches(0.80),
             "The verdict is portable.\nNumbers travel across labs.",
             size=14, bold=True, color=NEAR_BLACK, align=PP_ALIGN.CENTER)


def s_engibench_divider(prs):
    s_divider(prs,
              eyebrow="WHY WE BUILT IT",
              title="Enter EngiBench.",
              subtitle="A standardized framework for benchmarking generative AI in engineering design  —  NeurIPS 2025.",
              icon_shape=MSO_SHAPE.CAN, icon_color=BLUE_TXT_CHIP)


def s_paper_in_a_can(prs):
    s = slide_blank(prs)
    add_chrome(s, "EngiBench")
    add_eyebrow(s, "THE DESIGN PATTERN")
    add_title(s, "Each problem is a \"paper in a can\".", size=28, height=0.7)
    add_text(s, Inches(0.65), Inches(1.55), Inches(12.0), Inches(0.40),
             "Eight ingredients, one Python interface, ready to compare across labs.",
             size=13, italic=True, color=GREY)

    # central CAN shape
    cx = Inches(13.33 / 2); cy = Inches(4.40)
    can_w = Inches(2.60); can_h = Inches(3.0)
    can = s.shapes.add_shape(MSO_SHAPE.CAN,
                             cx - can_w / 2, cy - can_h / 2, can_w, can_h)
    can.fill.solid(); can.fill.fore_color.rgb = BLUE
    can.line.fill.background(); can.shadow.inherit = False
    add_text(s, cx - Inches(1.30), cy - Inches(0.35), Inches(2.60), Inches(0.40),
             "engibench", font=FONT_DISPLAY, size=20, bold=True, color=WHITE,
             align=PP_ALIGN.CENTER)
    add_text(s, cx - Inches(1.30), cy + Inches(0.05), Inches(2.60), Inches(0.30),
             "problem", size=12, color=BLUE_TXT_CHIP, align=PP_ALIGN.CENTER)

    # 8 satellite chips around the can
    parts = [
        ("Design space", BLUE),
        ("Conditions", GREEN),
        ("Objectives", ORANGE),
        ("Constraints", RED),
        ("Dataset", BLUE),
        ("Render", GREEN),
        ("Simulate", ORANGE),
        ("Optimize", RED),
    ]
    radius_x = Inches(5.20); radius_y = Inches(2.40)
    for i, (label, color) in enumerate(parts):
        angle = -math.pi / 2 + i * 2 * math.pi / len(parts)
        ex = cx + radius_x * math.cos(angle)
        ey = cy + radius_y * math.sin(angle)
        chip_w = Inches(1.85); chip_h = Inches(0.50)
        cx_c = ex - chip_w / 2; cy_c = ey - chip_h / 2
        add_shape_with_text(s, MSO_SHAPE.ROUNDED_RECTANGLE,
                            cx_c, cy_c, chip_w, chip_h,
                            color, label, color=WHITE, size=12, bold=True)


def s_coverage(prs):
    s = slide_blank(prs)
    add_chrome(s, "EngiBench")
    add_eyebrow(s, "COVERAGE TODAY")
    add_title(s, "Structural, thermal, aerodynamic, photonic, electronic.",
              size=24, height=0.7)
    img = Path(__file__).resolve().parent.parent / "assets" / "engibench_problems.png"
    if img.exists():
        s.shapes.add_picture(str(img), Inches(0.80), Inches(1.75),
                             width=Inches(11.73), height=Inches(4.70))
    else:
        image_placeholder(s, Inches(0.80), Inches(1.75), Inches(11.73), Inches(4.70),
                          "[IMAGE: engibench problems grid]")
    add_text(s, Inches(0.65), Inches(6.65), Inches(12.05), Inches(0.32),
             "Each comes with a dataset, a simulator, and a baseline optimizer.",
             size=12.5, italic=True, color=GREY, align=PP_ALIGN.CENTER)


def s_contract_code(prs):
    s = slide_blank(prs)
    add_chrome(s, "EngiBench")
    add_eyebrow(s, "THE CONTRACT, IN CODE")
    add_title(s, "Design-problem questions become Python calls.",
              size=26, height=0.7)

    # code block (dark)
    code_x = Inches(0.65); code_y = Inches(2.05)
    code_w = Inches(7.85); code_h = Inches(4.50)
    add_rect(s, code_x, code_y, code_w, code_h, NEAR_BLACK)
    # window dots
    for i, c in enumerate([RED, ORANGE, GREEN]):
        dot = s.shapes.add_shape(MSO_SHAPE.OVAL,
                                 code_x + Inches(0.25 + i * 0.30),
                                 code_y + Inches(0.20),
                                 Inches(0.18), Inches(0.18))
        dot.fill.solid(); dot.fill.fore_color.rgb = c
        dot.line.fill.background(); dot.shadow.inherit = False
    code_lines = [
        "from engibench.utils.all_problems import BUILTIN_PROBLEMS",
        "",
        "problem  = BUILTIN_PROBLEMS['beams2d'](seed=7)",
        "",
        "design   = my_generator.sample(problem.conditions_keys)",
        "ok       = problem.check_constraints(design, conds)",
        "score    = problem.simulate(design, conds)",
        "baseline = problem.optimize(starting=design, config=conds)",
        "problem.render(design)",
    ]
    tb = s.shapes.add_textbox(code_x + Inches(0.30), code_y + Inches(0.65),
                              code_w - Inches(0.5), code_h - Inches(0.85))
    tf = tb.text_frame
    tf.word_wrap = False
    tf.margin_left = Emu(0); tf.margin_right = Emu(0)
    tf.margin_top = Emu(0); tf.margin_bottom = Emu(0)
    for i, line in enumerate(code_lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run()
        r.text = line if line else " "
        r.font.name = "Consolas"
        r.font.size = Pt(14.5)
        r.font.color.rgb = WHITE

    # right column: 3 visual callouts
    rx = Inches(8.85); rw = Inches(4.0); ry = Inches(2.10)
    callouts = [
        (MSO_SHAPE.GEAR_6, BLUE,  "Same task contract",
         "Every paper compares on the same problem."),
        (MSO_SHAPE.HEXAGON, GREEN, "Swap the model, not the task",
         "Your generator is one cell. The rest is fixed."),
        (MSO_SHAPE.STAR_5_POINT, ORANGE, "Portable evidence",
         "Your numbers match a reviewer's numbers."),
    ]
    for shape, color, title, body in callouts:
        add_rect(s, rx, ry, rw, Inches(1.30), WHITE)
        # icon left
        shp = s.shapes.add_shape(shape, rx + Inches(0.18), ry + Inches(0.30),
                                 Inches(0.65), Inches(0.65))
        shp.fill.solid(); shp.fill.fore_color.rgb = color
        shp.line.fill.background(); shp.shadow.inherit = False
        add_text(s, rx + Inches(1.00), ry + Inches(0.22), rw - Inches(1.15), Inches(0.38),
                 title, size=13, bold=True, color=color)
        add_text(s, rx + Inches(1.00), ry + Inches(0.60), rw - Inches(1.15), Inches(0.65),
                 body, size=10.5, color=GREY)
        ry += Inches(1.50)


def s_what_today(prs):
    s = slide_blank(prs)
    add_chrome(s, "Workshop path")
    add_eyebrow(s, "FROM MOTIVATION TO PRACTICE")
    add_title(s, "Four notebooks. One contract.", size=28, height=0.7)
    add_text(s, Inches(0.65), Inches(1.55), Inches(12.0), Inches(0.40),
             "Run them in order. Then we come back together to discuss what is missing.",
             size=13, italic=True, color=GREY)

    # 4 stations + 3 arrows in a horizontal pipeline
    notebooks = [
        ("00", "Frame", MSO_SHAPE.HEXAGON, BLUE,
         "Inspect a benchmark contract."),
        ("01", "Train", MSO_SHAPE.GEAR_6, GREEN,
         "Fit a small conditional generator."),
        ("02", "Evaluate", MSO_SHAPE.STAR_5_POINT, ORANGE,
         "Score generated designs as engineering candidates."),
        ("03", "Extend", MSO_SHAPE.CAN, RED,
         "Wrap your own problem behind the same API."),
    ]
    y = Inches(2.50)
    h = Inches(3.50)
    n = len(notebooks)
    arrow_w = Inches(0.40)
    pad = Inches(0.20)
    total = Inches(13.33 - 1.30)
    station_w = (total - arrow_w * (n - 1) - pad * 2 * (n - 1)) / n
    x = Inches(0.65)
    for i, (num, name, shape, color, body) in enumerate(notebooks):
        # station
        add_rect(s, x, y, station_w, h, WHITE)
        add_rect(s, x, y, station_w, Inches(0.08), color)
        add_text(s, x + Inches(0.25), y + Inches(0.30), station_w - Inches(0.5), Inches(0.40),
                 f"NOTEBOOK {num}", size=10, bold=True, color=color)
        # big icon
        ic = Inches(1.10)
        ix = x + (station_w - ic) / 2
        iy = y + Inches(0.85)
        shp = s.shapes.add_shape(shape, ix, iy, ic, ic)
        shp.fill.solid(); shp.fill.fore_color.rgb = color
        shp.line.fill.background(); shp.shadow.inherit = False
        add_text(s, x, y + Inches(2.10), station_w, Inches(0.40),
                 name, font=FONT_DISPLAY, size=22, bold=True, color=NEAR_BLACK,
                 align=PP_ALIGN.CENTER)
        add_text(s, x + Inches(0.20), y + Inches(2.65), station_w - Inches(0.40), Inches(0.85),
                 body, size=11, color=GREY, align=PP_ALIGN.CENTER)
        x += station_w
        if i < n - 1:
            # arrow
            arrow = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW,
                                       x + pad, y + h / 2 - Inches(0.20),
                                       arrow_w, Inches(0.40))
            arrow.fill.solid(); arrow.fill.fore_color.rgb = GREY_LIGHT
            arrow.line.fill.background(); arrow.shadow.inherit = False
            x += arrow_w + pad * 2

    add_rect(s, Inches(0.65), Inches(6.40), Inches(12.05), Inches(0.50), BLUE)
    add_text(s, Inches(0.65), Inches(6.50), Inches(12.05), Inches(0.32),
             "Then: what is missing, and where do we go next as a community?",
             size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)


def s_closing(prs):
    s = slide_blank(prs)
    add_rect(s, Emu(0), Emu(0), SLIDE_W, SLIDE_H, BG)
    add_rect(s, Emu(0), Emu(0), Inches(0.12), SLIDE_H, BLUE)
    add_text(s, Inches(0.65), Inches(0.65), Inches(8.0), Inches(0.30),
             "DCC 2026 WORKSHOP  ·  OPENING KEYNOTE",
             size=11, bold=True, color=BLUE)
    # giant quote
    add_text(s, Inches(0.65), Inches(2.10), Inches(12.0), Inches(2.6),
             ["Benchmarks do not",
              "slow research down."],
             font=FONT_DISPLAY, size=52, bold=True, color=NEAR_BLACK,
             line_spacing=1.0)
    add_text(s, Inches(0.65), Inches(4.65), Inches(12.0), Inches(1.0),
             "They make it add up.",
             font=FONT_DISPLAY, size=52, bold=True, color=BLUE)
    # next-up bar
    add_rect(s, Inches(0.65), Inches(6.35), Inches(12.05), Inches(0.60), BLUE)
    add_text(s, Inches(0.85), Inches(6.50), Inches(11.8), Inches(0.30),
             "Up next  →  Notebook 00:  Frame your design problem as a benchmark contract.",
             size=14, bold=True, color=WHITE)


# ---------- main ----------

def build_deck(out_path: Path):
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    s_title(prs)
    s_promise(prs)
    s_renaissance(prs)
    s_honest_question(prs)
    s_repro_crisis(prs)
    s_design_harder(prs)
    s_concrete_case(prs)
    s_failure_modes(prs)
    s_divider(prs, eyebrow="SECTION 2  —  LESSONS FROM OTHER FIELDS",
              title="What benchmarks have done\nelsewhere.",
              subtitle="Vision, language, and biology have all been here. Three case studies, one recipe.",
              icon_shape=MSO_SHAPE.STAR_5_POINT)
    s_imagenet(prs)
    s_glue(prs)
    s_casp(prs)
    s_shared_dna(prs)
    s_divider(prs, eyebrow="SECTION 3  —  WHAT OUR FIELD NEEDS",
              title="What engineering design ML\nstill lacks.",
              subtitle="The gap is not theory or talent. It is shared, executable infrastructure.",
              icon_shape=MSO_SHAPE.GEAR_6)
    s_five_missing(prs)
    s_multifaceted(prs)
    s_dataset_not_enough(prs)
    s_engibench_divider(prs)
    s_paper_in_a_can(prs)
    s_coverage(prs)
    s_contract_code(prs)
    s_what_today(prs)
    s_closing(prs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out_path)
    print(f"Wrote {out_path} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    build_deck(here / "introduction-benchmarking-genai-engineering-design-visual.pptx")
