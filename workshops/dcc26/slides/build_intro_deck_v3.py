"""Build the DCC'26 intro deck — v3, the balanced version.

Versus v2:
- Decorative clipart removed (no more gears/stars/suns/hexagons used as labels
  for unrelated concepts). Visuals are kept only when they actually carry
  information: charts, the can-with-satellites metaphor, the spider chart,
  the timeline year markers, the engibench problems image, the code block.
- Real-image placeholders are tagged with specific sourcing notes
  ([IMAGE: ... source: ...]). See IMAGE_BRIEF.md for the full sourcing list.
- Each slide carries speaker notes (visible in PowerPoint presenter view)
  describing what the slide should convey and how to pace it.

Run:
    python3 workshops/dcc26/slides/build_intro_deck_v3.py

Output:
    workshops/dcc26/slides/introduction-benchmarking-genai-engineering-design-final.pptx
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


# ---------- design system ----------

BG = RGBColor(0xFB, 0xFA, 0xF7)
BLUE = RGBColor(0x22, 0x5E, 0x9B)
BLUE_DEEP = RGBColor(0x17, 0x42, 0x6E)
BLUE_LIGHT = RGBColor(0xD7, 0xE8, 0xF6)
BLUE_TINT = RGBColor(0xEC, 0xF3, 0xFA)
BLUE_TXT_LIGHT = RGBColor(0xEA, 0xF3, 0xFA)
BLUE_TXT_CHIP = RGBColor(0xCF, 0xE3, 0xF3)
NEAR_BLACK = RGBColor(0x17, 0x21, 0x2B)
GREY = RGBColor(0x5A, 0x66, 0x75)
GREY_SOFT = RGBColor(0x7A, 0x84, 0x93)
GREY_LIGHT = RGBColor(0xC8, 0xCE, 0xD6)
GREY_PALE = RGBColor(0xEC, 0xEE, 0xF1)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
RED = RGBColor(0xC8, 0x48, 0x37)
ORANGE = RGBColor(0xC4, 0x7B, 0x20)
GREEN = RGBColor(0x2F, 0x7D, 0x62)

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


def add_notes(slide, text):
    """Speaker notes on a slide."""
    ns = slide.notes_slide
    tf = ns.notes_text_frame
    tf.text = text


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


def add_title(slide, text, *, size=26, top=0.80, height=1.0, width=12.0,
              color=NEAR_BLACK):
    add_text(slide, Inches(0.65), Inches(top), Inches(width), Inches(height),
             text, font=FONT_DISPLAY, size=size, bold=True, color=color)


def add_kicker_band(slide, text):
    add_rect(slide, Inches(0.65), Inches(6.40), Inches(12.05), Inches(0.50), BLUE)
    add_text(slide, Inches(0.65), Inches(6.50), Inches(12.05), Inches(0.32),
             text, size=13.5, bold=True, color=WHITE, align=PP_ALIGN.CENTER)


# ---------- image placeholders ----------

def image_placeholder(slide, x, y, w, h, *, label, source=None, tint=BLUE_TINT,
                      dashed=True, icon_color=BLUE):
    """Tinted rectangle that says 'real image will go here' with a sourcing hint."""
    shp = add_rect(slide, x, y, w, h, tint)
    if dashed:
        nsmap = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
        spPr = shp.fill._xPr
        ln_elem = spPr.find("a:ln", nsmap)
        if ln_elem is None:
            ln_elem = etree.SubElement(spPr, qn("a:ln"))
        ln_elem.set("w", str(int(Pt(1).emu * 1.0)))
        # ensure solid fill on the line element
        for child in list(ln_elem):
            if child.tag.endswith("solidFill") or child.tag.endswith("prstDash"):
                ln_elem.remove(child)
        sf = etree.SubElement(ln_elem, qn("a:solidFill"))
        clr = etree.SubElement(sf, qn("a:srgbClr"))
        clr.set("val", "{:02X}{:02X}{:02X}".format(BLUE[0], BLUE[1], BLUE[2]))
        dash = etree.SubElement(ln_elem, qn("a:prstDash"))
        dash.set("val", "dash")

    # subtle photo-frame icon (mountain + sun) centered
    ix = x + w / 2 - Inches(0.55)
    iy = y + h / 2 - Inches(0.65)
    frame = add_rect(slide, ix, iy, Inches(1.10), Inches(0.80),
                     None, line=icon_color, shape=MSO_SHAPE.RECTANGLE)
    frame.line.color.rgb = icon_color
    frame.line.width = Pt(1.25)
    tri = slide.shapes.add_shape(MSO_SHAPE.ISOSCELES_TRIANGLE,
                                 ix + Inches(0.20), iy + Inches(0.25),
                                 Inches(0.70), Inches(0.50))
    tri.fill.solid(); tri.fill.fore_color.rgb = icon_color
    tri.line.fill.background(); tri.shadow.inherit = False
    sun = slide.shapes.add_shape(MSO_SHAPE.OVAL,
                                 ix + Inches(0.78), iy + Inches(0.12),
                                 Inches(0.18), Inches(0.18))
    sun.fill.solid(); sun.fill.fore_color.rgb = icon_color
    sun.line.fill.background(); sun.shadow.inherit = False

    # caption block
    cap_y = y + h - Inches(0.62)
    add_text(slide, x + Inches(0.15), cap_y, w - Inches(0.30), Inches(0.28),
             label, size=10, italic=True, bold=True, color=BLUE,
             align=PP_ALIGN.CENTER)
    if source:
        add_text(slide, x + Inches(0.15), cap_y + Inches(0.30),
                 w - Inches(0.30), Inches(0.26),
                 source, size=8.5, italic=True, color=GREY_SOFT,
                 align=PP_ALIGN.CENTER)


def section_divider(slide_factory, prs, *, eyebrow, title, subtitle, notes):
    """Clean typography section break, no decorative icon."""
    s = slide_factory(prs)
    add_rect(s, Emu(0), Emu(0), SLIDE_W, SLIDE_H, BLUE)
    # thin accent rule
    add_rect(s, Inches(0.85), Inches(2.55), Inches(0.80), Inches(0.06),
             BLUE_TXT_CHIP)
    add_text(s, Inches(0.85), Inches(2.75), Inches(11.5), Inches(0.36),
             eyebrow, size=12, bold=True, color=BLUE_TXT_CHIP)
    add_text(s, Inches(0.85), Inches(3.20), Inches(11.5), Inches(1.90),
             title, font=FONT_DISPLAY, size=46, bold=True, color=WHITE,
             line_spacing=1.0)
    add_text(s, Inches(0.85), Inches(5.40), Inches(11.5), Inches(0.55),
             subtitle, size=16, color=BLUE_TXT_LIGHT)
    add_notes(s, notes)
    return s


# ---------- shared visual helpers (chart, spider, can, timeline) ----------

def bar_chart(slide, x, y, w, h, values, labels, *, max_v=None,
              bar_color=BLUE, highlight_idx=None, highlight_color=RED,
              good_color=GREEN, good_threshold=None, show_values=True,
              annotation=None, annotation_idx=None):
    if max_v is None:
        max_v = max(values) * 1.10
    n = len(values)
    slot_w = w / n
    bar_w = slot_w * 0.55
    label_h = Inches(0.30)
    value_h = Inches(0.28)
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
            add_text(slide, x + slot_w * i, by - Inches(0.26),
                     slot_w, Inches(0.24),
                     f"{v:g}", size=9.5, bold=True, color=NEAR_BLACK,
                     align=PP_ALIGN.CENTER)
        add_text(slide, x + slot_w * i, y + value_h + chart_h + Inches(0.04),
                 slot_w, label_h,
                 str(lab), size=9.5, color=GREY, align=PP_ALIGN.CENTER)
    if annotation and annotation_idx is not None:
        ax = x + slot_w * annotation_idx + slot_w / 2
        add_text(slide, ax - Inches(1.0), y + Inches(0.0), Inches(2.0), Inches(0.28),
                 annotation, size=10.5, bold=True, color=highlight_color,
                 align=PP_ALIGN.CENTER)


def spider_chart(slide, cx, cy, r, axis_labels, *, n_rings=3, fill=BLUE_LIGHT,
                 line=BLUE, axis_color=GREY_LIGHT, label_color=NEAR_BLACK,
                 radii=None):
    n = len(axis_labels)
    for k in range(1, n_rings + 1):
        rr = r * k / n_rings
        ring = slide.shapes.add_shape(MSO_SHAPE.OVAL,
                                      int(cx - rr), int(cy - rr),
                                      int(2 * rr), int(2 * rr))
        ring.fill.background()
        ring.line.color.rgb = axis_color
        ring.line.width = Pt(0.75)
        ring.shadow.inherit = False
    for i, label in enumerate(axis_labels):
        angle = -math.pi / 2 + i * 2 * math.pi / n
        ax = int(cx + r * math.cos(angle))
        ay = int(cy + r * math.sin(angle))
        ln = slide.shapes.add_connector(1, int(cx), int(cy), ax, ay)
        ln.line.color.rgb = axis_color
        ln.line.width = Pt(0.75)
        lx_emu = int(cx + (r + Inches(0.45)) * math.cos(angle))
        ly_emu = int(cy + (r + Inches(0.45)) * math.sin(angle))
        box_w = Inches(2.6)
        box_h = Inches(0.35)
        add_text(slide,
                 int(lx_emu - box_w / 2), int(ly_emu - box_h / 2),
                 box_w, box_h,
                 label, size=12, bold=True, color=label_color,
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    if radii is None:
        radii = [0.85, 0.65, 0.78, 0.55, 0.80][:n]
    pts = []
    for i, rr in enumerate(radii):
        angle = -math.pi / 2 + i * 2 * math.pi / n
        pts.append((int(cx + r * rr * math.cos(angle)),
                    int(cy + r * rr * math.sin(angle))))
    _polygon(slide, pts, fill=fill, line=line, line_w=2.0)


def _polygon(slide, pts, *, fill, line, line_w=1.5):
    from lxml import etree as _etree
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = x_max - x_min; h = y_max - y_min
    spTree = slide.shapes._spTree
    sp = _etree.SubElement(spTree, qn("p:sp"))
    nvSpPr = _etree.SubElement(sp, qn("p:nvSpPr"))
    cNvPr = _etree.SubElement(nvSpPr, qn("p:cNvPr"))
    cNvPr.set("id", "9999"); cNvPr.set("name", "Polygon")
    _etree.SubElement(nvSpPr, qn("p:cNvSpPr"))
    _etree.SubElement(nvSpPr, qn("p:nvPr"))
    spPr = _etree.SubElement(sp, qn("p:spPr"))
    xfrm = _etree.SubElement(spPr, qn("a:xfrm"))
    off = _etree.SubElement(xfrm, qn("a:off"))
    off.set("x", str(int(x_min))); off.set("y", str(int(y_min)))
    ext = _etree.SubElement(xfrm, qn("a:ext"))
    ext.set("cx", str(int(w))); ext.set("cy", str(int(h)))
    custGeom = _etree.SubElement(spPr, qn("a:custGeom"))
    _etree.SubElement(custGeom, qn("a:avLst"))
    _etree.SubElement(custGeom, qn("a:gdLst"))
    _etree.SubElement(custGeom, qn("a:ahLst"))
    _etree.SubElement(custGeom, qn("a:cxnLst"))
    rect = _etree.SubElement(custGeom, qn("a:rect"))
    rect.set("l", "0"); rect.set("t", "0"); rect.set("r", "r"); rect.set("b", "b")
    pathLst = _etree.SubElement(custGeom, qn("a:pathLst"))
    path = _etree.SubElement(pathLst, qn("a:path"))
    path.set("w", str(int(w))); path.set("h", str(int(h)))
    for i, (px, py) in enumerate(pts):
        lx = int(px - x_min); ly = int(py - y_min)
        cmd = _etree.SubElement(path, qn("a:moveTo") if i == 0 else qn("a:lnTo"))
        pt = _etree.SubElement(cmd, qn("a:pt"))
        pt.set("x", str(lx)); pt.set("y", str(ly))
    _etree.SubElement(path, qn("a:close"))
    fillElem = _etree.SubElement(spPr, qn("a:solidFill"))
    clr = _etree.SubElement(fillElem, qn("a:srgbClr"))
    clr.set("val", "{:02X}{:02X}{:02X}".format(fill[0], fill[1], fill[2]))
    ln = _etree.SubElement(spPr, qn("a:ln"))
    ln.set("w", str(int(line_w * 12700)))
    fillL = _etree.SubElement(ln, qn("a:solidFill"))
    clrL = _etree.SubElement(fillL, qn("a:srgbClr"))
    clrL.set("val", "{:02X}{:02X}{:02X}".format(line[0], line[1], line[2]))


def year_dot(slide, cx, cy, r, color, *, year, year_color=WHITE, year_size=14):
    shp = slide.shapes.add_shape(MSO_SHAPE.OVAL,
                                 int(cx - r), int(cy - r),
                                 int(2 * r), int(2 * r))
    shp.fill.solid(); shp.fill.fore_color.rgb = color
    shp.line.fill.background(); shp.shadow.inherit = False
    tf = shp.text_frame
    tf.margin_left = Emu(0); tf.margin_right = Emu(0)
    tf.margin_top = Emu(0); tf.margin_bottom = Emu(0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r2 = p.add_run(); r2.text = year
    r2.font.name = FONT_DISPLAY; r2.font.size = Pt(year_size)
    r2.font.bold = True; r2.font.color.rgb = year_color


# ---------- slide builders ----------

def slide_blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def s_title(prs):
    s = slide_blank(prs)
    add_rect(s, Emu(0), Emu(0), SLIDE_W, SLIDE_H, BG)
    # left hero image area
    image_placeholder(s, Emu(0), Emu(0), Inches(6.65), SLIDE_H,
                      label="[HERO: Airbus A320 bionic partition  OR  EngiBench beams2d hero render]",
                      source="Source: Autodesk + Airbus 2016 press image; OR engibench.docs/_static/img/problems/beams2d.png",
                      tint=BLUE, icon_color=BLUE_TXT_CHIP)
    # caption labels on top of left panel
    add_text(s, Inches(0.65), Inches(0.55), Inches(5.5), Inches(0.30),
             "DCC 2026 PARIS  ·  OPENING KEYNOTE",
             size=11, bold=True, color=BLUE_TXT_CHIP)
    # right text panel
    add_text(s, Inches(7.05), Inches(1.50), Inches(5.85), Inches(0.30),
             "WORKSHOP OPENING", size=10, bold=True, color=BLUE)
    add_text(s, Inches(7.05), Inches(1.95), Inches(6.0), Inches(2.6),
             ["Benchmarking",
              "Generative AI",
              "for Engineering",
              "Design."],
             font=FONT_DISPLAY, size=40, bold=True, color=NEAR_BLACK,
             line_spacing=1.0)
    add_rect(s, Inches(7.05), Inches(5.05), Inches(0.50), Inches(0.06), BLUE)
    add_text(s, Inches(7.05), Inches(5.25), Inches(6.0), Inches(0.50),
             "Why shared, executable contracts are the missing infrastructure.",
             size=15, color=GREY)
    add_text(s, Inches(7.05), Inches(6.30), Inches(6.0), Inches(0.30),
             "Matthew Keeler  ·  Soheyl Massoudi  ·  Mark Fuge",
             size=12, bold=True, color=NEAR_BLACK)
    add_text(s, Inches(7.05), Inches(6.62), Inches(6.0), Inches(0.25),
             "D-MAVT, ETH Zürich  ·  EngiBench + EngiOpt",
             size=10.5, color=GREY)
    add_notes(s,
        "Open with the most striking visual you have — a real photo of a "
        "generatively-designed part that the audience can recognise as 'this "
        "is engineering, not a toy demo'. Don't read the title. Establish in "
        "one sentence: this workshop is about how to know whether new ML "
        "methods actually work for engineering design.")


def s_promise(prs):
    s = slide_blank(prs)
    add_chrome(s, "Why this matters")
    add_eyebrow(s, "THE PROMISE")
    add_title(s, "AI is already designing parts we manufacture and fly.",
              size=26, height=0.7)
    y = Inches(2.00)
    h = Inches(4.35)
    pad = Inches(0.30)
    n = 3
    total_w = Inches(13.33 - 1.30)
    w = (total_w - pad * (n - 1)) / n
    x = Inches(0.65)
    items = [
        ("Airbus A320 bionic partition",
         "45% lighter. Printed in titanium. Certified, flown.",
         "[PHOTO: Airbus A320 bionic partition installed in cabin]",
         "Source: Autodesk + Airbus 2016 press release (widely republished)."),
        ("GE Additive jet-engine bracket",
         "84% mass reduction. Topology-optimized + 3D-printed.",
         "[PHOTO: GE GEnx engine bracket in carbon-printed metal]",
         "Source: GE Additive Bracket Challenge (2013), winning entry."),
        ("Inverse-designed metasurface",
         "Optical devices beyond hand-crafted baselines.",
         "[SEM IMAGE: photonic metasurface unit cell array]",
         "Source: Stanford Fan-group inverse design (e.g., Sell et al. 2017)."),
    ]
    for title, caption, ph_label, ph_source in items:
        image_placeholder(s, x, y, w, h - Inches(1.10),
                          label=ph_label, source=ph_source)
        add_text(s, x, y + h - Inches(1.05), w, Inches(0.34),
                 title, size=14, bold=True, color=NEAR_BLACK,
                 align=PP_ALIGN.CENTER)
        add_text(s, x, y + h - Inches(0.65), w, Inches(0.55),
                 caption, size=11, color=GREY, align=PP_ALIGN.CENTER)
        x += w + pad
    add_notes(s,
        "Three real-world examples. Swap any of them for cases closer to the "
        "audience's domain if you have them. Point: generative methods aren't "
        "speculative for engineering — they make production parts. Pace: ~30s "
        "per example, name what changed materially (mass / cost / function).")


def s_renaissance(prs):
    s = slide_blank(prs)
    add_chrome(s, "Why this matters")
    add_eyebrow(s, "A RENAISSANCE OF METHODS")
    add_title(s, "Every few years, a new generative recipe.", size=28, height=0.7)
    add_text(s, Inches(0.65), Inches(1.55), Inches(12.0), Inches(0.40),
             "And every paper claims a new state of the art.",
             size=14, italic=True, color=GREY)

    # timeline axis
    y_axis = Inches(4.50)
    add_rect(s, Inches(0.95), y_axis, Inches(11.45), Inches(0.04), GREY_LIGHT)

    eras = [
        ("2017", "GAN era", "cGANs for layouts & shapes.", BLUE),
        ("2019", "VAE era", "Smooth latent design manifolds.", GREEN),
        ("2022", "Diffusion", "Score-based generative design.", ORANGE),
        ("2024", "LLMs + agents", "Code-writing design pipelines.", RED),
    ]
    n = len(eras)
    span = Inches(10.50)
    start_x = Inches(1.40)
    for i, (yr, name, blurb, color) in enumerate(eras):
        cx = start_x + span * i / (n - 1)
        year_dot(s, cx, y_axis + Inches(0.02), Inches(0.45), color, year=yr)
        # name above
        add_text(s, cx - Inches(1.60), Inches(2.45), Inches(3.20), Inches(0.40),
                 name, size=17, bold=True, color=color, align=PP_ALIGN.CENTER)
        # small thumbnail placeholder for sample output
        ph_w = Inches(1.80); ph_h = Inches(1.20)
        image_placeholder(s, cx - ph_w / 2, Inches(2.95), ph_w, ph_h,
                          label=f"[{name} sample]",
                          source="See IMAGE_BRIEF.md")
        # one-line blurb below the axis
        add_text(s, cx - Inches(1.60), Inches(5.30), Inches(3.20), Inches(0.40),
                 blurb, size=11.5, color=GREY, align=PP_ALIGN.CENTER)

    add_kicker_band(s,
        "The methods change quickly. The way we evaluate them barely changes at all.")
    add_notes(s,
        "Pace this fast — ~10 seconds per era. The point is volume, not "
        "method detail. Every couple of years a new generative recipe arrives, "
        "all of them get applied to engineering, all of them claim SOTA. "
        "Land the kicker: methods change quickly, evaluation doesn't.")


def s_honest_question(prs):
    s = slide_blank(prs)
    add_chrome(s, "Why this matters")
    add_eyebrow(s, "AN HONEST QUESTION")
    add_text(s, Inches(0.85), Inches(2.20), Inches(11.65), Inches(2.4),
             ["When a paper says",
              "“we beat the SOTA on topology optimization”,",
              "what actually changed?"],
             font=FONT_DISPLAY, size=38, bold=True, color=NEAR_BLACK,
             line_spacing=1.10)
    add_text(s, Inches(0.85), Inches(5.10), Inches(11.65), Inches(0.40),
             "Better model?  ·  Easier conditions?  ·  Weaker feasibility check?  ·  Some mix of all three?",
             size=15, color=GREY)
    add_text(s, Inches(0.85), Inches(6.20), Inches(11.65), Inches(0.40),
             "Today's literature does not give us a way to know.",
             size=15, italic=True, bold=True, color=BLUE)
    add_notes(s,
        "Pause after asking. Don't fill the silence. Most attendees will "
        "recognise the discomfort — they've all read a paper where they "
        "couldn't tell. Land softly: 'today's literature does not let us "
        "know.' This is the slide that sets up the entire workshop.")


def s_repro_crisis(prs):
    s = slide_blank(prs)
    add_chrome(s, "Why we cannot tell")
    add_eyebrow(s, "ML'S REPRODUCIBILITY CRISIS")
    add_title(s, "We are not the first field to face this.",
              size=26, height=0.7)
    add_text(s, Inches(0.65), Inches(1.55), Inches(12.0), Inches(0.40),
             "Mainstream ML had this exact problem a decade ago — and built the infrastructure to recover. "
             "We can copy the playbook.",
             size=13, italic=True, color=GREY)

    # 3 narrative columns: CRISIS → RESPONSE → US
    y = Inches(2.30); h = Inches(4.10)
    pad = Inches(0.20); arrow_w = Inches(0.55)
    total_w = Inches(13.33 - 1.30)
    col_w = (total_w - 2 * arrow_w - 4 * pad) / 3

    # ----- Column 1: Crisis -----
    x = Inches(0.65)
    add_rect(s, x, y, col_w, h, WHITE)
    add_rect(s, x, y, col_w, Inches(0.10), RED)
    add_text(s, x + Inches(0.30), y + Inches(0.30), col_w - Inches(0.6), Inches(0.32),
             "2017  ·  THE CRISIS", size=11, bold=True, color=RED)
    add_text(s, x + Inches(0.30), y + Inches(0.80), col_w - Inches(0.6), Inches(1.6),
             "≈30%",
             font=FONT_DISPLAY, size=84, bold=True, color=RED, line_spacing=1.0)
    add_text(s, x + Inches(0.30), y + Inches(2.55), col_w - Inches(0.6), Inches(0.85),
             "of accepted ML papers couldn't be reproduced from the text alone.",
             size=12.5, bold=True, color=NEAR_BLACK, line_spacing=1.10)
    add_text(s, x + Inches(0.30), y + Inches(3.55), col_w - Inches(0.6), Inches(0.32),
             "Pineau et al. 2019.", size=10.5, italic=True, color=GREY)

    # arrow 1
    x += col_w + pad
    arr1 = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW,
                              int(x), int(y + h / 2 - Inches(0.25)),
                              arrow_w, Inches(0.50))
    arr1.fill.solid(); arr1.fill.fore_color.rgb = GREY_LIGHT
    arr1.line.fill.background(); arr1.shadow.inherit = False
    x += arrow_w + pad

    # ----- Column 2: Response -----
    add_rect(s, x, y, col_w, h, WHITE)
    add_rect(s, x, y, col_w, Inches(0.10), GREEN)
    add_text(s, x + Inches(0.30), y + Inches(0.30), col_w - Inches(0.6), Inches(0.32),
             "2019–21  ·  THE RESPONSE", size=11, bold=True, color=GREEN)
    responses = [
        ("NeurIPS reproducibility checklist",
         "Every paper must document data, code, and compute."),
        ("Open code, weights, harnesses",
         "Default expectation, not optional supplement."),
        ("Shared benchmarks become the unit of progress",
         "HuggingFace, Papers-with-Code, leaderboards."),
    ]
    iy = y + Inches(0.95)
    for title, body in responses:
        add_text(s, x + Inches(0.30), iy, col_w - Inches(0.6), Inches(0.38),
                 title, size=13, bold=True, color=NEAR_BLACK)
        add_text(s, x + Inches(0.30), iy + Inches(0.40), col_w - Inches(0.6), Inches(0.55),
                 body, size=10.5, color=GREY, line_spacing=1.10)
        iy += Inches(1.05)

    # arrow 2
    x += col_w + pad
    arr2 = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW,
                              int(x), int(y + h / 2 - Inches(0.25)),
                              arrow_w, Inches(0.50))
    arr2.fill.solid(); arr2.fill.fore_color.rgb = GREY_LIGHT
    arr2.line.fill.background(); arr2.shadow.inherit = False
    x += arrow_w + pad

    # ----- Column 3: Us -----
    add_rect(s, x, y, col_w, h, WHITE)
    add_rect(s, x, y, col_w, Inches(0.10), BLUE)
    add_text(s, x + Inches(0.30), y + Inches(0.30), col_w - Inches(0.6), Inches(0.32),
             "2026  ·  OUR TURN", size=11, bold=True, color=BLUE)
    add_text(s, x + Inches(0.30), y + Inches(0.90), col_w - Inches(0.6), Inches(2.40),
             ["Engineering design ML",
              "is standing where ML",
              "stood in 2017."],
             font=FONT_DISPLAY, size=20, bold=True, color=NEAR_BLACK,
             line_spacing=1.10)
    add_text(s, x + Inches(0.30), y + Inches(3.05), col_w - Inches(0.6), Inches(0.95),
             "Same problem. None of the infrastructure yet.",
             size=12, italic=True, bold=True, color=BLUE, line_spacing=1.15)

    add_notes(s,
        "Tell the story left-to-right. (1) ML had its own crisis around 2017 — "
        "a study found 30% of papers couldn't be reproduced from text alone. "
        "(2) The community responded with checklists, open code/weights, and "
        "shared benchmarks. (3) Engineering design ML is now in the 2017 spot. "
        "The point of this workshop is to skip the 'flounder for two years' "
        "phase and go straight to the playbook that worked.")


def s_design_harder(prs):
    s = slide_blank(prs)
    add_chrome(s, "Why we cannot tell")
    add_eyebrow(s, "ENGINEERING DESIGN IS HARDER")
    add_title(s, "A design paper hides far more silent choices.",
              size=26, height=0.7)

    # left: annotated diagram placeholder
    image_placeholder(s, Inches(0.65), Inches(2.00), Inches(6.20), Inches(4.40),
                      label="[ANNOTATED FIGURE: a Beams2D problem with callouts for solver, "
                            "mesh, boundary conditions, volume fraction, baseline optimizer, units]",
                      source="Source: render via engibench problem.render(beams2d) and annotate, "
                             "or adapt Sigmund's 99-line code paper figures.")
    # right: ordered list of 6 dimensions
    rx = Inches(7.30); rw = Inches(5.55)
    add_text(s, rx, Inches(2.00), rw, Inches(0.30),
             "SIX KNOBS THAT FLIP CONCLUSIONS", size=11, bold=True, color=BLUE)
    items = [
        ("Simulator",  "Which solver? Which mesh? Which tolerance?"),
        ("Constraints", "Manufacturability, disconnected-material, volfrac slack."),
        ("Conditions",  "Distribution of loads and scenarios."),
        ("Baselines",   "Random search vs. SIMP vs. CMA-ES."),
        ("Representation", "Pixels, meshes, B-splines, Bézier curves."),
        ("Units & scaling", "Compliance vs. stress vs. dB."),
    ]
    iy = Inches(2.45)
    for name, body in items:
        add_text(s, rx, iy, Inches(1.95), Inches(0.30),
                 name, size=13, bold=True, color=NEAR_BLACK)
        add_text(s, rx + Inches(2.05), iy, rw - Inches(2.10), Inches(0.30),
                 body, size=11.5, color=GREY)
        iy += Inches(0.55)
    add_notes(s,
        "Walk the audience through the figure if you have a real one — point "
        "at each callout. The list on the right is the canonical taxonomy: "
        "any one of these can quietly flip a conclusion. Mention that vision "
        "papers hide at most a handful of choices; design papers hide all of "
        "these. Don't enumerate the whole list — pick two and move on.")


def s_concrete_case(prs):
    s = slide_blank(prs)
    add_chrome(s, "Why we cannot tell")
    add_eyebrow(s, "A CONCRETE CASE")
    add_title(s, "Same task name. Same headline number.",
              size=26, height=0.7)
    add_text(s, Inches(0.65), Inches(1.55), Inches(12.0), Inches(0.40),
             "Two recent \"Beams2D\" papers both report MSE = 0.04.  Are they comparable?  "
             "Each row below is a setup choice that quietly changes the problem.",
             size=12.5, italic=True, color=GREY)

    # two clean paper cards — each row has a plain-language label + an
    # italic sub-label naming what category of choice it actually is.
    y = Inches(2.20); doc_w = Inches(4.80); doc_h = Inches(4.10)
    x_a = Inches(0.85)
    x_b = SLIDE_W - Inches(0.85) - doc_w

    def paper_card(x, accent, label, rows):
        add_rect(s, x, y, doc_w, doc_h, WHITE)
        add_rect(s, x, y, doc_w, Inches(0.50), accent)
        add_text(s, x, y + Inches(0.13), doc_w, Inches(0.30),
                 label, size=11.5, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        add_text(s, x + Inches(0.30), y + Inches(0.65), doc_w - Inches(0.6), Inches(0.40),
                 "Reported  MSE = 0.04", size=14, bold=True, color=accent)
        ry = y + Inches(1.20)
        for human_label, sub_label, value in rows:
            # plain-language label
            add_text(s, x + Inches(0.30), ry, Inches(2.30), Inches(0.26),
                     human_label, size=11.5, bold=True, color=NEAR_BLACK)
            # value
            add_text(s, x + Inches(2.65), ry, doc_w - Inches(2.95), Inches(0.26),
                     value, size=11.5, color=accent, bold=True)
            # italic sub-label below the human label, explaining the category
            add_text(s, x + Inches(0.30), ry + Inches(0.26), Inches(2.30), Inches(0.22),
                     sub_label, size=9, italic=True, color=GREY_SOFT)
            ry += Inches(0.55)

    paper_card(x_a, BLUE, "PAPER A — diffusion for topology", [
        ("Material budget", "problem setup — volume fraction", "fixed at 0.50"),
        ("Smoothing scale", "solver — filter radius",         "1.5 px"),
        ("Validity check",  "scoring rule — feasibility",     "soft penalty"),
        ("Convergence",     "solver — stopping tolerance",    "1e-3 (loose)"),
        ("Test scenarios",  "evaluation — condition split",   "near training mean"),
    ])
    paper_card(x_b, ORANGE, "PAPER B — cVAE for inverse design", [
        ("Material budget", "problem setup — volume fraction", "sampled 0.30 – 0.55"),
        ("Smoothing scale", "solver — filter radius",         "2.5 px"),
        ("Validity check",  "scoring rule — feasibility",     "hard reject"),
        ("Convergence",     "solver — stopping tolerance",    "1e-5 (tight)"),
        ("Test scenarios",  "evaluation — condition split",   "OOD corners"),
    ])

    # big ≠ between
    add_text(s, Inches(5.85), Inches(3.85), Inches(1.65), Inches(1.40),
             "≠", font=FONT_DISPLAY, size=96, bold=True, color=RED,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    add_kicker_band(s, "Different problem, same number.")
    add_notes(s,
        "Each row is a different category of setup choice. The bold label is "
        "what the choice does in plain language; the italic line below names "
        "the technical term. Walk through 2-3 rows: 'material budget' (how "
        "much material is allowed), 'validity check' (when a design is "
        "called invalid), 'test scenarios' (whether evaluation looks like "
        "training). Point: every paper picks values for ALL of these, "
        "usually without saying so.")


def s_failure_modes(prs):
    s = slide_blank(prs)
    add_chrome(s, "Why we cannot tell")
    add_eyebrow(s, "THREE FAILURE MODES")
    add_title(s, "Without a shared contract, three patterns recur.",
              size=26, height=0.7)
    add_text(s, Inches(0.65), Inches(1.55), Inches(12.0), Inches(0.40),
             "These are the three patterns reviewers most often see when AI-for-design results "
             "cannot be reproduced or compared.",
             size=12.5, italic=True, color=GREY)

    y = Inches(2.20); h = Inches(4.30)
    pad = Inches(0.30)
    total_w = Inches(13.33 - 1.30)
    w = (total_w - pad * 2) / 3

    panels = [
        ("01", "CHERRY-PICKED CONDITIONS", RED,
         "Evaluated only where it's easy.",
         "Models tested only on conditions close to the training distribution. "
         "Numbers look strong on paper. Generalisation collapses on real scenarios.",
         "[OPTIONAL: scatter / t-SNE of train vs test conditions]"),
        ("02", "VISUAL-ONLY EVALUATION", ORANGE,
         "Looks good. Doesn't work.",
         "Generated designs look convincing in a paper figure. Run them through "
         "the simulator and many violate constraints or under-perform a baseline.",
         "[OPTIONAL: design grid + a simulator-failure overlay]"),
        ("03", "ONE-NUMBER SCORES", BLUE,
         "A single scalar hides everything else.",
         "Accuracy or MSE is reported alone. Feasibility, diversity, novelty, and "
         "warm-start value are not. Different choices among those flip the winner.",
         "[OPTIONAL: spider chart showing a model winning on objective only]"),
    ]

    x = Inches(0.65)
    for num, label, accent, headline, body, ph_hint in panels:
        # panel background
        add_rect(s, x, y, w, h, WHITE)
        add_rect(s, x, y, w, Inches(0.10), accent)

        # big numeral
        add_text(s, x + Inches(0.30), y + Inches(0.30), Inches(1.40), Inches(1.10),
                 num, font=FONT_DISPLAY, size=52, bold=True, color=accent,
                 line_spacing=1.0)
        # small all-caps category label
        add_text(s, x + Inches(0.30), y + Inches(1.40), w - Inches(0.6), Inches(0.30),
                 label, size=10.5, bold=True, color=accent)
        # headline (bold pull-quote)
        add_text(s, x + Inches(0.30), y + Inches(1.75), w - Inches(0.6), Inches(1.10),
                 headline, font=FONT_DISPLAY, size=20, bold=True, color=NEAR_BLACK,
                 line_spacing=1.10)
        # body explanation
        add_text(s, x + Inches(0.30), y + Inches(2.95), w - Inches(0.6), Inches(1.20),
                 body, size=11.5, color=GREY, line_spacing=1.20)
        # optional image hint in faint italic at the bottom — for the presenter,
        # not the audience: a reminder of what evidence would strengthen this slide.
        add_text(s, x + Inches(0.30), y + h - Inches(0.36), w - Inches(0.6), Inches(0.30),
                 ph_hint, size=8.5, italic=True, color=GREY_LIGHT)

        x += w + pad

    add_notes(s,
        "Three patterns reviewers flag most often. Big numeral, then the "
        "pull-quote — that's what the audience reads. Spend ~25 seconds per "
        "panel and give one real example you've seen for at least one of "
        "them. The grey italic line at the bottom of each panel is a hint "
        "for what evidence figure would strengthen this slide later; it's a "
        "presenter note, not for the audience.")


def s_imagenet(prs):
    s = slide_blank(prs)
    add_chrome(s, "Lessons from other fields")
    add_eyebrow(s, "LESSON 1  —  IMAGENET")
    add_title(s, "One frozen evaluation made breakthroughs unambiguous.",
              size=24, height=0.7)
    # sample strip placeholder above the chart
    image_placeholder(s, Inches(0.65), Inches(1.70), Inches(12.05), Inches(0.85),
                      label="[STRIP: 10 representative ImageNet sample images]",
                      source="Source: image-net.org/about (CC-BY); fall back to "
                             "Wikipedia ImageNet example tiles.")
    # chart panel
    cx = Inches(0.65); cy = Inches(2.75)
    cw = Inches(12.05); ch = Inches(3.55)
    add_rect(s, cx, cy, cw, ch, WHITE)
    add_text(s, cx + Inches(0.30), cy + Inches(0.18), Inches(10.0), Inches(0.30),
             "ImageNet top-5 classification error", size=12.5, bold=True,
             color=NEAR_BLACK)
    add_text(s, cx + Inches(0.30), cy + Inches(0.46), Inches(10.0), Inches(0.26),
             "A shared task plus a shared evaluation made a decade of progress legible.",
             size=10.5, italic=True, color=GREY)
    years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
    vals = [28.2, 25.8, 16.4, 11.7, 6.7, 3.6, 3.0, 2.3]
    bar_chart(s, cx + Inches(0.50), cy + Inches(0.85),
              cw - Inches(1.0), ch - Inches(1.05),
              vals, [str(y) for y in years],
              max_v=30, highlight_idx=2, highlight_color=RED,
              good_threshold=5, good_color=GREEN, bar_color=BLUE,
              annotation="AlexNet (2012)", annotation_idx=2)
    add_kicker_band(s,
        "Shared task + shared evaluation = a decade of compounding progress.")
    add_notes(s,
        "Two beats: the chart and the kicker. The 2012 AlexNet jump landed "
        "unambiguously because the benchmark was unambiguous. Without "
        "ImageNet, the deep learning revolution would have taken much longer "
        "to be recognised as a revolution. Engineering design has no such "
        "yardstick.")


def s_glue(prs):
    s = slide_blank(prs)
    add_chrome(s, "Lessons from other fields")
    add_eyebrow(s, "LESSON 2  —  GLUE / SUPERGLUE")
    add_title(s, "When one task saturates, a suite sustains the conversation.",
              size=20, top=0.74, height=0.55)

    assets = Path(__file__).resolve().parent.parent / "assets"

    # 1) how a GLUE score is computed — thin band across the top
    band_w = Inches(10.6)
    band = s.shapes.add_picture(str(assets / "glue_scoring.png"),
                                int((SLIDE_W - band_w) / 2), Inches(1.5), width=band_w)

    # 2) raising the bar — hero saturation chart, centred below the band
    chart_w = Inches(7.0)
    chart_x = int((SLIDE_W - chart_w) / 2)
    chart_y = band.top + band.height + Inches(0.20)
    s.shapes.add_picture(str(assets / "glue_saturation.png"),
                         chart_x, chart_y, width=chart_w)

    add_notes(s,
        "Two visuals, no wall of text. Left-to-right band: a GLUE score is just "
        "the unweighted average of 9 language tasks, each with its own metric — "
        "one number hiding nine. Then the chart: in ~a year BERT-class models "
        "cleared GLUE's human baseline, so the community didn't declare victory — "
        "it launched the harder SuperGLUE, which was then climbed too. Engineering "
        "design ML is where NLP was when GLUE first launched: it needs the suite, "
        "not one beam problem.")


def s_casp(prs):
    s = slide_blank(prs)
    add_chrome(s, "Lessons from other fields")
    add_eyebrow(s, "LESSON 3  —  CASP / ALPHAFOLD")
    add_title(s, "A 25-year community benchmark made a breakthrough legible.",
              size=22, height=1.0)
    image_placeholder(s, Inches(0.65), Inches(2.40), Inches(7.20), Inches(4.20),
                      label="[IMAGE: AlphaFold predicted structure overlaid on experimental, CASP14]",
                      source="Source: DeepMind AlphaFold blog (2020); CASP14 results page "
                             "(predictioncenter.org/casp14).")
    rx = Inches(8.40); rw = Inches(4.45); ry = Inches(2.40)
    add_text(s, rx, ry, rw, Inches(0.32),
             "CASP — THE PROTOCOL", size=11, bold=True, color=BLUE)
    add_text(s, rx, ry + Inches(0.40), rw, Inches(2.10),
             ["Held-out targets.",
              "Blind submissions.",
              "One shared score."],
             font=FONT_DISPLAY, size=22, bold=True, color=NEAR_BLACK,
             line_spacing=1.10)
    # mini timeline ruler
    rule_y = ry + Inches(2.75)
    add_rect(s, rx, rule_y, rw, Inches(0.04), GREY_LIGHT)
    marks = ["1994", "2000", "2010", "2020", "CASP14"]
    for i, yr in enumerate(marks):
        cx = rx + rw * i / (len(marks) - 1)
        c = GREEN if yr == "CASP14" else BLUE
        year_dot(s, cx, rule_y + Inches(0.02), Inches(0.10), c,
                 year="", year_size=1)
        add_text(s, cx - Inches(0.50), rule_y + Inches(0.20),
                 Inches(1.0), Inches(0.25),
                 yr, size=9.5, bold=True, color=GREY, align=PP_ALIGN.CENTER)
    add_text(s, rx, ry + Inches(3.55), rw, Inches(0.65),
             "Without CASP, AlphaFold is a press release. With it, a verdict.",
             size=13.5, italic=True, color=NEAR_BLACK)
    add_notes(s,
        "Land the protocol slowly: held-out targets, blind submissions, one "
        "shared score — for 25 years. That patience is what made AlphaFold "
        "credible to biologists overnight. Engineering design will need its "
        "own version of CASP for results to land the same way.")


def s_shared_dna(prs):
    s = slide_blank(prs)
    add_chrome(s, "Lessons from other fields")
    add_eyebrow(s, "WHAT GOOD BENCHMARKS SHARE")
    add_title(s, "Four ingredients turn a dataset into a benchmark.",
              size=26, height=0.7)
    add_text(s, Inches(0.65), Inches(1.55), Inches(12.0), Inches(0.40),
             "The recipe is consistent across ImageNet, GLUE, CASP, OGB, MuJoCo.",
             size=13, italic=True, color=GREY)

    items = [
        ("FIXED INPUTS", BLUE, "Same data, same splits, no silent re-curating."),
        ("STANDARD EVAL", GREEN, "Scoring is code, not prose."),
        ("OPEN ACCESS", ORANGE, "Low cost of entry. Anyone can submit."),
        ("COMMUNITY", RED, "Lives across years. Versioned. Maintained."),
    ]
    y = Inches(2.40); h = Inches(3.85)
    pad = Inches(0.25)
    total = Inches(13.33 - 1.30)
    w = (total - pad * 3) / 4
    x = Inches(0.65)
    for label, color, blurb in items:
        add_rect(s, x, y, w, h, WHITE)
        add_rect(s, x, y, w, Inches(0.10), color)
        # large numeral
        add_text(s, x + Inches(0.30), y + Inches(0.50), w - Inches(0.6), Inches(0.40),
                 label, size=12, bold=True, color=color)
        add_text(s, x + Inches(0.30), y + Inches(1.05), w - Inches(0.6), Inches(1.80),
                 blurb.split('.')[0] + ".",
                 font=FONT_DISPLAY, size=22, bold=True, color=NEAR_BLACK,
                 line_spacing=1.05)
        x += w + pad

    add_kicker_band(s,
        "Engineering design ML has almost none of this — yet.")
    add_notes(s,
        "Don't enumerate the four like a list — they're the recipe. The "
        "kicker is what matters: every mature ML field has these four, "
        "engineering design has almost none. That's the gap this workshop "
        "is about.")


def s_five_missing(prs):
    s = slide_blank(prs)
    add_chrome(s, "What the field needs")
    add_eyebrow(s, "FIVE MISSING PIECES")
    add_title(s, "Engineering design ML lacks shared infrastructure.",
              size=26, height=0.7)
    add_text(s, Inches(0.65), Inches(1.55), Inches(12.0), Inches(0.40),
             "Each can be rebuilt. No one should have to. That is the gap.",
             size=13, italic=True, color=GREY)

    items = [
        ("01", "Standard problems",  "A curated set of well-defined design tasks."),
        ("02", "Shared simulators",  "Installable, deterministic, citable."),
        ("03", "Curated datasets",   "Optimized designs + conditions + metadata, pre-split."),
        ("04", "Multi-faceted metrics", "Feasibility, gap, diversity, novelty, warm-start."),
        ("05", "Reproducible runners", "One harness for training and evaluation."),
    ]
    n = len(items)
    y = Inches(2.40); h = Inches(3.95)
    pad = Inches(0.18); total = Inches(13.33 - 1.30)
    w = (total - pad * (n - 1)) / n
    x = Inches(0.65)
    accents = [BLUE, GREEN, ORANGE, RED, BLUE]
    for (num, name, body), color in zip(items, accents):
        add_rect(s, x, y, w, h, WHITE)
        add_rect(s, x, y, w, Inches(0.08), color)
        add_text(s, x + Inches(0.25), y + Inches(0.35), w - Inches(0.5), Inches(1.0),
                 num, font=FONT_DISPLAY, size=46, bold=True, color=color)
        add_text(s, x + Inches(0.25), y + Inches(1.45), w - Inches(0.5), Inches(0.95),
                 name, font=FONT_DISPLAY, size=18, bold=True, color=NEAR_BLACK,
                 line_spacing=1.05)
        add_text(s, x + Inches(0.25), y + Inches(2.65), w - Inches(0.5), Inches(1.2),
                 body, size=11.5, color=GREY)
        x += w + pad
    add_notes(s,
        "Read out the names quickly — 5 to 10 seconds total. Don't expand on "
        "each. The point is to make the gap visible, not exhaustive. The "
        "next two slides motivate why each one matters.")


def s_multifaceted(prs):
    s = slide_blank(prs)
    add_chrome(s, "What the field needs")
    add_eyebrow(s, "EVALUATION, PROPERLY SCOPED")
    add_title(s, "Engineering quality has many axes. A single scalar will betray you.",
              size=22, height=1.0)

    spider_chart(s, Inches(4.20), Inches(4.65), Inches(2.15),
                 axis_labels=["Feasibility", "Performance", "Diversity",
                              "Novelty", "Warm-start"])

    # right legend
    rx = Inches(8.00); rw = Inches(4.85)
    axes_legend = [
        ("Feasibility", "Constraints actually satisfied.", BLUE),
        ("Performance", "Sim score vs. optimizer baseline.", GREEN),
        ("Diversity",   "Modes covered, not collapsed.", ORANGE),
        ("Novelty",     "New designs, not paraphrases.", RED),
        ("Warm-start",  "Does it help downstream search?", BLUE),
    ]
    ry = Inches(2.40)
    for name, body, color in axes_legend:
        add_rect(s, rx, ry, Inches(0.14), Inches(0.65), color)
        add_text(s, rx + Inches(0.30), ry + Inches(0.04), rw - Inches(0.4), Inches(0.30),
                 name, size=13.5, bold=True, color=NEAR_BLACK)
        add_text(s, rx + Inches(0.30), ry + Inches(0.36), rw - Inches(0.4), Inches(0.30),
                 body, size=10.5, color=GREY)
        ry += Inches(0.85)
    add_notes(s,
        "If you have real spider data from your own runs, overlay it on this "
        "chart. The point: a model can win on one axis and lose on another. "
        "Pick the 2-3 axes that matter most to the audience's domain.")


def s_dataset_not_enough(prs):
    s = slide_blank(prs)
    add_chrome(s, "What the field needs")
    add_eyebrow(s, "DATASETS ALONE DO NOT BENCHMARK")
    add_title(s, "Releasing the dataset is necessary, but not enough.",
              size=24, height=0.7)
    add_text(s, Inches(0.65), Inches(1.55), Inches(12.0), Inches(0.40),
             "Without the simulator and the scoring pipeline, you can display designs — you cannot score them.",
             size=13, italic=True, color=GREY)

    # two flow diagrams stacked
    flow_y = Inches(2.30)
    flow_h = Inches(2.00)
    pad = Inches(0.20)
    box_w = Inches(2.30)
    arrow_w = Inches(0.50)

    def flow(y, accent, label, stages, end_label, end_color):
        add_text(s, Inches(0.65), y, Inches(3.50), Inches(0.30),
                 label, size=11, bold=True, color=accent)
        x = Inches(0.65)
        for stage in stages:
            add_rect(s, x, y + Inches(0.40), box_w, flow_h - Inches(0.50), WHITE)
            add_rect(s, x, y + Inches(0.40), box_w, Inches(0.08), accent)
            add_text(s, x, y + Inches(0.40) + (flow_h - Inches(0.50)) / 2 - Inches(0.20),
                     box_w, Inches(0.40),
                     stage, size=13, bold=True, color=NEAR_BLACK,
                     align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
            x += box_w + pad
            # arrow
            arrow = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW,
                                       x, y + Inches(0.40) + (flow_h - Inches(0.50)) / 2 - Inches(0.15),
                                       arrow_w, Inches(0.30))
            arrow.fill.solid(); arrow.fill.fore_color.rgb = GREY_LIGHT
            arrow.line.fill.background(); arrow.shadow.inherit = False
            x += arrow_w + pad
        # ending box
        add_rect(s, x, y + Inches(0.40), Inches(3.60), flow_h - Inches(0.50),
                 end_color)
        add_text(s, x, y + Inches(0.40) + (flow_h - Inches(0.50)) / 2 - Inches(0.20),
                 Inches(3.60), Inches(0.40),
                 end_label, size=13, bold=True, color=WHITE,
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

    flow(flow_y, RED, "DATASET ONLY",
         ["Dataset", "Trained model", "Generated design"],
         "Display only", RED)
    flow(flow_y + flow_h + Inches(0.30), GREEN, "DATASET + SIMULATOR + SCORING",
         ["Dataset", "Trained model", "Simulator + score"],
         "Portable verdict", GREEN)
    add_notes(s,
        "This is the core argument for EngiBench versus just-release-the-"
        "dataset. The top flow ends at 'display only' — papers can show "
        "designs but cannot tell you whether they are good. The bottom flow "
        "ends at 'portable verdict' — numbers travel between labs. Land "
        "this slowly.")


def s_engibench_divider(prs):
    return section_divider(
        slide_blank, prs,
        eyebrow="WHY WE BUILT IT",
        title="Enter EngiBench.",
        subtitle="A standardized framework for benchmarking generative AI in engineering design.  Published at NeurIPS 2025.",
        notes="Brief transition. Pause for emphasis. From here on, the deck is "
              "concrete: what we built, what it gives you, and what you'll do with it in the next three hours.")


def s_paper_in_a_can(prs):
    s = slide_blank(prs)
    add_chrome(s, "EngiBench")
    add_eyebrow(s, "THE DESIGN PATTERN")
    add_title(s, "Each problem is a \"paper in a can\".", size=28, height=0.7)
    add_text(s, Inches(0.65), Inches(1.55), Inches(12.0), Inches(0.40),
             "Eight ingredients, one Python interface, ready to compare across labs.",
             size=13, italic=True, color=GREY)

    # central CAN — the metaphor is the visual
    cx = Inches(13.33 / 2); cy = Inches(4.40)
    can_w = Inches(2.60); can_h = Inches(3.0)
    can = s.shapes.add_shape(MSO_SHAPE.CAN,
                             int(cx - can_w / 2), int(cy - can_h / 2),
                             can_w, can_h)
    can.fill.solid(); can.fill.fore_color.rgb = BLUE
    can.line.fill.background(); can.shadow.inherit = False
    add_text(s, int(cx - Inches(1.30)), int(cy - Inches(0.35)),
             Inches(2.60), Inches(0.40),
             "engibench", font=FONT_DISPLAY, size=20, bold=True, color=WHITE,
             align=PP_ALIGN.CENTER)
    add_text(s, int(cx - Inches(1.30)), int(cy + Inches(0.05)),
             Inches(2.60), Inches(0.30),
             "problem", size=12, color=BLUE_TXT_CHIP, align=PP_ALIGN.CENTER)

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
        cx_c = int(ex - chip_w / 2); cy_c = int(ey - chip_h / 2)
        shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                 cx_c, cy_c, chip_w, chip_h)
        shp.fill.solid(); shp.fill.fore_color.rgb = color
        shp.line.fill.background(); shp.shadow.inherit = False
        tf = shp.text_frame
        tf.margin_left = Emu(0); tf.margin_right = Emu(0)
        tf.margin_top = Emu(0); tf.margin_bottom = Emu(0)
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = label
        r.font.name = FONT_BODY; r.font.size = Pt(12); r.font.bold = True
        r.font.color.rgb = WHITE
    add_notes(s,
        "The 'paper in a can' metaphor is the whole slide. Each satellite is a "
        "real Python attribute on a Problem object — when you say a name, "
        "say what it gives you. This is THE contract slide. Spend ~45 seconds.")


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
                          label="[IMAGE: EngiBench problems grid (assets/engibench_problems.png)]",
                          source="Already included in workshops/dcc26/assets/")
    add_text(s, Inches(0.65), Inches(6.65), Inches(12.05), Inches(0.32),
             "Each comes with a dataset, a simulator, and a baseline optimizer.",
             size=12.5, italic=True, color=GREY, align=PP_ALIGN.CENTER)
    add_notes(s,
        "Don't enumerate every problem — name the domains and move on. The "
        "image is the point. Mention briefly that there's a contribution path "
        "for new problems and that the next notebook walks them through one.")


def s_contract_code(prs):
    s = slide_blank(prs)
    add_chrome(s, "EngiBench")
    add_eyebrow(s, "THE CONTRACT, IN CODE")
    add_title(s, "Design-problem questions become Python calls.",
              size=26, height=0.7)

    code_x = Inches(0.65); code_y = Inches(2.05)
    code_w = Inches(7.85); code_h = Inches(4.50)
    add_rect(s, code_x, code_y, code_w, code_h, NEAR_BLACK)
    for i, c in enumerate([RED, ORANGE, GREEN]):
        dot = s.shapes.add_shape(MSO_SHAPE.OVAL,
                                 int(code_x + Inches(0.25 + i * 0.30)),
                                 int(code_y + Inches(0.20)),
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

    # right column: 3 typography callouts (no decorative icons)
    rx = Inches(8.85); rw = Inches(4.0); ry = Inches(2.20)
    callouts = [
        ("Same task contract",
         "Every paper compares on the same problem.", BLUE),
        ("Swap the model, not the task",
         "Your generator is one cell. The rest is fixed.", GREEN),
        ("Portable evidence",
         "Your numbers match a reviewer's numbers.", ORANGE),
    ]
    for title, body, color in callouts:
        add_rect(s, rx, ry, Inches(0.10), Inches(1.30), color)
        add_text(s, rx + Inches(0.30), ry + Inches(0.10), rw - Inches(0.4), Inches(0.40),
                 title, size=13.5, bold=True, color=color)
        add_text(s, rx + Inches(0.30), ry + Inches(0.55), rw - Inches(0.4), Inches(0.75),
                 body, size=11.5, color=GREY)
        ry += Inches(1.45)
    add_notes(s,
        "Don't read the code. Point at three lines (problem.check_constraints, "
        "problem.simulate, problem.optimize) and say what each gives you. The "
        "three callouts on the right are the takeaway claims — emphasise "
        "'portable evidence': numbers match across laptops.")


def s_what_today(prs):
    s = slide_blank(prs)
    add_chrome(s, "Workshop path")
    add_eyebrow(s, "FROM MOTIVATION TO PRACTICE")
    add_title(s, "Four notebooks. One contract.", size=28, height=0.7)
    add_text(s, Inches(0.65), Inches(1.55), Inches(12.0), Inches(0.40),
             "Run them in order. Then we come back together to discuss what is missing.",
             size=13, italic=True, color=GREY)

    notebooks = [
        ("00", "Frame", BLUE,
         "Inspect a benchmark contract end-to-end."),
        ("01", "Train", GREEN,
         "Fit a small conditional generator."),
        ("02", "Evaluate", ORANGE,
         "Score designs as engineering candidates."),
        ("03", "Extend", RED,
         "Wrap your own problem behind the same API."),
    ]
    y = Inches(2.30)
    h = Inches(4.00)
    n = len(notebooks)
    arrow_w = Inches(0.45)
    pad = Inches(0.20)
    total = Inches(13.33 - 1.30)
    station_w = (total - arrow_w * (n - 1) - pad * 2 * (n - 1)) / n
    x = Inches(0.65)
    for i, (num, name, color, body) in enumerate(notebooks):
        add_rect(s, x, y, station_w, h, WHITE)
        add_rect(s, x, y, station_w, Inches(0.10), color)
        add_text(s, x + Inches(0.30), y + Inches(0.30), station_w - Inches(0.6), Inches(0.40),
                 f"NOTEBOOK {num}", size=11, bold=True, color=color)
        add_text(s, x + Inches(0.30), y + Inches(0.80), station_w - Inches(0.6), Inches(0.60),
                 name, font=FONT_DISPLAY, size=26, bold=True, color=NEAR_BLACK)
        # small placeholder for a notebook screenshot
        image_placeholder(s, x + Inches(0.30), y + Inches(1.55),
                          station_w - Inches(0.60), Inches(1.55),
                          label=f"[SCREENSHOT: simple/0{i}_*.ipynb output]",
                          source="Source: capture from the notebook output cell")
        add_text(s, x + Inches(0.30), y + Inches(3.25), station_w - Inches(0.60), Inches(0.65),
                 body, size=11.5, color=GREY)
        x += station_w
        if i < n - 1:
            arrow = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW,
                                       int(x + pad),
                                       int(y + h / 2 - Inches(0.20)),
                                       arrow_w, Inches(0.40))
            arrow.fill.solid(); arrow.fill.fore_color.rgb = GREY_LIGHT
            arrow.line.fill.background(); arrow.shadow.inherit = False
            x += arrow_w + pad * 2

    add_kicker_band(s,
        "Then: what is missing, and where do we go next as a community?")
    add_notes(s,
        "Quick orientation, ~45 seconds. Don't go deep — they'll see it "
        "themselves. Mention runtime fallbacks: if training is slow, the "
        "evaluation notebook auto-rebuilds artifacts. The kicker reminds the "
        "audience that the second half is a real discussion, not a tutorial.")


def s_closing(prs):
    s = slide_blank(prs)
    add_rect(s, Emu(0), Emu(0), SLIDE_W, SLIDE_H, BG)
    add_rect(s, Emu(0), Emu(0), Inches(0.12), SLIDE_H, BLUE)
    add_text(s, Inches(0.65), Inches(0.65), Inches(8.0), Inches(0.30),
             "DCC 2026 WORKSHOP  ·  OPENING KEYNOTE",
             size=11, bold=True, color=BLUE)
    add_text(s, Inches(0.65), Inches(2.10), Inches(12.0), Inches(2.6),
             ["Benchmarks do not",
              "slow research down."],
             font=FONT_DISPLAY, size=52, bold=True, color=NEAR_BLACK,
             line_spacing=1.0)
    add_text(s, Inches(0.65), Inches(4.65), Inches(12.0), Inches(1.0),
             "They make it add up.",
             font=FONT_DISPLAY, size=52, bold=True, color=BLUE)
    add_rect(s, Inches(0.65), Inches(6.35), Inches(12.05), Inches(0.60), BLUE)
    add_text(s, Inches(0.85), Inches(6.50), Inches(11.8), Inches(0.30),
             "Up next  →  Notebook 00:  Frame your design problem as a benchmark contract.",
             size=14, bold=True, color=WHITE)
    add_notes(s,
        "Land the line. Long pause. Then redirect to Notebook 00 — physical "
        "transition (move from podium, change screens). Don't add more words "
        "here; the slide is the takeaway.")


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

    section_divider(slide_blank, prs,
        eyebrow="SECTION 2  —  LESSONS FROM OTHER FIELDS",
        title="What benchmarks have done\nelsewhere.",
        subtitle="Vision, language, and biology have all been here. Three case studies, one recipe.",
        notes="Section break. Brief pause. The next 4 slides step back and look at "
              "how vision, language, and biology each solved their version of this problem.")
    s_imagenet(prs)
    s_glue(prs)
    s_casp(prs)
    s_shared_dna(prs)

    section_divider(slide_blank, prs,
        eyebrow="SECTION 3  —  WHAT OUR FIELD NEEDS",
        title="What engineering design ML\nstill lacks.",
        subtitle="The gap is not theory or talent. It is shared, executable infrastructure.",
        notes="Now the punchline. What does our field still lack, and what does that imply "
              "we should build?")
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
    build_deck(here / "introduction-benchmarking-genai-engineering-design-final.pptx")
