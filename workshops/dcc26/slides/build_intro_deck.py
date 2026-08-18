"""Build the DCC'26 intro motivation deck.

Run:
    python3 workshops/dcc26/slides/build_intro_deck.py

Outputs:
    workshops/dcc26/slides/introduction-benchmarking-genai-engineering-design.pptx
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt, Emu


# ---------- design system ----------

BG = RGBColor(0xFB, 0xFA, 0xF7)
BLUE = RGBColor(0x22, 0x5E, 0x9B)
BLUE_LIGHT = RGBColor(0xD7, 0xE8, 0xF6)
BLUE_TXT_LIGHT = RGBColor(0xEA, 0xF3, 0xFA)
BLUE_TXT_CHIP = RGBColor(0xCF, 0xE3, 0xF3)
NEAR_BLACK = RGBColor(0x17, 0x21, 0x2B)
GREY = RGBColor(0x5A, 0x66, 0x75)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
RED = RGBColor(0xC8, 0x48, 0x37)
ORANGE = RGBColor(0xC4, 0x7B, 0x20)
GREEN = RGBColor(0x2F, 0x7D, 0x62)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

FONT_DISPLAY = "Aptos Display"
FONT_BODY = "Aptos"


# ---------- helpers ----------

def add_rect(slide, x, y, w, h, fill, line=None):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
    shp.shadow.inherit = False
    return shp


def add_text(slide, x, y, w, h, text, *, font=FONT_BODY, size=14, bold=False, color=NEAR_BLACK,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
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
        r = p.add_run()
        r.text = line
        r.font.name = font
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color
    return tb


def add_chrome(slide, section_label):
    """Background + left accent bar + footer."""
    add_rect(slide, Emu(0), Emu(0), SLIDE_W, SLIDE_H, BG)
    add_rect(slide, Emu(0), Emu(0), Inches(0.12), SLIDE_H, BLUE)
    add_text(slide, Inches(0.44), Inches(7.02), Inches(4.38), Inches(0.19),
             section_label, size=8.25, color=GREY)
    add_text(slide, Inches(8.54), Inches(7.02), Inches(4.06), Inches(0.19),
             "DCC 2026 workshop  |  EngiBench + EngiOpt",
             size=8.25, color=GREY, align=PP_ALIGN.RIGHT)


def add_eyebrow(slide, text):
    add_text(slide, Inches(0.65), Inches(0.48), Inches(8.0), Inches(0.23),
             text, size=9.75, bold=True, color=BLUE)


def add_title(slide, text, *, size=26, top=0.81, height=1.33):
    add_text(slide, Inches(0.65), Inches(top), Inches(11.04), Inches(height),
             text, font=FONT_DISPLAY, size=size, bold=True, color=NEAR_BLACK)


def add_lede(slide, text, *, top=2.23, height=0.7, size=14.25):
    add_text(slide, Inches(0.75), Inches(top), Inches(11.5), Inches(height),
             text, size=size, color=GREY)


def card(slide, x, y, w, h, *, num, num_color, title, body, title_color=None):
    """White card with colored number tag, title, body text."""
    add_rect(slide, x, y, w, h, WHITE)
    add_text(slide, x + Inches(0.25), y + Inches(0.31), Inches(0.54), Inches(0.35),
             num, size=19.5, bold=True, color=num_color)
    add_text(slide, x + Inches(0.90), y + Inches(0.35), w - Inches(1.0), Inches(0.38),
             title, size=14.25, bold=True, color=title_color or num_color)
    add_text(slide, x + Inches(0.27), y + Inches(0.98), w - Inches(0.4), h - Inches(1.1),
             body, size=10.5, color=GREY)


def chip_card(slide, x, y, w, h, *, year, title, body, color):
    """Card whose tag is a chip (year/era label) instead of a number."""
    add_rect(slide, x, y, w, h, WHITE)
    chip = add_rect(slide, x + Inches(0.25), y + Inches(0.30), Inches(1.20), Inches(0.36), color)
    add_text(slide, x + Inches(0.25), y + Inches(0.34), Inches(1.20), Inches(0.30),
             year, size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(slide, x + Inches(0.25), y + Inches(0.78), w - Inches(0.4), Inches(0.38),
             title, size=14, bold=True, color=NEAR_BLACK)
    add_text(slide, x + Inches(0.25), y + Inches(1.20), w - Inches(0.4), h - Inches(1.3),
             body, size=10.5, color=GREY)


def kicker_band(slide, text):
    """Blue band near bottom with one strong takeaway line."""
    add_rect(slide, Inches(1.31), Inches(5.92), Inches(10.71), Inches(0.65), BLUE)
    add_text(slide, Inches(1.75), Inches(6.11), Inches(9.83), Inches(0.28),
             text, size=15, bold=True, color=WHITE, align=PP_ALIGN.CENTER)


# ---------- slide builders ----------

def slide_blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def build_title(prs):
    s = slide_blank(prs)
    add_rect(s, Emu(0), Emu(0), SLIDE_W, SLIDE_H, BG)
    # blue lower panel
    add_rect(s, Emu(0), Inches(3.27), SLIDE_W, Inches(4.23), BLUE)
    add_text(s, Inches(0.81), Inches(3.71), Inches(8.0), Inches(0.25),
             "DCC 2026 Paris workshop  |  Opening keynote",
             size=12, bold=True, color=BLUE_TXT_CHIP)
    add_text(s, Inches(0.81), Inches(4.15), Inches(11.0), Inches(1.6),
             ["Benchmarking Generative AI",
              "for Engineering Design"],
             font=FONT_DISPLAY, size=38, bold=True, color=WHITE)
    add_text(s, Inches(0.85), Inches(5.85), Inches(11.0), Inches(0.50),
             "Why shared, executable contracts are the missing infrastructure for AI-driven design research.",
             size=17.25, color=BLUE_TXT_LIGHT)
    add_text(s, Inches(0.85), Inches(6.50), Inches(8.0), Inches(0.27),
             "Matthew Keeler, Soheyl Massoudi, Mark Fuge",
             size=13.5, bold=True, color=BLUE_TXT_LIGHT)
    add_text(s, Inches(0.85), Inches(6.80), Inches(8.0), Inches(0.23),
             "D-MAVT, ETH Zürich  ·  EngiBench + EngiOpt",
             size=11.25, color=BLUE_TXT_LIGHT)


def build_promise(prs):
    s = slide_blank(prs)
    add_chrome(s, "Why this matters")
    add_eyebrow(s, "The promise")
    add_title(s, "Generative AI is already rewriting how engineers design.", size=26, height=1.1)
    add_lede(s, "Across structures, propulsion, optics, and electronics, generative models now propose "
             "candidate designs that humans take seriously — and sometimes manufacture.", top=2.10, height=0.9)

    y = Inches(3.20)
    h = Inches(3.05)
    pad = Inches(0.18)
    w = Inches((13.33 - 0.65 - 0.65 - 0.36) / 3)
    x0 = Inches(0.65)
    x1 = x0 + w + pad
    x2 = x1 + w + pad

    # three example cards
    add_rect(s, x0, y, w, h, WHITE)
    add_rect(s, x0, y, w, Inches(0.55), BLUE)
    add_text(s, x0 + Inches(0.25), y + Inches(0.13), w - Inches(0.5), Inches(0.30),
             "STRUCTURAL", size=11, bold=True, color=WHITE)
    add_text(s, x0 + Inches(0.25), y + Inches(0.75), w - Inches(0.5), Inches(0.55),
             "Airbus A320 cabin partition", size=14, bold=True, color=NEAR_BLACK)
    add_text(s, x0 + Inches(0.25), y + Inches(1.30), w - Inches(0.5), h - Inches(1.5),
             "Generatively designed bionic partition, 45% lighter than the legacy "
             "part. Printed in titanium, certified, flown. A concrete proof that "
             "generative methods can clear aerospace bars.",
             size=11, color=GREY)

    add_rect(s, x1, y, w, h, WHITE)
    add_rect(s, x1, y, w, Inches(0.55), GREEN)
    add_text(s, x1 + Inches(0.25), y + Inches(0.13), w - Inches(0.5), Inches(0.30),
             "PROPULSION", size=11, bold=True, color=WHITE)
    add_text(s, x1 + Inches(0.25), y + Inches(0.75), w - Inches(0.5), Inches(0.55),
             "GE jet-engine bracket", size=14, bold=True, color=NEAR_BLACK)
    add_text(s, x1 + Inches(0.25), y + Inches(1.30), w - Inches(0.5), h - Inches(1.5),
             "Open challenge in 2013 reduced an engine bracket by 84% in mass with "
             "topology optimization. The legacy that triggered the ML-for-design wave "
             "we now ride.",
             size=11, color=GREY)

    add_rect(s, x2, y, w, h, WHITE)
    add_rect(s, x2, y, w, Inches(0.55), ORANGE)
    add_text(s, x2 + Inches(0.25), y + Inches(0.13), w - Inches(0.5), Inches(0.30),
             "PHOTONICS", size=11, bold=True, color=WHITE)
    add_text(s, x2 + Inches(0.25), y + Inches(0.75), w - Inches(0.5), Inches(0.55),
             "Inverse-designed metasurfaces", size=14, bold=True, color=NEAR_BLACK)
    add_text(s, x2 + Inches(0.25), y + Inches(1.30), w - Inches(0.5), h - Inches(1.5),
             "Stanford / Fan Group and others use gradient-based and learned inverse "
             "design to synthesize metasurfaces and photonic devices that beat "
             "hand-crafted baselines.",
             size=11, color=GREY)

    kicker_band(s, "These are not toy results — and they raise a follow-up question.")


def build_renaissance(prs):
    s = slide_blank(prs)
    add_chrome(s, "Why this matters")
    add_eyebrow(s, "A renaissance of methods")
    add_title(s, "Every year brings a new generative recipe for engineering design.", size=24, height=1.1)
    add_lede(s, "In under a decade the toolbox went from autoencoders to LLMs guiding solvers. "
             "Each method comes with a SOTA claim — and a slightly different setup.",
             top=2.10, height=0.9)

    # timeline cards
    y = Inches(3.30)
    h = Inches(2.95)
    pad = Inches(0.17)
    w = Inches((13.33 - 0.65 - 0.65 - 0.51) / 4)
    x = Inches(0.65)
    eras = [
        ("2017-2019", "GAN era", "Conditional GANs synthesize 2-D layouts and shapes; "
         "first 'inverse design with deep learning' papers appear.", BLUE),
        ("2019-2021", "VAE + cVAE era", "Variational autoencoders carry uncertainty, "
         "smooth latents, and learned design manifolds.", GREEN),
        ("2021-2024", "Diffusion era", "Score-based and conditional diffusion models "
         "dominate topology and shape generation benchmarks.", ORANGE),
        ("2024 →", "LLMs + agents", "Language models propose, critique, and run solvers; "
         "code-generating agents author entire design pipelines.", RED),
    ]
    for era, title, body, color in eras:
        chip_card(s, x, y, w, h, year=era, title=title, body=body, color=color)
        x += w + pad

    kicker_band(s, "The methods change quickly. The way we evaluate them barely changes at all.")


def build_honest_question(prs):
    s = slide_blank(prs)
    add_chrome(s, "Why this matters")
    add_eyebrow(s, "The honest question")
    add_title(s, "When a paper says \"we beat the SOTA on topology optimization\",\nwhat actually changed?",
              size=26, height=1.6, top=0.81)
    add_lede(s, "Pick any recent generative-design paper. Without leaving the abstract, you cannot tell which of these is true:",
             top=2.65, height=0.55)

    # four possibilities as horizontal pills
    y = Inches(3.55)
    h = Inches(2.6)
    pad = Inches(0.18)
    w = Inches((13.33 - 0.65 - 0.65 - 0.54) / 4)
    x = Inches(0.65)
    options = [
        ("A", "Better model", "A genuinely stronger generator that would also win on your problem.", BLUE),
        ("B", "Easier conditions", "A narrower or friendlier conditioning distribution than the comparison.", ORANGE),
        ("C", "Lower bar", "A weaker feasibility check or a coarser simulator than prior work.", RED),
        ("D", "All of the above", "Some mix of A, B, and C — which is what usually happens in practice.", GREY),
    ]
    for letter, title, body, color in options:
        add_rect(s, x, y, w, h, WHITE)
        # big letter
        add_text(s, x + Inches(0.25), y + Inches(0.28), Inches(0.6), Inches(0.55),
                 letter, font=FONT_DISPLAY, size=28, bold=True, color=color)
        add_text(s, x + Inches(0.25), y + Inches(0.95), w - Inches(0.5), Inches(0.38),
                 title, size=14, bold=True, color=color)
        add_text(s, x + Inches(0.25), y + Inches(1.40), w - Inches(0.5), h - Inches(1.55),
                 body, size=11, color=GREY)
        x += w + pad

    kicker_band(s, "Today's literature does not give us a way to know.")


def build_repro_crisis(prs):
    s = slide_blank(prs)
    add_chrome(s, "Why we cannot tell")
    add_eyebrow(s, "Background — ML's own crisis")
    add_title(s, "Mainstream machine learning had this fight first.", size=26, height=1.1)
    add_lede(s, "After a decade of \"SOTA\" claims that did not survive re-implementation, ML had to invent "
             "reproducibility infrastructure. Engineering design ML is now standing where ML stood in 2017.",
             top=2.10, height=1.0)

    # two stacked highlight rows
    y = Inches(3.35)
    h = Inches(1.30)
    add_rect(s, Inches(0.65), y, Inches(12.05), h, WHITE)
    add_text(s, Inches(0.95), y + Inches(0.22), Inches(2.3), Inches(0.38),
             "2017-2019", size=12, bold=True, color=BLUE)
    add_text(s, Inches(0.95), y + Inches(0.55), Inches(11.5), Inches(0.42),
             "NeurIPS adds a reproducibility checklist; ML Reproducibility Challenge launches.",
             size=14, bold=True, color=NEAR_BLACK)
    add_text(s, Inches(0.95), y + Inches(0.98), Inches(11.3), Inches(0.36),
             "Pineau et al. 2019 — independent replication of accepted RL papers found ~30% could not be reproduced "
             "from text alone.",
             size=11.5, color=GREY)

    y2 = y + h + Inches(0.20)
    add_rect(s, Inches(0.65), y2, Inches(12.05), h, WHITE)
    add_text(s, Inches(0.95), y2 + Inches(0.22), Inches(2.3), Inches(0.38),
             "2020 →", size=12, bold=True, color=GREEN)
    add_text(s, Inches(0.95), y2 + Inches(0.55), Inches(11.5), Inches(0.42),
             "Shared benchmarks (HuggingFace, OpenAI Gym, Papers-with-Code) become how progress is reported.",
             size=14, bold=True, color=NEAR_BLACK)
    add_text(s, Inches(0.95), y2 + Inches(0.98), Inches(11.3), Inches(0.36),
             "Method papers ship code, weights, and an evaluation harness by default. A claim without these is "
             "treated as an anecdote.",
             size=11.5, color=GREY)

    kicker_band(s, "Engineering design ML inherits all of this — and adds new failure modes.")


def build_design_is_harder(prs):
    s = slide_blank(prs)
    add_chrome(s, "Why we cannot tell")
    add_eyebrow(s, "Background — design is worse")
    add_title(s, "Engineering design adds degrees of freedom that ML never had to worry about.",
              size=24, height=1.3)
    add_lede(s, "A vision paper hides at most a handful of choices. A design paper hides far more — and any one "
             "of them can flip the conclusion.",
             top=2.30, height=0.7)

    y = Inches(3.30)
    h = Inches(1.45)
    pad_y = Inches(0.18)
    w = Inches(5.85)
    pad_x = Inches(0.20)

    items = [
        ("Simulator", "Which solver, which mesh, which tolerance? A FEM vs an analytical baseline are not the same evaluator.", BLUE),
        ("Constraints", "Volume fraction tolerance, manufacturability, disconnected-material penalties — all easy to silently relax.", RED),
        ("Conditions", "The distribution of loads, boundaries, scenarios. A narrower distribution makes any model look stronger.", ORANGE),
        ("Baselines", "Random search vs. SIMP topology optimizer vs. CMA-ES define very different bars to clear.", GREEN),
        ("Representation", "Pixels, meshes, B-splines, Bézier curves — the encoding determines what \"diversity\" even means.", BLUE),
        ("Units & scaling", "Compliance, stress, dB, dBi — silent normalization choices can move numbers by orders of magnitude.", ORANGE),
    ]
    for i, (title, body, color) in enumerate(items):
        col = i % 2
        row = i // 2
        x = Inches(0.65) + col * (w + pad_x)
        ypos = y + row * (h + pad_y)
        add_rect(s, x, ypos, w, h, WHITE)
        add_rect(s, x, ypos, Inches(0.10), h, color)
        add_text(s, x + Inches(0.30), ypos + Inches(0.20), w - Inches(0.5), Inches(0.36),
                 title, size=14.5, bold=True, color=color)
        add_text(s, x + Inches(0.30), ypos + Inches(0.60), w - Inches(0.5), h - Inches(0.75),
                 body, size=11, color=GREY)


def build_concrete_case(prs):
    s = slide_blank(prs)
    add_chrome(s, "Why we cannot tell")
    add_eyebrow(s, "A concrete case")
    add_title(s, "Two \"Beams2D\" papers. Same headline number. Different problem.",
              size=24, height=1.1)
    add_lede(s, "Imagine reading two recent papers that both report MSE = 0.04 on a Beams2D topology task. "
             "Are they comparable?",
             top=2.10, height=0.9)

    # two paper cards side by side, then a verdict bar
    y = Inches(3.20)
    h = Inches(2.75)
    w = Inches(5.95)
    pad = Inches(0.22)
    x_a = Inches(0.65)
    x_b = x_a + w + pad

    # paper A
    add_rect(s, x_a, y, w, h, WHITE)
    add_rect(s, x_a, y, w, Inches(0.50), BLUE)
    add_text(s, x_a + Inches(0.25), y + Inches(0.11), w - Inches(0.5), Inches(0.32),
             "PAPER A — \"Diffusion for topology\"", size=12, bold=True, color=WHITE)
    rows_a = [
        ("Volume fraction", "fixed at 0.50"),
        ("Filter radius", "1.5 px"),
        ("Feasibility check", "soft, mean penalty"),
        ("Solver tolerance", "1e-3"),
        ("Test conditions", "near training mean"),
    ]
    ry = y + Inches(0.70)
    for k, v in rows_a:
        add_text(s, x_a + Inches(0.25), ry, Inches(2.4), Inches(0.32),
                 k, size=11.5, bold=True, color=NEAR_BLACK)
        add_text(s, x_a + Inches(2.70), ry, w - Inches(2.95), Inches(0.32),
                 v, size=11.5, color=GREY)
        ry += Inches(0.38)

    # paper B
    add_rect(s, x_b, y, w, h, WHITE)
    add_rect(s, x_b, y, w, Inches(0.50), ORANGE)
    add_text(s, x_b + Inches(0.25), y + Inches(0.11), w - Inches(0.5), Inches(0.32),
             "PAPER B — \"cVAE for inverse design\"", size=12, bold=True, color=WHITE)
    rows_b = [
        ("Volume fraction", "sampled in [0.30, 0.55]"),
        ("Filter radius", "2.5 px"),
        ("Feasibility check", "hard, reject if violated"),
        ("Solver tolerance", "1e-5"),
        ("Test conditions", "out-of-distribution corners"),
    ]
    ry = y + Inches(0.70)
    for k, v in rows_b:
        add_text(s, x_b + Inches(0.25), ry, Inches(2.4), Inches(0.32),
                 k, size=11.5, bold=True, color=NEAR_BLACK)
        add_text(s, x_b + Inches(2.70), ry, w - Inches(2.95), Inches(0.32),
                 v, size=11.5, color=GREY)
        ry += Inches(0.38)

    kicker_band(s, "Same task name. Same MSE. Not the same benchmark.")


def build_failure_modes(prs):
    s = slide_blank(prs)
    add_chrome(s, "Why we cannot tell")
    add_eyebrow(s, "Three patterns of hidden failure")
    add_title(s, "Without a shared contract, three failure modes recur.",
              size=26, height=1.1)
    add_lede(s, "These are not strawmen — they are the patterns reviewers see most often when AI-for-design "
             "papers cannot be reproduced or compared.",
             top=2.10, height=0.9)

    y = Inches(3.27)
    h = Inches(2.6)
    pad = Inches(0.18)
    w = Inches((13.33 - 0.65 - 0.65 - 0.36) / 3)
    x0 = Inches(0.65)
    x1 = x0 + w + pad
    x2 = x1 + w + pad

    card(s, x0, y, w, h, num="01", num_color=RED,
         title="Cherry-picked conditions",
         body="Models are evaluated only on conditions near the training mean. The paper looks strong; "
              "the generator collapses on edge cases that engineers actually care about.")
    card(s, x1, y, w, h, num="02", num_color=ORANGE,
         title="Visual-only evaluation",
         body="Generated designs look convincing in a grid and the paper stops there. Run them through "
              "the simulator and a large fraction violate constraints or under-perform a baseline.")
    card(s, x2, y, w, h, num="03", num_color=BLUE,
         title="One-number scores",
         body="A single accuracy or MSE figure hides feasibility rate, diversity, novelty, and warm-start "
              "value. Different choices among these can flip which method \"wins\".")


def build_divider(prs, *, eyebrow, title, subtitle):
    s = slide_blank(prs)
    add_rect(s, Emu(0), Emu(0), SLIDE_W, SLIDE_H, BLUE)
    add_text(s, Inches(0.81), Inches(2.85), Inches(11.5), Inches(0.36),
             eyebrow, size=14, bold=True, color=BLUE_TXT_CHIP)
    add_text(s, Inches(0.81), Inches(3.35), Inches(11.5), Inches(1.6),
             title, font=FONT_DISPLAY, size=40, bold=True, color=WHITE)
    add_text(s, Inches(0.85), Inches(5.20), Inches(11.0), Inches(0.55),
             subtitle, size=17, color=BLUE_TXT_LIGHT)


def build_imagenet(prs):
    s = slide_blank(prs)
    add_chrome(s, "Lessons from other fields")
    add_eyebrow(s, "Lesson 1 — ImageNet")
    add_title(s, "One labeled dataset became the engine of a decade of compounding progress.",
              size=24, height=1.3)
    add_lede(s, "1.2M labeled images + an annual challenge with a frozen evaluation gave the field a "
             "way to measure progress that everyone trusted.",
             top=2.30, height=0.85)

    # chart-like visualization: bars showing top-5 error dropping
    chart_x = Inches(0.85)
    chart_y = Inches(3.55)
    chart_w = Inches(11.65)
    chart_h = Inches(2.50)
    add_rect(s, chart_x, chart_y, chart_w, chart_h, WHITE)

    add_text(s, chart_x + Inches(0.30), chart_y + Inches(0.18), Inches(8.0), Inches(0.30),
             "ImageNet top-5 classification error", size=12.5, bold=True, color=NEAR_BLACK)
    add_text(s, chart_x + Inches(0.30), chart_y + Inches(0.45), Inches(10.0), Inches(0.26),
             "Years where a single benchmark let everyone agree which method actually moved the field.",
             size=10.5, color=GREY)

    # bars
    years = [(2010, 28.2), (2011, 25.8), (2012, 16.4), (2013, 11.7), (2014, 6.7),
             (2015, 3.6), (2016, 3.0), (2017, 2.3)]
    bar_area_x = chart_x + Inches(0.50)
    bar_area_y = chart_y + Inches(0.90)
    bar_area_w = chart_w - Inches(1.00)
    bar_area_h = Inches(1.40)
    n = len(years)
    slot_w = bar_area_w / n
    bar_w = slot_w * 0.55
    max_v = 30.0
    for i, (yr, val) in enumerate(years):
        h = bar_area_h * (val / max_v)
        bx = bar_area_x + slot_w * i + (slot_w - bar_w) / 2
        by = bar_area_y + (bar_area_h - h)
        color = BLUE if i < 2 else (RED if val > 5 else GREEN)
        add_rect(s, bx, by, bar_w, h, color)
        add_text(s, bar_area_x + slot_w * i, by - Inches(0.30), slot_w, Inches(0.28),
                 f"{val:.1f}%", size=9.5, bold=True, color=NEAR_BLACK, align=PP_ALIGN.CENTER)
        add_text(s, bar_area_x + slot_w * i, bar_area_y + bar_area_h + Inches(0.05),
                 slot_w, Inches(0.25),
                 str(yr), size=9.5, color=GREY, align=PP_ALIGN.CENTER)

    # annotation arrows
    add_text(s, bar_area_x + slot_w * 2 - Inches(0.30), chart_y + Inches(0.95),
             Inches(2.0), Inches(0.30),
             "AlexNet (2012)", size=10, bold=True, color=RED)

    kicker_band(s, "A frozen evaluation made breakthroughs unambiguous.")


def build_glue(prs):
    s = slide_blank(prs)
    add_chrome(s, "Lessons from other fields")
    add_eyebrow(s, "Lesson 2 — GLUE / SuperGLUE")
    add_title(s, "When a single task saturates, a suite of tasks sustains the conversation.",
              size=24, height=1.3)
    add_lede(s, "Language understanding evaluation grew from one task to nine, then to a harder "
             "follow-on. Benchmarks have a lifecycle, and that is healthy.",
             top=2.30, height=0.85)

    y = Inches(3.35)
    h = Inches(2.7)
    pad = Inches(0.22)
    w = Inches((13.33 - 0.65 - 0.65 - 0.44) / 3)
    x = Inches(0.65)
    chip_card(s, x, y, w, h, year="2018", title="GLUE", color=BLUE,
              body="Nine sentence-level tasks collected under one leaderboard. Models had to be "
                   "broadly good, not narrowly tuned.")
    x += w + pad
    chip_card(s, x, y, w, h, year="2019", title="SuperGLUE", color=GREEN,
              body="When BERT-class models saturated GLUE in a year, the community designed harder tasks. "
                   "The benchmark evolved with the methods.")
    x += w + pad
    chip_card(s, x, y, w, h, year="LESSON", title="Suites, not snapshots", color=ORANGE,
              body="No single number captures \"language understanding\" — or \"design quality\". "
                   "Engineering design ML needs a suite, not one beam problem.")


def build_casp(prs):
    s = slide_blank(prs)
    add_chrome(s, "Lessons from other fields")
    add_eyebrow(s, "Lesson 3 — CASP and AlphaFold")
    add_title(s, "A 25-year community benchmark turned a breakthrough into a fact.",
              size=24, height=1.3)
    add_lede(s, "Critical Assessment of Structure Prediction (CASP) runs every two years with held-out "
             "protein structures, blind evaluation, and a shared scoring rubric.",
             top=2.30, height=0.85)

    y = Inches(3.35)
    h = Inches(2.7)
    pad = Inches(0.22)
    w = Inches((13.33 - 0.65 - 0.65 - 0.44) / 3)
    x = Inches(0.65)
    chip_card(s, x, y, w, h, year="1994 →", title="The shared rubric", color=BLUE,
              body="Held-out targets, blind submissions, GDT-TS scoring. The same protocol for 25 years "
                   "let the field compare across very different methods.")
    x += w + pad
    chip_card(s, x, y, w, h, year="CASP14", title="AlphaFold 2 verified", color=GREEN,
              body="DeepMind's claims landed on a benchmark biologists already trusted. The community "
                   "could see exactly how much better AlphaFold was, and on what.")
    x += w + pad
    chip_card(s, x, y, w, h, year="LESSON", title="Trust takes infrastructure", color=ORANGE,
              body="Without CASP, AlphaFold would have been a press release. With it, it was a verdict. "
                   "Engineering design needs the equivalent.")


def build_shared_dna(prs):
    s = slide_blank(prs)
    add_chrome(s, "Lessons from other fields")
    add_eyebrow(s, "What good benchmarks share")
    add_title(s, "Four ingredients turn a dataset into a benchmark — and a benchmark into a community.",
              size=24, height=1.3)
    add_lede(s, "ImageNet, GLUE, CASP, MuJoCo/Gym, OGB. Different communities, same recipe.",
             top=2.30, height=0.55)

    y = Inches(3.20)
    h = Inches(2.95)
    pad = Inches(0.18)
    w = Inches((13.33 - 0.65 - 0.65 - 0.54) / 4)
    x = Inches(0.65)
    items = [
        ("1", "Fixed inputs", "Everyone trains and evaluates on the same conditions, splits, and "
         "constraints. No silent re-curating.", BLUE),
        ("2", "Standardized evaluation", "One shared, executable scoring pipeline. The verdict is "
         "code, not a paragraph.", GREEN),
        ("3", "Open access", "Anyone can download, run, and submit. Low cost of entry is what "
         "creates a market for ideas.", ORANGE),
        ("4", "Community ownership", "Benchmarks evolve with their fields. Maintenance, versioning, "
         "and challenge rounds keep them alive.", RED),
    ]
    for num, title, body, color in items:
        card(s, x, y, w, h, num=num, num_color=color, title=title, body=body)
        x += w + pad

    kicker_band(s, "Engineering design ML has almost none of this — yet.")


def build_five_missing(prs):
    s = slide_blank(prs)
    add_chrome(s, "What the field needs")
    add_eyebrow(s, "What engineering design ML lacks")
    add_title(s, "Five pieces are missing from how we currently work.",
              size=26, height=1.1)
    add_lede(s, "Each missing piece is something a determined researcher can rebuild — but they should not "
             "have to. Shared infrastructure is faster than parallel effort.",
             top=2.10, height=0.9)

    y = Inches(3.35)
    h = Inches(1.35)
    pad_y = Inches(0.16)
    w = Inches(5.85)
    pad_x = Inches(0.20)
    items = [
        ("Standard problems", "A small, curated set of well-defined design tasks the community returns to.", BLUE),
        ("Shared simulators", "Installable, deterministic, and citable. The verdict has to run on any laptop.", GREEN),
        ("Curated datasets", "Optimized designs, conditions, and metadata — already split for fair evaluation.", ORANGE),
        ("Multi-faceted metrics", "Feasibility, performance gap, diversity, novelty, warm-start. Not one number.", RED),
        ("Reproducible runners", "Same harness for training and evaluation, scriptable and versioned.", BLUE),
        ("Contribution path", "A way for new problems and new domains to enter without forking the framework.", GREEN),
    ]
    for i, (title, body, color) in enumerate(items):
        col = i % 2
        row = i // 2
        x = Inches(0.65) + col * (w + pad_x)
        ypos = y + row * (h + pad_y)
        add_rect(s, x, ypos, w, h, WHITE)
        add_rect(s, x, ypos, Inches(0.10), h, color)
        add_text(s, x + Inches(0.30), ypos + Inches(0.20), w - Inches(0.5), Inches(0.36),
                 title, size=14, bold=True, color=color)
        add_text(s, x + Inches(0.30), ypos + Inches(0.60), w - Inches(0.5), h - Inches(0.75),
                 body, size=10.5, color=GREY)


def build_multifaceted(prs):
    s = slide_blank(prs)
    add_chrome(s, "What the field needs")
    add_eyebrow(s, "Evaluation, properly scoped")
    add_title(s, "Engineering quality is multi-faceted. A single scalar will betray you.",
              size=24, height=1.3)
    add_lede(s, "Each axis below can flip a paper's conclusion. Reporting only one is the design-ML "
             "equivalent of reporting only training loss.",
             top=2.30, height=0.85)

    # five horizontal bars with explanation
    y = Inches(3.30)
    items = [
        ("Feasibility", "What fraction of generated designs actually satisfy the constraints?", BLUE),
        ("Performance gap", "How does the simulated objective compare to the optimizer baseline?", GREEN),
        ("Diversity", "Does the generator explore the design space, or collapse to one mode?", ORANGE),
        ("Novelty", "Are generated designs new, or paraphrases of the training set?", RED),
        ("Warm-start utility", "Does the generator make a downstream optimizer faster, even when imperfect?", BLUE),
    ]
    h = Inches(0.65)
    pad_y = Inches(0.08)
    x = Inches(0.65)
    w = Inches(12.05)
    for title, body, color in items:
        add_rect(s, x, y, w, h, WHITE)
        add_rect(s, x, y, Inches(0.10), h, color)
        add_text(s, x + Inches(0.30), y + Inches(0.08), Inches(3.1), Inches(0.48),
                 title, size=14.5, bold=True, color=color, anchor=MSO_ANCHOR.MIDDLE)
        add_text(s, x + Inches(3.50), y + Inches(0.08), w - Inches(3.7), Inches(0.48),
                 body, size=12, color=GREY, anchor=MSO_ANCHOR.MIDDLE)
        y += h + pad_y


def build_data_not_enough(prs):
    s = slide_blank(prs)
    add_chrome(s, "What the field needs")
    add_eyebrow(s, "Datasets alone do not benchmark")
    add_title(s, "\"We released the dataset\" is necessary, but not enough.",
              size=26, height=1.1)
    add_lede(s, "If you cannot rerun the simulator and the scoring pipeline, you can display generated "
             "designs — you cannot score them.",
             top=2.10, height=0.9)

    # split: dataset-only vs dataset+pipeline
    y = Inches(3.20)
    h = Inches(2.95)
    pad = Inches(0.22)
    w = Inches(5.95)
    x_a = Inches(0.65)
    x_b = x_a + w + pad

    add_rect(s, x_a, y, w, h, WHITE)
    add_rect(s, x_a, y, w, Inches(0.50), RED)
    add_text(s, x_a + Inches(0.25), y + Inches(0.11), w - Inches(0.5), Inches(0.32),
             "DATASET ONLY", size=12, bold=True, color=WHITE)
    bullets_a = [
        ("Inspect", "Can plot ground-truth designs."),
        ("Train", "Can fit a model to recover them."),
        ("Compare", "Cannot. Reviewers re-derive their own metrics."),
        ("Reproduce", "Hard. Every group rebuilds the pipeline differently."),
    ]
    ry = y + Inches(0.75)
    for k, v in bullets_a:
        add_text(s, x_a + Inches(0.25), ry, Inches(1.7), Inches(0.32),
                 k, size=12, bold=True, color=NEAR_BLACK)
        add_text(s, x_a + Inches(2.00), ry, w - Inches(2.25), Inches(0.5),
                 v, size=11.5, color=GREY)
        ry += Inches(0.50)

    add_rect(s, x_b, y, w, h, WHITE)
    add_rect(s, x_b, y, w, Inches(0.50), GREEN)
    add_text(s, x_b + Inches(0.25), y + Inches(0.11), w - Inches(0.5), Inches(0.32),
             "DATASET + SIMULATOR + SCORING", size=12, bold=True, color=WHITE)
    bullets_b = [
        ("Inspect", "Plot, and verify against ground truth."),
        ("Train", "Fit a model with consistent conditions and splits."),
        ("Compare", "Run the shared evaluator and submit a number."),
        ("Reproduce", "One install, one command. The verdict is portable."),
    ]
    ry = y + Inches(0.75)
    for k, v in bullets_b:
        add_text(s, x_b + Inches(0.25), ry, Inches(1.7), Inches(0.32),
                 k, size=12, bold=True, color=NEAR_BLACK)
        add_text(s, x_b + Inches(2.00), ry, w - Inches(2.25), Inches(0.5),
                 v, size=11.5, color=GREY)
        ry += Inches(0.50)


def build_engibench_divider(prs):
    build_divider(
        prs,
        eyebrow="What we built, and why we are here",
        title="Enter EngiBench.",
        subtitle="A standardized framework for benchmarking generative AI in engineering design.  "
                 "Published at NeurIPS 2025.",
    )


def build_paper_in_a_can(prs):
    s = slide_blank(prs)
    add_chrome(s, "EngiBench")
    add_eyebrow(s, "The design pattern")
    add_title(s, "Each EngiBench problem is a \"paper in a can\".",
              size=26, height=1.1)
    add_lede(s, "One problem bundles every choice you would otherwise have to dig out of a methods section, "
             "and exposes them behind one Python interface.",
             top=2.10, height=0.9)

    # 4x2 grid of contract pieces
    parts = [
        ("Design space", "What values may a design take.", BLUE),
        ("Conditions", "What scenarios it must work under.", GREEN),
        ("Objectives", "What better means, quantitatively.", ORANGE),
        ("Constraints", "When a candidate is invalid.", RED),
        ("Dataset", "Curated examples with metadata.", BLUE),
        ("Render", "How to look at a design.", GREEN),
        ("Simulate", "How to score a design.", ORANGE),
        ("Optimize", "What classical baseline to beat.", RED),
    ]
    y = Inches(3.30)
    h = Inches(1.30)
    pad_y = Inches(0.18)
    w = Inches((13.33 - 0.65 - 0.65 - 0.60) / 4)
    pad_x = Inches(0.20)
    for i, (title, body, color) in enumerate(parts):
        col = i % 4
        row = i // 4
        x = Inches(0.65) + col * (w + pad_x)
        ypos = y + row * (h + pad_y)
        add_rect(s, x, ypos, w, h, WHITE)
        add_rect(s, x, ypos, w, Inches(0.10), color)
        add_text(s, x + Inches(0.22), ypos + Inches(0.22), w - Inches(0.4), Inches(0.40),
                 title, size=14, bold=True, color=color)
        add_text(s, x + Inches(0.22), ypos + Inches(0.66), w - Inches(0.4), h - Inches(0.78),
                 body, size=11, color=GREY)


def build_coverage(prs):
    s = slide_blank(prs)
    add_chrome(s, "EngiBench")
    add_eyebrow(s, "Coverage today")
    add_title(s, "Problems across structural, thermal, aerodynamic, photonic, and electronic design.",
              size=24, height=1.3)
    add_lede(s, "EngiBench ships a curated catalog of problems. Each comes with a dataset, a simulator, and "
             "a baseline optimizer — ready to compare against.",
             top=2.30, height=0.85)

    img = Path(__file__).resolve().parent.parent / "assets" / "engibench_problems.png"
    if img.exists():
        s.shapes.add_picture(str(img), Inches(1.10), Inches(3.40),
                             width=Inches(11.10), height=Inches(2.95))
    else:
        add_rect(s, Inches(1.10), Inches(3.40), Inches(11.10), Inches(2.95), WHITE)
        add_text(s, Inches(1.10), Inches(4.75), Inches(11.10), Inches(0.40),
                 "[engibench_problems.png — coverage figure]",
                 size=14, color=GREY, align=PP_ALIGN.CENTER)


def build_contract_code(prs):
    s = slide_blank(prs)
    add_chrome(s, "EngiBench")
    add_eyebrow(s, "The contract, in code")
    add_title(s, "From design-problem questions to Python calls.",
              size=26, height=1.1)
    add_lede(s, "The same eight calls work across every EngiBench problem. Your generator plugs into one "
             "interface; your evaluation becomes portable.",
             top=2.10, height=0.65)

    # code block on the left
    code_x = Inches(0.65)
    code_y = Inches(3.05)
    code_w = Inches(7.30)
    code_h = Inches(3.55)
    add_rect(s, code_x, code_y, code_w, code_h, NEAR_BLACK)
    code_lines = [
        "from engibench.utils.all_problems import BUILTIN_PROBLEMS",
        "",
        "problem = BUILTIN_PROBLEMS['beams2d'](seed=7)",
        "",
        "ds        = problem.dataset                # curated examples",
        "conds     = problem.conditions_keys        # scenario knobs",
        "design    = my_generator.sample(conditions)",
        "",
        "ok        = problem.check_constraints(design, conds)",
        "score     = problem.simulate(design, conds)",
        "baseline  = problem.optimize(starting=design, config=conds)",
        "problem.render(design)",
    ]
    tb = s.shapes.add_textbox(code_x + Inches(0.30), code_y + Inches(0.25),
                              code_w - Inches(0.5), code_h - Inches(0.5))
    tf = tb.text_frame
    tf.word_wrap = False
    tf.margin_left = Emu(0)
    tf.margin_right = Emu(0)
    tf.margin_top = Emu(0)
    tf.margin_bottom = Emu(0)
    for i, line in enumerate(code_lines):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        r = p.add_run()
        r.text = line if line else " "
        r.font.name = "Consolas"
        r.font.size = Pt(13)
        r.font.color.rgb = WHITE

    # callouts on the right
    side_x = Inches(8.20)
    side_y = Inches(3.05)
    side_w = Inches(4.50)
    rows = [
        ("Single source of truth", "Every paper that uses this contract is comparing on the same task."),
        ("Swap the model, not the task", "Your generator becomes one cell; the rest of the pipeline is fixed."),
        ("Portable evidence", "Numbers from your laptop match numbers from a reviewer's laptop."),
    ]
    rh = Inches(1.10)
    pad = Inches(0.10)
    for title, body in rows:
        add_rect(s, side_x, side_y, side_w, rh, WHITE)
        add_rect(s, side_x, side_y, Inches(0.10), rh, BLUE)
        add_text(s, side_x + Inches(0.25), side_y + Inches(0.16), side_w - Inches(0.4), Inches(0.32),
                 title, size=13, bold=True, color=BLUE)
        add_text(s, side_x + Inches(0.25), side_y + Inches(0.50), side_w - Inches(0.4), rh - Inches(0.6),
                 body, size=11, color=GREY)
        side_y += rh + pad


def build_what_today(prs):
    s = slide_blank(prs)
    add_chrome(s, "Workshop path")
    add_eyebrow(s, "From motivation to practice")
    add_title(s, "What you will do in the next three hours.",
              size=26, height=1.1)
    add_lede(s, "Four short notebooks turn the contract into muscle memory, then we come back together to "
             "discuss what the field needs next.",
             top=2.10, height=0.9)

    y = Inches(3.20)
    h = Inches(2.95)
    pad = Inches(0.18)
    w = Inches((13.33 - 0.65 - 0.65 - 0.54) / 4)
    x = Inches(0.65)
    notebooks = [
        ("00", "Frame", "Frame an engineering design problem as a benchmark. Load Beams2D, inspect "
         "the contract, render and check.", BLUE),
        ("01", "Train", "Train a small conditional generator on optimizer answers and export "
         "evaluation-ready artifacts.", GREEN),
        ("02", "Evaluate", "Run feasibility, performance, diversity, and warm-start evaluations. "
         "See how visual quality and engineering quality diverge.", ORANGE),
        ("03", "Extend", "Write your own minimal EngiBench-style problem so this contract works for "
         "your domain.", RED),
    ]
    for num, title, body, color in notebooks:
        card(s, x, y, w, h, num=num, num_color=color, title=title, body=body)
        x += w + pad

    kicker_band(s, "Then we discuss: what is missing, and where do we go next as a community?")


def build_closing(prs):
    s = slide_blank(prs)
    add_rect(s, Emu(0), Emu(0), SLIDE_W, SLIDE_H, BG)
    add_rect(s, Emu(0), Emu(0), Inches(0.12), SLIDE_H, BLUE)
    add_text(s, Inches(0.65), Inches(0.65), Inches(8.0), Inches(0.30),
             "DCC 2026 workshop  ·  Opening keynote",
             size=12, bold=True, color=BLUE)
    add_text(s, Inches(0.65), Inches(2.10), Inches(12.0), Inches(2.4),
             ["Benchmarks do not slow",
              "research down.",
              "They make it add up."],
             font=FONT_DISPLAY, size=46, bold=True, color=NEAR_BLACK)
    add_text(s, Inches(0.70), Inches(5.05), Inches(12.0), Inches(0.55),
             "For engineering design ML to compound, the task contract has to be as portable as the model.",
             size=17, color=GREY)
    # next-up bar
    add_rect(s, Inches(0.65), Inches(6.20), Inches(12.05), Inches(0.65), BLUE)
    add_text(s, Inches(0.95), Inches(6.38), Inches(11.5), Inches(0.30),
             "Up next  →  Notebook 00: Frame your design problem as a benchmark contract.",
             size=14, bold=True, color=WHITE)


# ---------- main ----------

def build_deck(out_path: Path):
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    build_title(prs)
    build_promise(prs)
    build_renaissance(prs)
    build_honest_question(prs)
    build_repro_crisis(prs)
    build_design_is_harder(prs)
    build_concrete_case(prs)
    build_failure_modes(prs)
    build_divider(prs, eyebrow="Section 2 — Lessons from other fields",
                  title="What benchmarks have done elsewhere.",
                  subtitle="Vision, language, and biology have all been here before. Three case studies, one recipe.")
    build_imagenet(prs)
    build_glue(prs)
    build_casp(prs)
    build_shared_dna(prs)
    build_divider(prs, eyebrow="Section 3 — What our field needs",
                  title="What engineering design ML still lacks.",
                  subtitle="The gap is not theory or talent. It is shared, executable infrastructure.")
    build_five_missing(prs)
    build_multifaceted(prs)
    build_data_not_enough(prs)
    build_engibench_divider(prs)
    build_paper_in_a_can(prs)
    build_coverage(prs)
    build_contract_code(prs)
    build_what_today(prs)
    build_closing(prs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out_path)
    print(f"Wrote {out_path} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    build_deck(here / "introduction-benchmarking-genai-engineering-design.pptx")
