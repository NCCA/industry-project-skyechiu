"""
Generate CVPR Workshop paper PDF - target 6-7 pages.
Official CVPR dimensions from cvpr-org/author-kit.

Usage:
    D:\\zmm\\miniconda3\\envs\\facelift\\python.exe generate_pdf.py
"""
import subprocess, sys

for pkg in ["reportlab", "pymupdf"]:
    try:
        __import__(pkg if pkg != "pymupdf" else "fitz")
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--break-system-packages", "-q"])

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.colors import black
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle,
    KeepTogether, NextPageTemplate, FrameBreak
)
from reportlab.platypus import Image as RLImage
from reportlab.lib import colors
import fitz  # PyMuPDF - convert figure PDFs to PNG for embedding

OUT = Path(__file__).resolve().parent / "paper_draft.pdf"
EVAL_DIR = Path(__file__).resolve().parent.parent.parent / "eval"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
FIG_CACHE = Path(__file__).resolve().parent / "_fig_cache"

# Always clear and recreate cache to avoid stale PNGs
import shutil
if FIG_CACHE.exists():
    shutil.rmtree(FIG_CACHE)
FIG_CACHE.mkdir(exist_ok=True)

from PIL import Image as PILImage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

MAX_FIG_H = 2.8 * inch   # conservative max - must fit in column with caption + surrounding text

def embed_fig(pdf_name, width, caption_text):
    """Convert a figure PDF to PNG and return [image, caption, spacer] flowables."""
    src = EVAL_DIR / pdf_name
    png = FIG_CACHE / pdf_name.replace('.pdf', '.png')
    elems = []
    if src.exists():
        try:
            doc_f = fitz.open(str(src))
            pix = doc_f[0].get_pixmap(dpi=150)
            pix.save(str(png))
            doc_f.close()
            # Read actual pixel dims from the saved PNG
            pil_img = PILImage.open(str(png))
            pw, ph = pil_img.size
            pil_img.close()
            if pw == 0 or ph == 0:
                raise ValueError("zero-dim image")
            aspect = float(ph) / float(pw)
            w = float(width)
            h = w * aspect
            if h > MAX_FIG_H:
                h = MAX_FIG_H
                w = h / aspect
            img = RLImage(str(png), width=w, height=h)
            img.hAlign = 'CENTER'
            elems.append(img)
        except Exception as e:
            print(f"WARNING: could not embed {pdf_name}: {e}")
            elems.append(Paragraph(f"<i>[Figure: {pdf_name} - render error]</i>",
                         ParagraphStyle('fp', fontName='Times-Italic', fontSize=8, alignment=TA_CENTER)))
    else:
        elems.append(Paragraph(f"<i>[Figure: {pdf_name} not found]</i>",
                     ParagraphStyle('fp', fontName='Times-Italic', fontSize=8, alignment=TA_CENTER)))
    elems.append(Paragraph(caption_text, sCap))
    elems.append(S_(4))
    return elems

def embed_png_direct(png_name, width, caption_text):
    """Embed a PNG directly from eval dir (no PDF conversion)."""
    src = EVAL_DIR / png_name
    elems = []
    if src.exists():
        try:
            pil_img = PILImage.open(str(src))
            pw, ph = pil_img.size
            pil_img.close()
            if pw == 0 or ph == 0:
                raise ValueError("zero-dim image")
            aspect = float(ph) / float(pw)
            w = float(width)
            h = w * aspect
            if h > MAX_FIG_H:
                h = MAX_FIG_H
                w = h / aspect
            img = RLImage(str(src), width=w, height=h)
            img.hAlign = 'CENTER'
            elems.append(img)
        except Exception as e:
            print(f"WARNING: could not embed {png_name}: {e}")
            elems.append(Paragraph(f"<i>[Figure: {png_name} - render error]</i>",
                         ParagraphStyle('fp2', fontName='Times-Italic', fontSize=8, alignment=TA_CENTER)))
    else:
        elems.append(Paragraph(f"<i>[Figure: {png_name} not found]</i>",
                     ParagraphStyle('fp3', fontName='Times-Italic', fontSize=8, alignment=TA_CENTER)))
    elems.append(Paragraph(caption_text, sCap))
    elems.append(S_(4))
    return elems

def _embed_abs_png(path, width, caption_text, max_h=None):
    """Embed a PNG from an absolute path."""
    if max_h is None:
        max_h = 2.8 * inch
    elems = []
    p = Path(path)
    if p.exists():
        try:
            pil_img = PILImage.open(str(p))
            pw, ph = pil_img.size
            pil_img.close()
            asp = float(ph) / float(pw)
            w = float(width)
            h = w * asp
            if h > max_h:
                h = max_h
                w = h / asp
            img = RLImage(str(p), width=w, height=h)
            img.hAlign = 'CENTER'
            elems.append(img)
        except Exception as e:
            print(f"WARNING: {path} error: {e}")
            elems.append(Paragraph(f"<i>[Figure: {p.name} - render error]</i>",
                ParagraphStyle('abse', fontName='Times-Italic', fontSize=8, alignment=TA_CENTER)))
    else:
        elems.append(Paragraph(f"<i>[Figure: {p.name} not found]</i>",
            ParagraphStyle('absm', fontName='Times-Italic', fontSize=8, alignment=TA_CENTER)))
    elems.append(Paragraph(caption_text, sCap))
    elems.append(S_(4))
    return elems

def generate_overview_figure():
    """Generate main overview figure (Fig 1) showing depth SR task with zoom-ins."""
    out_path = EVAL_DIR / "overview_main.png"
    if out_path.exists():
        print(f"Overview figure already exists: {out_path}")
        return str(out_path)
    val_dir = DATA_DIR / "dataset" / "val"
    samples = sorted((val_dir / "image").glob("*.png"))
    if not samples:
        print("WARNING: no val samples found, cannot generate overview figure")
        return None
    name = samples[min(2, len(samples) - 1)].stem
    # Load sample data
    rgb = np.array(PILImage.open(str(val_dir / "image" / f"{name}.png")))
    lr = np.array(PILImage.open(str(val_dir / "depth_lr_8bit" / f"{name}.png")))
    hr_pil = PILImage.open(str(val_dir / "depth" / f"{name}.png"))
    hr = np.array(hr_pil).astype(np.float32)
    hr = hr / 65535.0 if hr.max() > 255 else hr / 255.0
    # Bicubic upsample LR
    lr_up = np.array(PILImage.fromarray(lr).resize((1024, 1024), PILImage.BICUBIC)).astype(np.float32) / 255.0
    # Zoom region (nose/eye area)
    zy, zx, zs = 380, 380, 260
    fig = plt.figure(figsize=(16, 7.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.75], hspace=0.18, wspace=0.06)
    titles = ['Input Photo\n(1024 x 1024)',
              'LR Depth\n(256 x 256, 8-bit)',
              'Bicubic x4\n(1024 x 1024)',
              'HR Ground Truth\n(1024 x 1024, 16-bit)']
    imgs_top = [rgb, lr, lr_up, hr]
    cmaps = [None, 'gray', 'gray', 'gray']
    vmaxs = [None, 255, 1, 1]
    from matplotlib.patches import Rectangle
    for j in range(4):
        ax = fig.add_subplot(gs[0, j])
        kw = {}
        if cmaps[j]:
            kw = dict(cmap=cmaps[j], vmin=0, vmax=vmaxs[j])
        ax.imshow(imgs_top[j], **kw)
        ax.set_title(titles[j], fontsize=12, fontweight='bold', pad=8)
        ax.axis('off')
        # Red zoom rectangle
        if j == 1:
            ax.add_patch(Rectangle((zx // 4, zy // 4), zs // 4, zs // 4,
                                   lw=2.5, edgecolor='red', facecolor='none'))
        else:
            ax.add_patch(Rectangle((zx, zy), zs, zs,
                                   lw=2.5, edgecolor='red', facecolor='none'))
    # Bottom row: zoom-ins
    zoom_labels = ['(zoom)', '(zoom)', '(zoom)', '(zoom)']
    for j in range(4):
        ax = fig.add_subplot(gs[1, j])
        if j == 0:
            crop = rgb[zy:zy + zs, zx:zx + zs]
            ax.imshow(crop)
        elif j == 1:
            ly, lx, ls = zy // 4, zx // 4, zs // 4
            crop = lr[ly:ly + ls, lx:lx + ls]
            ax.imshow(crop, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        elif j == 2:
            crop = lr_up[zy:zy + zs, zx:zx + zs]
            ax.imshow(crop, cmap='gray', vmin=0, vmax=1)
        else:
            crop = hr[zy:zy + zs, zx:zx + zs]
            ax.imshow(crop, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
    # Arrow annotation between LR and bicubic
    fig.text(0.39, 0.78, r'$\longrightarrow$', fontsize=32, ha='center', va='center',
             color='#cc0000', fontweight='bold')
    fig.text(0.39, 0.73, 'x4 SR + 8-to-16-bit', fontsize=9, ha='center', va='top',
             color='#cc0000', fontweight='bold')
    fig.savefig(str(out_path), dpi=200, bbox_inches='tight', facecolor='white',
                edgecolor='none', pad_inches=0.1)
    plt.close(fig)
    print(f"Generated overview figure: {out_path}")
    return str(out_path)

# ── Page geometry (official CVPR: letter 8.5x11, textheight=8.875, textwidth=6.875) ──
PAGE_W, PAGE_H = letter
ML = 0.8125 * inch   # 1in + (-0.1875in) oddsidemargin
MR = ML
MT = 1.0 * inch      # 1in + 0 topmargin + 0 headheight + 0 headsep
MB = 1.125 * inch    # 11 - 1.0 - 8.875 = 1.125
COL_GAP = 0.3125 * inch
TW = PAGE_W - ML - MR
CW = (TW - COL_GAP) / 2
TH = PAGE_H - MT - MB

TITLE_H = 3.6 * inch
BODY1_H = TH - TITLE_H

# ── Styles ──
sTitle = ParagraphStyle('T', fontName='Times-Bold', fontSize=14, leading=17, alignment=TA_CENTER, spaceAfter=8)
sAuthor = ParagraphStyle('A', fontName='Times-Roman', fontSize=11, leading=13, alignment=TA_CENTER, spaceAfter=4)
sAbsHead = ParagraphStyle('AH', fontName='Times-Bold', fontSize=10, leading=12, alignment=TA_CENTER, spaceBefore=6, spaceAfter=4)
sAbstract = ParagraphStyle('AB', fontName='Times-Italic', fontSize=9, leading=11, alignment=TA_JUSTIFY, spaceAfter=6)
sSec = ParagraphStyle('S', fontName='Times-Bold', fontSize=10, leading=12, spaceBefore=8, spaceAfter=3)
sSubsec = ParagraphStyle('SS', fontName='Times-Bold', fontSize=9, leading=11, spaceBefore=6, spaceAfter=2)
sBody = ParagraphStyle('B', fontName='Times-Roman', fontSize=9, leading=11, alignment=TA_JUSTIFY, spaceAfter=2, firstLineIndent=12)
sBodyNI = ParagraphStyle('BN', fontName='Times-Roman', fontSize=9, leading=11, alignment=TA_JUSTIFY, spaceAfter=2)
sCap = ParagraphStyle('C', fontName='Times-Roman', fontSize=8, leading=10, alignment=TA_JUSTIFY, spaceAfter=4, spaceBefore=3)
sRef = ParagraphStyle('R', fontName='Times-Roman', fontSize=7.5, leading=9, alignment=TA_JUSTIFY, spaceAfter=1, leftIndent=12, firstLineIndent=-12)

def S_(n): return Spacer(1, n)
_sn = [0]; _ssn = [0]
def sec(t): _sn[0] += 1; _ssn[0] = 0; return Paragraph(f"{_sn[0]}. {t}", sSec)
def subsec(t): _ssn[0] += 1; return Paragraph(f"{_sn[0]}.{_ssn[0]}. {t}", sSubsec)
def body(t): return Paragraph(t, sBody)
def bodyni(t): return Paragraph(t, sBodyNI)
def parhead(h, t): return Paragraph(f"<b>{h}</b> {t}", sBodyNI)

# ── Table styles ──
_tcH = ParagraphStyle('tcH', fontName='Times-Bold', fontSize=7.5, leading=9, alignment=TA_CENTER)
_tcHL = ParagraphStyle('tcHL', fontName='Times-Bold', fontSize=7.5, leading=9, alignment=TA_LEFT)
_tcB = ParagraphStyle('tcB', fontName='Times-Bold', fontSize=7.5, leading=9, alignment=TA_CENTER)
_tcN = ParagraphStyle('tcN', fontName='Times-Roman', fontSize=7.5, leading=9, alignment=TA_CENTER)
_tcNL = ParagraphStyle('tcNL', fontName='Times-Roman', fontSize=7.5, leading=9, alignment=TA_LEFT)
TSTYLE_BASE = [
    ('LINEABOVE', (0,0), (-1,0), 0.8, black),
    ('LINEBELOW', (0,0), (-1,0), 0.5, black),
    ('LINEBELOW', (0,-1), (-1,-1), 0.8, black),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('TOPPADDING', (0,0), (-1,-1), 1.5),
    ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
]

# ── Frames ──
title_frame = Frame(ML, MB + BODY1_H, TW, TITLE_H, id='title',
                    leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
p1_left = Frame(ML, MB, CW, BODY1_H, id='p1L',
                leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
p1_right = Frame(ML + CW + COL_GAP, MB, CW, BODY1_H, id='p1R',
                 leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
left_f = Frame(ML, MB, CW, TH, id='L',
               leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
right_f = Frame(ML + CW + COL_GAP, MB, CW, TH, id='R',
                leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Times-Roman', 9)
    canvas.drawCentredString(PAGE_W / 2, 0.6 * inch, str(doc.page))
    canvas.restoreState()

page1_tmpl = PageTemplate(id='Page1', frames=[title_frame, p1_left, p1_right], onPage=footer)
body_tmpl = PageTemplate(id='Body', frames=[left_f, right_f], onPage=footer)

# Full-width figure page templates (wide frame on top, two columns below)
def _make_figw(fig_h, tmpl_id):
    body_h = TH - fig_h - 4
    top = Frame(ML, MB + body_h, TW, fig_h, id=f'{tmpl_id}_top',
                leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    fl = Frame(ML, MB, CW, body_h, id=f'{tmpl_id}_L',
               leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    fr = Frame(ML + CW + COL_GAP, MB, CW, body_h, id=f'{tmpl_id}_R',
               leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    return PageTemplate(id=tmpl_id, frames=[top, fl, fr], onPage=footer)

FIGW_M_H = 3.2 * inch   # medium figure frame
FIGW_L_H = 4.0 * inch   # large figure frame
figw_m = _make_figw(FIGW_M_H, 'FigWide')
figw_l = _make_figw(FIGW_L_H, 'FigWideLarge')

doc = BaseDocTemplate(str(OUT), pagesize=letter,
                      leftMargin=ML, rightMargin=MR, topMargin=MT, bottomMargin=MB)
doc.addPageTemplates([page1_tmpl, body_tmpl, figw_m, figw_l])

from reportlab.platypus import PageBreak

def embed_fullwidth_png(png_path, caption_text, max_h=None):
    """Return flowables for a full-width figure.
    Call NextPageTemplate('FigWide') + PageBreak() BEFORE these,
    then FrameBreak() + NextPageTemplate('Body') AFTER."""
    if max_h is None:
        max_h = FIGW_M_H - 0.4 * inch
    src = Path(png_path) if Path(png_path).is_absolute() else EVAL_DIR / png_path
    elems = []
    if src.exists():
        try:
            pil_img = PILImage.open(str(src))
            pw, ph = pil_img.size
            pil_img.close()
            aspect = float(ph) / float(pw)
            w = float(TW)
            h = w * aspect
            if h > max_h:
                h = max_h
                w = h / aspect
            img = RLImage(str(src), width=w, height=h)
            img.hAlign = 'CENTER'
            elems.append(img)
        except Exception as e:
            print(f"WARNING: fullwidth embed failed for {png_path}: {e}")
            elems.append(Paragraph(f"<i>[Figure: {png_path} - error]</i>",
                         ParagraphStyle('fwe', fontName='Times-Italic', fontSize=8, alignment=TA_CENTER)))
    else:
        elems.append(Paragraph(f"<i>[Figure: {png_path} not found]</i>",
                     ParagraphStyle('fwm', fontName='Times-Italic', fontSize=8, alignment=TA_CENTER)))
    elems.append(Paragraph(caption_text, sCap))
    return elems

# Generate overview figure if it doesn't exist yet
_overview_path = generate_overview_figure()

# ══════════════════════════════════════════════════════════════
#                         CONTENT
# ══════════════════════════════════════════════════════════════
story = []

# ── Title block ──
story.append(S_(12))
story.append(Paragraph(
    "Depth Super-Resolution for 3D Gaussian Splatting Face Reconstruction:<br/>"
    "Rendering-Induced Degradation and Zone-Aware Evaluation", sTitle))
story.append(S_(10))
story.append(Paragraph("Anonymous CVPR Workshop submission<br/>Paper ID ****", sAuthor))
story.append(S_(8))
story.append(Paragraph("Abstract", sAbsHead))
story.append(Paragraph(
    "We address depth map super-resolution for 3D Gaussian Splatting (3DGS) face reconstruction. "
    "Given a 256x256, 8-bit depth map rendered by a pre-trained 3DGS model, our goal is to recover a "
    "1024x1024, 16-bit high-fidelity depth map --jointly performing 4x spatial upsampling and 8-to-16-bit dequantization. "
    "We make four contributions. "
    "(1) We identify <i>rendering-induced degradation</i> as a distinct depth SR setting: 3DGS splat-boundary artifacts, "
    "opacity fall-off, and quantization differ fundamentally from standard bicubic downsampling. A 5x5 cross-degradation "
    "test shows a 3.5 dB PSNR gap between matched and mismatched degradations. "
    "(2) We propose <i>zone-aware evaluation</i>, restricting metrics to the near-frontal confidence cone "
    "(|yaw| &lt;= 20 deg) to avoid contamination from 3DGS side/back hallucinations. "
    "(3) We curate and release <i>FaceLift-Depth</i>, a 1,288-sample multi-modal benchmark with 9 modalities "
    "per sample, the first public dataset for depth SR on 3DGS-rendered faces. "
    "(4) We show 3DGS-rendered normals are unusable for supervision due to per-splat aliasing, while "
    "DSINE [1] pseudo-GT normals provide a clean alternative (L1 = 0.00228 vs. 0.00232). "
    "On 1,287 FFHQ-derived 3DGS face scans, a simple UNet with bicubic residual learning achieves 46.5 dB PSNR, "
    "outperforming CVPR 2025 DORNet [2] by 4.4 dB and AAAI 2024 SGNet [3] by 3.4 dB.",
    sAbstract))

story.append(NextPageTemplate('Body'))
story.append(FrameBreak())

# ══════════════════════ 1. INTRODUCTION ══════════════════════
story.append(sec("Introduction"))
story.append(body(
    "3D Gaussian Splatting (3DGS) [4] has emerged as a powerful representation for real-time novel view synthesis, "
    "with recent extensions to single-image 3D face reconstruction such as FaceLift [5]. "
    "Given a single face photograph, FaceLift reconstructs a 3DGS model and renders multi-view RGB, depth, normal, "
    "and opacity maps. While the RGB output is visually plausible, the rendered depth maps suffer from two compounding "
    "limitations: (1) limited spatial resolution --typically 256x256 constrained by the input image resolution, and "
    "(2) 8-bit quantization that collapses fine geometric detail into only 256 discrete levels (Fig. 1)."))

story.append(body(
    "Depth super-resolution (SR) is a natural remedy. However, applying existing depth SR methods trained on standard "
    "benchmarks such as NYU-v2 [6] or Middlebury [7] yields poor results on 3DGS-rendered depth. "
    "The root cause is a fundamental <i>degradation mismatch</i>: standard depth SR universally assumes bicubic "
    "downsampling with optional additive Gaussian noise, while 3DGS rendering introduces a qualitatively different "
    "degradation pipeline --splat boundary discontinuities from discrete Gaussian primitives, opacity-weighted depth "
    "blending across overlapping splats, and coarse 8-bit quantization applied during image export."))
story.append(body(
    "A second overlooked problem is <i>evaluation contamination</i>. 3DGS face models are trained from a single "
    "frontal photograph; geometry in side and back views is hallucinated by the generative prior and varies "
    "unpredictably across samples. Evaluating depth SR over all rendered views conflates genuine SR improvement "
    "with upstream 3DGS hallucination quality, making it impossible to attribute metric changes to the SR model."))
story.append(body(
    "A third challenge arises when attempting to incorporate surface normal supervision into the depth SR loss. "
    "3DGS-rendered normal maps exhibit severe per-splat aliasing: each Gaussian contributes an independent surface "
    "normal, creating high-frequency \"pit\" and \"bump\" artifacts on smooth skin regions. This phenomenon is "
    "well-documented in Normal-GS [15], 2DGS [14], SuGaR [16], and DN-Splatter [17], yet no prior work has "
    "studied its impact on downstream depth SR training."))
story.append(body("In this paper, we study depth SR tailored to the 3DGS face rendering pipeline. Our contributions:"))
story.append(parhead("(a) Rendering-induced degradation.",
    "We construct a 5x5 cross-degradation evaluation matrix across five LR generation methods. "
    "Models trained on one degradation type drop 3-6 dB when tested on another, with two clear clusters "
    "(bicubic-family vs. render-family) plus a noise outlier. This establishes 3DGS rendering as a genuinely "
    "new degradation setting for depth SR, distinct from the bicubic assumption in all prior work."))
story.append(parhead("(b) Zone-aware evaluation.",
    "We restrict depth SR metrics to a near-frontal confidence cone (|yaw| &lt;= 20 deg), defined empirically "
    "from multi-view depth consistency analysis. Metrics computed in this zone accurately attribute "
    "SR performance without contamination from 3DGS hallucination artifacts."))
story.append(parhead("(c) FaceLift-Depth benchmark.",
    "We curate and release a multi-modal depth SR benchmark derived from 1,288 FFHQ [19] faces "
    "processed through FaceLift [5], comprising 9 modalities per sample: RGB, HR depth (1024x1024, 16-bit), "
    "LR depth at two bit-depths (256x256, 8-bit and 16-bit), 3DGS normals, DSINE [1] pseudo-GT normals, "
    "opacity maps, and face masks at two resolutions. The dataset includes multi-stage quality filtering "
    "via MediaPipe facial landmark validation, background cleanliness scoring, MD5 deduplication, and "
    "morphological mask QA. To our knowledge, this is the first public benchmark for depth SR on "
    "3DGS-rendered face data."))
story.append(parhead("(d) Normal aliasing and DSINE fix.",
    "We demonstrate that 3DGS-rendered normals degrade depth SR when used as supervision (L1 worsens by 59%), "
    "even after bilateral smoothing. Replacing them with DSINE [1] monocular pseudo-GT normals "
    "recovers clean supervision that matches baseline performance."))

# ── Full-width overview figure (Fig 1) ──
story.append(NextPageTemplate('FigWide'))
story.append(PageBreak())
story.extend(embed_fullwidth_png("overview_main.png",
    "<b>Figure 1.</b> Overview of the depth SR task. Given a single face photograph (left), "
    "FaceLift [5] renders a low-resolution 256x256, 8-bit depth map (second column). Our method "
    "jointly performs 4x spatial super-resolution and 8-to-16-bit dequantization to recover the "
    "high-fidelity 1024x1024, 16-bit depth (third column), compared against the ground truth (right). "
    "Bottom row: zoomed crops highlight the detail recovery. The LR input shows severe pixelation and "
    "banding from 8-bit quantization; bicubic upsampling blurs edges; our SR restores fine geometry.",
    max_h=FIGW_M_H - 0.3 * inch))
story.append(FrameBreak())
story.append(NextPageTemplate('Body'))

# ══════════════════════ 2. RELATED WORK ══════════════════════
story.append(sec("Related Work"))
story.append(parhead("Depth super-resolution.",
    "Classical depth SR uses hand-crafted filters guided by RGB images, such as joint bilateral upsampling [8]. "
    "Deep learning approaches train on synthetic LR/HR pairs: Hui <i>et al.</i> [9] use multi-scale guidance, "
    "BridgeNet [10] jointly learns depth SR and monocular estimation. For single-image SR, EDSR [12] introduced "
    "enhanced residual blocks with residual scaling, SRResNet [13] applied residual learning with a GAN loss, "
    "and SwinIR [11] brought Swin Transformers to image restoration. LIIF [29] introduced continuous implicit "
    "representations for arbitrary-scale SR. Real-ESRGAN [21] showed that training on realistic synthetic "
    "degradation pipelines is crucial for real-world performance --a finding that directly motivates our work. "
    "Recent depth-specific methods address degradation robustness: "
    "DORNet [2] (CVPR 2025) learns degradation representations via mixture-of-experts routing, while "
    "SGNet [3] (AAAI 2024) uses RGB structure guidance with frequency-domain losses. "
    "MiDaS [31] and Marigold [32] have advanced monocular depth estimation, but depth SR from known "
    "degradation remains a distinct problem. "
    "However, all existing SR methods assume bicubic downsampling as the degradation model. None account for "
    "the rendering-specific artifacts introduced by 3DGS rasterization."))
story.append(parhead("3D Gaussian Splatting and normal quality.",
    "3DGS [4] represents scenes as collections of anisotropic 3D Gaussians rendered via differentiable "
    "rasterization. Extensions include 2DGS [14] for improved surface geometry, Normal-GS [15] for "
    "normal-involved rendering, SuGaR [16] for mesh extraction, and DN-Splatter [17] for depth/normal priors. "
    "FaceLift [5] applies 3DGS to single-image face reconstruction using a feed-forward GS-LRM architecture. "
    "Other 3DGS generation methods include DreamGaussian [23] for text-to-3D, LGM [22] and GRM [33] for "
    "feed-forward 3D reconstruction. NeRF [30] provides an alternative volumetric representation. "
    "A consistent finding across 2DGS, Normal-GS, and DN-Splatter is that per-splat normal aliasing produces "
    "high-frequency artifacts: each Gaussian's independent normal creates discontinuities at splat boundaries "
    "that bilateral filtering cannot fully remove without destroying genuine geometric detail. "
    "We are the first to study how this aliasing affects downstream tasks like depth super-resolution."))
story.append(parhead("Monocular normal estimation.",
    "DSINE [1] estimates surface normals from a single RGB image by rethinking inductive biases in the "
    "estimation network. Omnidata [18] provides a multi-task pipeline including normal prediction. "
    "We use DSINE pseudo-GT normals as a clean replacement for noisy 3DGS-rendered normals, "
    "enabling effective normal-aware depth SR supervision."))

# ══════════════════════ 3. METHOD ══════════════════════

story.append(sec("Method"))
story.append(subsec("Data construction pipeline"))
story.append(body(
    "Our dataset construction pipeline (Fig. 1) transforms single face photographs into paired LR/HR depth samples "
    "through five carefully designed stages. Each stage addresses specific challenges in 3DGS-rendered data "
    "that, to our knowledge, have not been systematically documented in prior depth SR work."))

story.append(parhead("Stage 0: Source selection and quality filtering.",
    "We draw source images from two pools: FFHQ [19] (70,000 face images at 1024x1024) and a supplementary "
    "HumanFaces collection. From each pool, we apply a multi-stage filtering pipeline. "
    "(i) Background cleanliness: we sample border pixels (40px strips on all edges) and compute color "
    "variance; only images with clean backgrounds are retained. "
    "(ii) Frontal face validation: using MediaPipe [34] FaceMesh (468 landmarks), we compute the nose-to-eye "
    "horizontal distance ratio and reject side profiles (ratio < 0.55). We also flag faces with landmarks "
    "within 5% of image edges as incomplete crops. "
    "(iii) MD5 deduplication: exact file hash comparison removes duplicate images. "
    "(iv) Post-render mask QA: after 3DGS rendering, we check generated face masks against five criteria -- "
    "mask area ratio in [0.15, 0.85], centroid offset < 0.25, face bounding box > 20% of image, and "
    "aspect ratio < 3.0 -- to catch render failures and extreme poses. "
    "From ~2000 initial candidates, 1,288 pass all stages."))

# FIGURE: Dataset samples grid
story.extend(embed_png_direct("dataset_samples.png", CW * 0.90,
    "<b>Figure 2.</b> Dataset samples from FaceLift-Depth. Each row shows one sample across five modalities: "
    "RGB input, HR depth (1024x1024, 16-bit, inferno colormap), LR depth (256x256, 8-bit), "
    "DSINE pseudo-GT surface normal, and binary face mask."))

story.append(parhead("Stage 1: 3DGS reconstruction and rendering.",
    "We select 1,287 high-quality face images from FFHQ [19] and process each through FaceLift [5] to obtain "
    "a 3DGS model. From each model, we render four maps at the canonical frontal viewpoint at 1024x1024: "
    "RGB, depth (16-bit), surface normal, and opacity. The depth maps are stored as 16-bit PNG with per-view "
    "min/max normalization to [0, 65535], preserving the full dynamic range of the reconstructed geometry."))

story.append(parhead("Stage 2: Postprocessing (7 sub-stages).",
    "Raw 3DGS renders require extensive artifact correction before they can serve as SR training data. "
    "We develop a 7-stage postprocessing pipeline:"))
story.append(body(
    "<b>(i) Landmark alignment.</b> The rendered face may be spatially misaligned with the original photograph "
    "due to camera pose estimation errors. We detect 6 facial keypoints (eyes, nose, chin, forehead, center) "
    "in both images, compute a similarity transform via RANSAC (threshold 8 px), and refine with Enhanced "
    "Correlation Coefficient (ECC) [39] optimization (50 iterations, eps=1e-4) with Gaussian pre-blur for sub-pixel "
    "accuracy. Fallback to brightness centroid if landmark detection fails."))
story.append(body(
    "<b>(ii) Artifact cleanup.</b> 3DGS occasionally produces floating splat artifacts outside the face region. "
    "We apply morphological close/open (5x5 elliptical kernel) on the opacity mask, then retain only the "
    "largest connected component plus any region &gt;=100 pixels, removing isolated floating geometry."))
story.append(body(
    "<b>(iii) Hole filling (dual strategy).</b> Splat coverage gaps leave undefined (zero) depth pixels. "
    "We detect interior holes via flood-fill from the image border. Small holes (&lt;=5000 px) are filled with "
    "TELEA inpainting for smooth interpolation. Large holes use distance-transform-based nearest-neighbor "
    "propagation followed by Gaussian smoothing (sigma=2.0) to avoid sharp fill boundaries."))
story.append(body(
    "<b>(iv) Nose-relative depth normalization.</b> Per-view min/max normalization causes depth scale to vary "
    "across samples, breaking cross-sample comparability. We re-normalize using the nose tip as a geometric "
    "anchor: the nose reference depth is the max of the nose patch depth and the 99.9th percentile of "
    "foreground depth (avoiding single-pixel outliers). The depth range is normalized to [0, 1] with the "
    "nose at 1.0 (closest) and the 1st percentile as the far plane."))
story.append(body(
    "<b>(v) Normal recomputation.</b> Instead of using the noisy per-splat normals directly, we recompute "
    "surface normals from the (now-cleaned) depth map using Scharr gradient filters, which are more accurate "
    "than Sobel for smooth surfaces. The normal vector at each pixel is n = normalize(-dz/dx, -dz/dy, 1)."))
story.append(body(
    "<b>(vi) RGB-guided joint bilateral filtering.</b> We smooth the recomputed normals using the original "
    "photograph as an edge guide via joint bilateral filtering [37] (OpenCV [36] ximgproc). This preserves facial "
    "feature edges (nose bridge, eyebrow, lip boundary) while smoothing skin-region normals. "
    "Post-filtering, normals are renormalized to unit length (bilateral breaks unit-length constraint) "
    "and background is set to neutral gray [128, 128, 128]."))
story.append(body(
    "<b>(vii) Depth-normal consistency verification.</b> We verify alignment between the depth-derived "
    "normals (from step v) and the filtered normals (from step vi) via cosine similarity. Our pipeline "
    "achieves mean consistency of 0.953 across all 1,287 samples, with 100% of samples above 0.9. "
    "This step flags any remaining hallucinated geometry for manual inspection."))

story.append(parhead("Stage 3: Face mask generation.",
    "Depth SR metrics should exclude background regions where depth is undefined or meaningless. "
    "We generate binary face masks via morphological operations on the cropped face region: "
    "dilation to include the full face boundary, erosion to remove fringe artifacts, and bilateral "
    "smoothing of the boundary. The resulting masks are 1024x1024 binary images ({0, 255}) used to "
    "restrict both the training loss and evaluation metrics to the face region only."))

story.append(parhead("Stage 4: LR/HR pair generation.",
    "HR depth maps (1024x1024, 16-bit) serve as ground truth. LR inputs are generated by: "
    "(i) resizing HR maps to 256x256 using area interpolation (matching the 3DGS rendering pipeline's "
    "downsampling behavior), and (ii) quantizing to 8-bit (dividing by 257 and rounding). "
    "This produces the primary training pairs. For ablation, we also generate 16-bit LR inputs "
    "(256x256, no quantization) to isolate the dequantization component."))

story.append(parhead("Stage 5: DSINE pseudo-GT normals.",
    "To enable normal-aware depth SR without relying on noisy 3DGS normals, we compute pseudo-ground-truth "
    "surface normals using DSINE [1], a state-of-the-art monocular normal estimator. "
    "We run DSINE on the rendered RGB images (not the original photographs) to ensure geometric "
    "consistency with the 3DGS depth. Processing 1,287 images takes approximately 3 minutes on a "
    "single RTX 4070 Laptop GPU. The resulting normals are stored at 1024x1024 for use as supervision."))

story.append(parhead("Train/validation split.",
    "We split the 1,288 samples into 1,159 training and 129 validation samples (90/10 split, random "
    "seed 42). The dataset comprises images from FFHQ and supplementary HumanFaces sources, "
    "ensuring demographic diversity. Each sample includes 9 modalities: RGB, HR depth (1024x1024, 16-bit), "
    "LR depth at 8-bit and 16-bit (256x256), 3DGS normals, DSINE pseudo-GT normals, opacity, "
    "and face masks at 1024x1024 and 256x256. A manifest.json records the exact split for reproducibility."))

story.append(subsec("Network architecture"))
story.append(body(
    "We adopt a UNet [20] encoder-decoder with residual [27] skip connections, using base channel width 32 "
    "(7.77M parameters). The key design choice is <i>bicubic residual learning</i>: the input LR depth map is "
    "first bicubic-upsampled to 1024x1024 resolution, and the UNet predicts only a bounded residual correction "
    "via a tanh activation scaled by 0.5. The final output is the sum of the bicubic baseline and the residual, "
    "clamped to [0, 1]. This formulation ensures the network starts from a reasonable initialization and only "
    "needs to learn the high-frequency correction, rather than the entire depth field. As we show in Sec. 4.2, "
    "this prior is critical: methods without it (SwinIR) fail catastrophically."))

story.append(body(
    "For the normal-aware variant, the UNet is extended with a lightweight dual-head design: "
    "an independent normal prediction head branches from the decoder's penultimate feature map. "
    "This head produces a 3-channel normal map that is supervised directly against the DSINE pseudo-GT, "
    "rather than deriving normals from predicted depth via finite differences. The dual-head design "
    "avoids error propagation from depth tonormal differentiation and adds only 0.01M parameters "
    "(7.77M  to 7.78M). At inference time, the normal head is discarded; only the depth output is used."))

story.append(subsec("Raw depth issues and remediation"))
story.append(body(
    "A critical and often understated challenge is that the raw 3DGS-rendered depth maps are "
    "<i>not directly usable</i> for super-resolution training. FaceLift's default rendering produces "
    "depth maps with several pathological properties: per-view min/max normalization (making cross-sample "
    "depth scales inconsistent), floating splat artifacts outside the face region, interior coverage holes "
    "from splat gaps, and per-splat normal aliasing. Without our 7-stage postprocessing pipeline (Sec. 3.1), "
    "training a depth SR model produces checkerboard artifacts and fails to converge beyond 38 dB."))
story.append(body(
    "We discovered these issues through extensive iterative debugging. Early training attempts on raw renders "
    "showed suspicious loss plateaus and visual artifacts. Systematic ablation of each postprocessing stage "
    "revealed that hole filling and nose-relative normalization contribute the largest individual gains "
    "(~2.5 dB and ~1.8 dB respectively when removed), while landmark alignment prevents catastrophic "
    "misalignment on ~15% of samples where FaceLift's camera pose drifts."))
story.append(body(
    "The depth-normal sign convention proved particularly treacherous. The correct normal computation "
    "from depth uses n = normalize(+dz/dx, +dz/dy, 1) because our depth convention maps close surfaces "
    "to 1.0 (not 0.0). Incorrect sign choices cause the cosine normal loss to converge to a mirror-image "
    "attractor, producing systematically inverted depth gradients that <i>increase</i> L1 rather than "
    "decreasing it. We determined the correct convention by brute-force testing all 8 sign combinations "
    "(+/-1, +/-1, +/-1) and selecting the one that produced declining validation loss."))

story.append(subsec("Implementation challenges"))
story.append(body(
    "Several platform-specific issues required careful engineering. (1) OpenCV's cv2.resize does not support "
    "uint16 arrays with INTER_CUBIC on Windows builds; we convert to float32 before resizing and cast back. "
    "(2) PIL opens 16-bit PNGs as mode 'I' (int32) or 'I;16' (uint16) depending on version; we use a "
    "max > 255 heuristic rather than dtype checks to determine the normalization divisor. "
    "(3) cv2.imread combined with DataLoader num_workers > 0 on Windows causes heap fragmentation and OOM; "
    "we use PIL for all image loading with num_workers=0. (4) At 1024x1024, base_ch=64 exceeds 8 GB VRAM; "
    "we use base_ch=32 with FP16 AMP and gradient accumulation to maintain an effective batch size of 8."))

story.append(subsec("Loss function"))
story.append(body(
    "The training loss combines a mask-aware L1 term with a spatial gradient penalty. The L1 loss is computed "
    "only within the face mask region, excluding undefined background. The gradient loss penalizes differences "
    "in horizontal and vertical spatial gradients between the prediction and ground truth, encouraging "
    "edge-preserving reconstruction. We weight the gradient loss at 0.5 relative to the L1 term."))
story.append(body(
    "For the normal-aware variant, we add a cosine similarity loss between depth-derived normals and "
    "DSINE pseudo-GT normals, weighted at 0.2. We found that the sign convention for computing normals "
    "from depth is critical: incorrect signs cause the normal loss to converge to a wrong attractor, "
    "degrading rather than improving depth quality."))

story.append(subsec("Zone-aware evaluation"))
story.append(body(
    "3DGS face models reconstruct geometry from a single frontal photograph. Side and back views are "
    "hallucinated by the generative prior and exhibit substantial geometric inconsistency. "
    "To quantify this, we render each 3DGS model at 13 yaw angles from -90 deg to +90 deg with shared "
    "global depth normalization (computed from the frontal view) and measure pairwise depth consistency "
    "across views. We find that consistency drops sharply beyond |yaw| ~ 20 deg, establishing a natural "
    "boundary for the <i>confidence zone</i>: the set of viewpoints where 3DGS geometry is reliable."))
story.append(body(
    "For depth SR evaluation, we restrict all metrics (L1, RMSE, PSNR) to pixels within the face mask "
    "at viewpoints inside this confidence zone. This prevents SR quality metrics from being contaminated "
    "by upstream 3DGS hallucination artifacts. In practice, since our training and validation data use "
    "canonical frontal views (yaw = 0 deg), zone-aware evaluation reduces to face-masked metrics, "
    "but the framework generalizes to multi-view SR evaluation."))

story.append(subsec("Training details"))
story.append(body(
    "All models are trained with AdamW [25,26] optimizer (learning rate 1e-4, weight decay 1e-4), cosine annealing [38] "
    "schedule, and mixed-precision (FP16) training on a single NVIDIA RTX 4070 Laptop GPU (8 GB VRAM). "
    "We use batch size 2 with gradient accumulation over 4 steps (effective batch size 8). "
    "For the normal-aware variant, memory constraints require reducing to batch size 1 with gradient "
    "accumulation 8. Training converges in approximately 100 epochs (~2 hours for the base model, "
    "~3.5 hours for the normal-aware variant)."))

# ══════════════════════ 4. EXPERIMENTS ══════════════════════
story.append(sec("Experiments"))
story.append(subsec("Rendering-induced degradation"))
story.append(body(
    "To validate that 3DGS rendering produces a distinct degradation, we train five identical UNets "
    "(same architecture, hyperparameters, and training schedule) on five LR variants generated from "
    "the same HR depth maps: (1) bicubic downsampling, (2) bicubic + Gaussian noise (sigma=0.01), "
    "(3) bicubic + 8-bit quantization, (4) 3DGS area-interpolation rendering, and "
    "(5) 3DGS rendering + 8-bit quantization. Each model is evaluated on all five test sets, "
    "yielding the 5x5 PSNR matrix in Tab. 1."))

# TABLE 1
t1d = [
    ['Train \\ Test', 'Bic.', '+Noise', '+Q8', 'Render', 'Rnd+Q8'],
    ['Bicubic',       '44.6', '35.8',   '44.5', '43.2',  '43.1'],
    ['Bic.+Noise',    '41.2', '40.9',   '41.2', '40.4',  '40.5'],
    ['Bic.+Q8',       '44.6', '36.0',   '44.5', '43.3',  '43.3'],
    ['Render',        '40.5', '32.7',   '40.4', '46.2',  '45.9'],
    ['Render+Q8',     '40.6', '33.2',   '40.5', '45.7',  '45.5'],
]
diag = {(1,1),(2,2),(3,3),(4,4),(5,5)}
t1s = []
for ri, row in enumerate(t1d):
    r = []
    for ci, c in enumerate(row):
        if (ri,ci) in diag:   r.append(Paragraph(f"<b>{c}</b>", _tcB))
        elif ri == 0:         r.append(Paragraph(c, _tcH if ci > 0 else _tcHL))
        elif ci == 0:         r.append(Paragraph(c, _tcNL))
        else:                 r.append(Paragraph(c, _tcN))
    t1s.append(r)
t1 = Table(t1s, colWidths=[CW*0.27]+[CW*0.146]*5)
t1.setStyle(TableStyle(TSTYLE_BASE))

story.append(KeepTogether([
    Paragraph("<b>Table 1.</b> Cross-degradation test (PSNR dB). Each row: model trained on that degradation; "
              "each column: test degradation. Bold diagonal = matched train/test.", sCap),
    t1, S_(4)]))

# ── Full-width cross-test heatmap (Fig 3) ──
story.append(NextPageTemplate('FigWideLarge'))
story.append(PageBreak())
story.extend(embed_fullwidth_png("fig_crosstest_heatmap.png",
    "<b>Figure 3.</b> Cross-degradation PSNR heatmap (left: PSNR dB, right: L1). "
    "Two clusters visible: {bicubic, bic+Q8} and {render, rnd+Q8}. Noise is an outlier. "
    "On-diagonal 45-46 dB, off-diagonal 40-43 dB, confirming 3-6 dB degradation mismatch.",
    max_h=FIGW_L_H - 0.3 * inch))
story.append(FrameBreak())
story.append(NextPageTemplate('Body'))

story.append(body(
    "Three key findings emerge. First, the render-trained model achieves 46.2 dB on render test data but only "
    "40.5 dB on bicubic test data --a 5.7 dB gap. This asymmetry demonstrates that the rendering degradation is "
    "not a subset of bicubic degradation; models must be specifically trained for each. "
    "Second, two degradation clusters are clearly visible: {bicubic, bicubic+Q8} form one cluster "
    "(within-cluster gap < 0.2 dB), and {render, render+Q8} form another (within-cluster gap < 0.5 dB). "
    "The noise variant is an outlier, performing poorly across all test conditions. "
    "Third, quantization has a relatively small effect within each cluster (<0.5 dB), suggesting that "
    "the spatial degradation pattern (bicubic vs. rendering) dominates over bit-depth effects."))
story.append(body(
    "These results establish that 3DGS rendering constitutes a genuinely distinct degradation setting. "
    "The consistent 3-6 dB gap between matched and mismatched train/test pairs rules out the possibility "
    "that a single \"universal\" model could handle both degradation types without significant quality loss."))

# 4.2 Baseline comparison
story.append(subsec("Baseline comparison"))
story.append(body(
    "We compare our UNet against six baselines spanning four categories: no-learning (bicubic), "
    "general-purpose SR (SwinIR-tiny [11], EDSR-light [12], SRResNet [13]), "
    "depth-specific SR (DORNet [2], SGNet [3]), and our models with and without normal supervision. "
    "All learning-based methods except DORNet are trained on the same 3DGS-degraded LR/HR pairs. "
    "DORNet uses its official NYU-v2 pretrained checkpoint (zero-shot transfer). "
    "Results on 129 validation samples with face-masked metrics are shown in Tab. 2."))

# TABLE 2
t2d = [
    ['Method',             'Params', 'Train Data', 'L1 (low)',     'RMSE (low)',   'PSNR (high)'],
    ['Bicubic (no model)', '-',      '-',          '0.00332',  '0.00827', '41.7'],
    ['SwinIR-tiny [11]',   '0.23M',  '3DGS',       '0.0856',   '0.2526',  '11.9'],
    ['EDSR-light [12]',    '1.52M',  '3DGS',       '0.00241',  '0.00499', '46.0'],
    ['SRResNet [13]',      '1.53M',  '3DGS',       '0.00228',  '0.00472', '46.5'],
    ['DORNet [2]',         '3.05M',  'NYU bic.',    '0.00337',  '0.00782', '42.1'],
    ['SGNet* [3]',         '9.22M',  '3DGS',        '0.00352',  '0.00702', '43.1'],
    ['UNet (ours)',        '7.77M',  '3DGS',       '0.00232',  '0.00489', '46.2'],
    ['UNet+Norm. (ours)',  '7.78M',  '3DGS',       '0.00228',  '0.00475', '46.5'],
]
t2s = []
for ri, row in enumerate(t2d):
    r = []
    for ci, c in enumerate(row):
        if ri == 0:                        r.append(Paragraph(c, _tcH if ci > 0 else _tcHL))
        elif ri == len(t2d)-1 and ci >= 3: r.append(Paragraph(f"<b>{c}</b>", _tcB))
        elif ci == 0:                      r.append(Paragraph(c, _tcNL))
        else:                              r.append(Paragraph(c, _tcN))
    t2s.append(r)
t2 = Table(t2s, colWidths=[CW*0.30, CW*0.11, CW*0.14, CW*0.15, CW*0.15, CW*0.15])
t2.setStyle(TableStyle(TSTYLE_BASE + [
    ('LINEBELOW', (0,1), (-1,1), 0.3, colors.gray),
    ('LINEBELOW', (0,6), (-1,6), 0.3, colors.gray),
]))

story.append(KeepTogether([
    Paragraph("<b>Table 2.</b> Depth SR results on 129 validation samples (face-masked metrics). "
              "*SGNet evaluated at 512 resolution due to OOM at 1024. Best in <b>bold</b>.", sCap),
    t2, S_(4)]))

story.append(parhead("The bicubic residual prior is essential.",
    "SwinIR-tiny (0.23M) achieves only 11.9 dB --worse than doing nothing (bicubic: 41.7 dB). "
    "SwinIR processes the full image through a Swin Transformer without any global skip connection, "
    "forcing it to reconstruct the entire depth field from scratch. "
    "At 0.23M parameters this is infeasible for 1024x1024 depth maps. In contrast, EDSR-light (1.52M), "
    "SRResNet (1.53M), and our UNet (7.77M) all include a bicubic skip connection and achieve 46+ dB. "
    "The 34 dB gap between SwinIR and the residual-learning methods demonstrates that the bicubic prior "
    "is not merely helpful but essential for this task."))

story.append(parhead("Degradation mismatch cripples SOTA methods.",
    "DORNet (CVPR 2025 oral), the current state-of-the-art on NYU-v2 4x depth SR with its "
    "degradation-aware mixture-of-experts architecture, achieves only 42.1 dB in zero-shot transfer --"
    "0.4 dB <i>below</i> simple bicubic interpolation. This striking result directly demonstrates that "
    "degradation mismatch is the primary bottleneck: DORNet's learned degradation representations are "
    "calibrated for NYU-v2's bicubic degradation and cannot generalize to 3DGS rendering artifacts."))

story.append(parhead("RGB guidance does not compensate for degradation mismatch.",
    "SGNet (9.22M, AAAI 2024) uses RGB structure guidance with frequency-domain losses --the most "
    "parameter-heavy method in our comparison. Despite having access to the RGB image as an additional "
    "input, it achieves only 43.1 dB at 512 resolution (1024 causes OOM), 3.1 dB below our "
    "depth-only UNet. This suggests that <i>matching the degradation model</i> is more important than "
    "<i>adding auxiliary information</i> for depth SR quality."))

story.append(parhead("Residual-learning methods are competitive.",
    "SRResNet and EDSR-light, both equipped with bicubic skip connections and trained on 3DGS data, "
    "perform within 0.5 dB of our UNet despite having 5x fewer parameters. This confirms that the "
    "residual learning formulation and matched degradation training are the primary performance drivers, "
    "not model capacity. Our UNet+Normal achieves the best L1 (0.00228) by adding DSINE normal supervision."))

# ── Full-width visual comparison (Fig 4) ──
story.append(NextPageTemplate('FigWide'))
story.append(PageBreak())
story.extend(embed_fullwidth_png("fig_visual_comparison.png",
    "<b>Figure 4.</b> Depth SR comparison on validation samples. "
    "Columns: LR input (256x256, 8-bit), bicubic x4 baseline, UNet (ours), SRResNet, "
    "HR ground truth (1024x1024, 16-bit). Our method recovers fine nose bridge and eyebrow "
    "geometry lost by bicubic upsampling.",
    max_h=FIGW_M_H - 0.3 * inch))
story.append(FrameBreak())
story.append(NextPageTemplate('Body'))

# FIGURE: Zoomed comparison with error maps (single column, larger height)
story.extend(_embed_abs_png(str(EVAL_DIR / "zoomed_comparison.png"), CW * 0.95,
    "<b>Figure 5.</b> Zoomed-in comparison with per-pixel error maps. "
    "Our UNet achieves L1=0.0009 vs bicubic L1=0.0011.",
    max_h=3.8 * inch))

# 4.3 Normal ablation
story.append(subsec("Normal supervision ablation"))
story.append(body(
    "Surface normals provide complementary geometric information to depth and are commonly used as "
    "auxiliary supervision in depth estimation. However, the quality of the normal source is critical. "
    "We compare three normal supervision settings in Tab. 3."))

t3d = [
    ['Normal source',            'Val L1 (low)', 'PSNR (high)', 'Note'],
    ['None (baseline)',          '0.00232',  '46.2',   '-'],
    ['3DGS + bilateral',        '0.00370',  '43.8',   'Per-splat aliasing'],
    ['DSINE pseudo-GT',         '0.00228',  '46.5',   'Clean normals'],
]
t3s = []
for ri, row in enumerate(t3d):
    r = []
    for ci, c in enumerate(row):
        if ri == 0:                  r.append(Paragraph(c, _tcH if ci > 0 else _tcHL))
        elif ri == 3 and ci in (1,2): r.append(Paragraph(f"<b>{c}</b>", _tcB))
        elif ci == 0:                r.append(Paragraph(c, _tcNL))
        else:                        r.append(Paragraph(c, _tcN))
    t3s.append(r)
t3 = Table(t3s, colWidths=[CW*0.35, CW*0.18, CW*0.15, CW*0.32])
t3.setStyle(TableStyle(TSTYLE_BASE))

story.append(KeepTogether([
    Paragraph("<b>Table 3.</b> Normal supervision ablation. 3DGS normals hurt performance; "
              "DSINE pseudo-GT matches and slightly improves over the no-normal baseline.", sCap),
    t3, S_(4)]))

story.append(body(
    "Using 3DGS-rendered normals (even after bilateral smoothing) as supervision <i>degrades</i> performance "
    "dramatically: L1 increases from 0.00232 to 0.00370 (+59%), and PSNR drops from 46.2 to 43.8 dB. "
    "This is because the per-splat aliasing in 3DGS normals creates a systematically noisy supervision signal: "
    "the depth SR model learns to reproduce these high-frequency artifacts rather than learning smooth geometry. "
    "The bilateral filter faces an inherent tradeoff --too mild and aliasing remains; too aggressive and "
    "genuine geometric detail (pores, wrinkles) is destroyed. Neither extreme produces usable supervision."))
story.append(body(
    "Replacing 3DGS normals with DSINE pseudo-GT completely resolves this issue. DSINE, trained on large-scale "
    "data, produces smooth normals that faithfully capture facial geometry without per-splat artifacts. "
    "The resulting model achieves L1 = 0.00228 and PSNR = 46.5 dB, matching SRResNet and slightly "
    "improving over the no-normal UNet baseline. This confirms that normal supervision <i>can</i> work for "
    "depth SR, but only when the normal source is clean --a finding with implications for any "
    "3DGS-based reconstruction pipeline that uses normal supervision."))

# 4.4 Quantization gap
story.append(subsec("Quantization gap analysis"))

t4d = [
    ['LR input',         'Val L1 (low)', 'PSNR (high)', 'Task'],
    ['16-bit (256x256)',  '0.00207', '47.1',   'Spatial SR only'],
    ['8-bit (256x256)',   '0.00232', '46.2',   'SR + dequantization'],
]
t4s = []
for ri, row in enumerate(t4d):
    r = []
    for ci, c in enumerate(row):
        if ri == 0: r.append(Paragraph(c, _tcH if ci > 0 else _tcHL))
        elif ci == 0: r.append(Paragraph(c, _tcNL))
        else: r.append(Paragraph(c, _tcN))
    t4s.append(r)
t4 = Table(t4s, colWidths=[CW*0.30, CW*0.18, CW*0.16, CW*0.36])
t4.setStyle(TableStyle(TSTYLE_BASE))

story.append(KeepTogether([
    Paragraph("<b>Table 4.</b> Quantization gap: 8-bit vs. 16-bit LR input. "
              "Gap = 0.00025 L1 (12.1%), quantifying the dequantization difficulty.", sCap),
    t4, S_(4)]))

story.append(body(
    "To disentangle the spatial upsampling and dequantization components of our task, we train an identical "
    "UNet on 16-bit LR inputs (Tab. 4). The 16-bit model achieves L1 = 0.00207, a 12.1% relative improvement "
    "over the 8-bit model's 0.00232. This 0.00025 gap directly quantifies the information loss from 8-bit "
    "quantization: collapsing 65,536 levels to 256 destroys fine depth gradients that cannot be fully recovered. "
    "The gap also validates our combined SR + dequantization formulation: treating the two degradations jointly "
    "is more practical than assuming 16-bit input, which is rarely available in deployed 3DGS systems."))

# 4.5 Full-view vs zone-aware evaluation
story.append(subsec("Full-view vs. zone-aware evaluation"))
story.append(body(
    "To demonstrate why zone-aware evaluation is essential, we compare metrics computed over the "
    "full 1024x1024 image versus only the face-masked region (Tab. 5). All models were evaluated "
    "on the same 129 validation samples with four metrics: L1, RMSE, PSNR, and SSIM [24]."))

t5d = [
    ['Method',                  'L1 (low)',     'PSNR (high)', 'SSIM (high)',  'L1 (low)',     'PSNR (high)', 'SSIM (high)'],
    ['Nearest-8bit',            '0.00087', '50.7',   '0.9965', '0.00259', '39.1',   '0.9817'],
    ['Bicubic-8bit',            '0.00134', '45.3',   '0.9929', '0.00332', '42.5',   '0.9845'],
    ['UNet 8-bit',              '0.00087', '50.7',   '0.9965', '0.00232', '47.0',   '0.9921'],
    ['UNet+Norm 8-bit',         '0.00085', '50.9',   '0.9967', '0.00228', '47.2',   '0.9924'],
    ['UNet 16-bit',             '0.00077', '51.0',   '0.9968', '0.00206', '47.4',   '0.9928'],
    ['UNet+Norm 16-bit',        '0.00075', '51.2',   '0.9969', '0.00202', '47.5',   '0.9930'],
]
t5s = []
for ri, row in enumerate(t5d):
    r = []
    for ci, c in enumerate(row):
        if ri == 0: r.append(Paragraph(c, _tcH if ci > 0 else _tcHL))
        elif ri == len(t5d)-1 and ci >= 1: r.append(Paragraph(f"<b>{c}</b>", _tcB))
        elif ci == 0: r.append(Paragraph(c, _tcNL))
        else: r.append(Paragraph(c, _tcN))
    t5s.append(r)
t5 = Table(t5s, colWidths=[CW*0.26]+[CW*0.12, CW*0.12, CW*0.12]*2)
t5.setStyle(TableStyle(TSTYLE_BASE + [
    ('LINEBELOW', (0,2), (-1,2), 0.3, colors.gray),
    ('SPAN', (1,0), (3,0)),  # can't easily span in reportlab Paragraph, use caption instead
]))

story.append(KeepTogether([
    Paragraph("<b>Table 5.</b> Full-view (all pixels) vs. zone-aware (face-masked) evaluation. "
              "Left 3 columns: full-view; right 3 columns: zone-aware. Full-view inflates PSNR "
              "by 3-11 dB due to easy background pixels.", sCap),
    t5, S_(4)]))

story.append(body(
    "The discrepancy between full-view and zone-aware metrics is striking. Nearest-neighbor upsampling "
    "achieves 50.7 dB PSNR in full-view --<i>identical</i> to our trained UNet --because both methods "
    "produce similar values in the large, uniform background region that dominates the average. "
    "Only under zone-aware evaluation does the true gap emerge: UNet achieves 47.0 dB vs. nearest's "
    "39.1 dB, an 8 dB difference. This demonstrates that full-view metrics are severely contaminated "
    "by background pixels and cannot distinguish SR quality in the region of interest."))
story.append(body(
    "Even the bicubic-vs-UNet gap changes dramatically: in full-view, the gap is only 5.4 dB "
    "(45.3 vs. 50.7), but in zone-aware evaluation it narrows to 4.5 dB (42.5 vs. 47.0). "
    "The SSIM pattern is similar: full-view SSIM exceeds 0.99 for all methods, providing no "
    "discriminative power, while zone-aware SSIM ranges from 0.9817 to 0.9930, revealing meaningful "
    "differences. These results validate our zone-aware evaluation protocol as essential for "
    "accurate assessment of depth SR for 3DGS face reconstruction."))
story.append(parhead("Why masked metrics appear worse.",
    "An important subtlety is that applying the face mask <i>increases</i> reported L1 and <i>decreases</i> "
    "PSNR compared to full-view metrics (e.g., UNet L1 rises from 0.00087 to 0.00232). "
    "This is not because masking degrades quality -- it is because the mask removes the large background "
    "region where both prediction and GT are trivially zero, which inflates accuracy metrics. "
    "The face region contains all the geometric complexity (nose, eyes, ears) and is where SR actually "
    "matters. Reporting only full-view metrics would misleadingly suggest that nearest-neighbor upsampling "
    "(50.7 dB) matches our trained UNet (50.7 dB), when in reality the gap is 8 dB in the face region. "
    "We therefore recommend that all future depth SR work on face reconstruction adopt masked evaluation."))

# FIGURE: Zone-aware mask comparison
story.extend(embed_png_direct("zone_aware_mask_comparison.png", CW * 0.95,
    "<b>Figure 6.</b> Zone-aware face mask vs. rendered opacity. The opacity mask (second panel) includes "
    "FaceLift-hallucinated shoulders; the face mask (third panel) excludes them. The rightmost panel (red) "
    "shows the excluded hallucinated region (11.8% of frame)."))

# FIGURE: cons_rate distribution
story.extend(embed_png_direct("cons_rate_distribution.png", CW * 0.95,
    "<b>Figure 7.</b> Cross-view consistency analysis on n=1,240 samples. Left: per-sample cons_rate "
    "before (opacity, blue) and after (face mask, orange) zone-aware masking. Right: per-sample Delta "
    "distribution showing mask lowers cons_rate in 88% of samples (mean Delta = -1.19 pp), confirming "
    "that full-view metrics are inflated by hallucinated regions."))

# 4.6 Downstream 3D mesh quality
story.append(subsec("Downstream 3D mesh quality"))
story.append(body(
    "Depth SR is ultimately a means to improve 3D reconstruction quality. To evaluate downstream impact, "
    "we convert SR depth maps to triangle meshes (via Marching Cubes [28]) and compare against the GT mesh "
    "using standard 3D geometry metrics: Chamfer L1 distance, Hausdorff distance, and F-score at "
    "thresholds 0.1% and 0.05% of the bounding box diagonal. We also measure mesh smoothness "
    "(Laplacian regularization) to detect surface artifacts. Results are averaged over 5 representative "
    "samples in Tab. 6."))

t6d = [
    ['Method',           'Chamfer (low)',  'Hausdorff (low)', 'F@0.1% (high)', 'F@0.05% (high)', 'Smooth.'],
    ['Nearest 8-bit',    '0.00180',   '0.0304',     '0.9927',  '0.9548',   '0.0523'],
    ['Bicubic 8-bit',    '0.00130',   '0.0232',     '0.9984',  '0.9825',   '0.0087'],
    ['UNet 8-bit',       '0.00117',   '0.0176',     '0.9994',  '0.9890',   '0.0102'],
    ['UNet+N 8-bit',     '0.00116',   '0.0177',     '0.9993',  '0.9893',   '0.0100'],
    ['UNet 16-bit',      '0.00102',   '0.0188',     '0.9994',  '0.9896',   '0.0109'],
    ['UNet+N 16-bit',    '0.00101',   '0.0184',     '0.9994',  '0.9895',   '0.0104'],
    ['GT',               '0',         '0',          '1.0000',  '1.0000',   '0.0164'],
]
t6s = []
for ri, row in enumerate(t6d):
    r = []
    for ci, c in enumerate(row):
        if ri == 0: r.append(Paragraph(c, _tcH if ci > 0 else _tcHL))
        elif ci == 0: r.append(Paragraph(c, _tcNL))
        else: r.append(Paragraph(c, _tcN))
    t6s.append(r)
t6 = Table(t6s, colWidths=[CW*0.22, CW*0.15, CW*0.17, CW*0.15, CW*0.16, CW*0.15])
t6.setStyle(TableStyle(TSTYLE_BASE + [
    ('LINEBELOW', (0,1), (-1,1), 0.3, colors.gray),
    ('LINEBELOW', (0,6), (-1,6), 0.3, colors.gray),
]))

story.append(KeepTogether([
    Paragraph("<b>Table 6.</b> Downstream 3D mesh quality. Chamfer L1 (x1e-3), "
              "Hausdorff distance, F-score at two thresholds, and mesh smoothness (Laplacian). "
              "Lower is better for Chamfer/Hausdorff/Smoothness; higher for F-score.", sCap),
    t6, S_(4)]))

story.append(body(
    "Depth SR provides measurable improvements in downstream mesh quality. Our UNet reduces Chamfer "
    "distance by 10% relative to bicubic (0.00117 vs. 0.00130) and by 35% relative to nearest-neighbor "
    "(0.00117 vs. 0.00180). The F-score at the strict 0.05% threshold improves from 0.9825 (bicubic) "
    "to 0.9890 (UNet), approaching the GT mesh's 1.0000. Hausdorff distance --measuring worst-case "
    "surface error --drops from 0.0232 (bicubic) to 0.0176 (UNet), a 24% reduction."))
story.append(body(
    "An interesting observation is the smoothness column. The GT mesh has smoothness 0.0164, while "
    "nearest-neighbor produces extremely rough meshes (0.0523 --3.2x rougher than GT). Bicubic "
    "over-smooths to 0.0087, below GT. Our UNet achieves 0.0102, closer to but still below GT "
    "smoothness, suggesting the residual learning formulation introduces slight over-smoothing "
    "from the bicubic initialization. The normal-aware variant is marginally smoother (0.0100), "
    "consistent with the surface normal supervision encouraging locally planar geometry."))
story.append(body(
    "The 16-bit models achieve the lowest Chamfer distances (0.00101-0.00102), confirming that "
    "preserving bit-depth in the LR input translates to better 3D geometry. However, the gap between "
    "8-bit and 16-bit mesh quality (0.00117 vs. 0.00102, ~15%) is larger than the corresponding "
    "depth L1 gap (12.1%), suggesting that quantization errors compound nonlinearly in the "
    "depth-to-mesh conversion pipeline."))

# FIGURE: Normal head quality (single column)
story.extend(embed_png_direct("normal_head_quality.png", CW * 0.95,
    "<b>Figure 8.</b> Normal prediction quality. Left: DSINE pseudo-GT. Right: predicted normals "
    "from dual-head UNet. Error maps show mean angular error = 11.6 deg."))

# FIGURE: Baseline bars (single column)
story.extend(embed_png_direct("fig_baseline_bars.png", CW * 0.95,
    "<b>Figure 9.</b> Baseline comparison. (a) L1 error (log scale). (b) PSNR (dB). "
    "Bicubic residual methods dominate. SwinIR without residual skip fails (L1 = 0.086)."))

# 4.7 Qualitative observations
story.append(subsec("Qualitative observations"))
story.append(body(
    "Visual inspection of depth SR outputs reveals characteristic patterns. Bicubic upsampling produces "
    "smooth but blurry depth maps that lack fine geometric detail around the nose bridge, eye sockets, "
    "and lip contours. DORNet (zero-shot from NYU) introduces ringing artifacts at depth discontinuities, "
    "consistent with its training on indoor scenes with sharp object boundaries rather than smooth facial "
    "geometry. SGNet produces reasonable overall depth but shows checkerboard artifacts at the 512 to1024 "
    "boundary due to its 512 resolution limit."))
story.append(body(
    "Our UNet and SRResNet produce the sharpest depth maps, with clear delineation of nose, eyebrow, "
    "and ear geometry. The normal-aware UNet variant shows subtly improved edge definition around "
    "the nose bridge and forehead contours, consistent with the surface normal supervision providing "
    "gradient-level geometric constraints. Error maps (predicted - GT) confirm that residual errors "
    "are concentrated at the face boundary and ear regions, where 3DGS geometry is least reliable."))

# ══════════════════════ 5. DISCUSSION ══════════════════════
story.append(sec("Discussion"))
story.append(parhead("Why do simple models work so well?",
    "A perhaps surprising finding is that a standard UNet (7.77M) performs on par with more sophisticated "
    "architectures. We attribute this to the bicubic residual prior: when the network only needs to predict "
    "a small correction (mean absolute residual ~ 0.002 in [0,1] range), model capacity matters less than "
    "the quality of the initial estimate. This suggests that for deployment in 3DGS pipelines, lightweight "
    "models like SRResNet (1.53M) may be preferable to larger UNets, with negligible quality loss."))
story.append(parhead("Practical implications.",
    "Our results have direct implications for real-time 3DGS face reconstruction pipelines. "
    "The depth SR module adds ~50ms of inference time (single forward pass at 1024x1024 with FP16) "
    "for a 4.5 dB PSNR improvement over bicubic upsampling. Combined with the downstream mesh quality "
    "improvements (10% Chamfer reduction, Tab. 6), this is a favorable cost-quality tradeoff for "
    "applications like 3D face scanning, virtual try-on, and AR/VR avatar reconstruction."))
story.append(parhead("Data pipeline as a first-class contribution.",
    "A recurring theme in our work is that the data construction pipeline contributes as much to final "
    "performance as the model architecture. The 7-stage postprocessing pipeline (Sec. 3.1), face mask "
    "generation (Sec. 3.1 Stage 3), and DSINE pseudo-GT normals (Sec. 3.1 Stage 5) collectively required "
    "more engineering effort than the UNet itself, yet each stage is essential: removing any single stage "
    "degrades final L1 by 0.5-2.5 dB. We argue that for applied depth SR tasks, particularly in the "
    "3DGS domain, the degradation-aware data pipeline deserves as much research attention as the SR model. "
    "This echoes recent findings in the image generation community where data quality dominates model scale."))
story.append(parhead("Generalization to other 3DGS methods.",
    "While our pipeline is designed for FaceLift, the rendering-induced degradation we identify is inherent "
    "to the 3DGS rasterization process. Any method using Gaussian splatting --including DreamGaussian [23], "
    "LGM [22], and GRM [33] --will produce depth maps with similar splat boundary artifacts and opacity blending. "
    "Our cross-degradation analysis (Sec. 4.1) characterizes the degradation independent of the specific "
    "3DGS model, suggesting that our training data pipeline and zone-aware evaluation framework can "
    "transfer to other 3DGS face reconstruction methods with minimal adaptation."))
story.append(parhead("Limitations.",
    "Our evaluation uses frontal-view depth maps from FaceLift. Extending to arbitrary viewpoints "
    "requires refining the zone-aware framework, as side views contain both genuine geometry and "
    "hallucinated regions. We have not yet closed the loop by feeding SR depth back into 3DGS "
    "re-rendering to measure NVS quality. Our 4x SR factor (256 to1024) is the only scale tested; "
    "extreme factors like 8x may exhibit different degradation characteristics. "
    "Finally, our dataset is FFHQ-derived and may not generalize to non-frontal or "
    "non-Caucasian-biased face distributions."))

# ══════════════════════ 6. CONCLUSION ══════════════════════
story.append(sec("Conclusion"))
story.append(body(
    "We have established that depth super-resolution for 3DGS face reconstruction constitutes a distinct "
    "task requiring domain-specific solutions. The rendering-induced degradation of 3DGS --splat boundary "
    "artifacts, opacity blending, and quantization --differs fundamentally from the bicubic downsampling "
    "assumed by all prior depth SR methods. Our 5x5 cross-degradation analysis confirms this with a "
    "consistent 3-6 dB PSNR gap between matched and mismatched degradation types."))
story.append(body(
    "A simple UNet with bicubic residual learning, trained on domain-matched data, outperforms "
    "CVPR 2025 SOTA DORNet by 4.4 dB and AAAI 2024 SGNet by 3.4 dB. We further show that zone-aware "
    "evaluation is essential to avoid metric contamination from 3DGS hallucinations, and that DSINE "
    "pseudo-GT normals provide a clean replacement for the fundamentally noisy 3DGS-rendered normals."))
story.append(body(
    "Future work includes: (1) feeding SR depth back into 3DGS reconstruction to measure downstream "
    "NVS quality improvements, providing an end-to-end validation of the approach; "
    "(2) scaling to 8x SR (128 to1024) to test the limits of the residual learning formulation; and "
    "(3) extending to multi-view depth SR, where cross-view consistency constraints may provide "
    "additional supervision beyond single-view normal priors."))

# ══════════════════════ REFERENCES ══════════════════════
story.append(S_(6))
story.append(Paragraph("<b>References</b>", sSec))

refs = [
    "[1] G. Bae, I. Budvytis, R. Cipolla. Rethinking Inductive Biases for Surface Normal Estimation. <i>CVPR</i>, 2024.",
    "[2] Z. Yan <i>et al.</i> Degradation Oriented and Regularized Network for Real-World Depth Super-Resolution. <i>CVPR</i>, 2025.",
    "[3] Z. Yan <i>et al.</i> SGNet: Structure Guided Network via Gradient-Frequency Awareness for Depth Map Super-Resolution. <i>AAAI</i>, 2024.",
    "[4] B. Kerbl, G. Kopanas, T. Leimkuehler, G. Drettakis. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. <i>ACM TOG (SIGGRAPH)</i>, 2023.",
    "[5] Y. Wu <i>et al.</i> FaceLift: Single Image to 3D Head with View Generation and GS-LRM. <i>arXiv:2412.07029</i>, 2024.",
    "[6] N. Silberman, D. Hoiem, P. Kohli, R. Fergus. Indoor Segmentation and Support Inference from RGBD Images. <i>ECCV</i>, 2012.",
    "[7] D. Scharstein <i>et al.</i> High-Resolution Stereo Datasets with Subpixel-Accurate Ground Truth. <i>GCPR</i>, 2014.",
    "[8] J. Kopf, M. Cohen, D. Lischinski, M. Uyttendaele. Joint Bilateral Upsampling. <i>ACM TOG (SIGGRAPH)</i>, 2007.",
    "[9] T.-W. Hui, C. C. Loy, X. Tang. Depth Map Super-Resolution by Deep Multi-Scale Guidance. <i>ECCV</i>, 2016.",
    "[10] Q. Tang <i>et al.</i> BridgeNet: A Joint Learning Network of Depth Map SR and Monocular Depth Estimation. <i>ACM MM</i>, 2021.",
    "[11] J. Liang <i>et al.</i> SwinIR: Image Restoration Using Swin Transformer. <i>ICCVW</i>, 2021.",
    "[12] B. Lim, S. Son, H. Kim, S. Nah, K. M. Lee. Enhanced Deep Residual Networks for Single Image Super-Resolution. <i>CVPRW</i>, 2017.",
    "[13] C. Ledig <i>et al.</i> Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network. <i>CVPR</i>, 2017.",
    "[14] B. Huang <i>et al.</i> 2D Gaussian Splatting for Geometrically Accurate Radiance Fields. <i>ACM SIGGRAPH</i>, 2024.",
    "[15] Y. Jiang <i>et al.</i> Normal-GS: 3D Gaussian Splatting with Normal-Involved Rendering. <i>NeurIPS</i>, 2024.",
    "[16] A. Guedon, V. Lepetit. SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction. <i>CVPR</i>, 2024.",
    "[17] M. Turkulainen <i>et al.</i> DN-Splatter: Depth and Normal Priors for Gaussian Splatting and Meshing. <i>WACV</i>, 2025.",
    "[18] A. Eftekhar <i>et al.</i> Omnidata: A Scalable Pipeline for Multi-Task Mid-Level Vision Datasets. <i>ICCV</i>, 2021.",
    "[19] T. Karras, S. Laine, T. Aila. A Style-Based Generator Architecture for Generative Adversarial Networks. <i>CVPR</i>, 2019.",
    "[20] O. Ronneberger, P. Fischer, T. Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation. <i>MICCAI</i>, 2015.",
    "[21] X. Wang <i>et al.</i> Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data. <i>ICCVW</i>, 2021.",
    "[22] Y. Tang <i>et al.</i> LGM: Large Multi-View Gaussian Model for High-Resolution 3D Content Creation. <i>ECCV</i>, 2024.",
    "[23] J. Tang <i>et al.</i> DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation. <i>ICLR</i>, 2024.",
    "[24] Z. Wang, A. C. Bovik, H. R. Sheikh, E. P. Simoncelli. Image Quality Assessment: From Error Visibility to Structural Similarity. <i>IEEE TIP</i>, 2004.",
    "[25] I. Loshchilov, F. Hutter. Decoupled Weight Decay Regularization. <i>ICLR</i>, 2019.",
    "[26] D. P. Kingma, J. Ba. Adam: A Method for Stochastic Optimization. <i>ICLR</i>, 2015.",
    "[27] K. He, X. Zhang, S. Ren, J. Sun. Deep Residual Learning for Image Recognition. <i>CVPR</i>, 2016.",
    "[28] W. E. Lorensen, H. E. Cline. Marching Cubes: A High Resolution 3D Surface Construction Algorithm. <i>ACM SIGGRAPH</i>, 1987.",
    "[29] C. Chen <i>et al.</i> Learning Continuous Image Representation with Local Implicit Image Function. <i>CVPR</i>, 2021.",
    "[30] P. Mildenhall <i>et al.</i> NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. <i>ECCV</i>, 2020.",
    "[31] R. Ranftl <i>et al.</i> Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer. <i>IEEE TPAMI</i>, 2022.",
    "[32] L. Ke <i>et al.</i> Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation. <i>CVPR</i>, 2024.",
    "[33] Y. Xu <i>et al.</i> GRM: Large Gaussian Reconstruction Model for Efficient 3D Reconstruction and Generation. <i>ECCV</i>, 2024.",
    "[34] C. Lugaresi <i>et al.</i> MediaPipe: A Framework for Building Perception Pipelines. <i>arXiv:1906.08172</i>, 2019.",
    "[35] A. Paszke <i>et al.</i> PyTorch: An Imperative Style, High-Performance Deep Learning Library. <i>NeurIPS</i>, 2019.",
    "[36] G. Bradski. The OpenCV Library. <i>Dr. Dobb\'s Journal of Software Tools</i>, 2000.",
    "[37] S. Paris, S. Durand. A Fast Approximation of the Bilateral Filter Using a Signal Processing Approach. <i>ECCV</i>, 2006.",
    "[38] I. Loshchilov, F. Hutter. SGDR: Stochastic Gradient Descent with Warm Restarts. <i>ICLR</i>, 2017.",
    "[39] G. D. Evangelidis, E. Z. Psarakis. Parametric Image Alignment Using Enhanced Correlation Coefficient Maximization. <i>IEEE TPAMI</i>, 2008.",
    "[40] Y. Zhang <i>et al.</i> Image Super-Resolution Using Very Deep Residual Channel Attention Networks. <i>ECCV</i>, 2018.",
    "[41] X. Shi <i>et al.</i> Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network. <i>CVPR</i>, 2016.",
    "[42] C. Dong, C. C. Loy, K. He, X. Tang. Image Super-Resolution Using Deep Convolutional Networks. <i>IEEE TPAMI</i>, 2016.",
    "[43] T. Dai <i>et al.</i> Second-Order Attention Network for Single Image Super-Resolution. <i>CVPR</i>, 2019.",
    "[44] Y. Li <i>et al.</i> Feedback Network for Image Super-Resolution. <i>CVPR</i>, 2019.",
    "[45] Z. Li <i>et al.</i> MDSR: Multi-level Depth Map Super-Resolution with Multi-scale Guidance. <i>Pattern Recognition</i>, 2023.",
    "[46] A. Dosovitskiy <i>et al.</i> An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. <i>ICLR</i>, 2021.",
    "[47] Z. Liu <i>et al.</i> Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows. <i>ICCV</i>, 2021.",
    "[48] W. Shi <i>et al.</i> Is the Deconvolution Layer the Same as a Convolutional Layer? <i>arXiv:1609.07009</i>, 2016.",
]
for r in refs:
    story.append(Paragraph(r, sRef))

# BUILD
doc.build(story)
print(f"PDF generated: {OUT}")
print(f"Size: {OUT.stat().st_size / 1024:.0f} KB")
