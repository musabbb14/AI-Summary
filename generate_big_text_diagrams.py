import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Ellipse, Rectangle, Polygon
import numpy as np

OUTPUT_DIR = r"c:\Users\user\Desktop\AI-Summary\report_diagrams"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global styling for crisp academic typography
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans']
plt.rcParams['axes.edgecolor'] = '#0F172A'
plt.rcParams['axes.linewidth'] = 1.4


# ==============================================================================
# FIGURE 3.1: UML USE CASE DIAGRAM (PERFECTED BIG TEXT)
# ==============================================================================
def generate_figure_3_1():
    fig, ax = plt.subplots(figsize=(11.5, 8.5), dpi=300)
    ax.set_xlim(-6, 118)
    ax.set_ylim(-7, 88)
    ax.axis('off')
    fig.patch.set_facecolor('#ffffff')

    # System Boundary Box
    sys_box = FancyBboxPatch((28, 1), 86, 83, boxstyle="round,pad=1.0,rounding_size=2",
                             edgecolor="#0F172A", facecolor="#F8FAFC", linewidth=2.5, zorder=1)
    ax.add_patch(sys_box)
    ax.text(71, 79.5, "«system» DocSummarize Application", ha='center', va='center',
            fontsize=15.5, fontweight='bold', color='#0F172A', zorder=6)

    # Actor: Stick Figure (x=11, y=41)
    actor_x, actor_y = 11, 41
    head = plt.Circle((actor_x, actor_y + 12), 3.8, edgecolor="#0F172A", facecolor="#E2E8F0", linewidth=2.5, zorder=5)
    ax.add_patch(head)
    ax.plot([actor_x, actor_x], [actor_y + 8.2, actor_y - 2], color="#0F172A", linewidth=2.5, zorder=5)
    ax.plot([actor_x - 6.5, actor_x + 6.5], [actor_y + 4.5, actor_y + 4.5], color="#0F172A", linewidth=2.5, zorder=5)
    ax.plot([actor_x, actor_x - 5.5], [actor_y - 2, actor_y - 13], color="#0F172A", linewidth=2.5, zorder=5)
    ax.plot([actor_x, actor_x + 5.5], [actor_y - 2, actor_y - 13], color="#0F172A", linewidth=2.5, zorder=5)
    ax.text(actor_x, actor_y - 19.5, "User\n(Student / Researcher)", ha='center', va='center',
            fontsize=13.0, fontweight='bold', color="#0F172A", multialignment='center', zorder=6)

    # Primary Use Case: Summarize Document
    p_uc = Ellipse((49, 41), 29, 12.5, edgecolor="#1D4ED8", facecolor="#DBEAFE", linewidth=2.5, zorder=3)
    ax.add_patch(p_uc)
    ax.text(49, 41, "Summarize Document\n(Primary Use Case)", ha='center', va='center',
            fontsize=12.5, fontweight='bold', color="#1E3A8A", multialignment='center', zorder=6)

    # Association Line
    ax.plot([actor_x + 6.5, 49 - 14.5], [actor_y + 4.5, 41], color="#0F172A", linewidth=2.2, zorder=2)

    # Included Sub-cases
    included_cases = [
        {"y": 66.5, "text": "Upload Document\n(PDF, DOCX, TXT)"},
        {"y": 50.0, "text": "Select Summary Length\n(Short, Medium, Long)"},
        {"y": 33.5, "text": "View Summary & Assets\n(Highlights, Figures, Code)"},
        {"y": 17.0, "text": "Download Summary\n(PDF, DOCX, TXT)"}
    ]

    for inc in included_cases:
        oval = Ellipse((94, inc["y"]), 35, 11.2, edgecolor="#2563EB", facecolor="#EFF6FF", linewidth=2.0, zorder=3)
        ax.add_patch(oval)
        ax.text(94, inc["y"], inc["text"], ha='center', va='center',
                fontsize=11.8, fontweight='bold', color="#1E40AF", multialignment='center', zorder=6)

        ax.annotate("", xy=(94 - 17.5, inc["y"]), xytext=(49 + 14.5, 41 + (inc["y"] - 41)*0.22),
                    arrowprops=dict(arrowstyle="->", linestyle="--", color="#2563EB", linewidth=1.8), zorder=2)
        lbl_x = 71.5
        lbl_y = 41 + (inc["y"] - 41) * 0.58
        ax.text(lbl_x, lbl_y, "«include»", ha='center', va='center',
                fontsize=11.0, fontweight='bold', color="#1D4ED8", zorder=6,
                bbox=dict(boxstyle="square,pad=0.2", facecolor="#F8FAFC", edgecolor="none"))

    # Extended Sub-case
    ext_oval = Ellipse((49, 11.5), 28, 10.0, edgecolor="#DC2626", facecolor="#FEF2F2", linewidth=2.0, zorder=3)
    ax.add_patch(ext_oval)
    ax.text(49, 11.5, "Cancel Processing\n(Interrupt Running Job)", ha='center', va='center',
            fontsize=11.8, fontweight='bold', color="#991B1B", multialignment='center', zorder=6)

    ax.annotate("", xy=(49, 41 - 6.25), xytext=(49, 11.5 + 5.0),
                arrowprops=dict(arrowstyle="->", linestyle="--", color="#DC2626", linewidth=1.8), zorder=2)
    ax.text(55.5, 26.0, "«extend»", ha='center', va='center',
            fontsize=11.0, fontweight='bold', color="#DC2626", zorder=6,
            bbox=dict(boxstyle="square,pad=0.2", facecolor="#F8FAFC", edgecolor="none"))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "Figure_3_1_UML_Use_Case.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 3.1 updated:", path)


# ==============================================================================
# FIGURE 3.2: SYSTEM ARCHITECTURE (PERFECTED BIG TEXT & NO OVERLAP)
# ==============================================================================
def generate_figure_3_2():
    # Compact but still spacious enough for all content
    fig, ax = plt.subplots(figsize=(15.0, 12.0), dpi=300)
    ax.set_xlim(0, 145)
    ax.set_ylim(-6.0, 145)
    ax.axis('off')
    fig.patch.set_facecolor('#ffffff')

    def draw_box(x, y, w, h, title, items=None, fc="#FFFFFF", ec="#1E293B", 
                 title_color="#0F172A", text_color="#0F172A", lw=1.8, badge=None, badge_color="#0284C7"):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4",
                             facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3)
        ax.add_patch(box)
        header_y = y + h - 2.0
        ax.text(x + w/2, header_y, title, ha='center', va='top',
                fontsize=13.5, fontweight='bold', color=title_color, zorder=5)
        if badge:
            badge_y = header_y - 2.8
            ax.text(x + w/2, badge_y, badge, ha='center', va='top',
                    fontsize=11.5, fontweight='bold', fontstyle='italic', color=badge_color, zorder=5)
            content_start_y = badge_y - 2.8
        else:
            content_start_y = header_y - 2.8
        ax.plot([x + 2, x + w - 2], [content_start_y + 0.8, content_start_y + 0.8], 
                color=ec, linewidth=1.0, alpha=0.4, zorder=4)
        if items:
            cur_y = content_start_y - 1.2
            for itm in items:
                ax.text(x + 2.5, cur_y, f"• {itm}", ha='left', va='top',
                        fontsize=11.5, fontweight='bold', color=text_color, zorder=5)
                cur_y -= 2.6

    def draw_layer(x, y, w, h, label, bg_color="#F8FAFC", border_color="#CBD5E1"):
        layer_box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=2.0",
                                  facecolor=bg_color, edgecolor=border_color, linewidth=2.0, 
                                  linestyle="solid", zorder=1)
        ax.add_patch(layer_box)
        # Place label comfortably inside top border with solid breathing room
        ax.text(x + 4.0, y + h - 2.0, label, ha='left', va='top',
                fontsize=14.0, fontweight='bold', color="#0F172A", zorder=2)

    # 1. Presentation Tier (y: 112 to 140, h=28)
    draw_layer(4, 112, 137, 28, "PRESENTATION TIER (Browser Client — HTML5 / CSS3 / JavaScript)", 
               bg_color="#F0F9FF", border_color="#0284C7")
    draw_box(7, 114.5, 41, 20.5, "Upload & Configuration",
             badge="User Interface", badge_color="#0369A1",
             items=[
                 "Document Dropzone (.pdf, .docx, .txt)",
                 "Summary Ratio Selector (25%, 45%, 65%)",
                 "Client-side File Validation"
             ],
             fc="#FFFFFF", ec="#0284C7", title_color="#0369A1")
    draw_box(52, 114.5, 41, 20.5, "Status Monitor",
             badge="Fixed-Interval AJAX Polling", badge_color="#0369A1",
             items=[
                 "1s Fixed Polling via /status/<job_id>",
                 "Live Progress Bar & Stage Indicator",
                 "Job Cancellation Handler"
             ],
             fc="#FFFFFF", ec="#0284C7", title_color="#0369A1")
    draw_box(97, 114.5, 41, 20.5, "Summary & Asset Viewer",
             badge="Results Display", badge_color="#0369A1",
             items=[
                 "Structured Text & Key Highlights",
                 "Preserved Technical Figures Gallery",
                 "Detected Code Listings Display"
             ],
             fc="#FFFFFF", ec="#0284C7", title_color="#0369A1")

    ax.annotate("", xy=(52, 124.7), xytext=(48, 124.7),
                arrowprops=dict(arrowstyle="->", color="#0284C7", linewidth=2.0), zorder=6)
    ax.annotate("", xy=(97, 124.7), xytext=(93, 124.7),
                arrowprops=dict(arrowstyle="->", color="#0284C7", linewidth=2.0), zorder=6)

    # Gap 1: Presentation <-> Application Logic (y: 104 to 112)
    ax.annotate("", xy=(72.5, 104), xytext=(72.5, 112),
                arrowprops=dict(arrowstyle="<->", color="#0F172A", linewidth=2.0), zorder=6)
    ax.text(75.0, 108, "REST API (JSON / HTTP POST & GET)", ha='left', va='center',
            fontsize=12.0, fontweight='bold', color="#0F172A", zorder=6)

    # 2. Application Logic Tier (y: 70 to 104, h=34)
    draw_layer(4, 70, 137, 34, "APPLICATION LOGIC & CONTROLLER TIER (Flask Web Server)",
               bg_color="#EEF2FF", border_color="#4F46E5")
    draw_box(7, 72.5, 41, 25.5, "AppController & Security",
             badge="app.py & file_validator.py", badge_color="#4338CA",
             items=[
                 "Endpoint Routing (/summarize, /status)",
                 "Rate Limiter (10 req/min per IP)",
                 "Upload Validator (Allowed Ext, <=20MB)",
                 "Session State Management"
             ],
             fc="#FFFFFF", ec="#4F46E5", title_color="#3730A3")
    draw_box(52, 72.5, 41, 25.5, "Asynchronous Job Worker",
             badge="_run_summarization_job()", badge_color="#4338CA",
             items=[
                 "Thread Task Dispatcher (Non-blocking)",
                 "Background Pipeline Execution",
                 "Atomic Progress State Tracking",
                 "Exception Trapping & Status Logging"
             ],
             fc="#FFFFFF", ec="#4F46E5", title_color="#3730A3")
    draw_box(97, 72.5, 41, 25.5, "Multi-Format Export",
             badge="download_generator.py", badge_color="#4338CA",
             items=[
                 "PDF Document Builder (ReportLab)",
                 "Word Document Builder (python-docx)",
                 "Plain Text Formatter (.txt)",
                 "Attachment Stream via /download"
             ],
             fc="#FFFFFF", ec="#4F46E5", title_color="#3730A3")

    ax.annotate("", xy=(52, 85.2), xytext=(48, 85.2),
                arrowprops=dict(arrowstyle="->", color="#4F46E5", linewidth=2.0), zorder=6)
    ax.annotate("", xy=(97, 85.2), xytext=(93, 85.2),
                arrowprops=dict(arrowstyle="->", linestyle="dashed", color="#4F46E5", linewidth=2.0), zorder=6)

    # Gap 2: Application Logic -> Processing Engine (y: 61 to 70)
    ax.annotate("", xy=(72.5, 61), xytext=(72.5, 70),
                arrowprops=dict(arrowstyle="->", color="#0F172A", linewidth=2.0), zorder=6)
    ax.text(75.0, 65.5, "Execution Pipeline Dispatch", ha='left', va='center',
            fontsize=12.0, fontweight='bold', color="#0F172A", zorder=6)

    # 3. Document Processing Engine Tier (y: 19 to 61, h=42)
    draw_layer(4, 19, 137, 42, "DOCUMENT PROCESSING & SUMMARIZATION ENGINE TIER",
               bg_color="#F8FAFC", border_color="#334155")
    draw_box(7, 21.5, 31, 33.5, "Document Ingestion",
             badge="pdf_extractor / docx", badge_color="#047857",
             items=[
                 "PyMuPDF Parser (PDF)",
                 "python-docx Parser (Word)",
                 "UTF-8 Stream Reader",
                 "Unicode Normalizer",
                 "Slide Header/Footer Stripping"
             ],
             fc="#FFFFFF", ec="#059669", title_color="#065F46")
    draw_box(40.5, 21.5, 36.5, 33.5, "Structural Preservation",
             badge="Novel Research Contribution", badge_color="#BE123C",
             items=[
                 "DiagramExtractor (PyMuPDF):",
                 "  - Image size filter >= 120x120 px",
                 "  - Deduplication via xref hash",
                 "  - Bounded cap (max 12 figures)",
                 "CodeListingDetector:",
                 "  - Monospace font coverage > 60%",
                 "  - Block span length >= 40 chars"
             ],
             fc="#FFF1F2", ec="#BE123C", title_color="#881337", text_color="#9F1239", lw=2.2)
    draw_box(79.5, 21.5, 31, 33.5, "Sliding-Window Chunker",
             badge="chunking.py", badge_color="#2563EB",
             items=[
                 "Regex Sentence Tokenizer",
                 "900-Token Chunk Window",
                 "60-Token Overlap Buffer",
                 "Cross-Boundary Continuity",
                 "Heading-Aware Structuring"
             ],
             fc="#FFFFFF", ec="#2563EB", title_color="#1E40AF")
    draw_box(113.0, 21.5, 25.5, 33.5, "Summarizer Core",
             badge="summarizer.py", badge_color="#7C3AED",
             items=[
                 "TF-IDF Vector Model",
                 "Sentence Centrality",
                 "Centroid Cosine",
                 "Position Bias Weighting",
                 "MMR Re-Ranking (0.65)"
             ],
             fc="#FFFFFF", ec="#7C3AED", title_color="#5B21B6")

    # Ingestion -> Structural
    ax.annotate("", xy=(40.5, 38), xytext=(38.0, 38),
                arrowprops=dict(arrowstyle="->", color="#BE123C", linewidth=2.0), zorder=6)
    ax.text(39.25, 39.5, "PDF Assets", ha='center', va='bottom', fontsize=10.5, fontweight='bold', color="#BE123C",
            bbox=dict(boxstyle="square,pad=0.15", facecolor="#F8FAFC", edgecolor="none"))

    # Ingestion -> Chunker
    ax.plot([38.0, 39.2, 39.2, 78.0, 78.0, 79.5], [29, 29, 20.0, 20.0, 38, 38], color="#059669", linewidth=1.8, zorder=5)
    ax.annotate("", xy=(79.5, 38), xytext=(78.0, 38),
                arrowprops=dict(arrowstyle="->", color="#059669", linewidth=1.8), zorder=6)
    ax.text(58.5, 20.6, "Cleaned Text Stream", ha='center', va='bottom', fontsize=11.0, fontweight='bold', color="#059669",
            bbox=dict(boxstyle="square,pad=0.15", facecolor="#F8FAFC", edgecolor="none"))

    # Chunker -> Summarizer
    ax.annotate("", xy=(113.0, 38), xytext=(110.5, 38),
                arrowprops=dict(arrowstyle="->", color="#2563EB", linewidth=2.0), zorder=6)

    # Structural -> Summarizer
    ax.plot([77.0, 78.5, 78.5, 111.5, 111.5, 113.0], [38, 38, 56.5, 56.5, 30, 30], color="#BE123C", linestyle="dotted", linewidth=1.8, zorder=5)
    ax.annotate("", xy=(113.0, 30), xytext=(111.5, 30),
                arrowprops=dict(arrowstyle="->", color="#BE123C", linewidth=1.8), zorder=6)
    ax.text(95.0, 57.0, "Preserved Assets (Figures & Code)", ha='center', va='bottom', fontsize=10.5, fontweight='bold', color="#BE123C",
            bbox=dict(boxstyle="square,pad=0.15", facecolor="#F8FAFC", edgecolor="none"))

    # Summarizer -> Export
    ax.annotate("", xy=(123, 72.5), xytext=(123, 55.0),
                arrowprops=dict(arrowstyle="->", color="#4338CA", linewidth=2.0), zorder=6)
    ax.text(124.5, 63.5, "Final Summary &\nIntegrated Assets", ha='left', va='center',
            fontsize=11.0, fontweight='bold', color="#3730A3")

    # 4. Storage & State Tier (y: -4.5 to 11.5, h=16.0)
    draw_layer(4, -4.5, 137, 16.0, "EPHEMERAL DATA & STATE TIER (Stateless Architecture)",
               bg_color="#ECFDF5", border_color="#10B981")
    box_s1 = FancyBboxPatch((7, -3.0), 63, 8.5, boxstyle="round,pad=0,rounding_size=1.2",
                            facecolor="#FFFFFF", edgecolor="#10B981", linewidth=1.6, zorder=3)
    ax.add_patch(box_s1)
    ax.text(38.5, 2.2, "In-Memory Session Store (summaries dictionary)", 
            ha='center', va='center', fontsize=11.5, fontweight='bold', color="#065F46", zorder=4)
    ax.text(38.5, -1.0, "Job state tracking: progress %, active stage, cached summary (1h TTL)", 
            ha='center', va='center', fontsize=10.2, fontweight='bold', color="#047857", zorder=4)

    box_s2 = FancyBboxPatch((75, -3.0), 63, 8.5, boxstyle="round,pad=0,rounding_size=1.2",
                            facecolor="#FFFFFF", edgecolor="#10B981", linewidth=1.6, zorder=3)
    ax.add_patch(box_s2)
    ax.text(106.5, 2.2, "Temporary File Workspace (uploads/ and outputs/)", 
            ha='center', va='center', fontsize=11.5, fontweight='bold', color="#065F46", zorder=4)
    ax.text(106.5, -1.0, "Isolated disk scratchpad for files & exports (cleaned post-job)", 
            ha='center', va='center', fontsize=10.2, fontweight='bold', color="#047857", zorder=4)

    # Clean inter-tier double arrows spanning exclusively within the gap (y: 11.5 to 19.0)
    ax.annotate("", xy=(38.5, 11.5), xytext=(38.5, 19.0),
                arrowprops=dict(arrowstyle="<->", color="#059669", linewidth=2.0), zorder=6)
    ax.text(41.0, 15.25, "State & Progress Sync", ha='left', va='center', fontsize=11.0, fontweight='bold', color="#059669")

    ax.annotate("", xy=(106.5, 11.5), xytext=(106.5, 19.0),
                arrowprops=dict(arrowstyle="<->", color="#059669", linewidth=2.0), zorder=6)
    ax.text(109.0, 15.25, "File Read / Write Buffer", ha='left', va='center', fontsize=11.0, fontweight='bold', color="#059669")

    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    out_path = os.path.join(OUTPUT_DIR, "Figure 3.2 System Architecture.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 3.2 updated:", out_path)


# ==============================================================================
# FIGURE 3.3: SYSTEM PROCESS FLOWCHART (PERFECTED BIG TEXT & NO OVERLAP)
# ==============================================================================
def generate_figure_3_3():
    fig, ax = plt.subplots(figsize=(12.0, 13.5), dpi=300)
    ax.set_xlim(0, 124)
    ax.set_ylim(-26, 165)
    ax.axis('off')
    fig.patch.set_facecolor('#ffffff')

    def draw_step(x, y, w, h, title, subtitle=None, fc="#FFFFFF", ec="#0284C7", 
                  title_color="#0F172A", lw=1.8, is_highlight=False):
        if is_highlight:
            fc = "#FFF1F2"
            ec = "#BE123C"
            lw = 2.2
            title_color = "#881337"
        box = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0,rounding_size=1.4",
                             facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3)
        ax.add_patch(box)
        if subtitle:
            ax.text(x, y + 1.8, title, ha='center', va='center',
                    fontsize=13.0, fontweight='bold', color=title_color, zorder=5)
            ax.text(x, y - 2.2, subtitle, ha='center', va='center',
                    fontsize=11.0, fontweight='bold', color="#334155" if not is_highlight else "#9F1239", zorder=5)
        else:
            ax.text(x, y, title, ha='center', va='center',
                    fontsize=13.5, fontweight='bold', color=title_color, zorder=5)

    def draw_decision(x, y, w, h, text):
        diamond = Polygon([(x, y + h/2), (x + w/2, y), (x, y - h/2), (x - w/2, y)],
                          closed=True, facecolor="#FEF3C7", edgecolor="#D97706", linewidth=1.8, zorder=3)
        ax.add_patch(diamond)
        ax.text(x, y, text, ha='center', va='center', fontsize=11.5, fontweight='bold', color="#92400E", zorder=5)

    # 1. Start Node
    ax.add_patch(FancyBboxPatch((48, 153), 24, 6.5, boxstyle="round,pad=0,rounding_size=3.2",
                                facecolor="#0F172A", edgecolor="#0F172A", zorder=3))
    ax.text(60, 156.25, "START", ha='center', va='center', fontsize=13.0, fontweight='bold', color="#FFFFFF", zorder=5)

    ax.annotate("", xy=(60, 145), xytext=(60, 153),
                arrowprops=dict(arrowstyle="->", color="#334155", linewidth=2.0), zorder=2)

    # 2. Upload & Configure
    draw_step(60, 139, 74, 10.5, "Upload Document & Select Parameters",
              "Select ratio (25%, 45%, 65%) & toggles (figures/code)",
              ec="#0284C7", title_color="#0369A1")

    ax.annotate("", xy=(60, 127.5), xytext=(60, 133.75),
                arrowprops=dict(arrowstyle="->", color="#334155", linewidth=2.0), zorder=2)

    # 3. Decision 1: Validation
    draw_decision(60, 121, 38, 12, "Valid Format\n& <= 20MB?")

    # Invalid branch (Right)
    ax.annotate("", xy=(100, 121), xytext=(79, 121),
                arrowprops=dict(arrowstyle="->", color="#DC2626", linewidth=2.0), zorder=2)
    ax.text(87, 123.5, "[No]", fontsize=11.5, fontweight='bold', color="#DC2626")
    draw_step(111, 121, 20, 9, "Return Error", "Alert user in UI", ec="#DC2626", fc="#FEF2F2", title_color="#DC2626")

    # Valid branch (Down)
    ax.annotate("", xy=(60, 107.5), xytext=(60, 115),
                arrowprops=dict(arrowstyle="->", color="#16A34A", linewidth=2.0), zorder=2)
    ax.text(62.5, 111.0, "[Yes]", fontsize=11.5, fontweight='bold', color="#16A34A")

    # 4. Asynchronous Job Spawn
    draw_step(60, 101, 74, 10.5, "Initialize Background Task & Job UUID",
              "Return HTTP 200 OK | Begin fixed-interval AJAX polling at 1s",
              ec="#4F46E5", title_color="#3730A3")

    ax.annotate("", xy=(60, 89.5), xytext=(60, 95.75),
                arrowprops=dict(arrowstyle="->", color="#334155", linewidth=2.0), zorder=2)

    # 5. Document Ingestion (Fork to Text & Figures)
    draw_step(60, 83.5, 74, 10.5, "Parse Source File Content",
              "PyMuPDF (.pdf) / python-docx (.docx) / Text Reader (.txt)",
              ec="#059669", title_color="#065F46")

    # Branching: Left for Text Pipeline, Right for Structural Extraction (with ample gap)
    ax.annotate("", xy=(32, 69.5), xytext=(48, 78.25),
                arrowprops=dict(arrowstyle="->", color="#059669", linewidth=2.0), zorder=2)
    ax.text(35, 75.0, "Text Content", fontsize=10.5, fontweight='bold', color="#059669",
            bbox=dict(boxstyle="square,pad=0.2", facecolor="#FFFFFF", edgecolor="none"))

    ax.annotate("", xy=(89, 69.5), xytext=(72, 78.25),
                arrowprops=dict(arrowstyle="->", color="#BE123C", linewidth=2.0), zorder=2)
    ax.text(85, 75.0, "Visual & Code Assets", fontsize=10.5, fontweight='bold', color="#BE123C",
            bbox=dict(boxstyle="square,pad=0.2", facecolor="#FFFFFF", edgecolor="none"))

    # 6A. Text Cleaner (y: 63)
    draw_step(32, 63, 52, 11.5, "Text Cleaning & Normalization",
              "Ligatures normalized, whitespace trimmed,\nheaders/footers stripped (text_cleaner.py)",
              ec="#059669", title_color="#065F46")

    # 6B. Structural Asset Extraction (y: 63)
    draw_step(89, 63, 56, 11.5, "Structural Asset Extraction",
              "Filter images >= 120x120px, deduplicate xref,\ndetect monospace code > 60% coverage",
              is_highlight=True)

    ax.annotate("", xy=(32, 50.5), xytext=(32, 57.25),
                arrowprops=dict(arrowstyle="->", color="#059669", linewidth=2.0), zorder=2)

    # 7. Sliding Window Chunking (y: 44)
    draw_step(32, 44, 52, 11.5, "Sliding-Window Text Chunking",
              "900-token window with 60-token sentence overlap\nPreserves section context (chunking.py)",
              ec="#2563EB", title_color="#1E40AF")

    ax.annotate("", xy=(32, 31.5), xytext=(32, 38.25),
                arrowprops=dict(arrowstyle="->", color="#2563EB", linewidth=2.0), zorder=2)

    # 8. TF-IDF & Scoring (y: 25)
    draw_step(32, 25, 52, 11.5, "Sentence Importance Scoring",
              "TF-IDF vector model, document centroid\ncosine similarity + position bias weighting",
              ec="#7C3AED", title_color="#5B21B6")

    ax.annotate("", xy=(32, 12.5), xytext=(32, 19.25),
                arrowprops=dict(arrowstyle="->", color="#7C3AED", linewidth=2.0), zorder=2)

    # 9. MMR Re-ranking (y: 6.0)
    draw_step(32, 6.0, 52, 11.5, "MMR Diversity Re-Ranking",
              "Maximal Marginal Relevance (lambda = 0.65)\nSelects non-redundant sentences per ratio",
              ec="#7C3AED", title_color="#5B21B6")

    # Join both pipelines into Final Synthesis
    # Left join arrow
    ax.plot([32, 32, 60], [0.25, -3.5, -3.5], color="#4F46E5", linewidth=2.0, zorder=2)
    # Right join arrow
    ax.plot([89, 89, 60], [57.25, -3.5, -3.5], color="#BE123C", linestyle="dashed", linewidth=2.0, zorder=2)
    ax.annotate("", xy=(60, -7.5), xytext=(60, -3.5),
                arrowprops=dict(arrowstyle="->", color="#4F46E5", linewidth=2.0), zorder=2)
    ax.text(80, -1.8, "Preserved Figures & Code", fontsize=10.5, fontweight='bold', color="#BE123C",
            bbox=dict(boxstyle="square,pad=0.2", facecolor="#FFFFFF", edgecolor="none"))

    # 10. Multi-Asset Reassembly & Export (y: -13.5)
    draw_step(60, -13.5, 78, 10.5, "Integrated Result Synthesis & Multi-Format Export",
              "In-memory session cached | View in UI | Download as PDF, DOCX, TXT",
              ec="#4F46E5", title_color="#3730A3")

    # Final End Node
    ax.annotate("", xy=(60, -22.0), xytext=(60, -18.75),
                arrowprops=dict(arrowstyle="->", color="#334155", linewidth=2.0), zorder=2)
    ax.add_patch(FancyBboxPatch((48, -27.0), 24, 6.5, boxstyle="round,pad=0,rounding_size=3.2",
                                facecolor="#0F172A", edgecolor="#0F172A", zorder=3))
    ax.text(60, -23.75, "END", ha='center', va='center', fontsize=13.0, fontweight='bold', color="#FFFFFF", zorder=5)

    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    out_path = os.path.join(OUTPUT_DIR, "Figure 3.3 End to End Process Flowchart.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 3.3 updated:", out_path)


# ==============================================================================
# FIGURE 3.4: SLIDING-WINDOW CHUNKING (PERFECTED BIG TEXT)
# ==============================================================================
def generate_figure_3_4():
    fig, ax = plt.subplots(figsize=(13.0, 7.5), dpi=300)
    ax.set_xlim(0, 130)
    ax.set_ylim(-6, 90)
    ax.axis('off')
    fig.patch.set_facecolor('#ffffff')

    # 1. Source Document Stream at top
    box_top = FancyBboxPatch((8, 69), 114, 15, boxstyle="round,pad=0,rounding_size=1.4",
                             facecolor="#F8FAFC", edgecolor="#475569", linewidth=1.8, zorder=2)
    ax.add_patch(box_top)
    ax.text(65, 79.0, "Source Document Sentence Stream (Raw Extracted Text)",
            ha='center', va='center', fontsize=14.0, fontweight='bold', color="#0F172A")
    ax.text(65, 73.5, "Sentence boundaries preserved using regex tokenizer — sentences are never split mid-way",
            ha='center', va='center', fontsize=11.5, fontweight='bold', color="#334155")

    # Downward arrows from Stream to Chunks
    for cx in [25, 65, 105]:
        ax.annotate("", xy=(cx, 58), xytext=(cx, 69),
                    arrowprops=dict(arrowstyle="->", color="#475569", linewidth=2.0), zorder=2)

    # 2. Chunks (Chunk 1, Chunk 2, Chunk 3)
    # Chunk 1
    box_c1 = FancyBboxPatch((8, 38), 38, 20, boxstyle="round,pad=0,rounding_size=1.4",
                            facecolor="#EFF6FF", edgecolor="#2563EB", linewidth=2.0, zorder=2)
    ax.add_patch(box_c1)
    ax.text(27, 53.0, "Chunk 1", ha='center', va='center', fontsize=13.5, fontweight='bold', color="#1E40AF")
    ax.text(27, 46.8, "Target: ~900 Tokens\nSentences: S1 ... Sk", ha='center', va='center', fontsize=12.0, fontweight='bold', color="#1E293B")
    ax.text(27, 41.2, "Scored independently via TF-IDF", ha='center', va='center', fontsize=11.0, fontstyle='italic', fontweight='bold', color="#2563EB")

    # Overlap Region 1-2
    ax.add_patch(Rectangle((40, 38), 12, 20, facecolor="#FEF3C7", edgecolor="#D97706", linewidth=1.6, linestyle="--", zorder=3))
    ax.text(46, 48.0, "60-Token\nOverlap", ha='center', va='center', fontsize=11.5, fontweight='bold', color="#92400E")

    # Chunk 2
    box_c2 = FancyBboxPatch((46, 38), 38, 20, boxstyle="round,pad=0,rounding_size=1.4",
                            facecolor="#EFF6FF", edgecolor="#2563EB", linewidth=2.0, zorder=2)
    ax.add_patch(box_c2)
    ax.text(65, 53.0, "Chunk 2", ha='center', va='center', fontsize=13.5, fontweight='bold', color="#1E40AF")
    ax.text(65, 46.8, "Target: ~900 Tokens\nSentences: Sk-1 ... Sm", ha='center', va='center', fontsize=12.0, fontweight='bold', color="#1E293B")
    ax.text(65, 41.2, "Scored independently via TF-IDF", ha='center', va='center', fontsize=11.0, fontstyle='italic', fontweight='bold', color="#2563EB")

    # Overlap Region 2-3
    ax.add_patch(Rectangle((78, 38), 12, 20, facecolor="#FEF3C7", edgecolor="#D97706", linewidth=1.6, linestyle="--", zorder=3))
    ax.text(84, 48.0, "60-Token\nOverlap", ha='center', va='center', fontsize=11.5, fontweight='bold', color="#92400E")

    # Chunk 3
    box_c3 = FancyBboxPatch((84, 38), 38, 20, boxstyle="round,pad=0,rounding_size=1.4",
                            facecolor="#EFF6FF", edgecolor="#2563EB", linewidth=2.0, zorder=2)
    ax.add_patch(box_c3)
    ax.text(103, 53.0, "Chunk 3", ha='center', va='center', fontsize=13.5, fontweight='bold', color="#1E40AF")
    ax.text(103, 46.8, "Target: ~900 Tokens\nSentences: Sm-1 ... Sn", ha='center', va='center', fontsize=12.0, fontweight='bold', color="#1E293B")
    ax.text(103, 41.2, "Scored independently via TF-IDF", ha='center', va='center', fontsize=11.0, fontstyle='italic', fontweight='bold', color="#2563EB")

    # Arrows down to MMR per chunk
    for cx in [27, 65, 103]:
        ax.annotate("", xy=(cx, 25), xytext=(cx, 38),
                    arrowprops=dict(arrowstyle="->", color="#1E40AF", linewidth=2.0), zorder=2)

    # 3. Batched MMR Diversity Selection per Chunk
    box_mmr = FancyBboxPatch((8, 14), 114, 11, boxstyle="round,pad=0,rounding_size=1.4",
                             facecolor="#FAF5FF", edgecolor="#7C3AED", linewidth=1.8, zorder=2)
    ax.add_patch(box_mmr)
    ax.text(65, 21.2, "Batched MMR Re-ranking per Chunk (\u03bb = 0.65 Diversity Parameter)",
            ha='center', va='center', fontsize=13.0, fontweight='bold', color="#5B21B6")
    ax.text(65, 16.5, "Extracts top non-redundant sentences from each chunk matching the target compression ratio",
            ha='center', va='center', fontsize=11.5, fontweight='bold', color="#6B21A8")

    # Arrow to Final Reassembly
    ax.annotate("", xy=(65, 5.5), xytext=(65, 14),
                arrowprops=dict(arrowstyle="->", color="#7C3AED", linewidth=2.0), zorder=2)

    # 4. Final Hierarchical Section Reassembly
    box_final = FancyBboxPatch((8, -4.5), 114, 11, boxstyle="round,pad=0,rounding_size=1.4",
                              facecolor="#ECFDF5", edgecolor="#059669", linewidth=1.8, zorder=2)
    ax.add_patch(box_final)
    ax.text(65, 2.8, "Hierarchical Section Reassembly & Bounded Output Grouping",
            ha='center', va='center', fontsize=13.0, fontweight='bold', color="#065F46")
    ax.text(65, -1.8, "Sentences are recombined in chronological document sequence under detected section headings",
            ha='center', va='center', fontsize=11.0, fontweight='bold', color="#047857")

    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    out_path = os.path.join(OUTPUT_DIR, "Figure 3.4 Chunking Strategy.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 3.4 updated:", out_path)


# ==============================================================================
# FIGURE 3.5: ASYNCHRONOUS SYSTEM EXECUTION FLOW (PERFECTED BIG TEXT)
# ==============================================================================
def generate_figure_3_5():
    fig, ax = plt.subplots(figsize=(14.0, 10.8), dpi=300)
    ax.set_xlim(0, 144)
    ax.set_ylim(-3, 110)
    ax.axis('off')
    fig.patch.set_facecolor('#ffffff')

    lanes = [
        {"name": "Client Browser\n(HTML5 / JS)", "x": 6, "w": 28, "color": "#0284C7", "bg": "#F0F9FF"},
        {"name": "Flask App Controller\n(app.py)", "x": 39, "w": 29, "color": "#4F46E5", "bg": "#EEF2FF"},
        {"name": "Background Worker\n(_run_summarization_job)", "x": 73, "w": 32, "color": "#7C3AED", "bg": "#FAF5FF"},
        {"name": "Session Store & Disk\n(summaries dict / uploads)", "x": 109, "w": 31, "color": "#059669", "bg": "#ECFDF5"},
    ]

    for lane in lanes:
        hdr = FancyBboxPatch((lane["x"], 96), lane["w"], 12.0, boxstyle="round,pad=0,rounding_size=1.4",
                             facecolor=lane["bg"], edgecolor=lane["color"], linewidth=2.0, zorder=3)
        ax.add_patch(hdr)
        ax.text(lane["x"] + lane["w"]/2, 102.0, lane["name"], ha='center', va='center',
                fontsize=12.2, fontweight='bold', color=lane["color"], zorder=4)
        cx = lane["x"] + lane["w"]/2
        ax.plot([cx, cx], [3, 96], color=lane["color"], linestyle=":", linewidth=1.6, alpha=0.5, zorder=1)

    c_client = 20.0
    c_flask = 53.5
    c_worker = 89.0
    c_store = 124.5

    def draw_msg(y, x1, x2, text, sub="", is_async=False, color="#0F172A"):
        style = "--" if is_async else "-"
        ax.plot([x1, x2], [y, y], color=color, linestyle=style, linewidth=1.8, zorder=3)
        ax.annotate("", xy=(x2, y), xytext=(x2 - (2.5 if x2 > x1 else -2.5), y),
                    arrowprops=dict(arrowstyle="->", color=color, linewidth=1.8), zorder=4)
        mid_x = (x1 + x2) / 2
        ax.text(mid_x, y + 1.5, text, ha='center', va='bottom', fontsize=11.2, fontweight='bold', color=color, zorder=5)
        if sub:
            ax.text(mid_x, y - 1.2, sub, ha='center', va='top', fontsize=10.0, fontweight='bold', color="#475569", zorder=5)

    # 1. User Uploads Document
    draw_msg(88, c_client, c_flask, "1. HTTP POST /summarize", "File (.pdf/.docx/.txt), ratio (25%), toggles", color="#0284C7")

    # 2. Flask Validates & Saves File
    draw_msg(80, c_flask, c_store, "2. Validate & Save Upload", "UUID filename in uploads/ (file_validator.py)", color="#4F46E5")

    # 3. Flask Initializes Session & Spawns Worker
    draw_msg(72, c_flask, c_store, "3. Init State in summaries[job_id]", "status: 'processing', progress: 5%", color="#4F46E5")
    draw_msg(64, c_flask, c_worker, "4. threading.Thread(target=_run_job)", "Spawns background task non-blocking", color="#4F46E5")

    # 4. Immediate Return to Client
    draw_msg(56, c_flask, c_client, "5. HTTP 202 Accepted {job_id: uuid}", "Client immediately unlocks UI", is_async=True, color="#0284C7")

    # 5. Dynamic Polling Loop
    ax.add_patch(FancyBboxPatch((8, 34), 58, 17, boxstyle="round,pad=0,rounding_size=1.2",
                                facecolor="#F8FAFC", edgecolor="#94A3B8", linewidth=1.4, linestyle="--", zorder=2))
    ax.text(10, 47.5, "POLLING LOOP (~1.2s INTERVAL)", fontsize=10.5, fontweight='bold', color="#64748B")
    draw_msg(42, c_client, c_flask, "GET /status/<job_id>", color="#0284C7")
    draw_msg(37, c_flask, c_client, "JSON: {progress: 45%, stage: 'Chunking...'}", is_async=True, color="#0284C7")

    # 6. Worker Execution Pipeline
    draw_msg(48, c_worker, c_store, "Ingest & Extract Figures / Code", color="#7C3AED")
    draw_msg(30, c_worker, c_store, "Chunking, TF-IDF & MMR Selection", color="#7C3AED")
    draw_msg(20, c_worker, c_store, "Save Payload into summaries[job_id]", "status: 'completed', progress: 100%", color="#059669")

    # 7. Final Polling Returns Completed
    draw_msg(13, c_client, c_flask, "GET /status/<job_id>", color="#0284C7")
    draw_msg(8, c_flask, c_client, "JSON: {status: 'completed', result: {...}}", "Renders summary & figures", is_async=True, color="#0284C7")

    # 8. Export Download (well separated)
    draw_msg(0, c_client, c_flask, "GET /download/<job_id>/<format>", "Download PDF / DOCX / TXT", color="#4F46E5")

    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    out_path = os.path.join(OUTPUT_DIR, "Figure_3_5_System_Execution_Flow.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 3.5 updated:", out_path)


# ==============================================================================
# FIGURE 3.6: SYSTEM MODULE & DATAFLOW ARCHITECTURE (PERFECTED BIG TEXT)
# ==============================================================================
def generate_figure_3_6():
    fig, ax = plt.subplots(figsize=(14.5, 10.2), dpi=300)
    ax.set_xlim(0, 146)
    ax.set_ylim(-2, 98)
    ax.axis('off')
    fig.patch.set_facecolor('#ffffff')

    def draw_mod(x, y, w, h, name, filepath, items=None, fc="#FFFFFF", ec="#334155", 
                 title_color="#0F172A", badge_color="#0284C7", is_contrib=False):
        if is_contrib:
            fc = "#FFF1F2"
            ec = "#BE123C"
            title_color = "#881337"
            badge_color = "#BE123C"
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4",
                             facecolor=fc, edgecolor=ec, linewidth=1.8, zorder=3)
        ax.add_patch(box)
        ax.text(x + w/2, y + h - 2.2, name, ha='center', va='top',
                fontsize=13.0, fontweight='bold', color=title_color, zorder=5)
        ax.text(x + w/2, y + h - 5.0, filepath, ha='center', va='top',
                fontsize=11.0, fontweight='bold', fontstyle='italic', color=badge_color, zorder=5)
        ax.plot([x + 2, x + w - 2], [y + h - 6.2, y + h - 6.2], color=ec, linewidth=1.0, alpha=0.4, zorder=4)
        if items:
            cy = y + h - 8.0
            for itm in items:
                ax.text(x + 2.5, cy, f"• {itm}", ha='left', va='top',
                        fontsize=11.0, fontweight='bold', color="#0F172A" if not is_contrib else "#9F1239", zorder=5)
                cy -= 2.6

    # Row 1 (y: 52 to 84, h=32)
    # Module 1: app.py
    draw_mod(5, 52, 42, 32, "Application Controller", "app.py",
             items=[
                 "Flask web routing (/summarize)",
                 "Background thread orchestrator",
                 "IP rate limiter (10 req/min)",
                 "Session dict: summaries[job_id]",
                 "Attachment stream generator"
             ], ec="#4F46E5", title_color="#3730A3", badge_color="#4338CA")

    # Module 2: file_validator.py
    draw_mod(52, 52, 42, 32, "Security & File Validator", "utils/file_validator.py",
             items=[
                 "Allowed formats (.pdf, .docx, .txt)",
                 "File size limit check (<= 16 MB)",
                 "Secure filename sanitization",
                 "MIME type verification",
                 "Error trapping & user alerts"
             ], ec="#0284C7", title_color="#0369A1", badge_color="#0284C7")

    # Module 3: Document Extractors
    draw_mod(99, 52, 42, 32, "Document Text Extractors", "pdf_extractor & docx",
             items=[
                 "PyMuPDF / fitz page parsing",
                 "python-docx paragraph reader",
                 "Document table linearizer",
                 "UTF-8 stream reader & cleaner",
                 "Raw document text stream emitter"
             ], ec="#059669", title_color="#065F46", badge_color="#047857")

    # Connectors Row 1
    ax.annotate("", xy=(52, 68), xytext=(47, 68),
                arrowprops=dict(arrowstyle="->", color="#4F46E5", linewidth=2.0), zorder=6)
    ax.text(49.5, 70.0, "Upload", ha='center', va='bottom', fontsize=10.5, fontweight='bold', color="#4F46E5",
            bbox=dict(boxstyle="square,pad=0.15", facecolor="#FFFFFF", edgecolor="none"))

    ax.annotate("", xy=(99, 68), xytext=(94, 68),
                arrowprops=dict(arrowstyle="->", color="#0284C7", linewidth=2.0), zorder=6)
    ax.text(96.5, 70.0, "Validated", ha='center', va='bottom', fontsize=10.5, fontweight='bold', color="#0284C7",
            bbox=dict(boxstyle="square,pad=0.15", facecolor="#FFFFFF", edgecolor="none"))

    # Row 2 (y: 8 to 40, h=32)
    # Module 4: Structural Preservation (Novel Contribution)
    draw_mod(5, 8, 42, 32, "Structural Asset Extractor", "pdf_extractor.py",
             items=[
                 "DiagramExtractor (PyMuPDF):",
                 "  - Image size filter: >= 60x60 px",
                 "  - Deduplication via xref hash",
                 "CodeListingDetector:",
                 "  - Monospace font (coverage > 60%)",
                 "  - Block span length >= 40 chars"
             ], is_contrib=True)

    # Module 5: Text Cleaning & Chunking
    draw_mod(52, 8, 42, 32, "Cleaner & Chunker", "text_cleaner & chunking",
             items=[
                 "Unicode ligature normalizer",
                 "Slide header/footer stripper",
                 "Regex sentence tokenizer",
                 "900-token sliding window",
                 "60-token overlap buffer"
             ], ec="#16A34A", title_color="#166534", badge_color="#15803D")

    # Module 6: Summarizer Core & Download Generator
    draw_mod(99, 8, 42, 32, "Summarizer & Exporter", "summarizer & download",
             items=[
                 "TF-IDF vector representation",
                 "Centroid cosine scoring",
                 "Position bias weighting",
                 "MMR diversity re-ranking",
                 "Multi-format export (PDF/DOCX/TXT)"
             ], ec="#7C3AED", title_color="#5B21B6", badge_color="#7E22CE")

    # Inter-Row Connectors
    # Extractors -> Structural Extractor
    ax.plot([120, 120, 26, 26], [52, 46, 46, 40], color="#BE123C", linewidth=1.8, zorder=5)
    ax.annotate("", xy=(26, 40), xytext=(26, 42),
                arrowprops=dict(arrowstyle="->", color="#BE123C", linewidth=1.8), zorder=6)
    ax.text(73, 47.0, "Raw PDF Stream for Figure & Code Extraction", ha='center', va='bottom', 
            fontsize=10.5, fontweight='bold', color="#BE123C",
            bbox=dict(boxstyle="square,pad=0.2", facecolor="#FFFFFF", edgecolor="none"))

    # Extractors -> Cleaner & Chunker
    ax.annotate("", xy=(73, 40), xytext=(105, 52),
                arrowprops=dict(arrowstyle="->", color="#059669", linewidth=1.8), zorder=6)
    ax.text(83, 42.0, "Raw Text", ha='center', va='center', fontsize=10.5, fontweight='bold', color="#059669",
            bbox=dict(boxstyle="square,pad=0.2", facecolor="#FFFFFF", edgecolor="none"))

    # Chunker -> Summarizer
    ax.annotate("", xy=(99, 24), xytext=(94, 24),
                arrowprops=dict(arrowstyle="->", color="#16A34A", linewidth=2.0), zorder=6)
    ax.text(96.5, 26.0, "Chunks", ha='center', va='bottom', fontsize=10.5, fontweight='bold', color="#16A34A",
            bbox=dict(boxstyle="square,pad=0.15", facecolor="#FFFFFF", edgecolor="none"))

    # Structural Extractor -> Summarizer & Exporter
    ax.plot([47, 73, 73, 99], [18, 18, 13.5, 13.5], color="#BE123C", linestyle="dashed", linewidth=1.8, zorder=5)
    ax.annotate("", xy=(99, 13.5), xytext=(96, 13.5),
                arrowprops=dict(arrowstyle="->", color="#BE123C", linewidth=1.8), zorder=6)
    ax.text(73, 15.0, "Preserved Assets (Figures & Code)", ha='center', va='bottom', 
            fontsize=10.5, fontweight='bold', color="#BE123C",
            bbox=dict(boxstyle="square,pad=0.2", facecolor="#FFFFFF", edgecolor="none"))

    # Summarizer -> AppController (Final payload to Session)
    ax.plot([120, 143, 143, 26, 26], [40, 40, 91, 91, 84], color="#4F46E5", linestyle="dotted", linewidth=1.8, zorder=5)
    ax.annotate("", xy=(26, 84), xytext=(26, 86),
                arrowprops=dict(arrowstyle="->", color="#4F46E5", linewidth=1.8), zorder=6)
    ax.text(84.5, 92.0, "Compiled Summary Result Payload -> summaries[job_id]", 
            ha='center', va='bottom', fontsize=11.0, fontweight='bold', color="#4F46E5",
            bbox=dict(boxstyle="square,pad=0.2", facecolor="#FFFFFF", edgecolor="none"))

    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    out_path = os.path.join(OUTPUT_DIR, "Figure_3_6_Module_Dataflow.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 3.6 updated:", out_path)


if __name__ == "__main__":
    print("Regenerating all 6 diagrams with perfected big text & zero overlap...")
    generate_figure_3_1()
    generate_figure_3_2()
    generate_figure_3_3()
    generate_figure_3_4()
    generate_figure_3_5()
    generate_figure_3_6()
    print("All diagrams regenerated successfully!")
