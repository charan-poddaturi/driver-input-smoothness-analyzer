"""
DISA — PDF Report Generator
============================
Generates a professional PDF report for a driving session analysis.
Uses fpdf2 for layout and embeds matplotlib charts as images.
"""

from __future__ import annotations

import io
import os
import tempfile
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from fpdf import FPDF, XPos, YPos
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

from backend.scoring import SmoothnesResult
from backend.anomaly import AnomalyEvent


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

DARK_BG   = (18, 18, 35)
ACCENT    = (99, 102, 241)
GREEN     = (52, 211, 153)
YELLOW    = (251, 191, 36)
RED       = (239, 68, 68)
TEXT_MAIN = (235, 235, 245)
TEXT_SUB  = (160, 160, 185)


def _score_color(score: float):
    if score >= 80:
        return GREEN
    if score >= 55:
        return YELLOW
    return RED


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _radar_chart_bytes(sub_scores: Dict[str, float]) -> bytes:
    """Generate a radar chart of sub-scores and return PNG bytes."""
    labels = list(sub_scores.keys())
    values = [sub_scores[l] for l in labels]
    N = len(labels)

    angles = [2 * np.pi * i / N for i in range(N)]
    angles += angles[:1]   # close polygon
    values_plot = values + values[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True),
                           facecolor="#12122a")
    ax.set_facecolor("#12122a")
    ax.plot(angles, values_plot, color="#6366f1", linewidth=2)
    ax.fill(angles, values_plot, color="#6366f1", alpha=0.3)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([l.title() for l in labels], color="white", size=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], color="#a0a0b9", size=7)
    ax.grid(color="#3a3a5c", linewidth=0.5)
    ax.spines["polar"].set_color("#3a3a5c")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="#12122a")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _bar_chart_bytes(contributions: Dict[str, float]) -> bytes:
    """Generate a feature contribution bar chart and return PNG bytes."""
    keys = list(contributions.keys())
    values = list(contributions.values())

    colors = ["#6366f1", "#34d399", "#fbbf24", "#f87171"]
    fig, ax = plt.subplots(figsize=(5, 2.5), facecolor="#12122a")
    ax.set_facecolor("#12122a")
    bars = ax.barh(keys, values, color=colors[:len(keys)], edgecolor="none", height=0.5)
    ax.set_xlim(0, max(values) * 1.3 if values else 10)
    ax.set_xlabel("Penalty Points", color="#a0a0b9", fontsize=9)
    ax.tick_params(colors="white", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#3a3a5c")
    ax.spines["left"].set_color("#3a3a5c")
    for bar, v in zip(bars, values):
        ax.text(v + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}", va="center", color="white", fontsize=8)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#12122a")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# PDF document
# ---------------------------------------------------------------------------

class DISAReport:
    """PDF report for a DISA driving session analysis."""

    def __init__(self):
        if not FPDF_AVAILABLE:
            raise ImportError("fpdf2 is required. Install with: pip install fpdf2")
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)
        self.pdf.add_page()

    def _header(self, session_id: str, timestamp: str):
        pdf = self.pdf
        # Title bar
        pdf.set_fill_color(*DARK_BG)
        pdf.rect(0, 0, 210, 40, style="F")
        pdf.set_font("Helvetica", "B", 20)
        pdf.set_text_color(*TEXT_MAIN)
        pdf.set_y(10)
        pdf.cell(0, 10, "Driver Input Smoothness Analyzer (DISA)", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*TEXT_SUB)
        pdf.cell(0, 8, f"Session: {session_id}   |   Generated: {timestamp}", align="C",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(5)

    def _section_title(self, title: str):
        pdf = self.pdf
        pdf.set_fill_color(*ACCENT)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 8, f"  {title}", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(3)

    def _key_metric(self, label: str, value: str, color=None):
        pdf = self.pdf
        color = color or TEXT_MAIN
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*TEXT_SUB)
        pdf.cell(60, 7, label)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(*color)
        pdf.cell(0, 7, value, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def build(
        self,
        result: SmoothnesResult,
        anomalies: List[AnomalyEvent],
        session_id: str = "session_001",
    ) -> bytes:
        """
        Build and return PDF bytes for a given analysis result.
        """
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._header(session_id, ts)

        pdf = self.pdf

        # --- Overall score section ---
        self._section_title("Overall Smoothness Score")
        sc = _score_color(result.overall_score)
        self._key_metric("Score:", f"{result.overall_score:.1f} / 100", color=sc)
        self._key_metric("Grade:", result.grade)
        self._key_metric("Weights used:", f"S={result.weights[0]:.2f}  J={result.weights[1]:.2f}  V={result.weights[2]:.2f}  P={result.weights[3]:.2f}")
        pdf.ln(3)

        # --- Sub-scores ---
        self._section_title("Sub-Scores by Channel Group")
        for group, score in result.sub_scores.items():
            gc = _score_color(score)
            self._key_metric(f"{group.title()}:", f"{score:.1f} / 100", color=gc)
        pdf.ln(3)

        # --- Radar chart ---
        radar_bytes = _radar_chart_bytes(result.sub_scores)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(radar_bytes)
            tmp_path = tmp.name
        pdf.image(tmp_path, x=60, w=90)
        os.unlink(tmp_path)
        pdf.ln(3)

        # --- Feature contributions ---
        self._section_title("Feature Penalty Breakdown")
        bar_bytes = _bar_chart_bytes(result.feature_contributions)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(bar_bytes)
            tmp_path = tmp.name
        pdf.image(tmp_path, x=15, w=130)
        os.unlink(tmp_path)
        pdf.ln(3)

        # Feature table
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*TEXT_MAIN)
        for feat, val in result.normalized_features.items():
            self._key_metric(f"  {feat} (normalized):", f"{val:.4f}")

        # --- Anomaly events ---
        pdf.ln(3)
        self._section_title(f"Detected Anomaly Events ({len(anomalies)})")
        if not anomalies:
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(*GREEN)
            pdf.cell(0, 7, "  No significant rough events detected.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        else:
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(*TEXT_SUB)
            pdf.cell(30, 6, "Time (s)")
            pdf.cell(30, 6, "Duration (s)")
            pdf.cell(40, 6, "Type")
            pdf.cell(30, 6, "Severity")
            pdf.cell(0, 6, "Channels", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(*TEXT_MAIN)
            for ev in anomalies[:20]:
                pdf.cell(30, 6, f"{ev.start_time:.1f}–{ev.end_time:.1f}")
                pdf.cell(30, 6, f"{ev.end_time - ev.start_time:.2f}")
                pdf.cell(40, 6, ev.event_type)
                pdf.cell(30, 6, f"{ev.severity:.2f}")
                pdf.cell(0, 6, ", ".join(ev.affected_channels[:2]), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # Footer
        pdf.set_y(-15)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(*TEXT_SUB)
        pdf.cell(0, 6, "DISA — Driver Input Smoothness Analyzer | Research MVP", align="C")

        return bytes(pdf.output())


def generate_report(
    result: SmoothnesResult,
    anomalies: List[AnomalyEvent],
    session_id: str = "session_001",
) -> bytes:
    """Convenience wrapper: generate and return PDF bytes."""
    report = DISAReport()
    return report.build(result, anomalies, session_id=session_id)
