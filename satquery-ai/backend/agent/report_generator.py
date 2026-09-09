"""
SatQuery AI — PDF Report Generator
Produces a structured PDF summarising the analysis.
"""
from __future__ import annotations

import base64
import io
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("satquery.report")


class ReportGenerator:
    """Generate PDF analysis reports using fpdf2."""

    PAGE_WIDTH = 210  # A4 mm
    MARGIN = 15
    CONTENT_WIDTH = 180

    def generate_pdf(
        self,
        analysis_response,
        images_data: List[Dict],
        session_id: str,
    ) -> bytes:
        """
        Generate a PDF report and return it as bytes.
        Falls back to a minimal report if fpdf2 is unavailable.
        """
        try:
            from fpdf import FPDF
            return self._build_pdf(FPDF, analysis_response, images_data, session_id)
        except ImportError:
            logger.warning("fpdf2 not available, generating text report")
            return self._fallback_text(analysis_response, session_id)

    # ── PDF Builder ──────────────────────────────────────────────────────────

    def _build_pdf(self, FPDF, response, images_data, session_id) -> bytes:
        from fpdf import FPDF

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # ── Header ──────────────────────────────────────────────────────────
        pdf.set_font("Helvetica", "B", 20)
        pdf.set_text_color(30, 80, 160)
        pdf.cell(0, 12, "SatQuery AI - Analysis Report", new_x="LMARGIN", new_y="NEXT", align="C")

        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(0, 6, self._safe(f"Session: {session_id}  |  Generated: {self._timestamp()}"), new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(4)

        # ── Divider ─────────────────────────────────────────────────────────
        pdf.set_draw_color(200, 200, 200)
        pdf.line(self.MARGIN, pdf.get_y(), self.PAGE_WIDTH - self.MARGIN, pdf.get_y())
        pdf.ln(6)

        # ── Task & Answer ────────────────────────────────────────────────────
        self._section_header(pdf, "Task & Answer")

        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(60, 60, 60)
        pdf.cell(40, 7, "Task Type:", new_x="RIGHT")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, response.task, new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("Helvetica", "B", 10)
        conf_pct = f"{response.confidence * 100:.1f}%"
        pdf.cell(40, 7, "Confidence:", new_x="RIGHT")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, conf_pct, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(30, 80, 160)
        pdf.cell(0, 7, "Answer:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(0, 6, self._safe(response.answer))
        pdf.ln(4)

        # ── Execution Summary ────────────────────────────────────────────────
        self._section_header(pdf, "Execution Summary")
        es = response.execution_summary

        rows = [
            ("Selected Task", es.selected_task),
            ("Task Confidence", f"{es.task_confidence * 100:.1f}%"),
            ("Models Used", ", ".join(es.models_used)),
            ("Processing Time", f"{es.processing_time_ms:.1f} ms"),
            ("Steps", " -> ".join(es.steps)),
        ]

        for label, value in rows:
            self._kv_row(pdf, label, value)

        # Parameters
        if es.parameters:
            param_str = ", ".join(f"{k}={v}" for k, v in es.parameters.items())
            self._kv_row(pdf, "Parameters", param_str)
        pdf.ln(4)

        # ── Change Percentage ────────────────────────────────────────────────
        if response.change_percentage is not None:
            self._section_header(pdf, "Change Metrics")
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(40, 40, 40)
            pdf.cell(0, 7, f"Changed area: {response.change_percentage:.2f}% of the scene", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(4)

        # ── Grounding Boxes ──────────────────────────────────────────────────
        if response.grounding_boxes:
            self._section_header(pdf, f"Detected Objects ({len(response.grounding_boxes)} found)")
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(40, 40, 40)
            for i, box in enumerate(response.grounding_boxes[:10], 1):
                line = (
                    f"{i}. {box.label}  score={box.score:.2f}  "
                    f"bbox=[{box.x1:.3f}, {box.y1:.3f}, {box.x2:.3f}, {box.y2:.3f}]"
                )
                pdf.cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(4)

        # ── Visual Evidence ──────────────────────────────────────────────────
        b64_images = [
            ("Visual Evidence", response.visual_evidence),
            ("Change Map", response.change_map),
            ("Fusion Map", response.fusion_map),
        ]
        for title, b64 in b64_images:
            if b64:
                self._embed_image(pdf, title, b64)

        # ── Footer ───────────────────────────────────────────────────────────
        pdf.set_y(-20)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(160, 160, 160)
        pdf.cell(0, 6, "Generated by SatQuery AI - Agentic Remote Sensing Analysis", align="C")

        return bytes(pdf.output())

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _section_header(self, pdf, title: str):
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(30, 80, 160)
        pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(200, 220, 255)
        pdf.line(self.MARGIN, pdf.get_y(), self.PAGE_WIDTH - self.MARGIN, pdf.get_y())
        pdf.ln(3)

    def _kv_row(self, pdf, label: str, value: str, label_w: float = 50):
        """Render a 'label: value' row where the value wraps safely.

        Resets x to the left margin and pins the value cell's exit position
        (new_x=LMARGIN, new_y=NEXT). Without this, each multi_cell leaves x at
        the right margin, so the next row's multi_cell(0, ...) computes a
        zero/negative width and fpdf raises "Not enough horizontal space to
        render a single character".
        """
        pdf.set_x(self.MARGIN)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(label_w, 6, self._safe(f"{label}:"), new_x="RIGHT", new_y="TOP")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(0, 6, self._safe(str(value)), new_x="LMARGIN", new_y="NEXT")

    def _embed_image(self, pdf, title: str, b64: str):
        """Embed a base64-encoded image into the PDF."""
        try:
            img_bytes = base64.b64decode(b64)
            img_buf = io.BytesIO(img_bytes)

            self._section_header(pdf, title)
            # Place image centred, max 160mm wide
            x = self.MARGIN + (self.CONTENT_WIDTH - 160) / 2
            pdf.image(img_buf, x=x, w=160)
            pdf.ln(4)
        except Exception as exc:
            logger.warning("Could not embed image '%s': %s", title, exc)

    @staticmethod
    def _timestamp() -> str:
        return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())

    # Common Unicode punctuation the core Helvetica font cannot render, mapped
    # to safe ASCII so answers keep their meaning instead of turning into "?".
    _UNICODE_MAP = {
        "—": "-", "–": "-",       # em / en dash
        "‘": "'", "’": "'",       # smart single quotes
        "“": '"', "”": '"',       # smart double quotes
        "…": "...", "•": "-",     # ellipsis, bullet
        " ": " ",                      # non-breaking space
    }

    @staticmethod
    def _safe(text) -> str:
        """Make text renderable by fpdf's core Latin-1 fonts.

        Maps common Unicode punctuation to ASCII, then drops anything still
        outside Latin-1 (e.g. emoji) rather than substituting '?'.
        """
        if text is None:
            return ""
        text = str(text)
        for uni, ascii_eq in ReportGenerator._UNICODE_MAP.items():
            text = text.replace(uni, ascii_eq)
        return text.encode("latin-1", errors="ignore").decode("latin-1")

    def _fallback_text(self, response, session_id: str) -> bytes:
        """Minimal text-based fallback if fpdf2 unavailable."""
        lines = [
            "SatQuery AI Analysis Report",
            f"Session: {session_id}",
            f"Generated: {self._timestamp()}",
            "",
            f"Task: {response.task}",
            f"Confidence: {response.confidence * 100:.1f}%",
            "",
            "Answer:",
            response.answer,
            "",
            "Execution Summary:",
            f"  Models: {', '.join(response.execution_summary.models_used)}",
            f"  Processing time: {response.execution_summary.processing_time_ms:.1f} ms",
        ]
        return "\n".join(lines).encode("utf-8")
