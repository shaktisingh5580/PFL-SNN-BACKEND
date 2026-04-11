"""
PDF report generation using ReportLab.
Generates professional compliance reports with imagery, spike maps, and violation details.
"""
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from io import BytesIO

logger = logging.getLogger(__name__)

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image as RLImage, PageBreak, HRFlowable,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

try:
    from PIL import Image as PILImage
    import numpy as np
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class PDFReportGenerator:
    """Generate professional PDF compliance reports."""

    def __init__(self, output_dir: str = "outputs/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if HAS_REPORTLAB:
            self.styles = getSampleStyleSheet()
            self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Create custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name="ReportTitle",
            parent=self.styles["Title"],
            fontSize=24,
            spaceAfter=20,
            textColor=colors.HexColor("#0a0e27"),
        ))
        self.styles.add(ParagraphStyle(
            name="SectionHeader",
            parent=self.styles["Heading2"],
            fontSize=14,
            spaceAfter=10,
            spaceBefore=15,
            textColor=colors.HexColor("#1a237e"),
        ))
        self.styles.add(ParagraphStyle(
            name="SubHeader",
            parent=self.styles["Heading3"],
            fontSize=11,
            spaceAfter=6,
            textColor=colors.HexColor("#283593"),
        ))
        self.styles.add(ParagraphStyle(
            name="BodyText2",
            parent=self.styles["Normal"],
            fontSize=10,
            spaceAfter=6,
            leading=14,
        ))
        self.styles.add(ParagraphStyle(
            name="ViolationAlert",
            parent=self.styles["Normal"],
            fontSize=11,
            textColor=colors.red,
            spaceAfter=8,
            fontName="Helvetica-Bold",
        ))

    def generate_report(
        self,
        detection: Dict,
        violations: List[Dict],
        spectral_data: Optional[Dict] = None,
        evidence_hash: Optional[str] = None,
        merkle_root: Optional[str] = None,
        before_image: Optional[np.ndarray] = None,
        after_image: Optional[np.ndarray] = None,
        spike_map: Optional[np.ndarray] = None,
    ) -> str:
        """
        Generate a full compliance report PDF.

        Args:
            detection: Detection metadata dict
            violations: List of violation dicts
            spectral_data: Spectral index values
            evidence_hash: Blockchain evidence hash
            before_image, after_image: RGB numpy arrays (H, W, 3)
            spike_map: Confidence heatmap (H, W)

        Returns:
            Path to generated PDF file
        """
        if not HAS_REPORTLAB:
            logger.error("ReportLab not installed. Cannot generate PDF.")
            return self._generate_text_report(detection, violations)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        det_id = detection.get("id", "unknown")[:8]
        filename = f"compliance_report_{det_id}_{timestamp}.pdf"
        filepath = self.output_dir / filename

        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=A4,
            rightMargin=20 * mm,
            leftMargin=20 * mm,
            topMargin=25 * mm,
            bottomMargin=20 * mm,
        )

        elements = []

        # === HEADER ===
        elements.append(Paragraph(
            "SATELLITE COMPLIANCE REPORT",
            self.styles["ReportTitle"],
        ))
        elements.append(Paragraph(
            "Surat Municipal Corporation — Urban Planning Department",
            self.styles["Normal"],
        ))
        elements.append(HRFlowable(
            width="100%", thickness=2, color=colors.HexColor("#1a237e"),
        ))
        elements.append(Spacer(1, 12))

        # === DETECTION SUMMARY ===
        elements.append(Paragraph("1. Detection Summary", self.styles["SectionHeader"]))
        summary_data = [
            ["Detection ID", str(detection.get("id", "N/A"))],
            ["Date Range", str(detection.get("date_range", "N/A"))],
            ["Coordinates", f"{detection.get('lat', 'N/A')}°N, {detection.get('lon', 'N/A')}°E"],
            ["Change Type", str(detection.get("change_type", "N/A")).replace("_", " ").title()],
            ["SNN Confidence", f"{detection.get('confidence', 0) * 100:.1f}%"],
            ["Area", f"{detection.get('area_hectares', 0):.3f} hectares"],
            ["Status", str(detection.get("status", "pending")).upper()],
        ]
        elements.append(self._create_table(summary_data, col_widths=[150, 350]))
        elements.append(Spacer(1, 15))

        # === IMAGERY (if available) ===
        if before_image is not None or after_image is not None:
            elements.append(Paragraph("2. Satellite Imagery", self.styles["SectionHeader"]))
            img_elements = self._create_image_panel(before_image, after_image, spike_map)
            elements.extend(img_elements)
            elements.append(Spacer(1, 15))

        # === SPECTRAL INDICES ===
        if spectral_data:
            elements.append(Paragraph("3. Spectral Index Analysis", self.styles["SectionHeader"]))
            spectral_table = [
                ["Index", "Before", "After", "Change (Δ)", "Interpretation"],
            ]
            for idx_name in ["ndvi", "ndbi", "mndwi"]:
                before_val = spectral_data.get(f"before_{idx_name}", 0)
                after_val = spectral_data.get(f"after_{idx_name}", 0)
                delta = spectral_data.get(f"delta_{idx_name}", after_val - before_val)
                interp = self._interpret_spectral(idx_name, delta)
                spectral_table.append([
                    idx_name.upper(),
                    f"{before_val:.3f}",
                    f"{after_val:.3f}",
                    f"{delta:+.3f}",
                    interp,
                ])
            elements.append(self._create_table(
                spectral_table,
                col_widths=[60, 70, 70, 80, 220],
                header=True,
            ))
            elements.append(Spacer(1, 15))

        # === COMPLIANCE VIOLATIONS ===
        elements.append(Paragraph("4. Compliance Assessment", self.styles["SectionHeader"]))
        if violations:
            elements.append(Paragraph(
                f"⚠ {len(violations)} VIOLATION(S) DETECTED",
                self.styles["ViolationAlert"],
            ))
            for i, v in enumerate(violations, 1):
                elements.append(Paragraph(
                    f"Violation {i}: {v.get('rule_name', 'N/A')}",
                    self.styles["SubHeader"],
                ))
                v_data = [
                    ["Rule ID", v.get("rule_id", "N/A")],
                    ["Severity", v.get("severity", "N/A")],
                    ["Legal Reference", v.get("legal_ref", "N/A")],
                    ["Description", v.get("description", "N/A")],
                ]
                if v.get("zone_name"):
                    v_data.append(["Affected Zone", v["zone_name"]])
                if v.get("distance_m"):
                    v_data.append(["Distance to Protected Area", f"{v['distance_m']:.0f}m"])
                elements.append(self._create_table(v_data, col_widths=[150, 350]))
                elements.append(Spacer(1, 8))
        else:
            elements.append(Paragraph(
                "✓ No compliance violations detected.",
                self.styles["BodyText2"],
            ))

        # === DIGITAL EVIDENCE CERTIFICATE ===
        if evidence_hash or merkle_root:
            elements.append(PageBreak())
            elements.append(Paragraph(
                "5. Digital Evidence Certificate",
                self.styles["SectionHeader"],
            ))
            elements.append(Paragraph(
                "This detection is secured with a blockchain-based audit trail.",
                self.styles["BodyText2"],
            ))
            cert_data = []
            if evidence_hash:
                cert_data.append(["Evidence Hash (SHA-256)", evidence_hash])
            if merkle_root:
                cert_data.append(["Merkle Root", merkle_root])
            cert_data.append(["Hash Algorithm", "SHA-256"])
            cert_data.append(["Timestamp", datetime.utcnow().isoformat() + "Z"])
            elements.append(self._create_table(cert_data, col_widths=[160, 340]))

        # === FOOTER ===
        elements.append(Spacer(1, 30))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
        elements.append(Paragraph(
            f"Report generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} | "
            f"Surat Satellite Compliance Engine (SSCE) v1.0",
            ParagraphStyle("Footer", parent=self.styles["Normal"], fontSize=8, textColor=colors.grey),
        ))

        doc.build(elements)
        logger.info(f"Report generated: {filepath}")
        return str(filepath)

    def _create_table(
        self, data: list, col_widths: list = None, header: bool = False
    ):
        """Create a styled table."""
        table = Table(data, colWidths=col_widths)
        style_commands = [
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8eaf6")),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#1a1a2e")),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c5cae9")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]
        if header:
            style_commands.extend([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a237e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ])
        table.setStyle(TableStyle(style_commands))
        return table

    def _create_image_panel(self, before, after, spike_map):
        """Create before/after/spike image panel."""
        elements = []
        # Placeholder — in production, convert numpy arrays to ReportLab images
        elements.append(Paragraph(
            "Before → After imagery and SNN spike heatmap available in interactive dashboard.",
            self.styles["BodyText2"],
        ))
        return elements

    def _interpret_spectral(self, index: str, delta: float) -> str:
        """Interpret spectral index change."""
        if index == "ndvi":
            if delta < -0.3:
                return "Severe vegetation loss"
            elif delta < -0.15:
                return "Moderate vegetation decline"
            elif delta > 0.15:
                return "Vegetation growth"
            return "Stable vegetation"
        elif index == "ndbi":
            if delta > 0.2:
                return "Significant new built-up area"
            elif delta > 0.1:
                return "Minor construction activity"
            return "Stable built environment"
        elif index == "mndwi":
            if delta < -0.2:
                return "Water body reduction"
            elif delta > 0.2:
                return "Water body expansion"
            return "Stable water body"
        return "N/A"

    def _generate_text_report(self, detection: Dict, violations: List[Dict]) -> str:
        """Fallback text report when ReportLab is not available."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"compliance_report_{timestamp}.txt"
        filepath = self.output_dir / filename

        lines = [
            "=" * 60,
            "SATELLITE COMPLIANCE REPORT",
            "Surat Municipal Corporation",
            "=" * 60,
            "",
            f"Detection ID: {detection.get('id', 'N/A')}",
            f"Change Type: {detection.get('change_type', 'N/A')}",
            f"Confidence: {detection.get('confidence', 0) * 100:.1f}%",
            f"Area: {detection.get('area_hectares', 0):.3f} hectares",
            "",
        ]

        if violations:
            lines.append(f"VIOLATIONS: {len(violations)}")
            for v in violations:
                lines.append(f"  - {v.get('rule_id')}: {v.get('rule_name')}")
                lines.append(f"    Legal Ref: {v.get('legal_ref')}")
                lines.append(f"    Severity: {v.get('severity')}")
        else:
            lines.append("No violations detected.")

        filepath.write_text("\n".join(lines))
        return str(filepath)
