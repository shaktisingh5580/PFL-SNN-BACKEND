"""
Automated enforcement notice generation.
Creates legal notices (Show Cause, Stop Work, Demolition) with satellite evidence.
"""
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional

logger = logging.getLogger(__name__)

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
    )
    from reportlab.lib.enums import TA_CENTER
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

from config.chatbot_prompts import NOTICE_TEMPLATES


class NoticeGenerator:
    """Generate enforcement notice PDFs."""

    def __init__(self, output_dir: str = "outputs/notices"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if HAS_REPORTLAB:
            self.styles = getSampleStyleSheet()
            self._setup_styles()

    def _setup_styles(self):
        self.styles.add(ParagraphStyle(
            name="NoticeTitle",
            parent=self.styles["Title"],
            fontSize=20,
            spaceAfter=5,
            textColor=colors.HexColor("#b71c1c"),
            alignment=TA_CENTER,
        ))
        self.styles.add(ParagraphStyle(
            name="NoticeSubtitle",
            parent=self.styles["Normal"],
            fontSize=12,
            spaceAfter=15,
            textColor=colors.HexColor("#424242"),
            alignment=TA_CENTER,
        ))
        self.styles.add(ParagraphStyle(
            name="NoticeBody",
            parent=self.styles["Normal"],
            fontSize=11,
            spaceAfter=10,
            leading=16,
        ))

    def generate_notice(
        self,
        notice_type: str,
        detection: Dict,
        violations: list,
        evidence_hash: Optional[str] = None,
        merkle_root: Optional[str] = None,
    ) -> str:
        """
        Generate an enforcement notice PDF.

        Args:
            notice_type: 'show_cause', 'stop_work', or 'demolition'
            detection: Detection metadata
            violations: List of violation dicts
            evidence_hash: Blockchain evidence hash
            merkle_root: Merkle tree root hash

        Returns:
            Path to generated PDF
        """
        template = NOTICE_TEMPLATES.get(notice_type)
        if not template:
            logger.error(f"Unknown notice type: {notice_type}")
            return ""

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        det_id = str(detection.get("id", "unknown"))[:8]
        filename = f"notice_{notice_type}_{det_id}_{timestamp}.pdf"
        filepath = self.output_dir / filename

        if not HAS_REPORTLAB:
            return self._generate_text_notice(filepath, template, detection, violations, evidence_hash)

        doc = SimpleDocTemplate(
            str(filepath), pagesize=A4,
            rightMargin=25 * mm, leftMargin=25 * mm,
            topMargin=30 * mm, bottomMargin=25 * mm,
        )

        elements = []

        # Header
        elements.append(Paragraph(
            "SURAT MUNICIPAL CORPORATION",
            ParagraphStyle("SMC", parent=self.styles["Normal"],
                         fontSize=16, fontName="Helvetica-Bold", alignment=TA_CENTER),
        ))
        elements.append(Paragraph(
            "Urban Planning & Development Department",
            ParagraphStyle("Dept", parent=self.styles["Normal"],
                         fontSize=11, alignment=TA_CENTER),
        ))
        elements.append(Spacer(1, 10))
        elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#b71c1c")))
        elements.append(Spacer(1, 15))

        # Title
        elements.append(Paragraph(template["title"], self.styles["NoticeTitle"]))
        elements.append(Paragraph(template["subtitle"], self.styles["NoticeSubtitle"]))

        # Reference number
        ref_num = f"SMC/UPD/{notice_type.upper()}/{datetime.now().strftime('%Y/%m/%d')}/{det_id}"
        elements.append(Paragraph(f"<b>Reference No:</b> {ref_num}", self.styles["NoticeBody"]))
        elements.append(Paragraph(
            f"<b>Date:</b> {datetime.now().strftime('%d %B %Y')}",
            self.styles["NoticeBody"],
        ))
        elements.append(Spacer(1, 15))

        # Notice body
        body_text = template["body"].format(
            change_type=detection.get("change_type", "unauthorized construction").replace("_", " "),
            lat=detection.get("lat", "N/A"),
            lon=detection.get("lon", "N/A"),
            zone_name=violations[0].get("zone_name", "Unknown Zone") if violations else "Unknown Zone",
            detection_date=detection.get("detection_date", datetime.now().strftime("%Y-%m-%d")),
            area_hectares=detection.get("area_hectares", 0),
            confidence=detection.get("confidence", 0) * 100,
            legal_ref=violations[0].get("legal_ref", "Gujarat GDCR 2017") if violations else "Gujarat GDCR 2017",
            evidence_hash=evidence_hash or "N/A",
            merkle_root=merkle_root or "N/A",
            response_days=template["response_days"],
            before_date=detection.get("before_date", "N/A"),
            after_date=detection.get("after_date", "N/A"),
        )

        for para in body_text.strip().split("\n"):
            para = para.strip()
            if para:
                elements.append(Paragraph(para, self.styles["NoticeBody"]))

        elements.append(Spacer(1, 15))

        # Violations table
        if violations:
            elements.append(Paragraph("<b>Violated Regulations:</b>", self.styles["NoticeBody"]))
            v_data = [["#", "Rule", "Legal Reference", "Severity"]]
            for i, v in enumerate(violations, 1):
                v_data.append([
                    str(i),
                    v.get("rule_name", "N/A"),
                    v.get("legal_ref", "N/A"),
                    v.get("severity", "N/A"),
                ])

            table = Table(v_data, colWidths=[30, 150, 200, 80])
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#b71c1c")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
            ]))
            elements.append(table)

        # Response deadline
        if template["response_days"] > 0:
            deadline = (datetime.now() + timedelta(days=template["response_days"])).strftime("%d %B %Y")
            elements.append(Spacer(1, 20))
            elements.append(Paragraph(
                f"<b>Response Deadline: {deadline}</b>",
                ParagraphStyle("Deadline", parent=self.styles["NoticeBody"],
                             textColor=colors.HexColor("#b71c1c"), fontSize=12),
            ))

        # Signature block
        elements.append(Spacer(1, 40))
        elements.append(Paragraph(
            "Town Planning Officer<br/>Surat Municipal Corporation",
            ParagraphStyle("Sig", parent=self.styles["Normal"], fontSize=11),
        ))

        # Footer
        elements.append(Spacer(1, 30))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
        elements.append(Paragraph(
            "This notice is generated by the Surat Satellite Compliance Engine (SSCE). "
            "Evidence is secured via blockchain-based audit trail.",
            ParagraphStyle("NFooter", parent=self.styles["Normal"],
                         fontSize=7, textColor=colors.grey),
        ))

        doc.build(elements)
        logger.info(f"Notice generated: {filepath}")
        return str(filepath)

    def _generate_text_notice(self, filepath, template, detection, violations, evidence_hash):
        """Fallback text notice."""
        lines = [
            "=" * 60,
            "SURAT MUNICIPAL CORPORATION",
            template["title"],
            template["subtitle"],
            "=" * 60,
            "",
            f"Change Type: {detection.get('change_type', 'N/A')}",
            f"Location: {detection.get('lat', 'N/A')}°N, {detection.get('lon', 'N/A')}°E",
            f"Confidence: {detection.get('confidence', 0) * 100:.1f}%",
            f"Evidence Hash: {evidence_hash or 'N/A'}",
            "",
        ]
        for v in violations:
            lines.append(f"Violation: {v.get('rule_name')} — {v.get('legal_ref')}")

        filepath.write_text("\n".join(lines))
        return str(filepath)
