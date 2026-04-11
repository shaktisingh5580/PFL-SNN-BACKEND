"""
Legal-Grade PDF Compliance Report Generator.

Generates professional urban compliance reports with:
  - Embedded Before/After satellite imagery
  - Change detection mask visualization
  - Spectral classification results with area measurements
  - Compliance rule violations with legal references
  - Blockchain evidence hash for tamper-proof audit trail
  - Surat Municipal Corporation branding

Uses fpdf2 for reliable cross-platform PDF generation.
Install: pip install fpdf2
"""
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False
    logger.warning("fpdf2 not installed. Install: pip install fpdf2")


def _sanitize(text: str) -> str:
    """Replace Unicode characters unsupported by Helvetica with ASCII equivalents."""
    if not isinstance(text, str):
        return str(text)
        
    replacements = {
        "\u2014": "-",   # em-dash
        "\u2013": "-",   # en-dash
        "\u00a7": "S.",  # section sign
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2026": "...", # ellipsis
        "\u00b0": "deg", # degree
        "\u2192": "->",  # arrow
        "\u2022": "*",   # bullet
    }
    for uni, ascii_char in replacements.items():
        text = text.replace(uni, ascii_char)
        
    # Ultimate safeguard for fpdf: force translatable characters, replace others with '?'
    return text.encode('latin-1', 'replace').decode('latin-1')


class ComplianceReportPDF(FPDF):
    """Custom FPDF subclass with header/footer branding."""

    DARK_BLUE = (10, 15, 40)
    BLUE = (26, 35, 126)
    LIGHT_BLUE = (232, 234, 246)
    RED = (220, 50, 50)
    GREEN = (34, 139, 34)
    ORANGE = (245, 158, 11)
    GRAY = (120, 120, 120)
    WHITE = (255, 255, 255)

    def __init__(self, city: str = "Surat", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.city = city
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(*self.DARK_BLUE)
        self.cell(0, 8, "SATELLITE COMPLIANCE REPORT", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*self.GRAY)
        self.cell(0, 5, f"{self.city} Municipal Corporation - Urban Planning & Compliance Division", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*self.BLUE)
        self.set_line_width(0.8)
        self.line(10, self.get_y() + 2, 200, self.get_y() + 2)
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*self.GRAY)
        self.cell(0, 5, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} | "
                  f"Agentic Geospatial Compliance Engine v1.0 | Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, num: int, title: str):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*self.BLUE)
        self.cell(0, 8, f"{num}. {title}", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def kv_row(self, key: str, value: str, bold_value: bool = False):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*self.DARK_BLUE)
        self.cell(55, 6, key, border=1, fill=True, new_x="RIGHT")
        self.set_font("Helvetica", "B" if bold_value else "", 9)
        self.set_text_color(40, 40, 40)
        self.set_fill_color(*self.WHITE)
        self.cell(0, 6, _sanitize(str(value)), border=1, new_x="LMARGIN", new_y="NEXT")
        self.set_fill_color(*self.LIGHT_BLUE)

    def severity_badge(self, severity: str):
        color_map = {
            "CRITICAL": self.RED,
            "HIGH": (220, 80, 50),
            "MEDIUM": self.ORANGE,
            "LOW": self.GREEN,
        }
        color = color_map.get(severity, self.GRAY)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*color)
        badge = f"[{severity}]"
        return badge


def generate_compliance_pdf(
    scan_dir: str,
    report_data: Dict,
    violations: List[Dict],
    output_path: Optional[str] = None,
    city: str = "Surat",
) -> str:
    """
    Generate a professional compliance report PDF from scan results.

    Args:
        scan_dir: Path to scan output directory (contains .png files)
        report_data: The JSON report from the scanner
        violations: List of violation dicts from the rule engine
        output_path: Custom output path (optional)
        city: City name for branding

    Returns:
        Path to generated PDF file
    """
    if not HAS_FPDF:
        logger.error("fpdf2 not installed. Run: pip install fpdf2")
        return _generate_text_fallback(scan_dir, report_data, violations)

    scan_dir = Path(scan_dir)

    if output_path is None:
        output_path = str(scan_dir / "compliance_report.pdf")

    pdf = ComplianceReportPDF(city=city, orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_fill_color(*ComplianceReportPDF.LIGHT_BLUE)

    # ═══════════════════════════════════════════════════
    #  SECTION 1: Scan Summary
    # ═══════════════════════════════════════════════════
    pdf.section_title(1, "Scan Summary")

    scan_id = report_data.get("scan_id", report_data.get("city", "N/A"))
    coords = report_data.get("coordinates", {})
    center = coords.get("center", [0, 0])
    classification = report_data.get("classification", {})

    pdf.kv_row("Scan ID", str(scan_id))
    pdf.kv_row("City / Region", report_data.get("city", city))
    pdf.kv_row("Timestamp", report_data.get("timestamp", datetime.utcnow().isoformat()))
    pdf.kv_row("Center Coordinates", f"{center[0]:.4f} N, {center[1]:.4f} E")

    bounds = coords.get("bounds", [])
    if bounds and len(bounds) == 2:
        pdf.kv_row("Bounding Box", f"[{bounds[0][0]:.4f}, {bounds[0][1]:.4f}] to [{bounds[1][0]:.4f}, {bounds[1][1]:.4f}]")

    pdf.kv_row("CRS", coords.get("crs", "EPSG:4326"))

    total_area = classification.get("total_changed_area_m2", 0)
    pdf.kv_row("Total Changed Area", f"{total_area:,.0f} m2 ({total_area/10000:.2f} hectares)", bold_value=True)
    pdf.kv_row("Total Changed Pixels", f"{classification.get('total_changed_pixels', 0):,}")
    pdf.ln(4)

    # ═══════════════════════════════════════════════════
    #  SECTION 2: Satellite Imagery
    # ═══════════════════════════════════════════════════
    before_img = scan_dir / "before_rgb.png"
    after_img = scan_dir / "after_rgb.png"
    mask_img = scan_dir / "change_mask.png"
    overlay_img = scan_dir / "classification_overlay.png"

    has_images = before_img.exists() or after_img.exists()

    if has_images:
        pdf.section_title(2, "Satellite Imagery Comparison")

        img_w = 88  # mm width for each image
        img_gap = 4

        # Before / After side by side
        if before_img.exists() and after_img.exists():
            y_start = pdf.get_y()
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(80, 80, 80)
            pdf.cell(img_w, 5, "BEFORE (T1)", align="C")
            pdf.cell(img_gap, 5, "")
            pdf.cell(img_w, 5, "AFTER (T2)", align="C", new_x="LMARGIN", new_y="NEXT")

            y_img = pdf.get_y()
            pdf.image(str(before_img), x=10, y=y_img, w=img_w)
            pdf.image(str(after_img), x=10 + img_w + img_gap, y=y_img, w=img_w)
            pdf.set_y(y_img + img_w * 0.85)
            pdf.ln(4)

        # Change Mask / Classification side by side
        if mask_img.exists() and overlay_img.exists():
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(80, 80, 80)
            pdf.cell(img_w, 5, "SNN CHANGE MASK", align="C")
            pdf.cell(img_gap, 5, "")
            pdf.cell(img_w, 5, "CLASSIFICATION OVERLAY", align="C", new_x="LMARGIN", new_y="NEXT")

            y_img = pdf.get_y()
            pdf.image(str(mask_img), x=10, y=y_img, w=img_w)
            pdf.image(str(overlay_img), x=10 + img_w + img_gap, y=y_img, w=img_w)
            pdf.set_y(y_img + img_w * 0.85)
            pdf.ln(4)

    # ═══════════════════════════════════════════════════
    #  SECTION 3: Change Classification
    # ═══════════════════════════════════════════════════
    section_num = 3
    pdf.add_page()
    pdf.section_title(section_num, "Change Classification Results")

    findings = classification.get("findings", [])

    if findings:
        # Table header
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(26, 35, 126)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(65, 7, "Change Type", border=1, fill=True)
        pdf.cell(25, 7, "Pixels", border=1, fill=True, align="C")
        pdf.cell(30, 7, "Area (m2)", border=1, fill=True, align="C")
        pdf.cell(30, 7, "Hectares", border=1, fill=True, align="C")
        pdf.cell(20, 7, "%", border=1, fill=True, align="C")
        pdf.cell(20, 7, "Severity", border=1, fill=True, align="C", new_x="LMARGIN", new_y="NEXT")

        # Table rows
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(40, 40, 40)

        severity_colors = {
            "high": (255, 230, 230),
            "medium": (255, 245, 220),
            "low": (230, 255, 230),
        }

        for f in findings:
            sev = f.get("severity", "low")
            bg = severity_colors.get(sev, (255, 255, 255))
            pdf.set_fill_color(*bg)

            pdf.cell(65, 6, f.get("class_name", "")[:30], border=1, fill=True)
            pdf.cell(25, 6, f"{f.get('pixel_count', 0):,}", border=1, fill=True, align="C")
            pdf.cell(30, 6, f"{f.get('area_m2', 0):,.0f}", border=1, fill=True, align="C")
            pdf.cell(30, 6, f"{f.get('area_hectares', 0):.2f}", border=1, fill=True, align="C")
            pdf.cell(20, 6, f"{f.get('percentage', 0):.1f}", border=1, fill=True, align="C")
            pdf.cell(20, 6, sev.upper(), border=1, fill=True, align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(6)

    # ═══════════════════════════════════════════════════
    #  SECTION 4: Compliance Violations
    # ═══════════════════════════════════════════════════
    section_num = 4
    pdf.section_title(section_num, "Compliance Assessment")

    if not violations:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(*ComplianceReportPDF.GREEN)
        pdf.cell(0, 8, "No compliance violations detected.", new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(*ComplianceReportPDF.RED)
        pdf.cell(0, 8, f"{len(violations)} COMPLIANCE VIOLATION(S) DETECTED", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

        for i, v in enumerate(violations, 1):
            # Check if we need a new page
            if pdf.get_y() > 240:
                pdf.add_page()

            severity = v.get("severity", "LOW")
            badge = pdf.severity_badge(severity)

            # Violation header
            pdf.set_font("Helvetica", "B", 11)
            sev_color = {
                "CRITICAL": ComplianceReportPDF.RED,
                "HIGH": (220, 80, 50),
                "MEDIUM": ComplianceReportPDF.ORANGE,
            }.get(severity, ComplianceReportPDF.GREEN)
            pdf.set_text_color(*sev_color)
            pdf.cell(0, 7, _sanitize(f"Violation #{i}: {v.get('rule_name', 'Unknown')} {badge}"), new_x="LMARGIN", new_y="NEXT")

            pdf.set_fill_color(*ComplianceReportPDF.LIGHT_BLUE)
            pdf.kv_row("Rule ID", v.get("rule_id", "N/A"))
            pdf.kv_row("Severity", severity, bold_value=True)
            pdf.kv_row("Legal Reference", v.get("legal_reference", "N/A"))
            pdf.kv_row("Description", v.get("description", "N/A"))
            pdf.kv_row("Violation Type", v.get("violation_type", "N/A").replace("_", " ").title())

            details = v.get("details", {})
            if "distance_to_protected_m" in details:
                pdf.kv_row("Distance to Protected Zone", f"{details['distance_to_protected_m']} m")
                pdf.kv_row("Required Buffer", f"{details.get('buffer_required_m', '?')} m")
            if "affected_area_hectares" in details:
                pdf.kv_row("Affected Area", f"{details['affected_area_hectares']} hectares")
            if "protected_zone" in details:
                pdf.kv_row("Protected Zone", details["protected_zone"].replace("_", " ").title())
            if "vegetation_loss_percent" in details:
                pdf.kv_row("Vegetation Loss", f"{details['vegetation_loss_percent']}%")
                pdf.kv_row("Minimum Required", f"{details.get('minimum_required_percent', 30)}%")

            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(*ComplianceReportPDF.RED)
            pdf.cell(55, 6, "Recommended Action", border=1, fill=True, new_x="RIGHT")
            pdf.set_fill_color(255, 240, 240)
            pdf.cell(0, 6, _sanitize(v.get("action", "N/A")), border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
            pdf.set_fill_color(*ComplianceReportPDF.LIGHT_BLUE)
            pdf.ln(6)

    # ═══════════════════════════════════════════════════
    #  SECTION 5: Blockchain Evidence
    # ═══════════════════════════════════════════════════
    section_num = 5
    if pdf.get_y() > 220:
        pdf.add_page()

    pdf.section_title(section_num, "Digital Evidence Certificate")

    blockchain = report_data.get("blockchain", {})
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(40, 40, 40)

    pdf.kv_row("Evidence Hash (SHA-256)", blockchain.get("hash", "N/A"))
    pdf.kv_row("Hash Algorithm", "SHA-256 (FIPS 180-4)")
    pdf.kv_row("Timestamp", blockchain.get("timestamp", datetime.utcnow().isoformat()))
    pdf.kv_row("Verification Status", "VERIFIED" if blockchain.get("verified") else "PENDING")
    pdf.kv_row("Admissibility", "Indian Evidence Act, 1872 - Section 65B (Electronic Records)")

    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*ComplianceReportPDF.GRAY)
    pdf.multi_cell(0, 4,
        "This report is generated by an automated satellite imagery analysis system. "
        "The SHA-256 hash above cryptographically seals the scan data, model output, "
        "and classification results. Any modification to the underlying evidence will "
        "produce a different hash, making tampering immediately detectable. "
        "This digital evidence is admissible under Section 65B of the Indian Evidence Act, 1872."
    )

    # ═══════════════════════════════════════════════════
    #  SECTION 6: Model Information
    # ═══════════════════════════════════════════════════
    model_info = report_data.get("model_info", {})
    if model_info:
        pdf.ln(6)
        pdf.section_title(6, "AI Model Information")
        pdf.kv_row("Model Architecture", model_info.get("name", "Siamese-SNN v3"))
        pdf.kv_row("Parameters", f"{model_info.get('parameters', 0):,}")
        pdf.kv_row("Inference Time", f"{model_info.get('inference_time_seconds', 0):.1f} seconds")
        pdf.kv_row("Throughput", f"{model_info.get('throughput_patches_per_hour', 0):,} patches/hour")
        pdf.kv_row("Training Dataset", "OSCD (Onera Satellite Change Detection)")
        pdf.kv_row("Spectral Bands", "B02 (Blue), B03 (Green), B04 (Red), B08 (NIR), B11 (SWIR)")

    # === SAVE ===
    pdf.output(output_path)
    logger.info(f"PDF report saved: {output_path}")
    print(f"  PDF saved: {output_path}")
    return output_path


def _generate_text_fallback(scan_dir, report_data, violations):
    """Fallback when fpdf2 is not installed."""
    output_path = Path(scan_dir) / "compliance_report.txt"
    lines = [
        "=" * 60,
        "SATELLITE COMPLIANCE REPORT",
        "=" * 60,
        f"City: {report_data.get('city', 'N/A')}",
        f"Timestamp: {report_data.get('timestamp', 'N/A')}",
        "",
    ]
    classification = report_data.get("classification", {})
    for f in classification.get("findings", []):
        lines.append(f"  {f['class_name']}: {f.get('area_hectares', 0)} ha [{f.get('severity', '')}]")

    if violations:
        lines.append(f"\nVIOLATIONS: {len(violations)}")
        for v in violations:
            lines.append(f"  [{v['severity']}] {v['rule_name']} - {v.get('legal_reference', '')}")

    output_path.write_text("\n".join(lines))
    return str(output_path)
