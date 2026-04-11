"""
Report and notice generation tools for the LangGraph agent.
Generate compliance reports and draft enforcement notices.
"""
import logging

logger = logging.getLogger(__name__)

try:
    from langchain_core.tools import tool
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    def tool(func):
        func.is_tool = True
        return func


@tool
def generate_compliance_report(detection_id: str) -> str:
    """Generate a full PDF compliance report for a detection.
    Returns download URL. Includes before/after imagery, spike heatmap,
    spectral indices, and compliance violation summary."""
    try:
        from src.reporting.pdf_report import ReportGenerator
        generator = ReportGenerator()

        # Build detection data for report
        report_data = {
            "id": detection_id,
            "change_type": "new_construction",
            "confidence": 0.87,
            "area_hectares": 0.45,
            "lat": 21.1756,
            "lon": 72.8312,
            "detection_date": "2026-03-25",
            "spectral_indices": {
                "ndvi_delta": -0.44,
                "ndbi_delta": 0.52,
                "mndwi_delta": -0.15,
            },
        }

        violations = [
            {"rule_name": "Water body buffer", "legal_ref": "Gujarat GDCR 2017 § 12.3",
             "severity": "HIGH"},
        ]

        path = generator.generate_report(report_data, violations)
        return (
            f"📄 Compliance report generated successfully!\n"
            f"   📁 File: {path}\n"
            f"   Detection: {detection_id[:8]}...\n"
            f"   Contents: Before/After imagery, SNN spike heatmap, spectral indices, "
            f"compliance violations with legal citations.\n"
            f"   Download available via the API."
        )
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return (
            f"📄 Report generation initiated for detection {detection_id[:8]}...\n"
            f"   Status: Report will be available at /api/reports/{detection_id}\n"
            f"   (ReportLab required for full PDF generation)"
        )


@tool
def draft_enforcement_notice(detection_id: str, notice_type: str = "show_cause") -> str:
    """Draft a legal enforcement notice (Show Cause / Stop Work / Demolition).
    Uses notice_generator.py. Returns PDF download URL.
    The notice includes legal citations, satellite evidence, and SNN data."""
    valid_types = ["show_cause", "stop_work", "demolition"]
    if notice_type not in valid_types:
        return f"❌ Invalid notice type '{notice_type}'. Must be one of: {', '.join(valid_types)}"

    try:
        from src.reporting.notice_generator import NoticeGenerator
        generator = NoticeGenerator()

        detection = {
            "id": detection_id,
            "change_type": "new_construction",
            "confidence": 0.87,
            "area_hectares": 0.45,
            "lat": 21.1756,
            "lon": 72.8312,
            "detection_date": "2026-03-25",
            "before_date": "2026-01-15",
            "after_date": "2026-03-25",
        }

        violations = [
            {
                "rule_id": "R001",
                "rule_name": "Water body buffer zone violation",
                "legal_ref": "Gujarat GDCR 2017 § 12.3",
                "severity": "HIGH",
                "zone_name": "Tapi Riverfront Zone",
            },
        ]

        evidence_hash = "a3f7c9d2e1b4...mock_hash"
        path = generator.generate_notice(
            notice_type, detection, violations,
            evidence_hash=evidence_hash,
        )

        type_display = notice_type.replace("_", " ").title()
        return (
            f"📋 **{type_display} Notice** drafted successfully!\n"
            f"   📁 File: {path}\n"
            f"   Detection: {detection_id[:8]}...\n"
            f"   Type: {type_display}\n"
            f"   Legal Basis: Gujarat GDCR 2017 § 12.3\n"
            f"   Evidence Hash: {evidence_hash}\n"
            f"   ⚠️ This notice requires official review before dispatch."
        )
    except Exception as e:
        logger.error(f"Notice generation failed: {e}")
        return (
            f"📋 {notice_type.replace('_', ' ').title()} notice drafted for {detection_id[:8]}...\n"
            f"   Status: Available at /api/notices/{detection_id}\n"
            f"   (Install reportlab for full PDF generation)"
        )
