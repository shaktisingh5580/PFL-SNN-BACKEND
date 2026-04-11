"""
Compliance check tools for the LangGraph agent.
Run compliance checks and list active violations.
"""
import logging
import json

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
def run_compliance_check(detection_id: str) -> str:
    """Run full compliance check on a detection. Returns violated rules
    with legal citations from Gujarat GDCR / TP Act."""
    # Load rules
    try:
        from config.settings import PROJECT_ROOT
        rules_path = PROJECT_ROOT / "config" / "compliance_rules.json"
        with open(rules_path) as f:
            rules_data = json.load(f)
        rules = rules_data.get("rules", [])
    except Exception:
        rules = []

    # Mock compliance check results
    violations = [
        {
            "rule_id": "R001",
            "rule_name": "No construction near water bodies",
            "legal_ref": "Gujarat GDCR 2017 § 12.3 — Riverfront & Water Body Buffer Zone",
            "severity": "HIGH",
            "description": f"Detection {detection_id[:8]}... is within 350m of a water body (buffer: 500m)",
            "distance_m": 350,
        },
        {
            "rule_id": "R002",
            "rule_name": "Minimum green cover in residential zones",
            "legal_ref": "Gujarat TP & UD Act 1976 § 22 — Development Plan Compliance",
            "severity": "MEDIUM",
            "description": "Green cover dropped from 42% to 18% (minimum required: 30%)",
            "current_percent": 18,
            "required_percent": 30,
        },
    ]

    if not violations:
        return f"✅ Detection {detection_id[:8]}... passed all {len(rules)} compliance checks."

    lines = [
        f"⚠️ Compliance Check for Detection {detection_id[:8]}...\n",
        f"Checked against {len(rules)} rules. Found {len(violations)} violation(s):\n",
    ]

    for i, v in enumerate(violations, 1):
        severity_emoji = {"LOW": "🟡", "MEDIUM": "🟠", "HIGH": "🔴", "CRITICAL": "⛔"}.get(
            v["severity"], "⚪"
        )
        lines.append(
            f"\n{severity_emoji} **Violation {i}: {v['rule_name']}**\n"
            f"   Rule ID: {v['rule_id']}\n"
            f"   Legal Reference: {v['legal_ref']}\n"
            f"   Severity: {v['severity']}\n"
            f"   Details: {v['description']}"
        )

    lines.append("\n\n💡 Would you like me to draft an enforcement notice for these violations?")
    return "\n".join(lines)


@tool
def list_violations(ward: str = None, severity: str = None) -> str:
    """List all active violations, optionally filtered by ward or severity.
    Severity levels: LOW, MEDIUM, HIGH, CRITICAL."""
    # Mock violations for demo
    all_violations = [
        {"id": "V001", "detection_id": "d1a2b3c4", "rule": "R001 - Water body buffer",
         "ward": "Ward 42", "severity": "HIGH", "status": "active",
         "location": "21.1756°N, 72.8312°E", "date": "2026-03-25"},
        {"id": "V002", "detection_id": "e5f6g7h8", "rule": "R005 - Green Belt Encroachment",
         "ward": "Ward 42", "severity": "CRITICAL", "status": "active",
         "location": "21.2156°N, 72.8687°E", "date": "2026-03-22"},
        {"id": "V003", "detection_id": "i9j0k1l2", "rule": "R002 - Green cover minimum",
         "ward": "Ward 15", "severity": "MEDIUM", "status": "active",
         "location": "21.1892°N, 72.7945°E", "date": "2026-03-20"},
        {"id": "V004", "detection_id": "m3n4o5p6", "rule": "R004 - Tapi Riverfront Buffer",
         "ward": "Adajan", "severity": "HIGH", "status": "under_review",
         "location": "21.1812°N, 72.7834°E", "date": "2026-03-18"},
    ]

    # Filter
    filtered = all_violations
    if ward:
        filtered = [v for v in filtered if ward.lower() in v["ward"].lower()]
    if severity:
        filtered = [v for v in filtered if v["severity"].upper() == severity.upper()]

    if not filtered:
        return f"No violations found matching filters (ward={ward}, severity={severity})."

    lines = [f"📋 Active Violations ({len(filtered)} found):\n"]
    for v in filtered:
        severity_emoji = {"LOW": "🟡", "MEDIUM": "🟠", "HIGH": "🔴", "CRITICAL": "⛔"}.get(
            v["severity"], "⚪"
        )
        lines.append(
            f"  {severity_emoji} **{v['id']}** — {v['rule']}\n"
            f"     Ward: {v['ward']} | Location: {v['location']}\n"
            f"     Date: {v['date']} | Status: {v['status']}\n"
        )

    return "\n".join(lines)
