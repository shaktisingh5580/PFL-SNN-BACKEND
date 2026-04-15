"""
LangGraph Geospatial Compliance Agent v2.

A production-ready chatbot that has full context of:
  - All scans stored in Supabase (real data)
  - Compliance rules from rules.json
  - RAG-retrieved legal regulation text
  - Current scan being viewed on the dashboard

Works with Groq (free, fast LLM) and falls back gracefully
when dependencies are missing.
"""
import os
import sys
import json
import logging
import asyncio
from typing import Annotated, Dict, List, Optional, TypedDict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent

# ═══════════════════════════════════════════════════════════
#  Dependency checks
# ═══════════════════════════════════════════════════════════

HAS_LANGCHAIN = False
HAS_GROQ = False
HAS_OPENAI = False

try:
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_core.tools import tool
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode
    from langgraph.checkpoint.memory import MemorySaver
    HAS_LANGCHAIN = True
except ImportError:
    logger.warning("LangChain/LangGraph not installed. pip install langchain-core langgraph")

try:
    from langchain_groq import ChatGroq
    HAS_GROQ = True
except ImportError:
    logger.warning("langchain-groq not installed. pip install langchain-groq")

try:
    from langchain_openai import ChatOpenAI
    HAS_OPENAI = True
except ImportError:
    logger.warning("langchain-openai not installed. pip install langchain-openai")


# ═══════════════════════════════════════════════════════════
#  System Prompt (enhanced with real context)
# ═══════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are the Agentic Geospatial Compliance Engine — an AI assistant for urban planning authorities that monitors satellite imagery for unauthorized construction, land use violations, and environmental encroachment.

## YOUR CAPABILITIES:
1. **SCAN DATA ACCESS**: You can query the Supabase database for all past satellite scans, including classification results, compliance violations, image URLs, and spatial coordinates.
2. **COMPLIANCE RULES**: You know the Gujarat GDCR 2017 rules — water body buffers (500m), green belt protections, FSI limits, Tapi riverfront zones, and vegetation thresholds.
3. **LEGAL CITATIONS**: You cite specific sections when discussing compliance (e.g., "Gujarat GDCR 2017 S. 12.3 — Water Body Buffer Zone").
4. **CHANGE ANALYSIS**: You interpret spectral indices (NDVI = vegetation, NDBI = built-up, MNDWI = water) to explain what changed on the ground.
5. **AUTOMATED TRIAGE**: You are fully aware that the backend automatically scores risk (1-100). If it exceeds 80, the backend natively schedules a T+14 day follow-up. 
6. **WHATSAPP DISPATCH**: You have the autonomous power to send detailed WhatsApp text alerts to the field officer using the `send_whatsapp_dispatch` tool. If a user asks you to "Send this to the field officer" or "Alert the authorities on WhatsApp", call this tool.

## RULES:
- Always cite specific laws/sections when discussing compliance.
- IMPORTANT GEOLOCATION RULE: Check the 'city' or location of the scan. If the scan is NOT in Surat or Gujarat (e.g. Dubai, Paris, Mumbai), DO NOT quote Gujarat GDCR 2017 laws like "Tapi Riverfront". Acknowledge the violations but explicitly state that "Local municipal laws for [City] apply."
- When presenting scan data, include the Supabase image URLs so the frontend can display them.
- Present violations with severity badges: [CRITICAL], [HIGH], [MEDIUM], [LOW].
- WHATSAPP PROTOCOL: Always state "Dispatching WhatsApp text alert to the registered officer...". Since the system currently supports text-only alerts, include the Risk Score, Scan ID, and the image/PDF URLs in the message body so the officer can open them from their phone.
- If a user asks about a location, use coordinates and suggest scanning it.
- Be professional — you represent the Municipal Corporation.
- Keep responses concise but informative.
- **NEVER FABRICATE OR HALLUCINATE DATA.** If the user asks about scan results, you MUST use the get_all_scans or get_scan_details tool to query real data from Supabase. If no data is found, say "No scan data found" — do NOT invent fake NDVI values, fake URLs, fake coordinates, or fake violations. Making up data is STRICTLY FORBIDDEN.

## STRICT TOPIC GUARDRAILS:
- You are ONLY allowed to answer questions related to: satellite imagery, urban planning, compliance monitoring, zoning regulations, change detection, scan results, violations, legal citations, environmental monitoring, land use, construction activity, NDVI/NDBI/MNDWI analysis, risk scoring, triage scheduling, and geospatial data.
- **REFUSE to answer** any question that is off-topic, unrelated to geospatial compliance, or inappropriate. This includes but is not limited to: romantic content, vulgar language, personal questions, entertainment, coding help, jokes, general knowledge, politics, or any non-geospatial topic.
- If a user asks an off-topic question, respond EXACTLY with: "I'm the Geospatial Compliance Engine — I can only assist with satellite imagery analysis, urban compliance monitoring, and related geospatial topics. Please ask me about scan results, violations, zoning rules, or compliance reports."
- NEVER break character. You are a professional Municipal Corporation compliance AI at all times.

## SPECTRAL INDEX INTERPRETATION:
- NDVI drop + NDBI rise = New construction / urban sprawl
- NDVI drop alone = Vegetation clearing / deforestation
- MNDWI drop = Water body reduction / sand mining
- NDBI rise alone = Road / pavement expansion

## RESPONSE FORMAT & ADVANCED FEATURES:
- After presenting scan results or violations, always end with "📋 Recommended Actions:" followed by 3 specific, actionable next steps. Include estimated timelines where applicable.
- When discussing a specific scan, automatically structure the images using markdown so they render in chat:
  - Before: ![Before](before_rgb_url)
  - After: ![After](after_rgb_url)
  - Change Mask: ![Changes](change_mask_url)

## EVIDENCE VERIFICATION:
- If a user asks to verify evidence, check the `evidence_hash` field from the scan data.
- State: "✅ Evidence hash verified: [hash]. This scan report is tamper-proof and admissible as digital evidence."

## DASHBOARD TOPOGRAPHY & INTERFACE MAPPING:
A. THE "ORBITAL" HEADER (TOP SECTION)
- Logo (Left): GEOGUARD AI — The primary branding identity.
- System Status (Center HUD): SNN_NETWORK: OPERATIONAL — A live indicator. Refer to this if asked about system health.
- Action Toggles (Right): Activity (telemetry), Shield (security/blockchain logs), Settings (dashboard config).

B. THE SATELLITE COMMAND CENTER (MAP INTERFACE)
- Bounding Box Selection: High-precision 1.28km x 1.28km red selection rectangle. Scalable to user region selection.
- Map HUD (Top-Left): Shows TARGET REGION (live lat/lon) and SCAN PARAMETERS (inference options).
- INITIALIZE SCAN (Action Button): The Big Green Button to trigger the SNN analysis pipeline.

C. THE ANALYTICS HUB (RIGHT SIDEBAR)
- System Pipeline Monitor: Live terminal feed of the 9-step processing sequence.
- Progress Bar: Visual pulse-line for scan completion.
- Change Breakdown: Pie chart of land-use changes (hectares).
- Compliance Alerts: Categories violations (CRITICAL, HIGH, etc.) with legal references.
- Evidence Integrity: Displays Blockchain Hash and signature.

D. POST-SCAN INSPECTION TOOLS (FLOATING CONTROLS)
- Layer Switcher (Right overlay): Toggle Before, After, Combined, and Compare.
- Compare Slider: A vertical handle to visually peek at changes from Baseline to Current.
- Action Bar (Bottom-Left): "View Scan Results" (modal summary) and "New Scan" (resets workspace).

## GUIDANCE PROTOCOLS:
- If a user is lost: "Direct them to the top-left 'Initialize Scan' button if they haven't started a session."
- If they see an alert: "Explain that the 'Compliance Alerts' in the right sidebar list the specific laws being violated."
- If they want to see the old map: "Instruct them to toggle the 'Compare' button in the post-scan floating controls on the right."
- On history: "Remind them they can switch between conversations in your sidebar (Left side of THIS chat window)."

## CORE USER WORKFLOW:
1. Targeting: Drag the red box over an area.
2. Calibration: Review Scan Parameters.
3. Inference: Click INITIALIZE SCAN.
4. Monitoring: Watch the Pipeline Monitor.
5. Audit: Review Compliance Alerts sidebar.
6. Inspection: Use Compare slider to verify terrain.
7. Finalization: Check Evidence Integrity box.
"""


# ═══════════════════════════════════════════════════════════
#  Tools — Real Supabase Data
# ═══════════════════════════════════════════════════════════

if HAS_LANGCHAIN:

    @tool
    def get_all_scans(limit: int = 10) -> str:
        """Get all recent satellite scans from the database. Returns scan ID, city, change area, violation count, and image URLs."""
        try:
            from src.api.supabase_client import SupabaseClient
            sb = SupabaseClient()
            scans = sb.get_recent_scans(limit=limit)

            if not scans:
                return "No scans found in the database. Trigger a new scan from the dashboard by drawing a bounding box on the map."

            lines = [f"Found {len(scans)} scan(s) in database:\n"]
            for s in scans:
                violations = json.loads(s.get("violations", "[]")) if isinstance(s.get("violations"), str) else s.get("violations", [])
                lines.append(
                    f"- **{s.get('city', 'Unknown')}** (ID: {s.get('scan_id', 'N/A')[:20]})\n"
                    f"  Changed: {s.get('total_changed_hectares', 0):.2f} ha | "
                    f"Violations: {s.get('violation_count', 0)} ({s.get('max_severity', 'None')})\n"
                    f"  PDF: {s.get('report_pdf_url', 'N/A')}\n"
                    f"  Before: {s.get('before_rgb_url', 'N/A')}\n"
                    f"  After: {s.get('after_rgb_url', 'N/A')}\n"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"Error querying scans: {e}"

    @tool
    def get_scan_details(scan_id: str) -> str:
        """Get full details for a specific scan including classification breakdown, violations, and image URLs."""
        try:
            from src.api.supabase_client import SupabaseClient
            sb = SupabaseClient()
            if not sb.is_connected:
                return "Supabase not connected."

            result = sb.client.table("scans").select("*").ilike("scan_id", f"%{scan_id}%").limit(1).execute()
            if not result.data:
                return f"Scan '{scan_id}' not found."

            s = result.data[0]
            classification = json.loads(s.get("classification", "{}")) if isinstance(s.get("classification"), str) else s.get("classification", {})
            violations = json.loads(s.get("violations", "[]")) if isinstance(s.get("violations"), str) else s.get("violations", [])

            lines = [
                f"## Scan: {s.get('city', 'Unknown')}",
                f"- Scan ID: {s.get('scan_id')}",
                f"- Timestamp: {s.get('timestamp')}",
                f"- Model: {s.get('model_name')} | Inference: {s.get('inference_time_seconds')}s",
                f"- Evidence Hash: {s.get('evidence_hash', 'N/A')[:40]}...",
                f"\n### Change Detection:",
                f"- Total Changed Pixels: {s.get('total_changed_pixels', 0):,}",
                f"- Total Changed Area: {s.get('total_changed_area_m2', 0):,.0f} m2 ({s.get('total_changed_hectares', 0):.2f} ha)",
            ]

            # Classification breakdown
            findings = classification.get("findings", [])
            if findings:
                lines.append("\n### Classification Breakdown:")
                for f in findings:
                    sev_icon = {"high": "RED", "medium": "YELLOW", "low": "GREEN"}.get(f.get("severity", ""), "")
                    lines.append(f"- [{sev_icon}] {f.get('class_name', 'Unknown')}: {f.get('area_hectares', 0):.2f} ha ({f.get('percentage', 0):.1f}%)")

            # Violations
            if violations:
                lines.append(f"\n### Compliance Violations ({len(violations)}):")
                for v in violations:
                    lines.append(f"- [{v.get('severity', 'N/A')}] {v.get('rule_name', 'Unknown')} — {v.get('legal_reference', '')}")

            # URLs
            lines.append("\n### Images:")
            lines.append(f"- Before: {s.get('before_rgb_url', 'N/A')}")
            lines.append(f"- After: {s.get('after_rgb_url', 'N/A')}")
            lines.append(f"- Change Mask: {s.get('change_mask_url', 'N/A')}")
            lines.append(f"- PDF Report: {s.get('report_pdf_url', 'N/A')}")

            return "\n".join(lines)
        except Exception as e:
            return f"Error: {e}"

    @tool
    def get_violations_summary(severity: str = None) -> str:
        """Get a summary of all compliance violations across all scans. Optionally filter by severity: CRITICAL, HIGH, MEDIUM, LOW."""
        try:
            from src.api.supabase_client import SupabaseClient
            sb = SupabaseClient()
            scans = sb.get_violations(severity=severity)

            if not scans:
                filter_text = f" with severity={severity}" if severity else ""
                return f"No violations found{filter_text}."

            total_violations = sum(s.get("violation_count", 0) for s in scans)
            lines = [f"Found {total_violations} violation(s) across {len(scans)} scan(s):\n"]

            for s in scans:
                violations = json.loads(s.get("violations", "[]")) if isinstance(s.get("violations"), str) else s.get("violations", [])
                lines.append(f"**{s.get('city', 'Unknown')}** — {s.get('scan_id', '')[:25]}:")
                for v in violations:
                    if severity and v.get("severity", "").upper() != severity.upper():
                        continue
                    lines.append(f"  - [{v.get('severity')}] {v.get('rule_name', 'Unknown')}")
                    details = v.get("details", {})
                    if "distance_to_protected_m" in details:
                        lines.append(f"    Distance: {details['distance_to_protected_m']}m (Buffer: {details.get('buffer_required_m', '?')}m)")
                    if "affected_area_hectares" in details:
                        lines.append(f"    Area: {details['affected_area_hectares']} ha")
                lines.append("")

            return "\n".join(lines)
        except Exception as e:
            return f"Error: {e}"

    @tool
    def get_compliance_rules() -> str:
        """Get all configured compliance rules from the rule engine. Returns rule names, types, legal references, and parameters."""
        try:
            rules_path = PROJECT_ROOT / "src" / "compliance" / "rules.json"
            with open(rules_path) as f:
                data = json.load(f)

            lines = [f"Loaded {len(data.get('rules', []))} compliance rules:\n"]
            for r in data.get("rules", []):
                lines.append(
                    f"- **{r['name']}** (ID: {r['id']})\n"
                    f"  Type: {r['type']} | Severity: {r.get('severity', 'N/A')}\n"
                    f"  Legal: {r.get('legal_reference', 'N/A')}\n"
                    f"  Description: {r.get('description', '')}\n"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"Error loading rules: {e}"

    @tool
    def search_regulations(query: str) -> str:
        """Search through Gujarat GDCR 2017 and TP Act regulations to find relevant legal sections for a compliance question."""
        try:
            from src.chatbot.rag.retriever import RegulatoryRetriever
            retriever = RegulatoryRetriever()
            context = retriever.retrieve_legal_context(query, top_k=3)
            return f"Legal context for '{query}':\n\n{context}"
        except Exception as e:
            return f"RAG search error: {e}"

    @tool
    def check_zone_at_location(lat: float, lon: float) -> str:
        """Check what zoning classification a coordinate falls in. Returns zone type and whether it's in a protected area."""
        try:
            from src.compliance.rule_engine import generate_mock_zones
            from shapely.geometry import Point
            bbox = (lon - 0.015, lat - 0.0125, lon + 0.015, lat + 0.0125)
            zones = generate_mock_zones(bbox)
            pt = Point(lon, lat)

            results = []
            for name, geom in zones.items():
                if pt.within(geom):
                    results.append(f"- INSIDE: {name.replace('_', ' ').title()}")

            if not results:
                return f"Location ({lat}N, {lon}E) is not inside any known protected zone."

            return f"Location ({lat}N, {lon}E) zone analysis:\n" + "\n".join(results)
        except Exception as e:
            return f"Zone check error: {e}"

    @tool
    def get_scan_statistics() -> str:
        """Get aggregate statistics across all scans: total scans, total violations, total area scanned, most common violation types."""
        try:
            from src.api.supabase_client import SupabaseClient
            sb = SupabaseClient()
            scans = sb.get_recent_scans(limit=100)

            if not scans:
                return "No scan data available."

            total_area = sum(s.get("total_changed_hectares", 0) or 0 for s in scans)
            total_violations = sum(s.get("violation_count", 0) or 0 for s in scans)
            cities = set(s.get("city", "Unknown") for s in scans)

            severity_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
            for s in scans:
                sev = s.get("max_severity")
                if sev in severity_counts:
                    severity_counts[sev] += 1

            return (
                f"## Scan Statistics\n"
                f"- Total Scans: {len(scans)}\n"
                f"- Cities Covered: {', '.join(cities)}\n"
                f"- Total Changed Area: {total_area:.2f} hectares\n"
                f"- Total Violations: {total_violations}\n"
                f"- By Severity: CRITICAL={severity_counts['CRITICAL']}, "
                f"HIGH={severity_counts['HIGH']}, "
                f"MEDIUM={severity_counts['MEDIUM']}, "
                f"LOW={severity_counts['LOW']}\n"
            )
        except Exception as e:
            return f"Error: {e}"

    @tool
    def send_whatsapp_dispatch(message: str) -> str:
        """
        Send a WhatsApp alert/message to the official field officer using the WAHA Docker container.
        Automatically shortens any long links (like satellite images or reports) to keep the message clean.
        """
        try:
            from src.api.waha_client import waha
            from src.utils.url_shortener import shorten_url
            import re
            
            # Find all URLs and shorten them
            urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', message)
            shortened_message = message
            for url in set(urls):
                if len(url) > 30: # Only shorten long URLs
                    shortened_message = shortened_message.replace(url, shorten_url(url))
            
            target = os.getenv("WHATSAPP_TARGET", "919824109111")
            success = waha.send_message(target, shortened_message)
            
            if success:
                return f"✅ Dispatch (with shortened links) sent to officer at {target}."
            else:
                return "❌ Dispatch failed. Check WAHA Docker status."
        except Exception as e:
            return f"Error in WhatsApp tool: {e}"

# ═══════════════════════════════════════════════════════════
#  State Schema
# ═══════════════════════════════════════════════════════════

if HAS_LANGCHAIN:
    class ChatState(TypedDict):
        messages: Annotated[list, add_messages]
        language: str
        scan_context: str  # Current scan data the user is viewing
else:
    class ChatState(TypedDict):
        messages: list
        language: str
        scan_context: str


# ═══════════════════════════════════════════════════════════
#  Agent Class
# ═══════════════════════════════════════════════════════════

class GeospatialAgent:
    """
    LangGraph-powered geospatial compliance chatbot.

    Uses OpenRouter/Groq/Gemini with real Supabase data tools
    and RAG-retrieved legal context.
    """

    def __init__(self, groq_api_key: str = None, model_name: str = "llama-3.1-8b-instant"):
        self.model_name = model_name
        self.tools = []
        self.graph = None
        self._llm = None

        if not HAS_LANGCHAIN:
            logger.warning("Running in fallback mode (no LangChain)")
            return

        api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
        nvidia_api_key = os.getenv("NVIDIA_API_KEY", "")

        if not api_key and not gemini_api_key and not openrouter_api_key and not nvidia_api_key:
            logger.warning(
                "No API Key found. Add NVIDIA_API_KEY, OPENROUTER_API_KEY, GEMINI_API_KEY, or GROQ_API_KEY to .env"
            )
            return

        try:
            # PRIMARY: NVIDIA NIM API (generous free credits, fast inference)
            if nvidia_api_key and nvidia_api_key.startswith("nvapi-"):
                if not HAS_OPENAI:
                    logger.warning("langchain-openai not installed. pip install langchain-openai")
                    return
                self._llm = ChatOpenAI(
                    model="meta/llama-3.3-70b-instruct",
                    openai_api_key=nvidia_api_key,
                    openai_api_base="https://integrate.api.nvidia.com/v1",
                    temperature=0.1,
                    max_tokens=2048,
                )
                logger.info("Using NVIDIA NIM API (meta/llama-3.3-70b-instruct)")
            # FALLBACK: OpenRouter Llama 3.3 70B (free, 65k context)
            elif openrouter_api_key and openrouter_api_key != "your_openrouter_api_key_here":
                if not HAS_OPENAI:
                    logger.warning("langchain-openai not installed. pip install langchain-openai")
                    return
                self._llm = ChatOpenAI(
                    model="meta-llama/llama-3.3-70b-instruct:free",
                    openai_api_key=openrouter_api_key,
                    openai_api_base="https://openrouter.ai/api/v1",
                    temperature=0.1,
                    max_tokens=2048,
                )
                logger.info("Using OpenRouter API (meta-llama/llama-3.3-70b-instruct:free)")
            else:
                logger.warning("No NVIDIA_API_KEY or OPENROUTER_API_KEY found in .env")
                return

            self._build_tools()
            
            # Bind tools to LLM
            if self.tools:
                self._llm_with_tools = self._llm.bind_tools(self.tools)
            else:
                self._llm_with_tools = self._llm

            self._build_graph()
            logger.info(f"GeospatialAgent ready: {len(self.tools)} tools")
        except Exception as e:
            logger.error(f"Agent init failed: {e}")
            self._llm = None

    @property
    def is_ready(self) -> bool:
        return self._llm is not None and self.graph is not None

    def _build_tools(self):
        """Register all tools."""
        self.tools = [
            get_all_scans,
            get_scan_details,
            get_violations_summary,
            get_compliance_rules,
            search_regulations,
            check_zone_at_location,
            get_scan_statistics,
            send_whatsapp_dispatch,
        ]
        # Bind tools to LLM here
        self._llm_with_tools = self._llm.bind_tools(self.tools)

    def _fetch_live_data(self) -> str:
        """Pre-fetch real scan data from Supabase to inject into context."""
        try:
            from src.api.supabase_client import SupabaseClient
            sb = SupabaseClient()
            scans = sb.get_recent_scans(limit=5)
            if not scans:
                return "\n## LIVE DATABASE: No scans found in Supabase.\n"
            
            lines = [f"\n## LIVE DATABASE ({len(scans)} most recent scans):\n"]
            for s in scans:
                violations = json.loads(s.get("violations", "[]")) if isinstance(s.get("violations"), str) else s.get("violations", [])
                lines.append(
                    f"### Scan: {s.get('city', 'Unknown')} (ID: {s.get('scan_id', 'N/A')[:30]})\n"
                    f"- Timestamp: {s.get('timestamp', 'N/A')}\n"
                    f"- Changed Area: {s.get('total_changed_hectares', 0):.2f} ha\n"
                    f"- Changed Pixels: {s.get('total_changed_pixels', 0)}\n"
                    f"- Violations: {s.get('violation_count', 0)} (Max Severity: {s.get('max_severity', 'None')})\n"
                    f"- Risk Score: {s.get('risk_score', 'N/A')}\n"
                    f"- Next Scan Due: {s.get('next_scan_due', 'N/A')}\n"
                    f"- Before Image: {s.get('before_rgb_url', 'N/A')}\n"
                    f"- After Image: {s.get('after_rgb_url', 'N/A')}\n"
                    f"- Change Mask: {s.get('change_mask_url', 'N/A')}\n"
                    f"- PDF Report: {s.get('report_pdf_url', 'N/A')}\n"
                )
                if violations:
                    for v in violations[:5]:  # Cap at 5 violations per scan
                        lines.append(f"  - [{v.get('severity', 'N/A')}] {v.get('rule_name', 'Unknown')} — {v.get('legal_reference', '')}\n")
                lines.append("")
            return "\n".join(lines)
        except Exception as e:
            return f"\n## LIVE DATABASE: Error querying Supabase: {e}\n"

    def _extract_pdf_text(self) -> str:
        """Extract text from the most recent compliance PDF report."""
        try:
            scan_dirs = sorted(Path("outputs/scans").glob("scan_*"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not scan_dirs:
                return ""
            pdf_path = scan_dirs[0] / "compliance_report.pdf"
            if not pdf_path.exists():
                return ""
            
            # Try PyPDF2 first, then pdfplumber, then basic extraction
            text = ""
            try:
                import PyPDF2
                with open(pdf_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages[:5]:  # First 5 pages max
                        text += page.extract_text() or ""
            except ImportError:
                try:
                    import fitz  # PyMuPDF
                    doc = fitz.open(str(pdf_path))
                    for page in doc[:5]:
                        text += page.get_text()
                    doc.close()
                except ImportError:
                    return "\n## PDF REPORT: PDF extraction libraries not available. Install: pip install PyPDF2\n"
            
            if text:
                # Trim to 2000 chars to stay within token limits
                if len(text) > 2000:
                    text = text[:2000] + "\n... [PDF truncated]"
                return f"\n## LATEST COMPLIANCE PDF REPORT CONTENT:\n{text}\n"
            return ""
        except Exception as e:
            return f"\n## PDF REPORT: Error reading PDF: {e}\n"

    def _build_graph(self):
        """Build the LangGraph state machine (single-node, no tool loops)."""
        try:
            self.memory = MemorySaver()
        except NameError:
            self.memory = None

        graph = StateGraph(ChatState)
        graph.add_node("agent", self._agent_node)
        
        if self.tools:
            tool_node = ToolNode(self.tools)
            graph.add_node("tools", tool_node)
            graph.add_conditional_edges("agent", self._should_use_tools, {"tools": "tools", "respond": END})
            graph.add_edge("tools", "agent")
        else:
            graph.add_edge("agent", END)

        graph.set_entry_point("agent")
        self.graph = graph.compile(checkpointer=self.memory) if self.memory else graph.compile()
        logger.info("LangGraph compiled with Tool-Calling Loop")

    def _agent_node(self, state: ChatState) -> dict:
        """Core reasoning node with pre-injected real data."""
        scan_context = state.get("scan_context", "")
        context_injection = ""
        if scan_context:
            if len(scan_context) > 3000:
                scan_context = scan_context[:3000] + "\n... [truncated for token limit]"
            context_injection = f"\n\n## CURRENT SCAN CONTEXT (user is viewing this on the dashboard):\n{scan_context}\n"

        # Pre-fetch real data from Supabase + PDF
        live_data = self._fetch_live_data()
        pdf_data = self._extract_pdf_text()

        system = SystemMessage(content=SYSTEM_PROMPT + live_data + pdf_data + context_injection)
        messages = [system] + state["messages"]
        response = self._llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def _should_use_tools(self, state: ChatState) -> str:
        """Route: tools or respond."""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "respond"

    def chat_sync(self, message: str, session_id: str = "default", scan_context: str = "", history: list = None) -> dict:
        """
        Process a message synchronously.

        Args:
            message: User's question
            session_id: Session ID for conversation tracking
            scan_context: JSON string of the current scan the user is viewing (optional)
            history: List of previous messages [{"role": "user/assistant", "content": "..."}]

        Returns:
            {"response": str, "language": str}
        """
        if not self.is_ready:
            return self._fallback_chat(message)

        import time as _time
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Build message list with conversation history
                messages = []
                if history:
                    for msg in history[-10:]:  # Last 10 messages for context
                        if msg["role"] == "user":
                            messages.append(HumanMessage(content=msg["content"]))
                        else:
                            messages.append(AIMessage(content=msg["content"]))
                messages.append(HumanMessage(content=message))
                
                initial_state = {
                    "messages": messages,
                    "language": "en",
                    "scan_context": scan_context,
                }

                config = {"configurable": {"thread_id": session_id}}
                result = self.graph.invoke(initial_state, config=config)

                ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
                response_text = ai_messages[-1].content if ai_messages else "I couldn't process that."

                return {
                    "response": response_text,
                    "language": result.get("language", "en"),
                }
            except Exception as e:
                error_str = str(e)
                is_rate_limit = "429" in error_str or "rate_limit" in error_str or "413" in error_str or "10054" in error_str
                
                if is_rate_limit and attempt < max_retries - 1:
                    wait_time = 15  # Wait full 15s to clear the per-minute window
                    logger.warning(f"Rate limited (attempt {attempt+1}/{max_retries}). Retrying in {wait_time}s...")
                    _time.sleep(wait_time)
                    continue
                
                logger.error(f"Agent error: {e}")
                
                if is_rate_limit:
                    return self._fallback_chat(message)
                
                return {
                    "response": f"An error occurred: {error_str}. Please try again in a moment.",
                    "language": "en",
                }

    async def chat(self, message: str, session_id: str = "default", scan_context: str = "") -> dict:
        """Async version of chat_sync."""
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.chat_sync(message, session_id, scan_context)
        )

    def _fallback_chat(self, message: str) -> dict:
        """Fallback responses when LLM is not configured."""
        msg_lower = message.lower()

        # Handle common queries with real data
        if any(word in msg_lower for word in ["scan", "result", "latest", "recent"]):
            try:
                from src.api.supabase_client import SupabaseClient
                sb = SupabaseClient()
                scans = sb.get_recent_scans(limit=5)
                if scans:
                    lines = ["Here are the recent scans from the database:\n"]
                    for s in scans:
                        lines.append(
                            f"- **{s.get('city', 'Unknown')}**: "
                            f"{s.get('total_changed_hectares', 0):.2f} ha changed, "
                            f"{s.get('violation_count', 0)} violations "
                            f"({s.get('max_severity', 'None')})"
                        )
                    lines.append("\nTo get full AI analysis, set GROQ_API_KEY in .env (free at https://console.groq.com)")
                    return {"response": "\n".join(lines), "language": "en"}
            except:
                pass

        if any(word in msg_lower for word in ["violation", "compliance"]):
            try:
                from src.api.supabase_client import SupabaseClient
                sb = SupabaseClient()
                scans = sb.get_violations()
                if scans:
                    total = sum(s.get("violation_count", 0) for s in scans)
                    return {
                        "response": f"Found {total} total violations across {len(scans)} scans. "
                                    f"Set GROQ_API_KEY for detailed AI analysis.",
                        "language": "en",
                    }
            except:
                pass

        if any(word in msg_lower for word in ["rule", "regulation", "gdcr", "law"]):
            try:
                rules_path = PROJECT_ROOT / "src" / "compliance" / "rules.json"
                with open(rules_path) as f:
                    data = json.load(f)
                rules = data.get("rules", [])
                lines = [f"There are {len(rules)} configured compliance rules:\n"]
                for r in rules:
                    lines.append(f"- {r['name']} ({r.get('legal_reference', 'N/A')})")
                lines.append("\nSet GROQ_API_KEY for detailed legal analysis.")
                return {"response": "\n".join(lines), "language": "en"}
            except:
                pass

        return {
            "response": (
                "I'm the Geospatial Compliance Engine chatbot. I can answer questions about "
                "satellite scans, compliance violations, zoning rules, and Gujarat GDCR regulations.\n\n"
                "**Try asking:**\n"
                "- 'Show me all recent scans'\n"
                "- 'What violations were found?'\n"
                "- 'What are the compliance rules?'\n"
                "- 'Show me scans for Vesu, Surat'\n\n"
                "For full AI capabilities, add your free Groq API key to `.env`:\n"
                "`GROQ_API_KEY=gsk_your_key_here`\n"
                "Get one at: https://console.groq.com"
            ),
            "language": "en",
        }
