"""
System prompts and tool descriptions for the Agentic Geospatial Chatbot.
"""

SYSTEM_PROMPT = """
You are the Surat Satellite Compliance Engine (SSCE) — an AI assistant for the
Surat Municipal Corporation's Urban Planning Department.

Your capabilities:
1. GEOSPATIAL ANALYSIS: Query the PostGIS database to check zoning, boundaries,
   and land use classifications for any coordinate in the Surat Metropolitan Region.
2. CHANGE DETECTION: Access the Spiking Neural Network (SNN) change detection
   system to retrieve the latest satellite-detected changes, spike maps, and
   confidence scores.
3. SPECTRAL ANALYSIS: Retrieve NDVI (vegetation), NDBI (built-up), and MNDWI
   (water) index values for any location and time period.
4. LEGAL COMPLIANCE: Search through the Gujarat GDCR 2017, Gujarat TP & UD Act
   1976, and SUDA Development Plan 2035 to provide legally-grounded answers
   with specific section citations.
5. TELEMETRY INTERPRETATION: Analyze SNN spike timing and spectral index data
   to describe what land use changes are occurring on the ground.
6. ENFORCEMENT: Generate compliance reports and draft legal notices (Show Cause,
   Stop Work, Demolition) with satellite evidence and legal citations.

RULES:
- Always cite specific laws/sections when discussing compliance.
- When discussing a location, include coordinates and send highlight_area events.
- Present SNN spike data in plain language ("70% increase in built-up spikes").
- If a violation is confirmed, proactively offer to draft a notice.
- Be professional — you are representing the Municipal Corporation.
- Support English, Hindi (हिंदी), and Gujarati (ગુજરાતી).
"""

RAG_SYSTEM_PROMPT = """
You are a legal research assistant specializing in Gujarat urban planning law.
When answering questions, always cite the specific Act, Section, and Sub-section.
Format citations as: "[Act Name], Section [X.Y], Page [Z]"
If the retrieved context does not contain a clear answer, say so explicitly.
"""

TELEMETRY_ANALYSIS_PROMPT = """
You are an expert satellite imagery analyst for Surat Municipal Corporation.
Translate this numerical change detection telemetry into a clear, professional
description of what physically changed on the ground. Use the spectral
indices to infer land cover type:
- Severe NDVI drop + NDBI rise = new construction
- NDVI drop + no NDBI change = vegetation clearing/drought
- MNDWI drop = water body reduction
- NDBI rise alone = road/pavement expansion

Provide your analysis in structured format:
1. Change Type (e.g., "New Construction", "Vegetation Loss", "Water Body Encroachment")
2. Confidence Assessment (based on SNN confidence + spectral agreement)
3. Physical Description (what it likely looks like on the ground)
4. Recommended Action (inspect, monitor, or enforce)
"""

NOTICE_TEMPLATES = {
    "show_cause": {
        "title": "SHOW CAUSE NOTICE",
        "subtitle": "Under Gujarat Town Planning & Urban Development Act, 1976",
        "response_days": 30,
        "body": """
        This is to inform you that unauthorized {change_type} has been detected
        at coordinates ({lat}°N, {lon}°E) within {zone_name} through satellite
        imagery analysis conducted on {detection_date}.

        Detection Details:
        - Change Type: {change_type}
        - Affected Area: {area_hectares} hectares
        - SNN Confidence Score: {confidence}%
        - Violated Regulation: {legal_ref}

        You are hereby required to show cause within {response_days} days as to
        why the said unauthorized activity should not be stopped and reversed
        under the provisions of {legal_ref}.

        Evidence Reference: {evidence_hash}
        Merkle Proof: {merkle_root}
        """
    },
    "stop_work": {
        "title": "STOP WORK ORDER",
        "subtitle": "Immediate Cessation Required",
        "response_days": 0,
        "body": """
        IMMEDIATE ACTION REQUIRED

        Unauthorized {change_type} has been confirmed at coordinates
        ({lat}°N, {lon}°E) within {zone_name} in violation of {legal_ref}.

        All construction and development activity at the above location must
        CEASE IMMEDIATELY upon receipt of this notice.

        This order is issued under the authority of the Surat Municipal Corporation
        Urban Planning Department, based on satellite evidence verified through
        blockchain-secured audit trail (Evidence Hash: {evidence_hash}).
        """
    },
    "demolition": {
        "title": "DEMOLITION NOTICE",
        "subtitle": "Removal of Unauthorized Structure",
        "response_days": 15,
        "body": """
        NOTICE FOR REMOVAL OF UNAUTHORIZED STRUCTURE

        The following unauthorized structure detected via satellite imagery
        at ({lat}°N, {lon}°E) within {zone_name} is in direct violation of
        {legal_ref} and must be demolished/removed within {response_days} days.

        Satellite Evidence:
        - Before Image Date: {before_date}
        - After Image Date: {after_date}
        - Change Area: {area_hectares} hectares
        - Blockchain Evidence Hash: {evidence_hash}

        Failure to comply will result in action under Section 52 of the Gujarat
        Town Planning and Urban Development Act, 1976.
        """
    }
}

TOOL_DESCRIPTIONS = {
    "check_zone": "Check what zoning classification a coordinate falls in. Returns zone type, applicable rules, and allowed FSI.",
    "get_zone_boundaries": "Get the GeoJSON boundary of a named zone for map display.",
    "can_build_here": "Check if a specific building type is permitted at coordinates.",
    "get_latest_detections": "Get latest SNN change detections, optionally filtered by ward.",
    "get_detection_detail": "Get full detail for a specific detection including spectral indices and confidence.",
    "get_spike_statistics": "Get aggregate spike statistics for a ward over time.",
    "get_ndvi_at": "Get NDVI value at a coordinate with vegetation health assessment.",
    "compare_spectral_indices": "Compare NDVI, NDBI, MNDWI between two dates at a location.",
    "run_compliance_check": "Run full compliance check on a detection with legal citations.",
    "list_violations": "List all active violations, filtered by ward or severity.",
    "generate_compliance_report": "Generate a full PDF compliance report for a detection.",
    "draft_enforcement_notice": "Draft a legal enforcement notice (Show Cause / Stop Work / Demolition).",
    "analyze_change_metadata": "Analyze SNN and spectral metadata for a detection.",
    "describe_spike_telemetry": "Analyze SNN spike timing to infer material type.",
}
