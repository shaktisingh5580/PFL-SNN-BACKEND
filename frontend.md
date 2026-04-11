# 🛰️ Agentic Geospatial Compliance Dashboard — Frontend Blueprint

> **For:** Frontend Developer  
> **Stack:** Next.js 14 (App Router) + React-Leaflet + Vanilla CSS  
> **Design:** Futuristic Dark Command Center with Glassmorphism  
> **Map:** Leaflet with CartoDB Dark Matter tiles (FREE, no API key needed)

---

## 📋 Table of Contents

1. [Project Setup](#1-project-setup)
2. [Design System & Theme](#2-design-system--theme)
3. [Layout Architecture](#3-layout-architecture)
4. [Page Structure](#4-page-structure)
5. [Component Breakdown](#5-component-breakdown)
6. [API Contract (Backend ↔ Frontend)](#6-api-contract)
7. [Map Integration Guide](#7-map-integration-guide)
8. [Animations & Micro-interactions](#8-animations--micro-interactions)
9. [Assets & Resources](#9-assets--resources)
10. [Implementation Checklist](#10-implementation-checklist)

---

## 1. Project Setup

### Initialize

```bash
npx -y create-next-app@latest geo-compliance --app --src-dir --no-tailwind --eslint --no-turbopack
cd geo-compliance
npm install leaflet react-leaflet lucide-react recharts framer-motion
npm install -D @types/leaflet
```

### Folder Structure

```
geo-compliance/
├── src/
│   ├── app/
│   │   ├── layout.js          # Root layout with dark theme
│   │   ├── page.js            # Main dashboard (single-page app)
│   │   ├── globals.css        # Design system tokens + global styles
│   │   └── api/
│   │       ├── scan/route.js        # POST — triggers SNN inference
│   │       ├── report/route.js      # GET  — returns classification report
│   │       └── agent/route.js       # POST — streams agent chat response
│   ├── components/
│   │   ├── layout/
│   │   │   ├── Sidebar.js           # Left control panel
│   │   │   ├── MapView.js           # Center map stage
│   │   │   └── AnalyticsPanel.js    # Right analytics sidebar
│   │   ├── map/
│   │   │   ├── SatelliteMap.js      # Leaflet map wrapper
│   │   │   ├── ChangeOverlay.js     # Classification overlay on map
│   │   │   ├── BeforeAfterSlider.js # Image comparison slider
│   │   │   └── RegionSelector.js    # Draw ROI on map
│   │   ├── agent/
│   │   │   ├── ChatWindow.js        # AI agent chat interface
│   │   │   ├── ChatMessage.js       # Single message bubble
│   │   │   └── TypingIndicator.js   # Animated typing dots
│   │   ├── analytics/
│   │   │   ├── ClassificationDonut.js  # Change type breakdown chart
│   │   │   ├── SeverityAlerts.js       # HIGH/MEDIUM/LOW alert cards
│   │   │   ├── AreaStats.js            # Area in hectares display
│   │   │   └── BlockchainHash.js       # Audit trail hash display
│   │   ├── controls/
│   │   │   ├── ScanButton.js        # Main "INITIALIZE SCAN" CTA
│   │   │   ├── ImageUploader.js     # Before/After image upload
│   │   │   ├── CitySelector.js      # Dropdown to pick target city
│   │   │   └── BandSelector.js      # Toggle satellite bands
│   │   └── ui/
│   │       ├── GlassCard.js         # Reusable glassmorphism container
│   │       ├── GlowButton.js        # Animated glowing button
│   │       ├── ProgressRing.js      # Circular progress indicator
│   │       └── StatusBadge.js       # Online/Processing/Error badge
│   ├── hooks/
│   │   ├── useAgent.js              # WebSocket/SSE hook for agent chat
│   │   ├── useScan.js               # Scan trigger + polling hook
│   │   └── useMapBounds.js          # Track current map viewport
│   ├── lib/
│   │   ├── api.js                   # Fetch wrapper for backend calls
│   │   └── constants.js             # Colors, class names, tile URLs
│   └── styles/
│       ├── sidebar.module.css
│       ├── map.module.css
│       ├── analytics.module.css
│       └── agent.module.css
```

---

## 2. Design System & Theme

### Color Palette

```css
/* globals.css */
:root {
  /* ── Base (Dark Navy) ── */
  --bg-primary:     #0B1120;
  --bg-secondary:   #111827;
  --bg-card:        rgba(15, 23, 42, 0.75);
  --bg-card-hover:  rgba(30, 41, 59, 0.85);

  /* ── Text ── */
  --text-primary:   #F1F5F9;
  --text-secondary: #94A3B8;
  --text-muted:     #64748B;

  /* ── Accent (Cyan/Teal Glow) ── */
  --accent:         #06B6D4;
  --accent-glow:    rgba(6, 182, 212, 0.4);
  --accent-dim:     #0E7490;

  /* ── Change Type Colors ── */
  --color-construction:  #EF4444;  /* Red */
  --color-vegetation:    #22C55E;  /* Green */
  --color-water:         #3B82F6;  /* Blue */
  --color-other:         #EAB308;  /* Yellow */

  /* ── Severity ── */
  --severity-high:    #EF4444;
  --severity-medium:  #F59E0B;
  --severity-low:     #22C55E;

  /* ── Borders & Surfaces ── */
  --border:         rgba(148, 163, 184, 0.12);
  --border-glow:    rgba(6, 182, 212, 0.3);
  --glass-blur:     16px;

  /* ── Typography ── */
  --font-main:  'Inter', 'SF Pro', -apple-system, sans-serif;
  --font-mono:  'JetBrains Mono', 'Fira Code', monospace;

  /* ── Spacing ── */
  --sidebar-width:    340px;
  --analytics-width:  300px;
  --header-height:    56px;
  --radius:           12px;
  --radius-sm:        8px;
}
```

### Google Fonts (add to `layout.js`)

```js
import { Inter, JetBrains_Mono } from 'next/font/google'

const inter = Inter({ subsets: ['latin'], variable: '--font-main' })
const jetbrains = JetBrains_Mono({ subsets: ['latin'], variable: '--font-mono' })
```

### Glassmorphism Utility

```css
.glass {
  background: var(--bg-card);
  backdrop-filter: blur(var(--glass-blur));
  -webkit-backdrop-filter: blur(var(--glass-blur));
  border: 1px solid var(--border);
  border-radius: var(--radius);
}

.glass:hover {
  background: var(--bg-card-hover);
  border-color: var(--border-glow);
  box-shadow: 0 0 20px var(--accent-glow);
}
```

---

## 3. Layout Architecture

The dashboard is a **single-page, 3-column layout** that fills the entire viewport. The map is the "hero" behind everything.

```
┌──────────────────────────────────────────────────────────────────────┐
│  HEADER BAR (56px) — Logo | "GeoGuard AI" | Status | Settings      │
├──────────┬──────────────────────────────────────┬───────────────────┤
│          │                                      │                   │
│  LEFT    │         CENTER MAP STAGE             │  RIGHT            │
│  SIDEBAR │                                      │  ANALYTICS        │
│  (340px) │   ┌──────────────────────────┐       │  PANEL            │
│          │   │     LEAFLET MAP          │       │  (300px)          │
│  Controls│   │                          │       │                   │
│  Upload  │   │  Before ◄──┼──► After    │       │  Donut Chart      │
│  City    │   │       (Slider)           │       │  Severity Alerts  │
│  Scan    │   │                          │       │  Area Stats       │
│          │   │  [Classification         │       │  Blockchain Hash  │
│ ──────── │   │   Overlay on Map]        │       │                   │
│  AGENT   │   │                          │       │  ──────────────   │
│  CHAT    │   └──────────────────────────┘       │  Timeline         │
│  WINDOW  │                                      │  (scan history)   │
│          │                                      │                   │
│          │                                      │                   │
└──────────┴──────────────────────────────────────┴───────────────────┘
```

### CSS Grid Implementation

```css
/* page.module.css */
.dashboard {
  display: grid;
  grid-template-columns: var(--sidebar-width) 1fr var(--analytics-width);
  grid-template-rows: var(--header-height) 1fr;
  grid-template-areas:
    "header  header    header"
    "sidebar map       analytics";
  height: 100vh;
  overflow: hidden;
  background: var(--bg-primary);
}

.header    { grid-area: header; }
.sidebar   { grid-area: sidebar; }
.map       { grid-area: map; }
.analytics { grid-area: analytics; }

/* Mobile: stack vertically */
@media (max-width: 1024px) {
  .dashboard {
    grid-template-columns: 1fr;
    grid-template-rows: var(--header-height) 50vh auto auto;
    grid-template-areas:
      "header"
      "map"
      "sidebar"
      "analytics";
  }
}
```

---

## 4. Page Structure

### `app/page.js` (Main Dashboard)

```jsx
'use client'
import { useState } from 'react'
import Header from '@/components/layout/Header'
import Sidebar from '@/components/layout/Sidebar'
import MapView from '@/components/layout/MapView'
import AnalyticsPanel from '@/components/layout/AnalyticsPanel'
import styles from './page.module.css'

export default function Dashboard() {
  const [scanResult, setScanResult] = useState(null)
  const [isScanning, setIsScanning] = useState(false)
  const [selectedCity, setSelectedCity] = useState('abudhabi')
  const [agentMessages, setAgentMessages] = useState([])

  const handleScan = async () => {
    setIsScanning(true)
    // POST to /api/scan → triggers Python backend
    const res = await fetch('/api/scan', {
      method: 'POST',
      body: JSON.stringify({ city: selectedCity }),
    })
    const data = await res.json()
    setScanResult(data)
    setIsScanning(false)
  }

  return (
    <div className={styles.dashboard}>
      <Header />
      <Sidebar
        selectedCity={selectedCity}
        onCityChange={setSelectedCity}
        onScan={handleScan}
        isScanning={isScanning}
        agentMessages={agentMessages}
      />
      <MapView
        scanResult={scanResult}
        selectedCity={selectedCity}
      />
      <AnalyticsPanel
        scanResult={scanResult}
      />
    </div>
  )
}
```

---

## 5. Component Breakdown

### 5.1 Header Bar

A thin, elegant top bar with a glowing logo and system status.

```
┌─────────────────────────────────────────────────────────────┐
│  🛰️ GeoGuard AI          ● System Online    ⚙ Settings    │
└─────────────────────────────────────────────────────────────┘
```

Key elements:
- **Logo**: Satellite emoji + "GeoGuard AI" in bold Inter font
- **Status Badge**: Pulsing green dot + "System Online" / "Scanning..." / "Error"
- **Settings Icon**: Opens a modal for API key config, theme toggle

---

### 5.2 Left Sidebar — Controls + Agent Chat

Split into two sections vertically:

**Top Section: Controls (40% of sidebar height)**

```
┌─────────────────────────┐
│  📍 Target Region       │
│  ┌───────────────────┐  │
│  │ Abu Dhabi      ▼  │  │  ← City Dropdown
│  └───────────────────┘  │
│                         │
│  📤 Upload Imagery      │
│  ┌───────────────────┐  │
│  │  Before Image  📎 │  │  ← Drag & drop zone
│  ├───────────────────┤  │
│  │  After Image   📎 │  │
│  └───────────────────┘  │
│                         │
│  ┌───────────────────┐  │
│  │ ⚡ INITIALIZE SCAN │  │  ← Glowing CTA button
│  └───────────────────┘  │
│                         │
│  ── Scan Parameters ──  │
│  Patch Size: 128        │
│  SNN Steps:  10         │
│  Model: Siamese-SNN v3  │
└─────────────────────────┘
```

**Bottom Section: AI Agent Chat (60% of sidebar height)**

```
┌─────────────────────────┐
│  🤖 Compliance Agent    │
│ ─────────────────────── │
│                         │
│  ┌─── Agent ────────┐   │
│  │ Analyzing sector  │   │
│  │ 4... Detected 1.1 │   │
│  │ hectares of new   │   │
│  │ construction in   │   │
│  │ restricted zone.  │   │
│  │ ⚠️ HIGH SEVERITY  │   │
│  └───────────────────┘   │
│                         │
│  ┌─── You ──────────┐   │
│  │ What zoning rules │   │
│  │ does this violate?│   │
│  └───────────────────┘   │
│                         │
│  ┌───────────────────┐  │
│  │  Ask about this   │  │  ← Input field
│  │  scan...     ➤    │  │
│  └───────────────────┘  │
└─────────────────────────┘
```

**Chat Message Styling:**
- Agent messages: Left-aligned, dark glass card, cyan accent border-left
- User messages: Right-aligned, slightly brighter glass card
- **Typing effect**: Agent text appears character-by-character (like ChatGPT) using `setInterval` with 20ms delay per character

---

### 5.3 Center — Interactive Map

This is the hero of the dashboard. The Leaflet map fills the entire center area.

**Layers (bottom to top):**
1. **Base Layer**: CartoDB Dark Matter tiles
2. **Before Image**: `ImageOverlay` — satellite imagery at T1
3. **After Image**: `ImageOverlay` — satellite imagery at T2
4. **Classification Overlay**: `ImageOverlay` — the RGBA color map from our classifier (semi-transparent)
5. **Interactive Markers**: `Marker` or `CircleMarker` at the centroid of each detected change cluster

**Before/After Slider:**
Use a CSS `clip-path` approach:
- The "After" image layer is clipped by a vertical line
- A draggable handle lets the user slide left/right
- Left side shows "Before", right side shows "After"
- The classification overlay only shows on the "After" side

**Map Controls:**
- Layer toggle button (top-right): Show/hide classification overlay
- Opacity slider: Adjust overlay transparency
- Zoom to fit: Auto-zoom to the scanned region

**Click Interaction:**
When a user clicks inside a colored change zone on the map:
```
┌─────────────────────────┐
│ 🔴 New Construction     │
│ Area: 0.45 hectares     │
│ Severity: HIGH          │
│ Zone: Residential-B     │
│ Status: ⚠️ VIOLATION     │
│                         │
│ [View Evidence] [Flag]  │
└─────────────────────────┘
```

---

### 5.4 Right — Analytics Panel

A scrollable panel with glassmorphism cards stacked vertically:

**Card 1: Classification Breakdown (Donut Chart)**
```
┌─────────────────────────┐
│  📊 Change Breakdown    │
│                         │
│      ┌───────┐          │
│     /  25.7%  \  🔴     │
│    | Construct |        │
│     \  60.5%  /  🟢     │
│      └───────┘          │
│       11.7% 🔵  2.1% 🟡│
│                         │
│  🔴 Construction  110px │
│  🟢 Vegetation    259px │
│  🔵 Water Change   50px │
│  🟡 Other           9px │
└─────────────────────────┘
```
Use `recharts` `<PieChart>` with custom colors matching our `COLORS` dict.

**Card 2: Severity Alerts**
```
┌─────────────────────────┐
│  🚨 Compliance Alerts   │
│                         │
│  ┌── HIGH ────────────┐ │
│  │ 🔴 1.1 ha illegal  │ │
│  │    construction     │ │
│  │    Zone: Green Belt │ │
│  └────────────────────┘ │
│                         │
│  ┌── MEDIUM ──────────┐ │
│  │ 🟢 2.6 ha veg loss │ │
│  │    near wetland     │ │
│  └────────────────────┘ │
└─────────────────────────┘
```
Use colored left-border (4px solid) to indicate severity.

**Card 3: Area Statistics**
```
┌─────────────────────────┐
│  📐 Scan Statistics     │
│                         │
│  Total Scanned    12 ha │
│  Total Changed   4.3 ha │
│  Change Rate     35.8%  │
│  Patches Analyzed  132  │
│  Inference Time   50.1s │
│  Throughput    3,593/hr  │
└─────────────────────────┘
```

**Card 4: Blockchain Audit Trail**
```
┌─────────────────────────┐
│  🔗 Evidence Integrity  │
│                         │
│  SHA-256 Hash:          │
│  ┌────────────────────┐ │
│  │ 0x7f3a...b2c1     │ │  ← Monospace, truncated, click to copy
│  └────────────────────┘ │
│  Timestamp:             │
│  2026-04-10 15:42:00    │
│  Status: ✅ Verified     │
│                         │
│  [📋 Copy] [📥 Export]  │
└─────────────────────────┘
```

**Card 5: Scan Timeline (History)**
```
┌─────────────────────────┐
│  📅 Recent Scans        │
│                         │
│  • 15:42  Abu Dhabi     │
│    F1=0.42  3 alerts    │
│  • 14:10  Mumbai        │
│    F1=0.38  1 alert     │
│  • 12:05  Beirut        │
│    F1=0.41  2 alerts    │
└─────────────────────────┘
```

---

## 6. API Contract (Backend - Frontend)

The Python backend (FastAPI) exposes these endpoints. The backend runs on your laptop at `http://<YOUR_IP>:8000`. Start it with `python start_server.py`.

### `POST /api/scan/stream` - Live Scan with Real-Time Progress (SSE)

This is the **primary endpoint**. The user draws a rectangle on the map. The frontend sends the bounding box coordinates. The backend streams progress events in real-time.

**Request:**
```json
{
  "bbox": [72.7750, 21.1450, 72.8050, 21.1700],
  "city": "Vesu, Surat",
  "date_before": ["2025-01-01", "2025-03-31"],
  "date_after": ["2025-10-01", "2026-03-31"],
  "resolution": 10
}
```

**Response: Server-Sent Events (SSE) stream**

Each event is a JSON object with these fields:
```
step          - Current step number (1-9)
total_steps   - Always 9
status        - "running" | "complete" | "error" | "finished"
message       - Human-readable status text for the loader
progress      - Integer 0-100 (for progress bar)
data          - Step-specific data (optional)
```

**Sample Event Stream:**
```
data: {"step":1,"status":"running","message":"Authenticating with Copernicus Data Space (CDSE)...","progress":5}

data: {"step":1,"status":"complete","message":"Authenticated. Image size: 315x272 px at 10m","progress":10,"data":{"image_size":[315,272],"coverage_km":[3.1,2.7]}}

data: {"step":2,"status":"running","message":"Downloading BEFORE image (2025-01-01 to 2025-03-31)...","progress":15}

data: {"step":2,"status":"running","message":"BEFORE downloaded (6.8s). Downloading AFTER image...","progress":25}

data: {"step":4,"status":"running","message":"Loading Siamese-SNN model...","progress":40}

data: {"step":4,"status":"complete","message":"SNN detected 401 changed pixels (0.5%) in 2.6s","progress":55,"data":{"changed_pixels":401,"change_percent":0.47,"inference_time":2.6}}

data: {"step":6,"status":"complete","message":"3 compliance violation(s) detected","progress":73,"data":{"violation_count":3,"violations":["[CRITICAL] Water Body Buffer Zone","[HIGH] Green Belt Protection"]}}

data: {"step":9,"status":"finished","message":"Scan complete: Vesu, Surat","progress":100,"data":{"scan_id":"scan_vesu_surat_20260410_181228","change_hectares":3.60,"violation_count":3,"pdf_url":"https://...supabase.co/.../compliance_report.pdf","before_url":"https://...","after_url":"https://...","overlay_url":"https://...","report":{...}}}
```

**Frontend Usage (React hook):**
```jsx
// hooks/useScanStream.js
'use client'
import { useState, useCallback } from 'react'

export function useScanStream() {
  const [events, setEvents] = useState([])
  const [progress, setProgress] = useState(0)
  const [status, setStatus] = useState('idle')  // idle | scanning | complete | error
  const [result, setResult] = useState(null)

  const startScan = useCallback(async (bbox, city, dateBefore, dateAfter) => {
    setEvents([])
    setProgress(0)
    setStatus('scanning')
    setResult(null)

    const response = await fetch('http://<BACKEND_IP>:8000/api/scan/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        bbox,
        city,
        date_before: dateBefore,
        date_after: dateAfter,
      }),
    })

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const event = JSON.parse(line.slice(6))
          setEvents(prev => [...prev, event])
          setProgress(event.progress)

          if (event.status === 'finished') {
            setResult(event.data)
            setStatus('complete')
          } else if (event.status === 'error') {
            setStatus('error')
          }
        }
      }
    }
  }, [])

  return { events, progress, status, result, startScan }
}
```

### `POST /api/scan/quick` - Non-Streaming Scan

Same request body as `/api/scan/stream`, but returns a single JSON response after the pipeline completes. Useful for testing.

### `GET /api/scans` - Get Scan History (from Supabase)

**Query params:** `?city=Surat&limit=20`

Returns stored scan results from the Supabase PostGIS database.

### `GET /api/scans/violations` - Get Scans with Violations

**Query params:** `?severity=CRITICAL`

Returns only scans that have compliance violations, filterable by severity.

### `GET /api/health` - Server Health Check

Returns `{"status": "healthy"}`.

### `POST /api/chat` - Chat with Compliance Agent (LangGraph + Groq)

The agent has full context of all Supabase scan data, compliance rules, and RAG-retrieved legal regulations. Send `scan_context` to tell the agent what the user is currently viewing on the dashboard.

**Request:**
```json
{
  "message": "What violations were found in the latest Vesu scan?",
  "session_id": "user_123",
  "scan_context": "{\"scan_id\":\"scan_vesu_surat_20260410\",\"city\":\"Vesu, Surat\",\"violation_count\":3}"
}
```

**Response:**
```json
{
  "response": "The latest Vesu scan detected **3 compliance violations**:\n\n1. **[CRITICAL] Water Body Buffer Zone** — Gujarat GDCR 2017 S. 12.3\n   Construction detected within 243m of a designated water body (500m buffer required).\n\n2. **[CRITICAL] Water Body Buffer Zone** (second instance)\n   Same rule violated in adjacent sector.\n\n3. **[HIGH] Green Belt Protection** — Gujarat GDCR 2017 S. 15.1\n   Change detected within protected green belt zone.\n\nTotal affected area: 2.76 hectares. Recommended action: Issue Stop Work Order.\n\n[View PDF Report](https://...supabase.co/.../compliance_report.pdf)",
  "language": "en"
}
```

**Frontend Chat Component Usage:**
```jsx
const [messages, setMessages] = useState([])

async function sendMessage(text) {
  setMessages(prev => [...prev, { role: 'user', content: text }])
  
  const res = await fetch('http://<BACKEND_IP>:8000/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message: text,
      session_id: sessionId,
      scan_context: JSON.stringify(currentScanResult),  // from useScanStream
    }),
  })
  const data = await res.json()
  setMessages(prev => [...prev, { role: 'assistant', content: data.response }])
}
```

**Agent Tools Available:**
| Tool | What It Does |
|------|-------------|
| `get_all_scans` | Query all scans from Supabase |
| `get_scan_details` | Get full details for a specific scan |
| `get_violations_summary` | Aggregate violations across scans |
| `get_compliance_rules` | Load rules from rules.json |
| `search_regulations` | RAG search through Gujarat GDCR |
| `check_zone_at_location` | Check zoning for coordinates |
| `get_scan_statistics` | Aggregate stats across all scans |

### `GET /api/report/:scanId` - Download PDF Report

Returns the PDF report URL from Supabase Storage.

---

### NEW: Pipeline Progress Loader Component

```jsx
// components/controls/ScanLoader.js
'use client'

const STEP_LABELS = {
  1: 'Authenticating CDSE...',
  2: 'Downloading Sentinel-2 imagery...',
  3: 'Normalizing spectral bands...',
  4: 'Running Siamese-SNN inference...',
  5: 'Classifying changes (NDVI/NDBI/MNDWI)...',
  6: 'Evaluating compliance rules...',
  7: 'Generating PDF report...',
  8: 'Uploading to Supabase...',
  9: 'Generating visualization...',
}

export default function ScanLoader({ events, progress, status }) {
  return (
    <div className="scan-loader">
      <div className="progress-bar">
        <div className="progress-fill" style={{ width: `${progress}%` }} />
      </div>
      <div className="log-feed">
        {events.map((e, i) => (
          <div key={i} className={`log-line status-${e.status}`}>
            <span className="step-badge">STEP {e.step}/{e.total_steps}</span>
            <span className="message">{e.message}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
```

```css
/* Hacker Terminal Loader Style */
.scan-loader {
  background: rgba(10, 14, 39, 0.95);
  border: 1px solid rgba(0, 255, 170, 0.2);
  border-radius: 12px;
  padding: 20px;
  font-family: 'JetBrains Mono', monospace;
}

.progress-bar {
  height: 4px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 2px;
  overflow: hidden;
  margin-bottom: 16px;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #00ffaa, #0088ff);
  transition: width 0.3s ease;
  box-shadow: 0 0 10px rgba(0, 255, 170, 0.5);
}

.log-feed {
  max-height: 300px;
  overflow-y: auto;
}

.log-line {
  padding: 4px 0;
  font-size: 12px;
  color: rgba(255, 255, 255, 0.7);
  display: flex;
  gap: 8px;
  align-items: center;
}

.log-line.status-complete { color: #00ffaa; }
.log-line.status-error { color: #ff4444; }
.log-line.status-running { color: #00aaff; }

.step-badge {
  background: rgba(0, 136, 255, 0.2);
  color: #0088ff;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 10px;
  white-space: nowrap;
}
```

### NEW: Map Draw Tool for Custom Bounding Box

```jsx
// components/map/RegionSelector.js
'use client'
import { useMap } from 'react-leaflet'
import { useEffect, useState } from 'react'
import L from 'leaflet'

export default function RegionSelector({ onBboxSelected }) {
  const map = useMap()
  const [rectangle, setRectangle] = useState(null)

  useEffect(() => {
    let startLatLng = null
    let rect = null

    const onMouseDown = (e) => {
      startLatLng = e.latlng
      map.dragging.disable()
    }

    const onMouseMove = (e) => {
      if (!startLatLng) return
      if (rect) map.removeLayer(rect)
      rect = L.rectangle([startLatLng, e.latlng], {
        color: '#00ffaa',
        weight: 2,
        fillOpacity: 0.15,
        dashArray: '5, 5',
      }).addTo(map)
    }

    const onMouseUp = (e) => {
      map.dragging.enable()
      if (!startLatLng) return
      const bounds = L.latLngBounds(startLatLng, e.latlng)
      const bbox = [
        bounds.getWest(),   // lon_min
        bounds.getSouth(),  // lat_min
        bounds.getEast(),   // lon_max
        bounds.getNorth(),  // lat_max
      ]
      onBboxSelected(bbox)
      setRectangle(rect)
      startLatLng = null
    }

    map.on('mousedown', onMouseDown)
    map.on('mousemove', onMouseMove)
    map.on('mouseup', onMouseUp)

    return () => {
      map.off('mousedown', onMouseDown)
      map.off('mousemove', onMouseMove)
      map.off('mouseup', onMouseUp)
    }
  }, [map, onBboxSelected])

  return null
}
```

---

## 7. Map Integration Guide

### Install & Setup

```bash
npm install leaflet react-leaflet
```

### Dynamic Import (Required for Next.js — Leaflet uses `window`)

```jsx
// components/map/SatelliteMap.js
'use client'
import dynamic from 'next/dynamic'

const MapContainer = dynamic(
  () => import('react-leaflet').then(mod => mod.MapContainer),
  { ssr: false }
)
const TileLayer = dynamic(
  () => import('react-leaflet').then(mod => mod.TileLayer),
  { ssr: false }
)
const ImageOverlay = dynamic(
  () => import('react-leaflet').then(mod => mod.ImageOverlay),
  { ssr: false }
)
```

### Import Leaflet CSS in `globals.css`

```css
@import 'leaflet/dist/leaflet.css';
```

### Base Map Tiles (FREE — No API Key!)

```jsx
<TileLayer
  url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
  attribution='&copy; OSM &copy; CARTO'
  maxZoom={19}
/>
```

### Alternative Free Tile Providers

| Provider | Style | URL |
|----------|-------|-----|
| **CartoDB Dark Matter** ← Use this | Dark, sleek | `https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png` |
| CartoDB Voyager | Light, clean | `https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png` |
| Esri World Imagery | Satellite photo | `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}` |
| OpenStreetMap | Standard | `https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png` |

### Classification Overlay on Map

```jsx
{scanResult && (
  <ImageOverlay
    url={scanResult.images.classification_overlay}
    bounds={scanResult.coordinates.bounds}
    opacity={overlayOpacity}  // controlled by slider
    zIndex={400}
  />
)}
```

### City Coordinates (for auto-zoom)

```js
// lib/constants.js
export const CITY_COORDS = {
  abudhabi:  { center: [24.4539, 54.3773], zoom: 15 },
  beirut:    { center: [33.8938, 35.5018], zoom: 15 },
  mumbai:    { center: [19.0760, 72.8777], zoom: 15 },
  paris:     { center: [48.8566, 2.3522],  zoom: 15 },
  hongkong:  { center: [22.3193, 114.1694],zoom: 15 },
  bordeaux:  { center: [44.8378, -0.5792], zoom: 15 },
  pisa:      { center: [43.7228, 10.4017], zoom: 15 },
  cupertino: { center: [37.3230, -122.0322], zoom: 15 },
  nantes:    { center: [47.2184, -1.5536], zoom: 15 },
  rennes:    { center: [48.1173, -1.6778], zoom: 15 },
  beihai:    { center: [21.4811, 109.1200], zoom: 15 },
}
```

---

## 8. Animations & Micro-interactions

### Scan Button — Pulse Glow

```css
.scanButton {
  background: linear-gradient(135deg, var(--accent), #0891B2);
  color: white;
  font-size: 1rem;
  font-weight: 700;
  letter-spacing: 0.05em;
  padding: 14px 28px;
  border: none;
  border-radius: var(--radius);
  cursor: pointer;
  position: relative;
  overflow: hidden;
  transition: transform 0.2s, box-shadow 0.3s;
}

.scanButton::before {
  content: '';
  position: absolute;
  inset: -2px;
  border-radius: var(--radius);
  background: linear-gradient(135deg, var(--accent), #0891B2, var(--accent));
  z-index: -1;
  animation: pulseGlow 2s ease-in-out infinite;
}

@keyframes pulseGlow {
  0%, 100% { opacity: 0.4; filter: blur(8px); }
  50%      { opacity: 0.8; filter: blur(16px); }
}

.scanButton:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 32px var(--accent-glow);
}

.scanButton:active {
  transform: translateY(0);
}

/* Scanning state */
.scanButton.scanning {
  pointer-events: none;
  background: var(--bg-secondary);
  border: 2px solid var(--accent);
}
```

### Agent Chat — Typewriter Effect

```js
// hooks/useTypewriter.js
import { useState, useEffect } from 'react'

export function useTypewriter(text, speed = 20) {
  const [displayed, setDisplayed] = useState('')
  const [isDone, setIsDone] = useState(false)

  useEffect(() => {
    setDisplayed('')
    setIsDone(false)
    let i = 0
    const timer = setInterval(() => {
      if (i < text.length) {
        setDisplayed(prev => prev + text[i])
        i++
      } else {
        setIsDone(true)
        clearInterval(timer)
      }
    }, speed)
    return () => clearInterval(timer)
  }, [text, speed])

  return { displayed, isDone }
}
```

### Severity Alert Cards — Slide In

```css
.alertCard {
  animation: slideInRight 0.4s ease-out forwards;
  opacity: 0;
  transform: translateX(20px);
}

.alertCard:nth-child(1) { animation-delay: 0.1s; }
.alertCard:nth-child(2) { animation-delay: 0.25s; }
.alertCard:nth-child(3) { animation-delay: 0.4s; }

@keyframes slideInRight {
  to { opacity: 1; transform: translateX(0); }
}
```

### Numbers — Count Up Animation

```js
// hooks/useCountUp.js
import { useState, useEffect } from 'react'

export function useCountUp(target, duration = 1500) {
  const [count, setCount] = useState(0)

  useEffect(() => {
    const start = performance.now()
    const animate = (now) => {
      const progress = Math.min((now - start) / duration, 1)
      const eased = 1 - Math.pow(1 - progress, 3) // easeOutCubic
      setCount(Math.floor(eased * target))
      if (progress < 1) requestAnimationFrame(animate)
    }
    requestAnimationFrame(animate)
  }, [target, duration])

  return count
}
```

---

## 9. Assets & Resources

### Icons (use `lucide-react`)

```jsx
import {
  Satellite, Shield, AlertTriangle, MapPin,
  Upload, Search, BarChart3, Lock,
  ChevronDown, Settings, Send, Loader2
} from 'lucide-react'
```

### Classification Color Constants

```js
// lib/constants.js
export const CHANGE_TYPES = {
  1: { name: 'Construction / Urban Sprawl', color: '#EF4444', severity: 'high',   icon: '🔴' },
  2: { name: 'Vegetation Clearance',        color: '#22C55E', severity: 'medium', icon: '🟢' },
  3: { name: 'Water Body Change',           color: '#3B82F6', severity: 'high',   icon: '🔵' },
  4: { name: 'Other Land Alteration',       color: '#EAB308', severity: 'low',    icon: '🟡' },
}

export const SEVERITY_CONFIG = {
  high:   { color: '#EF4444', bg: 'rgba(239, 68, 68, 0.1)',  border: 'rgba(239, 68, 68, 0.3)' },
  medium: { color: '#F59E0B', bg: 'rgba(245, 158, 11, 0.1)', border: 'rgba(245, 158, 11, 0.3)' },
  low:    { color: '#22C55E', bg: 'rgba(34, 197, 94, 0.1)',  border: 'rgba(34, 197, 94, 0.3)' },
}
```

---

## 10. Implementation Checklist

### Phase 1: Shell (Day 1)
- [ ] Initialize Next.js project with the folder structure above
- [ ] Set up `globals.css` with the complete design system
- [ ] Build the 3-column grid layout (`page.js`)
- [ ] Create `GlassCard` and `GlowButton` reusable components
- [ ] Implement the Header with logo and status badge

### Phase 2: Map (Day 1-2)
- [ ] Integrate Leaflet with dynamic imports (NO SSR)
- [ ] Add CartoDB Dark Matter base tiles
- [ ] Add `ImageOverlay` component for before/after/classification layers
- [ ] Implement opacity slider control
- [ ] Add city coordinate auto-zoom

### Phase 3: Controls (Day 2)
- [ ] Build `CitySelector` dropdown (styled select)
- [ ] Build `ImageUploader` with drag-and-drop
- [ ] Build `ScanButton` with glow animation + scanning state
- [ ] Connect to `/api/scan` endpoint

### Phase 4: Analytics (Day 2-3)
- [ ] Build `ClassificationDonut` with recharts
- [ ] Build `SeverityAlerts` with staggered slide-in animation
- [ ] Build `AreaStats` with count-up number animation
- [ ] Build `BlockchainHash` display with copy-to-clipboard

### Phase 5: Agent Chat (Day 3)
- [ ] Build `ChatWindow` with message list
- [ ] Build `ChatMessage` with agent/user styling
- [ ] Implement typewriter effect for agent responses
- [ ] Connect to `/api/agent` SSE endpoint

### Phase 6: Polish (Day 3-4)
- [ ] Test responsive layout on different screen sizes
- [ ] Verify all animations are smooth (60fps)
- [ ] Test with real scan data from the Python backend
- [ ] Add loading skeletons while data is fetching
- [ ] Add error states (backend offline, scan failed, etc.)
- [ ] Final visual QA pass

---

> **Note for teammate:** The Python backend is being built in parallel. For now, mock the API responses using the exact JSON format in Section 6. When the backend is ready, just swap the `fetch` URL from mock data to the real FastAPI endpoint.
