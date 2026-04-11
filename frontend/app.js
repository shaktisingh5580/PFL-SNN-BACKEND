/**
 * SSCE — Leaflet map, time-slider, layer controls, HITL UI, and scan visualization.
 */

// ===== MAP INITIALIZATION =====

const SURAT_CENTER = [21.17, 72.83];
const SURAT_BOUNDS = [[21.08, 72.72], [21.28, 72.92]];

const map = L.map('map', {
    center: SURAT_CENTER,
    zoom: 12,
    zoomControl: true,
    attributionControl: false,
});

// Satellite basemap
const satellite = L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    { maxZoom: 19 }
).addTo(map);

// Light basemap for toggle
const light = L.tileLayer(
    'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
    { maxZoom: 19, subdomains: 'abcd' }
);

// ===== LAYER GROUPS =====

const changeLayer = L.layerGroup().addTo(map);
const zoneLayer = L.layerGroup();
const violationLayer = L.layerGroup();
const scanLayer = L.layerGroup();
const highlightLayer = L.layerGroup().addTo(map);

// ===== TIME SLIDER =====

const months = [
    'Apr 2025', 'May 2025', 'Jun 2025', 'Jul 2025',
    'Aug 2025', 'Sep 2025', 'Oct 2025', 'Nov 2025',
    'Dec 2025', 'Jan 2026', 'Feb 2026', 'Mar 2026'
];

const timeSlider = document.getElementById('time-slider');
const timeDisplay = document.getElementById('time-display');

timeSlider.addEventListener('input', () => {
    timeDisplay.textContent = months[timeSlider.value];
    // In production, this would reload temporal layers
});

// ===== LAYER CONTROLS =====

document.getElementById('layer-changes').addEventListener('change', (e) => {
    e.target.checked ? map.addLayer(changeLayer) : map.removeLayer(changeLayer);
});

document.getElementById('layer-zones').addEventListener('change', (e) => {
    e.target.checked ? map.addLayer(zoneLayer) : map.removeLayer(zoneLayer);
});

document.getElementById('layer-ndvi').addEventListener('change', () => {
    // Toggle NDVI tile layer
});

document.getElementById('layer-violations').addEventListener('change', (e) => {
    e.target.checked ? map.addLayer(violationLayer) : map.removeLayer(violationLayer);
});

// ===== SIDEBAR TABS =====

document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById(`tab-${tab.dataset.tab}`).classList.add('active');
    });
});

// ===== DEMO DATA =====

const demoDetections = [
    {
        id: 'd1a2b3c4', type: 'new_construction', confidence: 0.87,
        lat: 21.1756, lon: 72.8312, area: 0.45, date: '2026-03-25',
        severity: 'HIGH',
    },
    {
        id: 'e5f6g7h8', type: 'vegetation_loss', confidence: 0.91,
        lat: 21.2156, lon: 72.8687, area: 1.2, date: '2026-03-22',
        severity: 'CRITICAL',
    },
    {
        id: 'i9j0k1l2', type: 'new_construction', confidence: 0.73,
        lat: 21.1892, lon: 72.7945, area: 0.28, date: '2026-03-20',
        severity: 'MEDIUM',
    },
    {
        id: 'm3n4o5p6', type: 'road_expansion', confidence: 0.65,
        lat: 21.1612, lon: 72.8034, area: 0.8, date: '2026-03-18',
        severity: 'LOW',
    },
];

// ===== POPULATE DETECTIONS =====

function renderDetections() {
    const list = document.getElementById('detections-list');
    list.innerHTML = '';

    demoDetections.forEach(d => {
        const card = document.createElement('div');
        card.className = 'detection-card';
        card.innerHTML = `
            <div class="card-header">
                <span class="card-title">${d.type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</span>
                <span class="card-badge badge-${d.severity.toLowerCase()}">${d.severity}</span>
            </div>
            <div class="card-meta">
                <strong>Confidence:</strong> ${(d.confidence * 100).toFixed(0)}%<br>
                <strong>Location:</strong> ${d.lat.toFixed(4)}°N, ${d.lon.toFixed(4)}°E<br>
                <strong>Area:</strong> ${d.area} ha | <strong>Date:</strong> ${d.date}
            </div>
        `;

        card.addEventListener('click', () => selectDetection(d));
        list.appendChild(card);

        // Add marker to map
        const color = d.severity === 'CRITICAL' ? '#ff5252' :
                      d.severity === 'HIGH' ? '#ff9800' :
                      d.severity === 'MEDIUM' ? '#ffb74d' : '#76b900';

        const marker = L.circleMarker([d.lat, d.lon], {
            radius: 10,
            color: color,
            fillColor: color,
            fillOpacity: 0.6,
            weight: 2,
        }).addTo(changeLayer);

        marker.bindPopup(`
            <strong>${d.type.replace(/_/g, ' ')}</strong><br>
            Confidence: ${(d.confidence * 100).toFixed(0)}%<br>
            Area: ${d.area} ha
        `);
    });

    // Update stats
    document.getElementById('detection-count').textContent = `${demoDetections.length} detections`;
    document.getElementById('stat-total-changes').textContent = demoDetections.length;
}

function renderViolations() {
    const list = document.getElementById('violations-list');
    list.innerHTML = '';

    const violations = [
        { id: 'V001', rule: 'Water body buffer', ref: 'GDCR § 12.3', severity: 'HIGH',
          location: '21.1756°N, 72.8312°E' },
        { id: 'V002', rule: 'Green Belt encroachment', ref: 'GDCR § 15.1', severity: 'CRITICAL',
          location: '21.2156°N, 72.8687°E' },
        { id: 'V003', rule: 'Green cover minimum', ref: 'TP Act § 22', severity: 'MEDIUM',
          location: '21.1892°N, 72.7945°E' },
    ];

    violations.forEach(v => {
        const card = document.createElement('div');
        card.className = 'violation-card';
        card.innerHTML = `
            <div class="card-header">
                <span class="card-title">${v.rule}</span>
                <span class="card-badge badge-${v.severity.toLowerCase()}">${v.severity}</span>
            </div>
            <div class="card-meta">
                <strong>Rule:</strong> ${v.ref}<br>
                <strong>Location:</strong> ${v.location}
            </div>
        `;
        list.appendChild(card);
    });

    document.getElementById('violation-count').textContent = `${violations.length} violations`;
    document.getElementById('stat-violations').textContent = violations.length;
}

// ===== HITL REVIEW =====

let selectedDetection = null;

function selectDetection(detection) {
    selectedDetection = detection;
    const hitl = document.getElementById('hitl-review');
    hitl.classList.remove('hidden');

    document.getElementById('hitl-detail').innerHTML = `
        <strong>${detection.type.replace(/_/g, ' ')}</strong> at
        ${detection.lat.toFixed(4)}°N, ${detection.lon.toFixed(4)}°E<br>
        Confidence: ${(detection.confidence * 100).toFixed(0)}% |
        Area: ${detection.area} ha
    `;

    // Fly to location
    map.flyTo([detection.lat, detection.lon], 15, { duration: 1.5 });

    // Add pulsing highlight
    highlightArea({
        type: 'Feature',
        geometry: { type: 'Point', coordinates: [detection.lon, detection.lat] },
    });
}

document.getElementById('btn-approve').addEventListener('click', () => {
    if (!selectedDetection) return;
    submitFeedback('approve');
});

document.getElementById('btn-reject').addEventListener('click', () => {
    if (!selectedDetection) return;
    submitFeedback('reject');
});

document.getElementById('btn-dispatch').addEventListener('click', () => {
    if (!selectedDetection) return;
    alert(`📱 WhatsApp alert would be sent to field officer for detection ${selectedDetection.id}`);
});

async function submitFeedback(type) {
    try {
        const resp = await fetch('/api/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                detection_id: selectedDetection.id,
                feedback_type: type,
                reviewed_by: 'dashboard_user',
            }),
        });
        const result = await resp.json();
        const emoji = type === 'approve' ? '✅' : '❌';
        console.log(`${emoji} Feedback submitted:`, result);
    } catch (e) {
        console.log(`Feedback (demo): ${type} for ${selectedDetection.id}`);
    }
    document.getElementById('hitl-review').classList.add('hidden');
}

// ===== MAP HIGHLIGHT =====

function highlightArea(geojson) {
    highlightLayer.clearLayers();

    if (geojson.geometry.type === 'Point') {
        const [lon, lat] = geojson.geometry.coordinates;
        const pulse = L.circleMarker([lat, lon], {
            radius: 20,
            color: '#4fc3f7',
            fillColor: '#4fc3f7',
            fillOpacity: 0.2,
            weight: 2,
            className: 'pulse-marker',
        }).addTo(highlightLayer);

        // Animate pulse
        let growing = true;
        let r = 20;
        const anim = setInterval(() => {
            r += growing ? 0.5 : -0.5;
            if (r > 30) growing = false;
            if (r < 20) growing = true;
            pulse.setRadius(r);
        }, 50);

        setTimeout(() => clearInterval(anim), 5000);
    } else {
        L.geoJSON(geojson, {
            style: { color: '#4fc3f7', weight: 3, fillOpacity: 0.15, dashArray: '5, 5' },
        }).addTo(highlightLayer);
    }
}

// ===== CITY SCAN =====

document.getElementById('scan-city-btn').addEventListener('click', startCityScan);

function startCityScan() {
    const progress = document.getElementById('scan-progress');
    progress.classList.remove('hidden');

    let completed = 0;
    const total = 2400;

    const interval = setInterval(() => {
        completed += Math.floor(Math.random() * 30) + 10;
        if (completed >= total) {
            completed = total;
            clearInterval(interval);
            setTimeout(() => progress.classList.add('hidden'), 3000);
        }

        const pct = ((completed / total) * 100).toFixed(1);
        document.getElementById('scan-bar').style.width = `${pct}%`;
        document.getElementById('scan-percent').textContent = `${pct}%`;

        const eta = Math.max(0, ((total - completed) / 40)).toFixed(0);
        document.getElementById('scan-detail').textContent =
            `${completed} / ${total} tiles — ETA: ${eta}s`;
    }, 100);
}

// Expose highlightArea for chat.js
window.highlightArea = highlightArea;

// ===== INIT =====

renderDetections();
renderViolations();
