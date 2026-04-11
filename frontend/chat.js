/**
 * Chat widget — WebSocket real-time chat with streaming, map highlighting, and voice input.
 */

// ===== STATE =====

let chatOpen = false;
let ws = null;
let messageHistory = [];
let unreadCount = 0;
const API_BASE = window.location.origin;

// ===== DOM ELEMENTS =====

const chatToggle = document.getElementById('chat-toggle');
const chatWindow = document.getElementById('chat-window');
const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const chatSend = document.getElementById('chat-send');
const chatClose = document.getElementById('chat-close');
const chatTyping = document.getElementById('chat-typing');
const chatBadge = document.getElementById('chat-badge');
const voiceBtn = document.getElementById('voice-btn');

// ===== TOGGLE CHAT =====

chatToggle.addEventListener('click', () => {
    chatOpen = !chatOpen;
    chatWindow.classList.toggle('hidden', !chatOpen);
    chatToggle.classList.toggle('hidden', chatOpen);

    if (chatOpen) {
        unreadCount = 0;
        chatBadge.classList.add('hidden');
        chatInput.focus();

        if (messageHistory.length === 0) {
            addBotMessage(
                "Hello! I'm the **Surat Satellite Compliance Engine**. I can help you:\n\n" +
                "• Check zoning regulations for any location\n" +
                "• View satellite change detections\n" +
                "• Run compliance checks with legal citations\n" +
                "• Generate reports and enforcement notices\n\n" +
                "How can I assist you?",
                ['Check Ward 42', 'Latest violations', 'Scan Adajan area']
            );
        }

        connectWebSocket();
    }
});

chatClose.addEventListener('click', () => {
    chatOpen = false;
    chatWindow.classList.add('hidden');
    chatToggle.classList.remove('hidden');
});

// ===== SEND MESSAGE =====

chatSend.addEventListener('click', sendMessage);
chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

function sendMessage() {
    const text = chatInput.value.trim();
    if (!text) return;

    addUserMessage(text);
    chatInput.value = '';
    showTyping();

    // Send via WebSocket or HTTP fallback
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            message: text,
            language: document.getElementById('language-selector').value,
        }));
    } else {
        sendHTTP(text);
    }
}

async function sendHTTP(text) {
    try {
        const resp = await fetch(`${API_BASE}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: text,
                session_id: 'web_default',
                language: document.getElementById('language-selector').value,
            }),
        });
        const data = await resp.json();
        hideTyping();
        addBotMessage(data.response, getActionsForResponse(data));

        // Handle map highlights
        if (data.highlighted_areas) {
            data.highlighted_areas.forEach(area => {
                if (window.highlightArea) window.highlightArea(area);
            });
        }
    } catch (e) {
        hideTyping();
        addBotMessage(
            "🛰️ I'm running in demo mode. To enable full AI capabilities, " +
            "set your `GROQ_API_KEY` in `.env` and install the LangChain dependencies.\n\n" +
            `Your query: "${text}"`,
            ['Generate Report', 'Show Violations']
        );
    }
}

// ===== WEBSOCKET =====

function connectWebSocket() {
    if (ws && ws.readyState === WebSocket.OPEN) return;

    const wsUrl = `${API_BASE.replace('http', 'ws')}/ws/chat`;
    try {
        ws = new WebSocket(wsUrl);

        ws.onopen = () => console.log('WebSocket connected');

        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            handleWSMessage(msg);
        };

        ws.onclose = () => {
            console.log('WebSocket disconnected');
            ws = null;
        };

        ws.onerror = () => {
            console.log('WebSocket unavailable — using HTTP fallback');
            ws = null;
        };
    } catch (e) {
        console.log('WebSocket not available');
    }
}

function handleWSMessage(msg) {
    switch (msg.type) {
        case 'chat_response':
            hideTyping();
            const data = msg.data;
            addBotMessage(data.content, data.actions || []);

            if (data.highlighted_areas) {
                data.highlighted_areas.forEach(area => {
                    if (window.highlightArea) window.highlightArea(area);
                });
            }
            break;

        case 'alert':
            hideTyping();
            addAlertMessage(msg.data);
            if (!chatOpen) {
                unreadCount++;
                chatBadge.textContent = unreadCount;
                chatBadge.classList.remove('hidden');
            }
            break;

        case 'highlight_area':
            if (window.highlightArea) window.highlightArea(msg.data.geojson);
            break;

        case 'scan_progress':
            // Handled by app.js
            break;
    }
}

// ===== MESSAGE RENDERING =====

function addUserMessage(text) {
    const bubble = document.createElement('div');
    bubble.className = 'chat-bubble bubble-user';
    bubble.textContent = text;
    chatMessages.appendChild(bubble);
    messageHistory.push({ role: 'user', content: text });
    scrollToBottom();
}

function addBotMessage(text, actions = []) {
    const bubble = document.createElement('div');
    bubble.className = 'chat-bubble bubble-bot';

    // Simple markdown rendering
    let html = text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/\n/g, '<br>')
        .replace(/• /g, '&bull; ');
    bubble.innerHTML = html;

    // Action buttons
    if (actions.length > 0) {
        const actionsDiv = document.createElement('div');
        actionsDiv.className = 'chat-actions';

        actions.forEach(action => {
            const btn = document.createElement('button');
            btn.className = 'chat-action-btn';
            btn.textContent = action;
            btn.addEventListener('click', () => {
                chatInput.value = action;
                sendMessage();
            });
            actionsDiv.appendChild(btn);
        });

        bubble.appendChild(actionsDiv);
    }

    chatMessages.appendChild(bubble);
    messageHistory.push({ role: 'assistant', content: text });
    scrollToBottom();
}

function addAlertMessage(alert) {
    const bubble = document.createElement('div');
    bubble.className = 'chat-bubble bubble-alert';

    let html = alert.message
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n/g, '<br>');

    bubble.innerHTML = html;

    // Alert actions
    if (alert.actions) {
        const actionsDiv = document.createElement('div');
        actionsDiv.className = 'chat-actions';

        const actionLabels = {
            'generate_report': '📄 Generate Report',
            'draft_notice': '📋 Draft Notice',
            'show_on_map': '🗺️ Show on Map',
            'dispatch_officer': '📱 Dispatch Officer',
        };

        alert.actions.forEach(action => {
            const btn = document.createElement('button');
            btn.className = 'chat-action-btn';
            btn.textContent = actionLabels[action] || action;
            btn.addEventListener('click', () => {
                if (action === 'show_on_map' && alert.geojson) {
                    if (window.highlightArea) window.highlightArea(alert.geojson);
                } else {
                    chatInput.value = `${action.replace(/_/g, ' ')} for detection ${alert.detection_id}`;
                    sendMessage();
                }
            });
            actionsDiv.appendChild(btn);
        });

        bubble.appendChild(actionsDiv);
    }

    chatMessages.appendChild(bubble);
    scrollToBottom();
}

function showTyping() { chatTyping.classList.remove('hidden'); scrollToBottom(); }
function hideTyping() { chatTyping.classList.add('hidden'); }
function scrollToBottom() { chatMessages.scrollTop = chatMessages.scrollHeight; }

function getActionsForResponse(data) {
    const actions = [];
    if (data.pending_actions) {
        data.pending_actions.forEach(a => {
            if (a === 'generate_report') actions.push('📄 Generate Report');
            if (a === 'draft_notice') actions.push('📋 Draft Notice');
        });
    }
    return actions;
}

// ===== VOICE INPUT =====

let recognition = null;

voiceBtn.addEventListener('click', () => {
    if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
        alert('Voice input is not supported in this browser.');
        return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();

    const lang = document.getElementById('language-selector').value;
    recognition.lang = lang === 'hi' ? 'hi-IN' : lang === 'gu' ? 'gu-IN' : 'en-US';
    recognition.continuous = false;
    recognition.interimResults = false;

    recognition.onresult = (event) => {
        const text = event.results[0][0].transcript;
        chatInput.value = text;
        sendMessage();
        voiceBtn.style.color = '';
    };

    recognition.onerror = () => {
        voiceBtn.style.color = '';
    };

    recognition.start();
    voiceBtn.style.color = '#ff5252';
});
