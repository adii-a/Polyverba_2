const btnStart = document.getElementById('btnStart');
const btnStop = document.getElementById('btnStop');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const captionBox = document.getElementById('captionBox');
const targetLang = document.getElementById('targetLang');
const sourceLang = document.getElementById('sourceLang');

let ws = null;
let currentPartialLine = null;
let modelReady = false;

// Disable Start until model is confirmed ready
btnStart.disabled = true;
statusText.textContent = 'Loading model…';


/** Only auto-scroll if user is already near the bottom (within 120px). */
function smartScroll() {
    const threshold = 120;
    const distFromBottom = captionBox.scrollHeight - captionBox.scrollTop - captionBox.clientHeight;
    if (distFromBottom <= threshold) {
        captionBox.scrollTop = captionBox.scrollHeight;
    }
}

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws/captions`);

    ws.onopen = () => {
        console.log("Connected to WebSocket");
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);

        // Handle system events (not caption text)
        if (data.type === 'system') {
            if (data.event === 'model_ready') {
                modelReady = data.ready;
                if (modelReady) {
                    if (!btnStop.disabled) return; // already running
                    btnStart.disabled = false;
                    statusText.textContent = 'Ready';
                } else {
                    btnStart.disabled = true;
                    statusText.textContent = 'Model failed to load';
                }
            }
            return;
        }

        const text = data.text;
        const isFinal = data.is_final;
        const source = data.source || "Local";
        const latency = data.latency || 0.0;

        const sourceBadge = document.getElementById('sourceBadge');
        if (source === "Cloud") {
            sourceBadge.textContent = "Cloud Fallback";
            sourceBadge.style.backgroundColor = "#3b82f6";
        } else {
            sourceBadge.textContent = "Edge Local";
            sourceBadge.style.backgroundColor = "#ef4444";
        }

        // Removing placeholder if exists
        const placeholder = captionBox.querySelector('.placeholder');
        if (placeholder) placeholder.remove();

        if (isFinal) {
            // Remove any partial line
            if (currentPartialLine) {
                currentPartialLine.remove();
                currentPartialLine = null;
            }

            // Add final line
            const div = document.createElement('div');
            div.className = 'caption-line final';
            
            const textSpan = document.createElement('span');
            textSpan.appendChild(parseTextWithSpeakers(text));
            div.appendChild(textSpan);
            
            if (latency > 0) {
                const latencyBadge = document.getElementById('latencyBadge');
                if (latencyBadge) {
                    latencyBadge.style.display = 'inline-block';
                    latencyBadge.textContent = `Latency: ${latency}s`;
                    
                    // Color code latency
                    if (latency < 1.0) latencyBadge.style.backgroundColor = '#10b981'; // Green
                    else if (latency < 2.5) latencyBadge.style.backgroundColor = '#f59e0b'; // Orange
                    else latencyBadge.style.backgroundColor = '#ef4444'; // Red
                }
            }
            
            captionBox.appendChild(div);

            // Smart auto-scroll — respects user scrolling up to read history
            smartScroll();
        } else {
            // Update partial line
            if (!currentPartialLine) {
                currentPartialLine = document.createElement('div');
                currentPartialLine.className = 'caption-line partial';
                captionBox.appendChild(currentPartialLine);
            }
            // Clear children without touching textContent (preserves speaker-tag spans)
            while (currentPartialLine.firstChild) currentPartialLine.removeChild(currentPartialLine.firstChild);
            currentPartialLine.appendChild(parseTextWithSpeakers(text));
            smartScroll();
        }
    };

    ws.onclose = () => {
        console.log("WebSocket Disconnected");
    };
}

/**
 * Splits text on [Speaker N]: tags and wraps each in a bold colored pill.
 * The label is rendered in the currently selected target language.
 */
const SPEAKER_COLORS = [
    { solid: '#ea580c', bg: 'linear-gradient(135deg,#ea580c,#b45309)', shadow: 'rgba(234,88,12,0.35)'  }, // 1 orange
    { solid: '#2563eb', bg: 'linear-gradient(135deg,#2563eb,#1d4ed8)', shadow: 'rgba(37,99,235,0.35)'  }, // 2 blue
    { solid: '#7c3aed', bg: 'linear-gradient(135deg,#7c3aed,#5b21b6)', shadow: 'rgba(124,58,237,0.35)' }, // 3 purple
    { solid: '#16a34a', bg: 'linear-gradient(135deg,#16a34a,#15803d)', shadow: 'rgba(22,163,74,0.35)'  }, // 4 green
    { solid: '#e11d48', bg: 'linear-gradient(135deg,#e11d48,#be123c)', shadow: 'rgba(225,29,72,0.35)'  }, // 5 rose
    { solid: '#0891b2', bg: 'linear-gradient(135deg,#0891b2,#0e7490)', shadow: 'rgba(8,145,178,0.35)'  }, // 6 cyan
    { solid: '#d97706', bg: 'linear-gradient(135deg,#d97706,#b45309)', shadow: 'rgba(217,119,6,0.35)'  }, // 7 amber
    { solid: '#db2777', bg: 'linear-gradient(135deg,#db2777,#be185d)', shadow: 'rgba(219,39,119,0.35)' }, // 8 pink
];

// "Speaker" translated into each supported target language
const SPEAKER_WORD = {
    en: 'Speaker',
    hi: 'वक्ता',
    ta: 'பேச்சாளர்',
    te: 'వక్త',
    ml: 'വക്താവ്',
    mr: 'वक्ता',
    gu: 'વક્તા',
    kn: 'ವಕ್ತಾರ',
    bn: 'বক্তা',
    pa: 'ਬੁਲਾਰਾ',
    ur: 'مقرر',
    or: 'ବକ୍ତା',
    as: 'বক্তা',
    auto: 'Speaker',
};

function getSpeakerWord() {
    const lang = (targetLang && targetLang.value) ? targetLang.value : 'en';
    return SPEAKER_WORD[lang] || 'Speaker';
}

function parseTextWithSpeakers(text) {
    const fragment = document.createDocumentFragment();
    const parts = text.split(/(\[Speaker \d+\]:)/g);
    parts.forEach(part => {
        const match = part.match(/^\[Speaker (\d+)\]:$/);
        if (match) {
            const num = parseInt(match[1], 10);
            const color = SPEAKER_COLORS[(num - 1) % SPEAKER_COLORS.length];

            // Outer pill wrapper
            const tag = document.createElement('span');
            tag.className = 'speaker-tag';
            tag.style.background = color.bg;
            tag.style.boxShadow = `0 3px 10px ${color.shadow}`;
            tag.setAttribute('data-speaker', num);

            // Colored dot
            const dot = document.createElement('span');
            dot.className = 'speaker-dot';

            // Label: localized word + number
            const label = document.createElement('span');
            label.textContent = `${getSpeakerWord()} ${num}`;

            tag.appendChild(dot);
            tag.appendChild(label);
            fragment.appendChild(tag);
        } else {
            part.split('\n').forEach((line, i) => {
                if (i > 0) fragment.appendChild(document.createElement('br'));
                if (line) fragment.appendChild(document.createTextNode(line));
            });
        }
    });
    return fragment;
}

btnStart.addEventListener('click', async () => {
    btnStart.disabled = true;

    const response = await fetch('/api/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            source_lang: sourceLang.value,
            target_lang: targetLang.value,
            model: 'base'
        })
    });

    if (response.ok) {
        btnStop.disabled = false;
        statusDot.classList.add('active');
        statusText.textContent = "Listening...";

        // Add a session separator — history persists until page refresh
        const sep = document.createElement('div');
        sep.className = 'session-sep';
        sep.textContent = `▶ Session started · ${new Date().toLocaleTimeString()}`;
        captionBox.appendChild(sep);
        smartScroll();

        const latencyBadge = document.getElementById('latencyBadge');
        if (latencyBadge) {
            latencyBadge.style.display = 'inline-block';
            latencyBadge.textContent = 'Latency: --s';
            latencyBadge.style.backgroundColor = '#64748b';
        }

        if (!ws || ws.readyState === WebSocket.CLOSED) {
            connectWebSocket();
        }
    } else if (response.status === 503) {
        // Model still warming up
        btnStart.disabled = true; // keep disabled — WS event will re-enable
        statusText.textContent = 'Model still loading, please wait…';
    } else {
        btnStart.disabled = false;
        statusText.textContent = "Error starting";
    }
});


btnStop.addEventListener('click', async () => {
    btnStop.disabled = true;

    const response = await fetch('/api/stop', { method: 'POST' });

    if (response.ok) {
        btnStart.disabled = false;
        statusDot.classList.remove('active');
        statusText.textContent = "Stopped";
        const latencyBadge = document.getElementById('latencyBadge');
        if (latencyBadge) latencyBadge.style.display = 'none';
    } else {
        btnStop.disabled = false;
    }
});

// Initial connection
connectWebSocket();

async function restartSession() {
    // Only auto-restart if currently running (Stop button is enabled)
    if (btnStop.disabled) {
        return;
    }

    console.log("Language changed, restarting...");
    statusText.textContent = "Switching language...";

    // 1. Stop current session
    await fetch('/api/stop', { method: 'POST' });

    // 2. Start new session
    const response = await fetch('/api/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            source_lang: sourceLang.value,
            target_lang: targetLang.value,
            model: 'base'
        })
    });

    if (response.ok) {
        statusText.textContent = "Listening...";
        // Visual separator
        const div = document.createElement('div');
        div.className = 'caption-line partial';
        div.textContent = `--- Switched: ${sourceLang.options[sourceLang.selectedIndex].text} -> ${targetLang.options[targetLang.selectedIndex].text} ---`;
        div.style.textAlign = 'center';
        div.style.opacity = '0.5';
        div.style.margin = '1rem 0';
        captionBox.appendChild(div);
        smartScroll();
    } else {
        statusText.textContent = "Error switching";
        btnStart.disabled = false;
        btnStop.disabled = true;
    }
}

targetLang.addEventListener('change', restartSession);
sourceLang.addEventListener('change', restartSession);

// Font size control
const fontSizeSelect = document.getElementById('fontSize');
if (fontSizeSelect) {
    fontSizeSelect.addEventListener('change', (e) => {
        captionBox.style.fontSize = e.target.value;
    });
    // Set initial size
    captionBox.style.fontSize = fontSizeSelect.value;
}
