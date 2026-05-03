const btnStart = document.getElementById('btnStart');
const btnStop = document.getElementById('btnStop');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const captionBox = document.getElementById('captionBox');
const targetLang = document.getElementById('targetLang');
const sourceLang = document.getElementById('sourceLang');

let ws = null;
let currentPartialLine = null;

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws/captions`);

    ws.onopen = () => {
        console.log("Connected to WebSocket");
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        const text = data.text;
        const isFinal = data.is_final;
        const source = data.source || "Local";
        
        const sourceBadge = document.getElementById('sourceBadge');
        if (source === "Cloud") {
            sourceBadge.textContent = "Cloud Fallback";
            sourceBadge.style.backgroundColor = "#3b82f6"; // Blue
        } else {
            sourceBadge.textContent = "Edge Local";
            sourceBadge.style.backgroundColor = "#ef4444"; // Red
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
            div.textContent = text;
            captionBox.appendChild(div);

            // Auto scroll
            captionBox.scrollTop = captionBox.scrollHeight;
        } else {
            // Update partial line
            if (!currentPartialLine) {
                currentPartialLine = document.createElement('div');
                currentPartialLine.className = 'caption-line partial';
                captionBox.appendChild(currentPartialLine);
            }
            currentPartialLine.textContent = text;
        }
    };

    ws.onclose = () => {
        console.log("WebSocket Disconnected");
    };
}

btnStart.addEventListener('click', async () => {
    btnStart.disabled = true;

    const response = await fetch('/api/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            source_lang: sourceLang.value,
            target_lang: targetLang.value,
            model: 'base' // Use base model for better multilingual support
        })
    });

    if (response.ok) {
        btnStop.disabled = false;
        statusDot.classList.add('active');
        statusText.textContent = "Listening...";

        // Clear previous captions
        captionBox.innerHTML = '';

        if (!ws || ws.readyState === WebSocket.CLOSED) {
            connectWebSocket();
        }
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
        captionBox.scrollTop = captionBox.scrollHeight;
    } else {
        statusText.textContent = "Error switching";
        btnStart.disabled = false;
        btnStop.disabled = true;
    }
}

targetLang.addEventListener('change', restartSession);
sourceLang.addEventListener('change', restartSession);
