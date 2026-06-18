const API_BASE_URL = 'http://localhost:8000';
const POLL_INTERVAL = 2000; // 2 seconds

// DOM Elements
const connectionDot = document.getElementById('connection-dot');
const connectionText = document.getElementById('connection-text');
const annotatedImage = document.getElementById('annotated-image');
const imagePlaceholder = document.getElementById('image-placeholder');
const timeBadge = document.getElementById('time-badge');
const overallStatusChip = document.getElementById('overall-status-chip');
const summaryText = document.getElementById('summary-text');
const llmText = document.getElementById('llm-text');
const llmPlaceholder = document.getElementById('llm-placeholder');
const statsPanel = document.getElementById('stats-panel');
const itemsGrid = document.getElementById('items-grid');

let lastProcessedImage = null;

// Function to fetch the latest data
async function fetchLatestData() {
    try {
        const response = await fetch(`${API_BASE_URL}/latest`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Update connection status
        setConnectionStatus('connected', 'Live');
        
        // Only update UI if we have valid data that isn't just the waiting placeholder
        if (data && data.overall_status && data.summary) {
            updateDashboard(data);
        }
        
    } catch (error) {
        console.error('Error fetching data:', error);
        setConnectionStatus('error', 'Disconnected');
    }
}

// Function to update the dashboard UI
function updateDashboard(data) {
    // 1. Update Image
    if (data.annotated_image && data.annotated_image !== lastProcessedImage) {
        // Use cache-busting to ensure image updates if path is same but content changed
        const imageUrl = `${API_BASE_URL}${data.annotated_image}?t=${new Date().getTime()}`;
        annotatedImage.src = imageUrl;
        annotatedImage.style.display = 'block';
        imagePlaceholder.style.display = 'none';
        lastProcessedImage = data.annotated_image;
    }

    // 2. Update Status & Summary
    const isProblem = data.overall_status === 'Problem Detected';
    
    overallStatusChip.textContent = data.overall_status;
    overallStatusChip.className = `status-chip ${isProblem ? 'problem' : 'healthy'}`;
    
    summaryText.innerHTML = data.summary.replace(/\n/g, '<br><br>');
    
    if (data.processing_time_ms) {
        timeBadge.textContent = `${data.processing_time_ms}ms`;
    }

    // 3. Update LLM Advice
    if (data.llm) {
        llmPlaceholder.style.display = 'none';
        llmText.style.display = 'block';
        
        // Simple markdown parsing for bold text
        let formattedText = data.llm.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        formattedText = formattedText.replace(/\n/g, '<br>');
        
        llmText.innerHTML = formattedText;
    } else {
        llmPlaceholder.style.display = 'flex';
        llmText.style.display = 'none';
    }
}

// Helper to set connection status
function setConnectionStatus(status, text) {
    connectionDot.className = `pulse-dot ${status}`;
    connectionText.textContent = text;
    
    if (status === 'connected') {
        connectionDot.style.setProperty('--shadow-color', 'rgba(16, 185, 129, 0.7)');
    } else if (status === 'error') {
        connectionDot.style.setProperty('--shadow-color', 'rgba(239, 68, 68, 0.7)');
    } else {
        connectionDot.style.setProperty('--shadow-color', 'rgba(245, 158, 11, 0.7)');
    }
}

// Start polling
setInterval(fetchLatestData, POLL_INTERVAL);

// Initial fetch
fetchLatestData();
