const API_BASE_URL = 'http://localhost:8000';
const POLL_INTERVAL = 2000;

const connectionDot = document.getElementById('connection-dot');
const connectionText = document.getElementById('connection-text');
const largeItemsGrid = document.getElementById('large-items-grid');
const emptyState = document.getElementById('empty-state');

let lastProcessedImage = null;

async function fetchLatestData() {
    try {
        const response = await fetch(`${API_BASE_URL}/latest`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        setConnectionStatus('connected', 'Live');
        
        if (data && data.overall_status && data.summary) {
            // Check if image changed to prevent unnecessary re-renders
            if (data.annotated_image !== lastProcessedImage) {
                lastProcessedImage = data.annotated_image;
                updateItemsGrid(data.leaves || {}, data.fruits || {});
            }
        }
        
    } catch (error) {
        console.error('Error fetching data:', error);
        setConnectionStatus('error', 'Disconnected');
    }
}

function updateItemsGrid(leaves, fruits) {
    largeItemsGrid.innerHTML = '';
    
    const allItems = [
        ...Object.values(leaves),
        ...Object.values(fruits)
    ];

    if (allItems.length > 0) {
        emptyState.style.display = 'none';
        largeItemsGrid.style.display = 'grid';
        
        allItems.forEach(item => {
            const card = document.createElement('div');
            card.className = 'item-card';
            
            let statusClass = 'good';
            let statusText = 'Healthy';
            let detailText = '';

            if (item.type === 'leaf') {
                if (item.status !== 'Healthy') {
                    statusClass = 'bad';
                    statusText = 'Diseased';
                    detailText = item.disease_type || '';
                }
            } else if (item.type === 'fruit') {
                if (item.status !== 'Healthy' || item.ripeness !== 'Ripe') {
                    if (item.status !== 'Healthy') {
                        statusClass = 'bad';
                        statusText = 'Damaged';
                        detailText = item.damage_level || '';
                    } else if (item.ripeness === 'Unripe') {
                        statusClass = 'neutral';
                        statusText = 'Unripe';
                    }
                } else {
                    statusText = 'Ready to Harvest (Ripe)';
                }
            }

            const typeStr = item.type === 'leaf' ? 'Leaf' : 'Fruit';
            
            // Note: item.confidence is already a string like "95.0%" from backend
            card.innerHTML = `
                <span class="item-type">${typeStr} - Confidence: ${item.confidence}</span>
                <span class="item-status ${statusClass}">${statusText}</span>
                ${detailText ? `<span style="font-size:0.8rem; opacity:0.8;">Disease/Damage: ${detailText}</span>` : ''}
            `;
            
            largeItemsGrid.appendChild(card);
        });
    } else {
        emptyState.style.display = 'block';
        largeItemsGrid.style.display = 'none';
    }
}

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

// Initial fetch & poll
fetchLatestData();
setInterval(fetchLatestData, POLL_INTERVAL);
