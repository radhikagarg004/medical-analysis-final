// Set Current Date
const dateEl = document.getElementById('dateDisplay');
if (dateEl) {
    const options = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
    dateEl.innerText = new Date().toLocaleDateString('en-US', options);
}

// Preview Image on Upload
function previewImage(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const imgPreview = document.getElementById('imagePreview');
            imgPreview.src = e.target.result;
            imgPreview.style.display = 'block';
            document.getElementById('uploadPlaceholder').style.display = 'none';
            document.getElementById('analyzeBtn').classList.remove('hidden');
        }
        reader.readAsDataURL(file);
    }
}

// Run Diagnosis
function analyzeXray() {
    // Hide Button & Show Loader
    document.getElementById('analyzeBtn').classList.add('hidden');
    document.getElementById('loader').classList.remove('hidden');
    document.getElementById('emptyState').style.display = 'none';

    // Simulate API delay (2 seconds)
    setTimeout(() => {
        // Hide Loader & Show Results
        document.getElementById('loader').classList.add('hidden');
        document.getElementById('resultCard').style.display = 'block';
        document.getElementById('simulatorCard').style.display = 'block';
        
        // Dummy API Response
        const data = {
            label: "PNEUMONIA",
            risk: "HIGH",
            confidence: 93.4,
            opacity: 18.5,
            recommendation: "Immediate antibiotic therapy. Recommend Chest CT scan."
        };

        // Render Results
        const color = data.risk === 'HIGH' ? 'var(--danger)' : 'var(--success)';
        document.getElementById('diagnosisContent').innerHTML = `
            <h2 style="color: ${color}; margin-bottom: 10px;">⚠️ ${data.label}</h2>
            <span class="risk-badge ${data.risk === 'HIGH' ? 'risk-high' : 'risk-low'}">${data.risk} RISK</span>
            
            <div class="stat-grid">
                <div class="stat">
                    <h4>Confidence</h4>
                    <p>${data.confidence}%</p>
                </div>
                <div class="stat">
                    <h4>Opacity</h4>
                    <p>${data.opacity}%</p>
                </div>
            </div>

            <div class="progress-bar">
                <div class="progress-fill" style="background: ${color}; width: ${data.confidence}%;"></div>
            </div>

            <div class="recommendation-box">
                <b>👨‍⚕️ AI Recommendation:</b><br> ${data.recommendation}
            </div>
        `;

        // Initialize Chart
        updateChart();
        
    }, 2000);
}

// Digital Twin Chart Logic
let chartInstance = null;
function updateChart() {
    const delay = document.getElementById('delaySlider').value;
    document.getElementById('delayValue').innerText = delay + " Days";

    const labels = ['Day 1', 'Day 3', 'Day 5', 'Day 7'];
    const data = labels.map((_, i) => 20 + (i * 5) + (delay * 7)); 

    if (chartInstance) { chartInstance.destroy(); }
    
    const ctx = document.getElementById('recoveryChart').getContext('2d');
    chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Infection Level (%)',
                data: data,
                borderColor: '#ef4444',
                backgroundColor: 'rgba(239, 68, 68, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: { 
                y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { color: '#94a3b8' } },
                x: { grid: { display: false }, ticks: { color: '#94a3b8' } }
            }
        }
    });
}