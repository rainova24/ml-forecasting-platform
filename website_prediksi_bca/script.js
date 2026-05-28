// Inisialisasi Intersection Observer untuk animasi scroll
document.addEventListener('DOMContentLoaded', () => {
    const reveals = document.querySelectorAll('.reveal');

    const revealOnScroll = () => {
        for (let i = 0; i < reveals.length; i++) {
            const windowHeight = window.innerHeight;
            const elementTop = reveals[i].getBoundingClientRect().top;
            const elementVisible = 100;

            if (elementTop < windowHeight - elementVisible) {
                reveals[i].classList.add('active');
            }
        }
    };

    window.addEventListener('scroll', revealOnScroll);
    revealOnScroll(); // Trigger sekali saat halaman dimuat

    // Render Chart.js jika data tersedia
    if (typeof chartData !== 'undefined') {
        renderChart();
    } else {
        console.warn("Menunggu data dari file data.js yang akan di-generate oleh Python.");
    }
});

function renderChart() {
    const ctx = document.getElementById('predictionChart').getContext('2d');
    
    // Kita mengambil 100 data terakhir agar grafik tidak terlalu padat dan lebih cantik dilihat
    const displayLength = Math.min(chartData.labels.length, 100);
    const labels = chartData.labels.slice(-displayLength);
    const actual = chartData.actual.slice(-displayLength);
    const predicted = chartData.predicted.slice(-displayLength);

    // Konfigurasi Chart.js
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Harga Aktual',
                    data: actual,
                    borderColor: '#3b82f6', // Biru
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 2,
                    tension: 0.4, // Membuat garis lebih smooth
                    pointRadius: 0,
                    pointHoverRadius: 6
                },
                {
                    label: 'Rolling Forecast (ARIMA)',
                    data: predicted,
                    borderColor: '#f43f5e', // Merah
                    backgroundColor: 'rgba(244, 63, 94, 0.1)',
                    borderWidth: 2,
                    borderDash: [5, 5], // Garis putus-putus
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        color: '#cbd5e1',
                        font: {
                            family: "'Inter', sans-serif",
                            size: 14
                        }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    titleColor: '#fff',
                    bodyColor: '#cbd5e1',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    padding: 12
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    },
                    ticks: {
                        color: '#cbd5e1'
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    },
                    ticks: {
                        color: '#cbd5e1',
                        callback: function(value) {
                            return 'Rp ' + value.toLocaleString('id-ID');
                        }
                    }
                }
            }
        }
    });
}
