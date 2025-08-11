

document.addEventListener('DOMContentLoaded', () => {
    const ctx = document.getElementById('chart').getContext('2d');

    const temps = [];
    const hums = [];
    const times = [];

    document.querySelectorAll('.forecast-item').forEach(item => {
        const time = item.querySelector('.forecast-time')?.textContent.trim();
        const temp = parseFloat(item.querySelector('.forecast-temperatureValue')?.textContent.trim());
        // Support both .forecast-humidityValue and .forecast-humidityValue inside .forecast-humidity-text
        let humSpan = item.querySelector('.forecast-humidityValue');
        const hum = humSpan ? parseFloat(humSpan.textContent.trim()) : NaN;

        if (time && !isNaN(temp) && !isNaN(hum)) {
            times.push(time);
            temps.push(temp);
            hums.push(hum);
        }
        // Debug log for each item
        console.log('Forecast:', {time, temp, hum});
    });

    // Debug log for final arrays
    console.log('Times:', times);
    console.log('Temps:', temps);
    console.log('Hums:', hums);

    const tempGradient = ctx.createLinearGradient(0, 0, 0, 100);
    tempGradient.addColorStop(0, 'rgba(255, 99, 132, 1)');
    tempGradient.addColorStop(1, 'rgba(255, 159, 64, 1)');

    const humGradient = ctx.createLinearGradient(0, 0, 0, 100);
    humGradient.addColorStop(0, 'rgba(54, 162, 235, 1)');
    humGradient.addColorStop(1, 'rgba(75, 192, 192, 1)');

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: times,
            datasets: [
                {
                    label: 'Temperature (°C)',
                    data: temps,
                    borderColor: tempGradient,
                    borderWidth: 2,
                    tension: 0.4,
                    yAxisID: 'y',
                    pointRadius: 3
                },
                {
                    label: 'Humidity (%)',
                    data: hums,
                    borderColor: humGradient,
                    borderWidth: 2,
                    tension: 0.4,
                    yAxisID: 'y1',
                    pointRadius: 3
                }
            ]
        },
        options: {
            responsive: true,
            interaction: {
                mode: 'index',
                intersect: false
            },
            stacked: false,
            plugins: {
                legend: { display: true }
            },
            scales: {
                y: {
                    type: 'linear',
                    position: 'left',
                    title: { display: true, text: 'Temperature (°C)' }
                },
                y1: {
                    type: 'linear',
                    position: 'right',
                    title: { display: true, text: 'Humidity (%)' },
                    grid: { drawOnChartArea: false }
                }
            }
        }
    });
});