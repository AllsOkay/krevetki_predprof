// frontend/js/charts.js
// Модуль визуализации: создание и обновление графиков с Chart.js
// ✅ Поддержка масштабирования (zoom) и адаптивности

const charts = {};

/**
 * Инициализация всех графиков на странице
 * @param {Object} analyticsData - Данные от API /api/analytics
 */
function initCharts(analyticsData) {
    if (analyticsData.accuracy_vs_epochs) {
        initAccuracyChart(analyticsData.accuracy_vs_epochs);
    }
    
    if (analyticsData.class_distribution) {
        initClassDistributionChart(analyticsData.class_distribution);
    }
    
    if (analyticsData.per_record_accuracy) {
        initPerRecordChart(analyticsData.per_record_accuracy);
    }
    
    if (analyticsData.top5_classes) {
        initTop5Chart(analyticsData.top5_classes);
    }
    
    // ✅ Добавляем подсказку о масштабировании
    addZoomHint();
}

/**
 * Добавляет подсказку о возможности масштабирования графиков
 */
function addZoomHint() {
    const chartContainers = document.querySelectorAll('.chart-container');
    chartContainers.forEach(container => {
        if (!container.querySelector('.chart-zoom-hint')) {
            const hint = document.createElement('div');
            hint.className = 'chart-zoom-hint';
            hint.innerHTML = '<i class="fas fa-search-plus"></i> Масштабирование: колёсико мыши или щипок на сенсорном экране';
            container.appendChild(hint);
        }
    });
}

/**
 * График 1: Зависимость точности от количества эпох
 */
function initAccuracyChart(data) {
    const ctx = document.getElementById('accuracyChart');
    if (!ctx) return;
    
    if (charts.accuracy) charts.accuracy.destroy();
    
    charts.accuracy = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.epochs,
            datasets: [
                {
                    label: 'Точность (обучение)',
                    data: data.accuracy,
                    borderColor: '#FF6B9D',
                    backgroundColor: 'rgba(255, 107, 157, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.3,
                    pointRadius: 4,
                    pointHoverRadius: 6
                },
                {
                    label: 'Точность (валидация)',
                    data: data.val_accuracy || [],
                    borderColor: '#4FC3F7',
                    backgroundColor: 'rgba(79, 195, 247, 0.1)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.3,
                    pointRadius: 4,
                    pointHoverRadius: 6
                },
                // ✅ Добавляем точность на тесте если есть
                {
                    label: 'Точность (тест)',
                    data: data.test_accuracy || [],
                    borderColor: '#81C784',
                    backgroundColor: 'rgba(129, 199, 132, 0.1)',
                    borderWidth: 3,
                    borderDash: [10, 5],
                    fill: false,
                    tension: 0.3,
                    pointRadius: 5,
                    pointHoverRadius: 7
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Динамика обучения модели',
                    font: { size: 14 }
                },
                legend: { 
                    position: 'bottom',
                    labels: { usePointStyle: true }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            return context.dataset.label + ': ' + (context.parsed.y * 100).toFixed(2) + '%';
                        }
                    }
                },
                // ✅ Настройки зума
                zoom: {
                    pan: {
                        enabled: true,
                        mode: 'xy',
                    },
                    zoom: {
                        wheel: {
                            enabled: true,
                        },
                        pinch: {
                            enabled: true,
                        },
                        mode: 'xy',
                        onZoomComplete: function({chart}) {
                            chart.update('none');
                        }
                    },
                    limits: {
                        x: { min: 'original', max: 'original' },
                        y: { min: 0, max: 1 }
                    }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Эпоха' },
                    grid: { color: 'rgba(0,0,0,0.05)' }
                },
                y: {
                    title: { display: true, text: 'Точность' },
                    min: 0,
                    max: 1,
                    ticks: {
                        callback: function(value) {
                            return (value * 100).toFixed(0) + '%';
                        }
                    },
                    grid: { color: 'rgba(0,0,0,0.05)' }
                }
            },
            interaction: {
                mode: 'nearest',
                axis: 'x',
                intersect: false
            }
        }
    });
}

/**
 * График 2: Распределение классов в обучающих данных
 */
function initClassDistributionChart(data) {
    const ctx = document.getElementById('classDistributionChart');
    if (!ctx) return;
    
    if (charts.classDist) charts.classDist.destroy();
    
    const colors = [
        '#FF6B9D', '#FF8E53', '#FFB347', '#C2185B', '#4FC3F7',
        '#29B6F6', '#81C784', '#FFF176', '#BA68C8', '#7986CB'
    ];
    
    charts.classDist = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.classes.map(c => `Цивилизация #${c}`),
            datasets: [{
                label: 'Количество записей',
                data: data.counts,
                backgroundColor: data.counts.map((_, i) => colors[i % colors.length]),
                borderWidth: 1,
                borderColor: '#fff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Баланс классов в обучающей выборке',
                    font: { size: 14 }
                },
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: ctx => `Записей: ${ctx.parsed.y}`
                    }
                },
                // ✅ Зум только по Y для столбчатых диаграмм
                zoom: {
                    pan: { enabled: true, mode: 'y' },
                    zoom: {
                        wheel: { enabled: true },
                        pinch: { enabled: true },
                        mode: 'y'
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: { display: true, text: 'Количество' },
                    grid: { color: 'rgba(0,0,0,0.05)' }
                },
                x: {
                    grid: { display: false }
                }
            }
        }
    });
}

/**
 * График 3: Точность определения каждой записи из теста
 */
function initPerRecordChart(accuracies) {
    const ctx = document.getElementById('perRecordChart');
    if (!ctx) return;
    
    if (charts.perRecord) charts.perRecord.destroy();
    
    const displayCount = Math.min(100, accuracies.length);
    const labels = accuracies.slice(0, displayCount).map((_, i) => `#${i+1}`);
    const values = accuracies.slice(0, displayCount);
    
    charts.perRecord = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Правильно определено',
                data: values,
                backgroundColor: values.map(v => v === 1 ? '#81C784' : '#EF9A9A'),
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: `Точность по записям (показано ${displayCount} из ${accuracies.length})`,
                    font: { size: 14 }
                },
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: ctx => ctx.parsed.y === 1 ? '✅ Верно' : '❌ Ошибка'
                    }
                },
                // ✅ Зум по X для прокрутки записей
                zoom: {
                    pan: { enabled: true, mode: 'x' },
                    zoom: {
                        wheel: { enabled: true },
                        pinch: { enabled: true },
                        mode: 'x'
                    }
                }
            },
            scales: {
                y: {
                    min: 0,
                    max: 1,
                    ticks: {
                        callback: value => value === 1 ? '✓' : '✗'
                    },
                    grid: { color: 'rgba(0,0,0,0.05)' }
                },
                x: {
                    display: false
                }
            }
        }
    });
}

/**
 * График 4: Топ-5 наиболее частых классов в валидации
 */
function initTop5Chart(data) {
    const ctx = document.getElementById('top5Chart');
    if (!ctx) return;
    
    if (charts.top5) charts.top5.destroy();
    
    const total = data.counts.reduce((a, b) => a + b, 0);
    
    charts.top5 = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: data.classes.map(c => `Класс #${c}`),
            datasets: [{
                data: data.counts,
                backgroundColor: [
                    '#FF6B9D', '#FF8E53', '#FFB347', '#C2185B', '#4FC3F7'
                ],
                borderWidth: 2,
                borderColor: '#fff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Топ-5 классов в валидационной выборке',
                    font: { size: 14 }
                },
                legend: { 
                    position: 'right',
                    labels: { usePointStyle: true }
                },
                tooltip: {
                    callbacks: {
                        label: ctx => {
                            const value = ctx.parsed;
                            const percent = ((value / total) * 100).toFixed(1);
                            return `${ctx.label}: ${value} записей (${percent}%)`;
                        }
                    }
                }
            }
        }
    });
}

/**
 * Обновление всех графиков новыми данными
 */
function updateCharts(newData) {
    if (newData.accuracy_vs_epochs && charts.accuracy) {
        const d = newData.accuracy_vs_epochs;
        charts.accuracy.data.labels = d.epochs;
        charts.accuracy.data.datasets[0].data = d.accuracy;
        charts.accuracy.data.datasets[1].data = d.val_accuracy || [];
        if (d.test_accuracy) {
            charts.accuracy.data.datasets[2].data = d.test_accuracy;
        }
        charts.accuracy.update();
    }
    
    if (newData.class_distribution && charts.classDist) {
        const d = newData.class_distribution;
        charts.classDist.data.labels = d.classes.map(c => `Цивилизация #${c}`);
        charts.classDist.data.datasets[0].data = d.counts;
        charts.classDist.update();
    }
    
    if (newData.per_record_accuracy && charts.perRecord) {
        initPerRecordChart(newData.per_record_accuracy);
    }
}

/**
 * Уничтожение всех графиков
 */
function destroyCharts() {
    Object.values(charts).forEach(chart => chart?.destroy());
    Object.keys(charts).forEach(key => delete charts[key]);
}

/**
 * Сброс масштабирования всех графиков
 */
function resetZoom() {
    Object.values(charts).forEach(chart => {
        if (chart && chart.resetZoom) {
            chart.resetZoom();
        }
    });
}

window.Charts = {
    init: initCharts,
    update: updateCharts,
    destroy: destroyCharts,
    resetZoom: resetZoom
};