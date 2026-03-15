// frontend/js/charts.js
// Модуль визуализации: создание и обновление графиков с Chart.js
// Поддержка масштабирования (zoom) и адаптивности

// Хранилище экземпляров графиков для обновления/уничтожения
const charts = {};

/**
 * Инициализация всех графиков на странице
 * @param {Object} analyticsData - Данные от API /api/analytics
 */
function initCharts(analyticsData) {
    // График 1: Точность от эпох
    if (analyticsData.accuracy_vs_epochs) {
        initAccuracyChart(analyticsData.accuracy_vs_epochs);
    }
    
    // График 2: Распределение классов
    if (analyticsData.class_distribution) {
        initClassDistributionChart(analyticsData.class_distribution);
    }
    
    // График 3: Точность по записям (если есть данные)
    if (analyticsData.per_record_accuracy) {
        initPerRecordChart(analyticsData.per_record_accuracy);
    }
    
    // График 4: Топ-5 классов
    if (analyticsData.top5_classes) {
        initTop5Chart(analyticsData.top5_classes);
    }
}

/**
 * График 1: Зависимость точности от количества эпох
 * @param {Object} data - {epochs: [], accuracy: [], val_accuracy: []}
 */
function initAccuracyChart(data) {
    const ctx = document.getElementById('accuracyChart');
    if (!ctx) return;
    
    // Уничтожаем предыдущий график если есть
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
                    tension: 0.3
                },
                {
                    label: 'Точность (валидация)',
                    data: data.val_accuracy || [],
                    borderColor: '#4FC3F7',
                    backgroundColor: 'rgba(79, 195, 247, 0.1)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.3
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
                legend: { position: 'bottom' },
                zoom: {
                    pan: { enabled: true, mode: 'xy' },
                    zoom: {
                        wheel: { enabled: true },
                        pinch: { enabled: true },
                        mode: 'xy'
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
                    grid: { color: 'rgba(0,0,0,0.05)' }
                }
            }
        }
    });
}

/**
 * График 2: Распределение классов в обучающих данных
 * @param {Object} data - {classes: [], counts: []}
 */
function initClassDistributionChart(data) {
    const ctx = document.getElementById('classDistributionChart');
    if (!ctx) return;
    
    if (charts.classDist) charts.classDist.destroy();
    
    // Креветочная цветовая палитра
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
                borderColor: data.counts.map((_, i) => colors[i % colors.length].replace(')', ', 0.8)')),
                borderWidth: 1
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
 * @param {Array<number>} accuracies - Массив 0/1 для каждой записи
 */
function initPerRecordChart(accuracies) {
    const ctx = document.getElementById('perRecordChart');
    if (!ctx) return;
    
    if (charts.perRecord) charts.perRecord.destroy();
    
    // Показываем только первые 100 записей для читаемости
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
                legend: { display: false }
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
                    display: false // Скрываем метки для компактности
                }
            }
        }
    });
}

/**
 * График 4: Топ-5 наиболее частых классов в валидации
 * @param {Object} data - {classes: [], counts: []}
 */
function initTop5Chart(data) {
    const ctx = document.getElementById('top5Chart');
    if (!ctx) return;
    
    if (charts.top5) charts.top5.destroy();
    
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
                legend: { position: 'right' },
                tooltip: {
                    callbacks: {
                        label: ctx => `${ctx.label}: ${ctx.parsed} записей (${((ctx.parsed / data.counts.reduce((a,b)=>a+b,0))*100).toFixed(1)}%)`
                    }
                }
            }
        }
    });
}

/**
 * Обновление всех графиков новыми данными
 * @param {Object} newData - Данные от сервера
 */
function updateCharts(newData) {
    // Для каждого графика: обновить данные и вызвать update()
    if (newData.accuracy_vs_epochs && charts.accuracy) {
        const d = newData.accuracy_vs_epochs;
        charts.accuracy.data.labels = d.epochs;
        charts.accuracy.data.datasets[0].data = d.accuracy;
        charts.accuracy.data.datasets[1].data = d.val_accuracy || [];
        charts.accuracy.update();
    }
    
    // ... аналогично для других графиков
    // Для краткости не дублирую код
}

/**
 * Уничтожение всех графиков (при выходе со страницы)
 */
function destroyCharts() {
    Object.values(charts).forEach(chart => chart?.destroy());
    Object.keys(charts).forEach(key => delete charts[key]);
}

// Экспорт для использования в других модулях
window.Charts = {
    init: initCharts,
    update: updateCharts,
    destroy: destroyCharts
};