// frontend/js/charts.js
// Модуль визуализации: создание и обновление графиков с Chart.js

const charts = {};

/**
 * Инициализация всех графиков на странице
 * @param {Object} analyticsData - Данные от API /api/analytics
 */
function initCharts(analyticsData) {
    console.log('📊 initCharts вызван с:', analyticsData);
    
    // 🔥 Проверка на пустые данные
    if (!analyticsData?.accuracy_vs_epochs?.accuracy?.length) {
        console.warn('⚠️ Нет данных для accuracy_vs_epochs — график будет пустым');
    }
    
    // 🔥 Проверка что Chart доступен
    if (typeof Chart === 'undefined') {
        console.error('❌ Chart.js не загружен! Проверь CDN и сеть.');
        document.querySelectorAll('.chart-container').forEach(c => {
            c.innerHTML = '<div class="status-badge status-error">❌ Chart.js не загружен</div>';
        });
        return;
    }
    
    if (!analyticsData) {
        console.warn('⚠️ Нет данных для графиков');
        return;
    }
    
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
    
    console.log('✅ Графики инициализированы');
}

/**
 * График 1: Зависимость точности от количества эпох
 */
function initAccuracyChart(data) {
    const ctx = document.getElementById('accuracyChart');
    if (!ctx) {
        console.warn('⚠️ Canvas accuracyChart не найден');
        return;
    }
    
    if (charts.accuracy) {
        charts.accuracy.destroy();
    }
    
    charts.accuracy = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.epochs || [],
            datasets: [
                {
                    label: 'Точность (обучение)',
                    data: data.accuracy || [],
                    borderColor: '#FF6B9D',
                    backgroundColor: 'rgba(255, 107, 157, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.3,
                    pointRadius: 3,
                    pointHoverRadius: 5
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
                    pointRadius: 3,
                    pointHoverRadius: 5
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
                    callbacks: {
                        label: function(context) {
                            return context.dataset.label + ': ' + (context.parsed.y * 100).toFixed(2) + '%';
                        }
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
            }
        }
    });
    
    console.log('✅ График точности создан');
}

/**
 * График 2: Распределение классов в обучающих данных
 */
function initClassDistributionChart(data) {
    const ctx = document.getElementById('classDistributionChart');
    if (!ctx) {
        console.warn('⚠️ Canvas classDistributionChart не найден');
        return;
    }
    
    if (charts.classDist) {
        charts.classDist.destroy();
    }
    
    const colors = [
        '#FF6B9D', '#FF8E53', '#FFB347', '#C2185B', '#4FC3F7',
        '#29B6F6', '#81C784', '#FFF176', '#BA68C8', '#7986CB'
    ];
    
    charts.classDist = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.classes.map(c => `Класс #${c}`),
            datasets: [{
                label: 'Количество записей',
                data: data.counts || [],
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
    
    console.log('✅ График распределения создан');
}

/**
 * График 3: Точность определения каждой записи из теста
 */
function initPerRecordChart(accuracies) {
    const ctx = document.getElementById('perRecordChart');
    if (!ctx) {
        console.warn('⚠️ Canvas perRecordChart не найден');
        return;
    }
    
    if (charts.perRecord) {
        charts.perRecord.destroy();
    }
    
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
    
    console.log('✅ График по записям создан');
}

/**
 * График 4: Топ-5 наиболее частых классов в валидации
 */
function initTop5Chart(data) {
    const ctx = document.getElementById('top5Chart');
    if (!ctx) {
        console.warn('⚠️ Canvas top5Chart не найден');
        return;
    }
    
    if (charts.top5) {
        charts.top5.destroy();
    }
    
    const total = data.counts.reduce((a, b) => a + b, 0);
    
    charts.top5 = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: data.classes.map(c => `Класс #${c}`),
            datasets: [{
                data: data.counts || [],
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
                            const percent = total > 0 ? ((value / total) * 100).toFixed(1) : 0;
                            return `${ctx.label}: ${value} записей (${percent}%)`;
                        }
                    }
                }
            }
        }
    });
    
    console.log('✅ График топ-5 создан');
}

/**
 * Обновление всех графиков новыми данными
 */
/**
 * Обновление всех графиков новыми данными
 */
function updateCharts(newData) {
    console.log('🔄 Обновление графиков...', newData);
    
    if (!newData) return;
    
    // График 1: Точность по эпохам
    if (newData.accuracy_vs_epochs && charts.accuracy) {
        const d = newData.accuracy_vs_epochs;
        if (d.epochs?.length > 0) {
            charts.accuracy.data.labels = d.epochs;
            charts.accuracy.data.datasets[0].data = d.accuracy || [];
            charts.accuracy.data.datasets[1].data = d.val_accuracy || [];
            charts.accuracy.update('none');
            console.log('✅ График точности обновлён');
        }
    }
    
    // График 2: Распределение классов
    if (newData.class_distribution && charts.classDist) {
        const d = newData.class_distribution;
        if (d.classes?.length > 0) {
            charts.classDist.data.labels = d.classes.map(c => `Класс #${c}`);
            charts.classDist.data.datasets[0].data = d.counts || [];
            // Обновляем цвета динамически
            const colors = ['#FF6B9D', '#FF8E53', '#FFB347', '#C2185B', '#4FC3F7', '#29B6F6', '#81C784', '#FFF176', '#BA68C8', '#7986CB'];
            charts.classDist.data.datasets[0].backgroundColor = d.counts.map((_, i) => colors[i % colors.length]);
            charts.classDist.update('none');
            console.log('✅ График распределения обновлён');
        }
    }
    
    // График 3: Точность по записям — ✅ ПРАВИЛЬНОЕ ОБНОВЛЕНИЕ
    if (newData.per_record_accuracy && Array.isArray(newData.per_record_accuracy)) {
        if (charts.perRecord) {
            const displayCount = Math.min(100, newData.per_record_accuracy.length);
            charts.perRecord.data.labels = newData.per_record_accuracy.slice(0, displayCount).map((_, i) => `#${i+1}`);
            const values = newData.per_record_accuracy.slice(0, displayCount);
            charts.perRecord.data.datasets[0].data = values;
            charts.perRecord.data.datasets[0].backgroundColor = values.map(v => v === 1 ? '#81C784' : '#EF9A9A');
            // Обновляем заголовок
            charts.perRecord.options.plugins.title.text = `Точность по записям (показано ${displayCount} из ${newData.per_record_accuracy.length})`;
            charts.perRecord.update('none');
        } else if (newData.per_record_accuracy.length > 0) {
            // Если график ещё не создан, но есть данные — создаём
            initPerRecordChart(newData.per_record_accuracy);
        }
        console.log('✅ График по записям обновлён');
    }
    
    // График 4: Топ-5 классов
    if (newData.top5_classes && charts.top5) {
        const d = newData.top5_classes;
        if (d.classes?.length > 0) {
            charts.top5.data.labels = d.classes.map(c => `Класс #${c}`);
            charts.top5.data.datasets[0].data = d.counts || [];
            charts.top5.update('none');
            console.log('✅ График топ-5 обновлён');
        }
    }
}
/**
 * Уничтожение всех графиков
 */
function destroyCharts() {
    Object.values(charts).forEach(chart => {
        if (chart) chart.destroy();
    });
    Object.keys(charts).forEach(key => delete charts[key]);
    console.log('🗑️ Графики уничтожены');
}

/**
 * Проверка что Chart.js загружен
 */
function checkChartJS() {
    if (typeof Chart === 'undefined') {
        console.error('❌ Chart.js не загружен! Проверьте подключение CDN');
        return false;
    }
    console.log('✅ Chart.js загружен');
    return true;
}

// Экспорт для использования в других модулях
window.Charts = {
    init: initCharts,
    update: updateCharts,
    destroy: destroyCharts,
    checkChartJS: checkChartJS
};

// Автопроверка при загрузке
document.addEventListener('DOMContentLoaded', function() {
    checkChartJS();
});