/**
 * 🦐 Pokémon AI Search - Shrimp Style
 * Минималистичная версия с адаптацией под мобильные
 */

document.addEventListener('DOMContentLoaded', () => {
    console.log('🦐 Shrimp Style загружен');
    
    // DOM Элементы
    const addAllBtn = document.getElementById('add-all-btn');
    const trainCnnBtn = document.getElementById('train-cnn-btn');
    const trainRnnBtn = document.getElementById('train-rnn-btn');
    const trainAeBtn = document.getElementById('train-ae-btn');
    const trainMlpBtn = document.getElementById('train-mlp-btn');
    const clearBtn = document.getElementById('clear-btn');
    const refreshBtn = document.getElementById('refresh-btn');
    
    const cnnInput = document.getElementById('cnn-input');
    const cnnSearchBtn = document.getElementById('cnn-search-btn');
    const cnnResults = document.getElementById('cnn-results');
    const cnnLoading = document.getElementById('cnn-loading');
    
    const rnnInput = document.getElementById('rnn-input');
    const rnnClassifyBtn = document.getElementById('rnn-classify-btn');
    const rnnResults = document.getElementById('rnn-results');
    const rnnLoading = document.getElementById('rnn-loading');
    
    const aeInput = document.getElementById('ae-input');
    const aeCheckBtn = document.getElementById('ae-check-btn');
    const aeResults = document.getElementById('ae-results');
    const aeLoading = document.getElementById('ae-loading');
    
    const mlpHpInput = document.getElementById('mlp-hp');
    const mlpAtkInput = document.getElementById('mlp-atk');
    const mlpDefInput = document.getElementById('mlp-def');
    const mlpSpdInput = document.getElementById('mlp-spd');
    const mlpPredictBtn = document.getElementById('mlp-predict-btn');
    const mlpResults = document.getElementById('mlp-results');
    const mlpLoading = document.getElementById('mlp-loading');
    
    const graphSection = document.getElementById('graph-section');
    const graphModelSelect = document.getElementById('graph-model-select');
    const view2dBtn = document.getElementById('view-2d-btn');
    const view3dBtn = document.getElementById('view-3d-btn');
    const refreshGraphBtn = document.getElementById('refresh-graph-btn');
    const graphLoading = document.getElementById('graph-loading');
    const graph2dContainer = document.getElementById('graph-2d-container');
    const graph3dContainer = document.getElementById('graph-3d-container');
    const graphStats = document.getElementById('graph-stats');
    
    const progress = document.getElementById('progress');
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');
    
    const pokemonTableBody = document.getElementById('pokemon-table-body');
    const dbCount = document.getElementById('db-count');
    const statusText = document.getElementById('status-text');
    const statusDot = document.querySelector('.status-dot');
    
    let chart2d = null;
    let currentModelType = 'cnn';
    let currentView = '2d';
    
    // Инициализация
    checkServerHealth();
    loadPokemon();
    loadAutocomplete();
    loadTrainingGraph();
    
    // Обработчики
    if (addAllBtn) addAllBtn.addEventListener('click', addAllPokemon);
    if (trainCnnBtn) trainCnnBtn.addEventListener('click', () => trainModel('cnn'));
    if (trainRnnBtn) trainRnnBtn.addEventListener('click', () => trainModel('rnn'));
    if (trainAeBtn) trainAeBtn.addEventListener('click', () => trainModel('autoencoder'));
    if (trainMlpBtn) trainMlpBtn.addEventListener('click', () => trainModel('mlp'));
    if (clearBtn) clearBtn.addEventListener('click', clearAllData);
    if (refreshBtn) refreshBtn.addEventListener('click', loadPokemon);
    
    if (cnnSearchBtn) cnnSearchBtn.addEventListener('click', findSimilarCNN);
    if (cnnInput) cnnInput.addEventListener('keypress', e => { if (e.key === 'Enter') findSimilarCNN(); });
    
    if (rnnClassifyBtn) rnnClassifyBtn.addEventListener('click', classifyPokemonRNN);
    if (rnnInput) rnnInput.addEventListener('keypress', e => { if (e.key === 'Enter') classifyPokemonRNN(); });
    
    if (aeCheckBtn) aeCheckBtn.addEventListener('click', checkPokemonOrdinariness);
    if (aeInput) aeInput.addEventListener('keypress', e => { if (e.key === 'Enter') checkPokemonOrdinariness(); });
    
    if (mlpPredictBtn) mlpPredictBtn.addEventListener('click', predictWinProbability);
    
    if (graphModelSelect) graphModelSelect.addEventListener('change', e => { currentModelType = e.target.value; loadTrainingGraph(); });
    if (view2dBtn) view2dBtn.addEventListener('click', () => switchGraphView('2d'));
    if (view3dBtn) view3dBtn.addEventListener('click', () => switchGraphView('3d'));
    if (refreshGraphBtn) refreshGraphBtn.addEventListener('click', () => {
        console.log('🔄 Обновление графика для:', currentModelType);
        loadTrainingGraph();
    });
    
    // Функции
    async function checkServerHealth() {
        try {
            const res = await fetch('/api/health');
            const data = await res.json();
            if (data.status === 'ok') {
                statusText.textContent = `OK (${data.pokemon_count || 0})`;
                statusDot.classList.add('connected');
            } else {
                statusText.textContent = 'Ошибка';
                statusDot.classList.remove('connected');
            }
        } catch (e) {
            statusText.textContent = 'Нет связи';
            statusDot.classList.remove('connected');
        }
    }
    
    async function loadPokemon() {
        try {
            const res = await fetch('/api/pokemon');
            const list = await res.json();
            pokemonTableBody.innerHTML = '';
            dbCount.textContent = list.length;
            
            list.forEach(p => {
                const row = document.createElement('tr');
                row.innerHTML = `<td>${p.id}</td><td>${p.name}</td><td>${p.types || '-'}</td><td>${p.hp||0}</td><td>${p.attack||0}</td><td>${p.defense||0}</td><td>${p.speed||0}</td>`;
                pokemonTableBody.appendChild(row);
            });
        } catch (e) {
            console.error(e);
            dbCount.textContent = '0';
        }
    }
    
    async function addAllPokemon() {
        if (!addAllBtn) return;
        addAllBtn.disabled = true;
        progress.classList.remove('hidden');
        updateProgress(0, 'Загрузка...');
        
        try {
            const limit = 1302;
            const res = await fetch(`https://pokeapi.co/api/v2/pokemon?limit=${limit}`);
            const data = await res.json();
            
            const existing = await (await fetch('/api/pokemon')).json();
            const existingIds = existing.map(p => p.id);
            
            let processed = 0, success = 0;
            
            for (let i = 0; i < data.results.length; i += 5) {
                const batch = data.results.slice(i, i + 5);
                await Promise.all(batch.map(async (p, idx) => {
                    const id = i + idx + 1;
                    if (existingIds.includes(id)) { processed++; return; }
                    
                    try {
                        const detail = await (await fetch(`https://pokeapi.co/api/v2/pokemon/${id}`)).json();
                        await fetch('/api/pokemon', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                id: detail.id, name: detail.name,
                                types: detail.types.map(t => t.type.name).join(', '),
                                hp: detail.stats.find(s => s.stat.name === 'hp')?.base_stat || 0,
                                attack: detail.stats.find(s => s.stat.name === 'attack')?.base_stat || 0,
                                defense: detail.stats.find(s => s.stat.name === 'defense')?.base_stat || 0,
                                speed: detail.stats.find(s => s.stat.name === 'speed')?.base_stat || 0,
                                sprite_url: detail.sprites.front_default
                            })
                        });
                        success++;
                    } catch (e) { console.warn(e); }
                    processed++;
                    updateProgress(Math.round(processed / data.results.length * 100), `${processed}/${data.results.length}`);
                }));
                await sleep(100);
            }
            
            updateProgress(100, `✅ ${success}`);
            loadPokemon();
            setTimeout(() => progress.classList.add('hidden'), 2000);
        } catch (e) {
            alert('Ошибка: ' + e.message);
        } finally {
            addAllBtn.disabled = false;
        }
    }
    
    function updateProgress(percent, text) {
        progressFill.style.width = `${percent}%`;
        progressText.textContent = `${percent}% ${text}`;
    }
    
    async function trainModel(type) {
        const btn = document.getElementById(`train-${type}-btn`);
        if (!btn) return;
        
        // Маппинг типов к ID кнопок
        const btnIds = {
            'cnn': 'train-cnn-btn',
            'rnn': 'train-rnn-btn',
            'autoencoder': 'train-ae-btn',
            'mlp': 'train-mlp-btn'
        };
        
        btn.disabled = true;
        btn.textContent = '⏳';
        
        try {
            console.log(`📡 Отправка POST запроса: /api/${type}/train`);
            const res = await fetch(`/api/${type}/train`, { method: 'POST' });
            console.log(`📥 Ответ сервера: ${res.status} ${res.statusText}`);
            
            const result = await res.json();
            console.log('📦 Данные ответа:', result);
            
            if (result.success) {
                let message = `✅ ${type.toUpperCase()} обучена!\nЭпох: ${result.epochs}`;
                
                if (result.final_loss !== undefined && result.final_loss !== null) {
                    message += `\nLoss: ${result.final_loss.toFixed(4)}`;
                }
                if (result.final_accuracy !== undefined && result.final_accuracy !== null) {
                    message += `\nТочность: ${(result.final_accuracy * 100).toFixed(1)}%`;
                }
                
                alert(message + '\n\n💡 График обучения открыт ниже');
                showGraphSection(type);
            } else {
                console.warn('⚠️ Ошибка от сервера:', result);
                let errorMsg = `⚠️ Ошибка: ${result.reason || 'Неизвестная ошибка'}`;
                if (result.current_count && result.required_count) {
                    errorMsg += `\n\n📊 В БД: ${result.current_count} покемонов`;
                    errorMsg += `\n🎯 Нужно минимум: ${result.required_count}`;
                }
                alert(errorMsg);
            }
        } catch (e) {
            console.error(`❌ Ошибка обучения ${type}:`, e);
            alert(`Ошибка: ${e.message}`);
        } finally {
            btn.disabled = false;
            const icons = { cnn: '🎓', rnn: '🧠', autoencoder: '🔄', mlp: '⚔️' };
            btn.textContent = icons[type] || '🔄';
        }
    }
    
    function showGraphSection(modelType) {
        if (graphSection) {
            graphSection.classList.remove('hidden');
            currentModelType = modelType;
            if (graphModelSelect) graphModelSelect.value = modelType;
            loadTrainingGraph();
            graphSection.scrollIntoView({ behavior: 'smooth' });
        }
    }
    
    function switchGraphView(view) {
        currentView = view;
        if (view2dBtn) view2dBtn.classList.toggle('active', view === '2d');
        if (view3dBtn) view3dBtn.classList.toggle('active', view === '3d');
        
        if (view === '2d') {
            if (graph2dContainer) graph2dContainer.classList.remove('hidden');
            if (graph3dContainer) graph3dContainer.classList.add('hidden');
            loadGraph2D();
        } else {
            if (graph2dContainer) graph2dContainer.classList.add('hidden');
            if (graph3dContainer) graph3dContainer.classList.remove('hidden');
            loadGraph3D();
        }
    }
    
    async function loadTrainingGraph() {
        if (graphLoading) graphLoading.classList.remove('hidden');
        
        try {
            console.log(`📊 Загрузка графика для: ${currentModelType}`);
            
            // Загружаем 2D метрики
            const m1 = await fetch(`/api/metrics/${currentModelType}`);
            console.log(`2D статус: ${m1.status}`);
            
            if (!m1.ok) {
                throw new Error(`2D метрики: HTTP ${m1.status}`);
            }
            
            const data = await m1.json();
            console.log('📦 Данные 2D:', data);
            
            // Проверяем наличие данных
            if (!data.metrics || data.metrics.length === 0) {
                if (graphStats) {
                    graphStats.innerHTML = `
                        <p class="text-muted">
                            Нет данных обучения для <strong>${currentModelType.toUpperCase()}</strong>.
                            <br>Обучите модель, чтобы увидеть график.
                        </p>`;
                }
                showGraphPlaceholder('2d', `Нет данных для ${currentModelType.toUpperCase()}`);
                showGraphPlaceholder('3d', `Нет данных для ${currentModelType.toUpperCase()}`);
                if (graphLoading) graphLoading.classList.add('hidden');
                return;
            }
            
            // Обновить статистику
            if (data.summary) {
                updateGraphStats(data.summary, currentModelType);
            }
            
            // Отрисовать 2D график
            if (currentView === '2d') {
                loadGraph2D(data.metrics, currentModelType);
            }
            
            // 3D график - опционально (может не быть на сервере)
            if (currentView === '3d') {
                try {
                    const m2 = await fetch(`/api/metrics/3d/${currentModelType}`);
                    console.log(`3D статус: ${m2.status}`);
                    
                    if (m2.ok) {
                        const data3d = await m2.json();
                        console.log('📦 Данные 3D:', data3d);
                        loadGraph3D(data3d, currentModelType);
                    } else {
                        console.warn('⚠️ 3D endpoint не доступен, показываем 2D');
                        showGraphPlaceholder('3d', '3D график недоступен. Используйте 2D режим.');
                        // Автоматически переключаемся на 2D
                        if (view2dBtn) view2dBtn.click();
                    }
                } catch (e3d) {
                    console.warn('⚠️ Ошибка 3D:', e3d);
                    showGraphPlaceholder('3d', '3D график недоступен');
                    if (view2dBtn) view2dBtn.click();
                }
            }
            
        } catch (e) {
            console.error('❌ Ошибка загрузки графика:', e);
            if (graphStats) {
                graphStats.innerHTML = `
                    <p class="error">
                        ❌ Ошибка загрузки: ${e.message}<br>
                        <small>Проверьте консоль (F12) для деталей</small>
                    </p>`;
            }
        } finally {
            if (graphLoading) graphLoading.classList.add('hidden');
        }
    }
    
    function showGraphPlaceholder(view, message) {
        if (view === '2d') {
            const ctx = document.getElementById('training-chart-2d');
            if (ctx && ctx.parentElement) {
                ctx.parentElement.innerHTML = `<div class="graph-placeholder">${message}</div>`;
            }
        } else {
            const container = document.getElementById('training-chart-3d');
            if (container) {
                container.innerHTML = `<div class="graph-placeholder">${message}</div>`;
            }
        }
    }
    
    function loadGraph2D(metrics, modelType) {
        const ctx = document.getElementById('training-chart-2d');
        if (!ctx || typeof Chart === 'undefined') {
            console.error('❌ Chart.js не доступен или canvas не найден');
            return;
        }
        
        // Проверка данных
        if (!metrics || metrics.length === 0) {
            showGraphPlaceholder('2d', 'Нет данных для отображения');
            if (chart2d) {
                chart2d.destroy();
                chart2d = null;
            }
            return;
        }
        
        // Восстановить canvas если был заменен
        if (!ctx.tagName || ctx.tagName.toLowerCase() !== 'canvas') {
            ctx.parentElement.innerHTML = '<canvas id="training-chart-2d"></canvas>';
            return loadGraph2D(metrics, modelType);
        }
        
        if (chart2d) chart2d.destroy();
        
        const epochs = metrics.map(m => m.epoch);
        const losses = metrics.map(m => m.loss !== null ? m.loss : 0);
        const modelConfig = getModelChartConfig(modelType);
        
        console.log('📈 2D данные:', { epochs: epochs.length, losses: losses.length });
        
        const datasets = [{
            label: modelConfig.lossLabel,
            data: losses,
            borderColor: '#ff6b8a',
            backgroundColor: 'rgba(255, 107, 138, 0.1)',
            fill: true,
            tension: 0.4,
            yAxisID: 'y'
        }];
        
        // Добавить точность если есть
        const accuracies = metrics.map(m => m.accuracy !== null && m.accuracy !== undefined ? m.accuracy * 100 : null);
        const hasAccuracy = accuracies.some(a => a !== null && !isNaN(a));
        
        if (hasAccuracy) {
            datasets.push({
                label: modelConfig.accuracyLabel,
                data: accuracies,
                borderColor: '#22c55e',
                borderWidth: 2,
                borderDash: [5, 5],
                pointRadius: 3,
                tension: 0.4,
                yAxisID: 'y1'
            });
        }
        
        try {
            chart2d = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: epochs,
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: { mode: 'index', intersect: false },
                    scales: {
                        x: { 
                            title: { display: true, text: 'Эпоха' },
                            grid: { color: 'rgba(0, 0, 0, 0.05)' },
                            beginAtZero: true
                        },
                        y: { 
                            type: 'linear',
                            display: true,
                            position: 'left',
                            title: { display: true, text: modelConfig.lossLabel },
                            grid: { color: 'rgba(255, 107, 138, 0.1)' },
                            min: 0,
                            beginAtZero: true
                        },
                        y1: {
                            type: 'linear',
                            display: hasAccuracy,
                            position: 'right',
                            title: { display: true, text: modelConfig.accuracyLabel },
                            grid: { drawOnChartArea: false },
                            min: 0,
                            max: 100,
                            beginAtZero: true,
                            ticks: { callback: (value) => value + '%' }
                        }
                    },
                    plugins: {
                        legend: { position: 'top' },
                        tooltip: {
                            callbacks: {
                                label: (context) => {
                                    let label = context.dataset.label || '';
                                    if (label) label += ': ';
                                    if (context.parsed.y !== null && !isNaN(context.parsed.y)) {
                                        if (context.dataset.yAxisID === 'y1') {
                                            label += context.parsed.y.toFixed(1) + '%';
                                        } else {
                                            label += context.parsed.y.toFixed(4);
                                        }
                                    }
                                    return label;
                                }
                            }
                        }
                    }
                }
            });
            console.log('✅ 2D график создан');
        } catch (e) {
            console.error('❌ Ошибка создания 2D графика:', e);
            showGraphPlaceholder('2d', 'Ошибка отрисовки графика');
        }
    }
    
    function loadGraph3D(data3d, modelType) {
        const container = document.getElementById('training-chart-3d');
        if (!container || typeof Plotly === 'undefined') {
            console.error('❌ Plotly не доступен или контейнер не найден');
            return;
        }
        
        // Проверка данных перед передачей в Plotly
        if (!data3d || !data3d.data_3d || 
            !data3d.data_3d.x || data3d.data_3d.x.length === 0 ||
            !data3d.data_3d.y || data3d.data_3d.y.length === 0 ||
            !data3d.data_3d.z || data3d.data_3d.z.length === 0) {
            console.warn('⚠️ Нет данных для 3D графика');
            showGraphPlaceholder('3d', 'Нет данных для 3D отображения');
            Plotly.purge(container);
            return;
        }
        
        // Проверка на NaN и null значения
        const x = data3d.data_3d.x.filter(v => v !== null && v !== undefined && !isNaN(v));
        const y = data3d.data_3d.y.filter(v => v !== null && v !== undefined && !isNaN(v));
        const z = data3d.data_3d.z.filter(v => v !== null && v !== undefined && !isNaN(v));
        
        if (x.length === 0 || y.length === 0 || z.length === 0) {
            console.warn('⚠️ Все данные содержат NaN/null');
            showGraphPlaceholder('3d', 'Некорректные данные для 3D');
            Plotly.purge(container);
            return;
        }
        
        // Выравнивание массивов по минимальной длине
        const minLength = Math.min(x.length, y.length, z.length);
        const safeX = x.slice(0, minLength);
        const safeY = y.slice(0, minLength);
        const safeZ = z.slice(0, minLength);
        
        console.log('📈 3D данные:', { x: safeX.length, y: safeY.length, z: safeZ.length });
        
        const modelConfig = getModelChartConfig(modelType);
        
        try {
            Plotly.newPlot(container, [{
                x: safeX,
                y: safeY,
                z: safeZ,
                mode: 'lines+markers',
                type: 'scatter3d',
                name: modelConfig.lossLabel,
                marker: {
                    size: 4,
                    color: safeZ,
                    colorscale: data3d.layout_hints?.color_scale || 'Viridis',
                    opacity: 0.8,
                    colorbar: { 
                        title: modelConfig.accuracyLabel || 'Метрика',
                        titleside: 'right'
                    }
                },
                line: { 
                    color: '#ff6b8a', 
                    width: 2 
                },
                text: data3d.data_3d.text || safeX.map((v, i) => `Эпоха ${v}: Loss=${safeY[i]?.toFixed(4)}`),
                hoverinfo: 'text+x+y+z'
            }], {
                margin: { l: 0, r: 0, b: 0, t: 0 },
                scene: {
                    xaxis: { 
                        title: data3d.layout_hints?.x_title || 'Эпоха',
                        gridcolor: '#eee',
                        zeroline: false,
                        showbackground: true,
                        backgroundcolor: 'rgba(240, 240, 240, 0.5)'
                    },
                    yaxis: { 
                        title: data3d.layout_hints?.y_title || 'Loss',
                        gridcolor: '#eee',
                        zeroline: false,
                        showbackground: true,
                        backgroundcolor: 'rgba(240, 240, 240, 0.5)'
                    },
                    zaxis: { 
                        title: data3d.layout_hints?.z_title || 'Metric',
                        gridcolor: '#eee',
                        zeroline: false,
                        showbackground: true,
                        backgroundcolor: 'rgba(240, 240, 240, 0.5)'
                    },
                    camera: { 
                        eye: { x: 1.5, y: 1.5, z: 1.5 } 
                    },
                    aspectmode: 'data'
                },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { 
                    family: 'Segoe UI, sans-serif', 
                    size: 12 
                }
            }, {
                responsive: true,
                displayModeBar: true,
                modeBarButtonsToAdd: ['hoverClosest3d', 'zoomIn3d', 'zoomOut3d', 'resetCameraDefault3d'],
                scrollZoom: true
            });
            console.log('✅ 3D график создан');
        } catch (e) {
            console.error('❌ Ошибка создания 3D графика:', e);
            showGraphPlaceholder('3d', `Ошибка 3D: ${e.message}`);
            Plotly.purge(container);
        }
    }
    
    function updateGraphStats(summary, type) {
        if (!graphStats) return;
        graphStats.innerHTML = `
            <div class="stat-card"><div class="value">${summary.total_epochs || 0}</div><div class="label">Эпох</div></div>
            <div class="stat-card"><div class="value">${summary.final_loss?.toFixed(3) || '-'}</div><div class="label">Loss</div></div>
            <div class="stat-card"><div class="value">${summary.final_accuracy ? (summary.final_accuracy * 100).toFixed(1) + '%' : '-'}</div><div class="label">Точность</div></div>
        `;
    }
    
    function getModelChartConfig(modelType) {
        const configs = {
            'cnn': { 
                lossLabel: 'Contrastive Loss', 
                accuracyLabel: 'Embedding Distance' 
            },
            'rnn': { 
                lossLabel: 'Cross-Entropy Loss', 
                accuracyLabel: 'Classification Accuracy' 
            },
            'autoencoder': { 
                lossLabel: 'Reconstruction MSE', 
                accuracyLabel: 'Ordinariness Score' 
            },
            'mlp': { 
                lossLabel: 'Binary Cross-Entropy', 
                accuracyLabel: 'Win Prediction Accuracy' 
            }
        };
        return configs[modelType] || configs['cnn'];
    }
    
    async function clearAllData() {
        if (!confirm('Удалить всё?')) return;
        try {
            await fetch('/api/pokemon', { method: 'DELETE' });
            loadPokemon();
            if (graphSection) graphSection.classList.add('hidden');
            alert('✅ Удалено');
        } catch (e) {
            alert('Ошибка: ' + e.message);
        }
    }
    
    async function findSimilarCNN() {
        const name = cnnInput?.value.trim().toLowerCase();
        if (!name) { alert('Введите имя!'); return; }
        
        if (cnnLoading) cnnLoading.classList.remove('hidden');
        if (cnnResults) cnnResults.classList.remove('hidden');
        if (cnnResults) cnnResults.innerHTML = '<div class="loading">⏳</div>';
        
        try {
            const res = await fetch(`/api/cnn/similar?name=${encodeURIComponent(name)}&k=5`);
            const data = await res.json();
            
            if (data.error) {
                if (cnnResults) cnnResults.innerHTML = `<p>❌ ${data.error}</p>`;
                return;
            }
            
            let html = `<div class="result-card target-card"><img src="${data.target.sprite_url || ''}" alt="${data.target.name}"><div class="name">${data.target.name}</div></div>`;
            
            if (data.similar?.length) {
                data.similar.forEach((p, i) => {
                    html += `<div class="result-card"><img src="${p.sprite_url || ''}" alt="${p.name}"><div class="name">${p.name}</div><div class="stat">${Math.round(p.similarity * 100)}%</div></div>`;
                });
            } else {
                html += '<p>Нет похожих</p>';
            }
            
            if (cnnResults) cnnResults.innerHTML = html;
        } catch (e) {
            if (cnnResults) cnnResults.innerHTML = `<p>❌ ${e.message}</p>`;
        } finally {
            if (cnnLoading) cnnLoading.classList.add('hidden');
        }
    }
    
    async function classifyPokemonRNN() {
        const name = rnnInput?.value.trim().toLowerCase();
        if (!name) { alert('Введите имя!'); return; }
        
        if (rnnLoading) rnnLoading.classList.remove('hidden');
        if (rnnResults) rnnResults.classList.remove('hidden');
        if (rnnResults) rnnResults.innerHTML = '<div class="loading">⏳</div>';
        
        try {
            const res = await fetch(`/api/rnn/classify?name=${encodeURIComponent(name)}`);
            const data = await res.json();
            
            if (data.error) {
                if (rnnResults) rnnResults.innerHTML = `<p>❌ ${data.error}</p>`;
                return;
            }
            
            const prob = data.result.probabilities || [0, 0, 0];
            if (rnnResults) {
                rnnResults.innerHTML = `
                    <div class="result-card target-card">
                        <img src="${data.pokemon.sprite_url || ''}" alt="${data.pokemon.name}">
                        <div class="name">${data.pokemon.name}</div>
                        <div class="stat">${data.result.class_ru || ''}</div>
                        <div class="stat">BST: ${data.pokemon.bst}</div>
                        <div class="stat">🔹 ${Math.round(prob[0]*100)}% | 🔸 ${Math.round(prob[1]*100)}% | 🔴 ${Math.round(prob[2]*100)}%</div>
                    </div>
                `;
            }
        } catch (e) {
            if (rnnResults) rnnResults.innerHTML = `<p>❌ ${e.message}</p>`;
        } finally {
            if (rnnLoading) rnnLoading.classList.add('hidden');
        }
    }
    
    async function checkPokemonOrdinariness() {
        const name = aeInput?.value.trim().toLowerCase();
        if (!name) { alert('Введите имя!'); return; }
        
        if (aeLoading) aeLoading.classList.remove('hidden');
        if (aeResults) aeResults.classList.remove('hidden');
        if (aeResults) aeResults.innerHTML = '<div class="loading">⏳</div>';
        
        try {
            const res = await fetch(`/api/autoencoder/ordinariness?name=${encodeURIComponent(name)}`);
            const data = await res.json();
            
            if (data.error) {
                if (aeResults) aeResults.innerHTML = `<p>❌ ${data.error}</p>`;
                return;
            }
            
            const ord = data.result.ordinariness || 0;
            if (aeResults) {
                aeResults.innerHTML = `
                    <div class="result-card target-card">
                        <img src="${data.pokemon.sprite_url || ''}" alt="${data.pokemon.name}">
                        <div class="name">${data.pokemon.name}</div>
                        <div class="stat" style="font-size: 24px; color: #ff6b8a;">${ord}%</div>
                        <div class="stat">${data.result.description || 'обычности'}</div>
                    </div>
                `;
            }
        } catch (e) {
            if (aeResults) aeResults.innerHTML = `<p>❌ ${e.message}</p>`;
        } finally {
            if (aeLoading) aeLoading.classList.add('hidden');
        }
    }
    
    async function predictWinProbability() {
        const hp = parseFloat(mlpHpInput?.value) || 0;
        const atk = parseFloat(mlpAtkInput?.value) || 0;
        const def_ = parseFloat(mlpDefInput?.value) || 0;
        const spd = parseFloat(mlpSpdInput?.value) || 0;
        
        if (!hp && !atk && !def_ && !spd) { alert('Введите статы!'); return; }
        
        if (mlpLoading) mlpLoading.classList.remove('hidden');
        if (mlpResults) mlpResults.classList.remove('hidden');
        if (mlpResults) mlpResults.innerHTML = '<div class="loading">⏳</div>';
        
        try {
            const res = await fetch(`/api/mlp/predict?hp=${hp}&atk=${atk}&def=${def_}&spd=${spd}`);
            const data = await res.json();
            
            if (data.error) {
                if (mlpResults) mlpResults.innerHTML = `<p>❌ ${data.error}</p>`;
                return;
            }
            
            const prob = data.result.win_probability || 0;
            if (mlpResults) {
                mlpResults.innerHTML = `
                    <div class="result-card target-card">
                        <div class="stat" style="font-size: 32px; color: #ff6b8a;">${prob}%</div>
                        <div class="stat">победы</div>
                        <div class="stat">${data.result.classification || ''}</div>
                        <div class="stat">${data.result.description || ''}</div>
                    </div>
                `;
            }
        } catch (e) {
            if (mlpResults) mlpResults.innerHTML = `<p>❌ ${e.message}</p>`;
        } finally {
            if (mlpLoading) mlpLoading.classList.add('hidden');
        }
    }
    
    function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
    
    async function loadAutocomplete() {
        try {
            const res = await fetch('/api/pokemon');
            const list = await res.json();
            
            let dl = document.getElementById('pokemon-suggestions');
            if (dl) dl.remove();
            
            dl = document.createElement('datalist');
            dl.id = 'pokemon-suggestions';
            list.forEach(p => {
                const opt = document.createElement('option');
                opt.value = p.name;
                dl.appendChild(opt);
            });
            document.body.appendChild(dl);
            
            [cnnInput, rnnInput, aeInput].forEach(i => i?.setAttribute('list', 'pokemon-suggestions'));
        } catch (e) { console.warn(e); }
    }
    
    console.log('🦐 Готово!');
    console.log('📊 График обучения всегда виден');
    console.log('🔄 Кнопка обновления графика активна');
});