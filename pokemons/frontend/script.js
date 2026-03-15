/**
 * 🦐 Pokémon AI Search - Shrimp Style
 * Минималистичная версия с адаптацией под мобильные
 * Версия: 2.0 - Исправления графиков и Autoencoder
 */

document.addEventListener('DOMContentLoaded', () => {
    console.log('🦐 Shrimp Style загружен');
    
    // ==================== DOM ЭЛЕМЕНТЫ ====================
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
    
    // Переменные для графиков
    let chart2d = null;
    let currentModelType = 'cnn';
    let currentView = '2d';
    
    // Маппинг типов моделей к реальным ID кнопок
    const modelBtnMap = {
        'cnn': 'train-cnn-btn',
        'rnn': 'train-rnn-btn',
        'autoencoder': 'train-ae-btn',
        'mlp': 'train-mlp-btn'
    };
    
    const modelIcons = {
        'cnn': '🎓',
        'rnn': '🧠',
        'autoencoder': '🔄',
        'mlp': '⚔️'
    };
    
    // ==================== ИНИЦИАЛИЗАЦИЯ ====================
    checkServerHealth();
    loadPokemon();
    loadAutocomplete();
    loadTrainingGraph();
    
    // ==================== ОБРАБОТЧИКИ СОБЫТИЙ ====================
    
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
    
    if (graphModelSelect) graphModelSelect.addEventListener('change', e => { 
        currentModelType = e.target.value; 
        loadTrainingGraph(); 
    });
    
    if (view2dBtn) view2dBtn.addEventListener('click', () => switchGraphView('2d'));
    if (view3dBtn) view3dBtn.addEventListener('click', () => switchGraphView('3d'));
    if (refreshGraphBtn) refreshGraphBtn.addEventListener('click', () => {
        console.log('🔄 Обновление графика для:', currentModelType);
        loadTrainingGraph();
    });
    
    // ==================== ФУНКЦИИ ====================
    
    async function checkServerHealth() {
        try {
            const res = await fetch('/api/health');
            const data = await res.json();
            if (data.status === 'ok') {
                statusText.textContent = `OK (${data.pokemon_count || 0})`;
                statusDot?.classList.add('connected');
            } else {
                statusText.textContent = 'Ошибка';
                statusDot?.classList.remove('connected');
            }
        } catch (e) {
            statusText.textContent = 'Нет связи';
            statusDot?.classList.remove('connected');
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
        progress?.classList.remove('hidden');
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
            setTimeout(() => progress?.classList.add('hidden'), 2000);
        } catch (e) {
            alert('Ошибка: ' + e.message);
        } finally {
            addAllBtn.disabled = false;
        }
    }
    
    function updateProgress(percent, text) {
        if (progressFill) progressFill.style.width = `${percent}%`;
        if (progressText) progressText.textContent = `${percent}% ${text}`;
    }
    
    async function trainModel(type) {
        const btnId = modelBtnMap[type];
        const btn = document.getElementById(btnId);
        if (!btn) {
            console.error(`❌ Кнопка ${btnId} не найдена`);
            alert(`Ошибка: кнопка "${type}" не найдена`);
            return;
        }
        
        btn.disabled = true;
        btn.textContent = '⏳';
        
        try {
            console.log(`📡 Отправка POST: /api/${type}/train`);
            const res = await fetch(`/api/${type}/train`, { method: 'POST' });
            console.log(`📥 Ответ: ${res.status}`);
            
            const result = await res.json();
            console.log('📦 Результат:', result);
            
            if (result.success) {
                let message = `✅ ${type.toUpperCase()} обучена!\nЭпох: ${result.epochs}`;
                if (result.final_loss !== undefined && result.final_loss !== null) {
                    message += `\nLoss: ${result.final_loss.toFixed(4)}`;
                }
                if (result.final_accuracy !== undefined && result.final_accuracy !== null) {
                    message += `\nТочность: ${(result.final_accuracy * 100).toFixed(1)}%`;
                }
                alert(message + '\n\n💡 График обучения ниже');
                showGraphSection(type);
            } else {
                let errorMsg = `⚠️ Ошибка: ${result.reason || 'Неизвестная'}`;
                if (result.current_count && result.required_count) {
                    errorMsg += `\n\n📊 В БД: ${result.current_count} | Нужно: ${result.required_count}`;
                }
                alert(errorMsg);
            }
        } catch (e) {
            console.error(`❌ Ошибка ${type}:`, e);
            alert(`Ошибка: ${e.message}`);
        } finally {
            btn.disabled = false;
            btn.textContent = modelIcons[type] || '🔄';
        }
    }
    
    function showGraphSection(modelType) {
        currentModelType = modelType;
        if (graphModelSelect) graphModelSelect.value = modelType;
        loadTrainingGraph();
        graphSection?.scrollIntoView({ behavior: 'smooth' });
    }
    
    function switchGraphView(view) {
        currentView = view;
        if (view2dBtn) view2dBtn.classList.toggle('active', view === '2d');
        if (view3dBtn) view3dBtn.classList.toggle('active', view === '3d');
        
        if (view === '2d') {
            graph2dContainer?.classList.remove('hidden');
            graph3dContainer?.classList.add('hidden');
            loadGraph2D();
        } else {
            graph2dContainer?.classList.add('hidden');
            graph3dContainer?.classList.remove('hidden');
            loadGraph3D();
        }
    }
    
    async function loadTrainingGraph() {
        graphLoading?.classList.remove('hidden');
        
        try {
            console.log(`📊 Загрузка графика: ${currentModelType}`);
            
            const [m1, m2] = await Promise.all([
                fetch(`/api/metrics/${currentModelType}`),
                fetch(`/api/metrics/3d/${currentModelType}`)
            ]);
            
            const data = await m1.json();
            console.log('📦 2D данные:', data);
            
            // Нет данных — показываем заглушку
            if (!data.metrics || data.metrics.length === 0) {
                if (graphStats) {
                    graphStats.innerHTML = `<p class="text-muted">Нет данных для <strong>${currentModelType.toUpperCase()}</strong>.<br>Обучите модель.</p>`;
                }
                showGraphPlaceholder('2d', `Нет данных для ${currentModelType.toUpperCase()}`);
                showGraphPlaceholder('3d', `Нет данных для ${currentModelType.toUpperCase()}`);
                if (chart2d && typeof Chart !== 'undefined') { chart2d.destroy(); chart2d = null; }
                if (typeof Plotly !== 'undefined') Plotly.purge('training-chart-3d');
                return;
            }
            
            // Есть данные — обновляем
            if (data.summary) updateGraphStats(data.summary, currentModelType);
            
            if (currentView === '2d') {
                loadGraph2D(data.metrics, currentModelType);
            } else {
                const data3d = await m2.json();
                loadGraph3D(data3d, currentModelType);
            }
            
        } catch (e) {
            console.error('❌ Ошибка графика:', e);
            if (graphStats) graphStats.innerHTML = `<p class="error">❌ ${e.message}</p>`;
        } finally {
            graphLoading?.classList.add('hidden');
        }
    }
    
    function showGraphPlaceholder(view, message) {
        if (view === '2d') {
            const container = document.getElementById('graph-2d-container');
            const canvas = document.getElementById('training-chart-2d');
            if (container && canvas) {
                canvas.style.display = 'none';
                let placeholder = container.querySelector('.graph-placeholder');
                if (!placeholder) {
                    placeholder = document.createElement('div');
                    placeholder.className = 'graph-placeholder';
                    container.appendChild(placeholder);
                }
                placeholder.textContent = message;
                placeholder.style.display = 'flex';
            }
        } else {
            const container = document.getElementById('graph-3d-container');
            const plotDiv = document.getElementById('training-chart-3d');
            if (container && plotDiv) {
                plotDiv.style.display = 'none';
                let placeholder = container.querySelector('.graph-placeholder');
                if (!placeholder) {
                    placeholder = document.createElement('div');
                    placeholder.className = 'graph-placeholder';
                    container.appendChild(placeholder);
                }
                placeholder.textContent = message;
                placeholder.style.display = 'flex';
            }
        }
    }
    
    function hideGraphPlaceholder(view) {
        if (view === '2d') {
            const canvas = document.getElementById('training-chart-2d');
            const container = document.getElementById('graph-2d-container');
            if (canvas) canvas.style.display = 'block';
            if (container) {
                const p = container.querySelector('.graph-placeholder');
                if (p) p.style.display = 'none';
            }
        } else {
            const plotDiv = document.getElementById('training-chart-3d');
            const container = document.getElementById('graph-3d-container');
            if (plotDiv) plotDiv.style.display = 'block';
            if (container) {
                const p = container.querySelector('.graph-placeholder');
                if (p) p.style.display = 'none';
            }
        }
    }
    
    function loadGraph2D(metrics, modelType) {
        const ctx = document.getElementById('training-chart-2d');
        if (!ctx || typeof Chart === 'undefined') {
            console.error('❌ Chart.js не доступен или canvas не найден');
            return;
        }
        
        // Сначала скрываем заглушку
        hideGraphPlaceholder('2d');
        
        // Проверка данных
        if (!metrics || metrics.length === 0) {
            showGraphPlaceholder('2d', 'Нет данных');
            if (chart2d) { chart2d.destroy(); chart2d = null; }
            return;
        }
        
        // Уничтожаем старый график если есть
        if (chart2d) chart2d.destroy();
        
        const epochs = metrics.map(m => m.epoch);
        const losses = metrics.map(m => m.loss !== null ? m.loss : 0);
        const modelConfig = getModelChartConfig(modelType);
        
        const datasets = [{
            label: modelConfig.lossLabel,
            data: losses,  // ✅ Ключ "data:" внутри datasets
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
                data: accuracies,  // ✅ Ключ "data:" внутри datasets
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
                data: {  // ✅ ВАЖНО: Ключ "data:" для основного объекта
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
                            grid: { color: 'rgba(0,0,0,0.05)' },
                            beginAtZero: true
                        },
                        y: { 
                            type: 'linear',
                            display: true,
                            position: 'left',
                            title: { display: true, text: modelConfig.lossLabel },
                            grid: { color: 'rgba(255,107,138,0.1)' },
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
                            ticks: { callback: v => v + '%' }
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
                                        label += context.dataset.yAxisID === 'y1' 
                                            ? context.parsed.y.toFixed(1) + '%' 
                                            : context.parsed.y.toFixed(4);
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
            console.error('❌ Ошибка 2D:', e);
            showGraphPlaceholder('2d', 'Ошибка отрисовки');
        }
    }
    
    function loadGraph3D(data3d, modelType) {
        const container = document.getElementById('training-chart-3d');
        if (!container || typeof Plotly === 'undefined') return;
        
        hideGraphPlaceholder('3d');
        
        if (!data3d || !data3d.data_3d || !data3d.data_3d.x || data3d.data_3d.x.length === 0) {
            showGraphPlaceholder('3d', 'Нет данных для 3D');
            Plotly.purge(container);
            return;
        }
        
        const x = data3d.data_3d.x.filter(v => v !== null && v !== undefined && !isNaN(v));
        const y = data3d.data_3d.y.filter(v => v !== null && v !== undefined && !isNaN(v));
        const z = data3d.data_3d.z.filter(v => v !== null && v !== undefined && !isNaN(v));
        
        if (x.length === 0 || y.length === 0 || z.length === 0) {
            showGraphPlaceholder('3d', 'Некорректные данные');
            Plotly.purge(container);
            return;
        }
        
        const minLength = Math.min(x.length, y.length, z.length);
        const safeX = x.slice(0, minLength);
        const safeY = y.slice(0, minLength);
        const safeZ = z.slice(0, minLength);
        
        const modelConfig = getModelChartConfig(modelType);
        
        try {
            Plotly.newPlot(container, [{
                x: safeX, y: safeY, z: safeZ,
                mode: 'lines+markers',
                type: 'scatter3d',
                name: modelConfig.lossLabel,
                marker: { size: 4, color: safeZ, colorscale: data3d.layout_hints?.color_scale || 'Viridis', opacity: 0.8, colorbar: { title: modelConfig.accuracyLabel || 'Метрика' } },
                line: { color: '#ff6b8a', width: 2 },
                text: data3d.data_3d.text || safeX.map((v, i) => `Эпоха ${v}: Loss=${safeY[i]?.toFixed(4)}`),
                hoverinfo: 'text+x+y+z'
            }], {
                margin: { l: 0, r: 0, b: 0, t: 0 },
                scene: {
                    xaxis: { title: data3d.layout_hints?.x_title || 'Эпоха', gridcolor: '#eee', zeroline: false, showbackground: true, backgroundcolor: 'rgba(240,240,240,0.5)' },
                    yaxis: { title: data3d.layout_hints?.y_title || 'Loss', gridcolor: '#eee', zeroline: false, showbackground: true, backgroundcolor: 'rgba(240,240,240,0.5)' },
                    zaxis: { title: data3d.layout_hints?.z_title || 'Metric', gridcolor: '#eee', zeroline: false, showbackground: true, backgroundcolor: 'rgba(240,240,240,0.5)' },
                    camera: { eye: { x: 1.5, y: 1.5, z: 1.5 } }
                },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { family: 'Segoe UI, sans-serif', size: 12 }
            }, {
                responsive: true,
                displayModeBar: true,
                modeBarButtonsToAdd: ['hoverClosest3d', 'zoomIn3d', 'zoomOut3d', 'resetCameraDefault3d'],
                scrollZoom: true
            });
            console.log('✅ 3D график создан');
        } catch (e) {
            console.error('❌ Ошибка 3D:', e);
            showGraphPlaceholder('3d', `Ошибка: ${e.message}`);
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
            'cnn': { lossLabel: 'Contrastive Loss', accuracyLabel: 'Embedding Distance' },
            'rnn': { lossLabel: 'Cross-Entropy Loss', accuracyLabel: 'Classification Accuracy' },
            'autoencoder': { lossLabel: 'Reconstruction MSE', accuracyLabel: 'Ordinariness Score' },
            'mlp': { lossLabel: 'Binary Cross-Entropy', accuracyLabel: 'Win Prediction Accuracy' }
        };
        return configs[modelType] || configs['cnn'];
    }
    
    async function clearAllData() {
        if (!confirm('Удалить всё?')) return;
        try {
            await fetch('/api/pokemon', { method: 'DELETE' });
            loadPokemon();
            graphSection?.classList.add('hidden');
            alert('✅ Удалено');
        } catch (e) {
            alert('Ошибка: ' + e.message);
        }
    }
    
    async function findSimilarCNN() {
        const name = cnnInput?.value.trim().toLowerCase();
        if (!name) { alert('Введите имя!'); return; }
        
        cnnLoading?.classList.remove('hidden');
        cnnResults?.classList.remove('hidden');
        if (cnnResults) cnnResults.innerHTML = '<div class="loading">⏳</div>';
        
        try {
            const res = await fetch(`/api/cnn/similar?name=${encodeURIComponent(name)}&k=5`);
            const data = await res.json();
            
            if (data.error) { if (cnnResults) cnnResults.innerHTML = `<p>❌ ${data.error}</p>`; return; }
            
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
            cnnLoading?.classList.add('hidden');
        }
    }
    
    async function classifyPokemonRNN() {
        const name = rnnInput?.value.trim().toLowerCase();
        if (!name) { alert('Введите имя!'); return; }
        
        rnnLoading?.classList.remove('hidden');
        rnnResults?.classList.remove('hidden');
        if (rnnResults) rnnResults.innerHTML = '<div class="loading">⏳</div>';
        
        try {
            const res = await fetch(`/api/rnn/classify?name=${encodeURIComponent(name)}`);
            const data = await res.json();
            
            if (data.error) { if (rnnResults) rnnResults.innerHTML = `<p>❌ ${data.error}</p>`; return; }
            
            const prob = data.result.probabilities || [0, 0, 0];
            if (rnnResults) {
                rnnResults.innerHTML = `
                    <div class="result-card target-card">
                        <img src="${data.pokemon.sprite_url || ''}" alt="${data.pokemon.name}">
                        <div class="name">${data.pokemon.name}</div>
                        <div class="stat">${data.result.class_ru || ''}</div>
                        <div class="stat">BST: ${data.pokemon.bst}</div>
                        <div class="stat">🔹 ${Math.round(prob[0]*100)}% | 🔸 ${Math.round(prob[1]*100)}% | 🔴 ${Math.round(prob[2]*100)}%</div>
                    </div>`;
            }
        } catch (e) {
            if (rnnResults) rnnResults.innerHTML = `<p>❌ ${e.message}</p>`;
        } finally {
            rnnLoading?.classList.add('hidden');
        }
    }
    
    async function checkPokemonOrdinariness() {
        const name = aeInput?.value.trim().toLowerCase();
        if (!name) { alert('Введите имя!'); return; }
        
        aeLoading?.classList.remove('hidden');
        aeResults?.classList.remove('hidden');
        if (aeResults) aeResults.innerHTML = '<div class="loading">⏳</div>';
        
        try {
            const res = await fetch(`/api/autoencoder/ordinariness?name=${encodeURIComponent(name)}`);
            const data = await res.json();
            
            if (data.error) { if (aeResults) aeResults.innerHTML = `<p>❌ ${data.error}</p>`; return; }
            
            const ord = data.result.ordinariness || 0;
            if (aeResults) {
                aeResults.innerHTML = `
                    <div class="result-card target-card">
                        <img src="${data.pokemon.sprite_url || ''}" alt="${data.pokemon.name}">
                        <div class="name">${data.pokemon.name}</div>
                        <div class="stat" style="font-size:24px;color:#ff6b8a;">${ord}%</div>
                        <div class="stat">${data.result.description || 'обычности'}</div>
                    </div>`;
            }
        } catch (e) {
            if (aeResults) aeResults.innerHTML = `<p>❌ ${e.message}</p>`;
        } finally {
            aeLoading?.classList.add('hidden');
        }
    }
    
    async function predictWinProbability() {
        const hp = parseFloat(mlpHpInput?.value) || 0;
        const atk = parseFloat(mlpAtkInput?.value) || 0;
        const def_ = parseFloat(mlpDefInput?.value) || 0;
        const spd = parseFloat(mlpSpdInput?.value) || 0;
        
        if (!hp && !atk && !def_ && !spd) { alert('Введите статы!'); return; }
        
        mlpLoading?.classList.remove('hidden');
        mlpResults?.classList.remove('hidden');
        if (mlpResults) mlpResults.innerHTML = '<div class="loading">⏳</div>';
        
        try {
            const res = await fetch(`/api/mlp/predict?hp=${hp}&atk=${atk}&def=${def_}&spd=${spd}`);
            const data = await res.json();
            
            if (data.error) { if (mlpResults) mlpResults.innerHTML = `<p>❌ ${data.error}</p>`; return; }
            
            const prob = data.result.win_probability || 0;
            if (mlpResults) {
                mlpResults.innerHTML = `
                    <div class="result-card target-card">
                        <div class="stat" style="font-size:32px;color:#ff6b8a;">${prob}%</div>
                        <div class="stat">победы</div>
                        <div class="stat">${data.result.classification || ''}</div>
                        <div class="stat">${data.result.description || ''}</div>
                    </div>`;
            }
        } catch (e) {
            if (mlpResults) mlpResults.innerHTML = `<p>❌ ${e.message}</p>`;
        } finally {
            mlpLoading?.classList.add('hidden');
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
    console.log('📊 График всегда виден');
    console.log('🔄 Кнопка обновления активна');
});