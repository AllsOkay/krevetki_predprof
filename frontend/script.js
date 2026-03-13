/**
 * Pokémon AI Similarity Search - Frontend Logic
 * Полная версия со всеми функциями (CNN, RNN, Autoencoder, MLP)
 * 
 * Этот файл обрабатывает все взаимодействия пользователя с интерфейсом:
 * - Загрузка и отображение покемонов из базы данных
 * - Массовая загрузка всех покемонов из PokeAPI
 * - Обучение нейросетей (CNN, RNN, Autoencoder, MLP)
 * - Визуальный поиск похожих покемонов (CNN)
 * - Классификация силы покемонов (RNN)
 * - Определение "обычности" покемонов (Autoencoder)
 * - Предсказание вероятности победы (MLP)
 */

document.addEventListener('DOMContentLoaded', () => {
    console.log('✅ DOM загружен, инициализация...');
    
    // ==================== DOM ЭЛЕМЕНТЫ ====================
    // Кнопки управления данными
    const addAllBtn = document.getElementById('add-all-btn');
    const trainCnnBtn = document.getElementById('train-cnn-btn');
    const trainRnnBtn = document.getElementById('train-rnn-btn');
    const trainAeBtn = document.getElementById('train-ae-btn');
    const trainMlpBtn = document.getElementById('train-mlp-btn');
    const clearBtn = document.getElementById('clear-btn');
    const refreshBtn = document.getElementById('refresh-btn');
    
    // CNN элементы (визуальный поиск)
    const cnnInput = document.getElementById('cnn-input');
    const cnnSearchBtn = document.getElementById('cnn-search-btn');
    const cnnResults = document.getElementById('cnn-results');
    const cnnLoading = document.getElementById('cnn-loading');
    
    // RNN элементы (классификация силы)
    const rnnInput = document.getElementById('rnn-input');
    const rnnClassifyBtn = document.getElementById('rnn-classify-btn');
    const rnnDistBtn = document.getElementById('rnn-dist-btn');
    const rnnResults = document.getElementById('rnn-results');
    const rnnDistribution = document.getElementById('rnn-distribution');
    const rnnLoading = document.getElementById('rnn-loading');
    
    // Autoencoder элементы (обычность)
    const aeInput = document.getElementById('ae-input');
    const aeCheckBtn = document.getElementById('ae-check-btn');
    const aeStatsBtn = document.getElementById('ae-stats-btn');
    const aeResults = document.getElementById('ae-results');
    const aeStatistics = document.getElementById('ae-statistics');
    const aeLoading = document.getElementById('ae-loading');
    
    // MLP элементы (вероятность победы)
    const mlpHpInput = document.getElementById('mlp-hp');
    const mlpAtkInput = document.getElementById('mlp-atk');
    const mlpDefInput = document.getElementById('mlp-def');
    const mlpSpdInput = document.getElementById('mlp-spd');
    const mlpPredictBtn = document.getElementById('mlp-predict-btn');
    const mlpTrainBtn = document.getElementById('mlp-train-btn');
    const mlpResults = document.getElementById('mlp-results');
    const mlpLoading = document.getElementById('mlp-loading');
    
    // Прогресс бар (для массовой загрузки)
    const mainProgress = document.getElementById('main-progress');
    const progressFill = document.getElementById('progress-fill');
    const progressLabel = document.getElementById('progress-label');
    const progressPercent = document.getElementById('progress-percent');
    const progressDetails = document.getElementById('progress-details');
    
    // Таблица покемонов
    const pokemonTableBody = document.getElementById('pokemon-table-body');
    const dbCount = document.getElementById('db-count');
    
    // Статус приложения
    const statusText = document.getElementById('status-text');
    const statusDot = document.querySelector('.status-dot');
    
    // ==================== ПРОВЕРКА ЭЛЕМЕНТОВ ====================
    if (!addAllBtn || !clearBtn) {
        console.error('❌ Не найдены кнопки управления! Проверьте index.html');
    }
    
    // ==================== ИНИЦИАЛИЗАЦИЯ ====================
    checkServerHealth();
    loadPokemon();
    loadAutocomplete();
    
    // ==================== ОБРАБОТЧИКИ СОБЫТИЙ ====================
    
    // Кнопка "Добавить всех покемонов"
    if (addAllBtn) {
        addAllBtn.addEventListener('click', async () => {
            console.log('🔘 Нажата кнопка: Добавить всех покемонов');
            await addAllPokemon();
        });
    }
    
    // Кнопка "Обучить CNN"
    if (trainCnnBtn) {
        trainCnnBtn.addEventListener('click', async () => {
            console.log('🔘 Нажата кнопка: Обучить CNN');
            await trainCNN();
        });
    }
    
    // Кнопка "Обучить RNN"
    if (trainRnnBtn) {
        trainRnnBtn.addEventListener('click', async () => {
            console.log('🔘 Нажата кнопка: Обучить RNN');
            await trainRNN();
        });
    }
    
    // Кнопка "Обучить Autoencoder"
    if (trainAeBtn) {
        trainAeBtn.addEventListener('click', async () => {
            console.log('🔘 Нажата кнопка: Обучить Autoencoder');
            await trainAE();
        });
    }
    
    // Кнопка "Обучить MLP"
    if (trainMlpBtn) {
        trainMlpBtn.addEventListener('click', async () => {
            console.log('🔘 Нажата кнопка: Обучить MLP');
            await trainMLP();
        });
    }
    
    // Кнопка "Очистить всё"
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            console.log('🔘 Нажата кнопка: Очистить всё');
            if (confirm('⚠️ Вы уверены? Все данные и модели будут удалены!')) {
                clearAllData();
            }
        });
    }
    
    // Кнопка "Найти похожих" (CNN поиск)
    if (cnnSearchBtn) {
        cnnSearchBtn.addEventListener('click', () => {
            console.log('🔘 Нажата кнопка: Найти похожих (CNN)');
            findSimilarCNN();
        });
    }
    
    // Поиск по Enter в поле CNN
    if (cnnInput) {
        cnnInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                console.log('🔘 Нажат Enter в поле CNN поиска');
                findSimilarCNN();
            }
        });
    }
    
    // Кнопка "Определить силу" (RNN классификация)
    if (rnnClassifyBtn) {
        rnnClassifyBtn.addEventListener('click', () => {
            console.log('🔘 Нажата кнопка: Определить силу (RNN)');
            classifyPokemonRNN();
        });
    }
    
    // Кнопка "Распределение" (RNN статистика)
    if (rnnDistBtn) {
        rnnDistBtn.addEventListener('click', () => {
            console.log('🔘 Нажата кнопка: Распределение (RNN)');
            showRNNDistribution();
        });
    }
    
    // Поиск по Enter в поле RNN
    if (rnnInput) {
        rnnInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                console.log('🔘 Нажат Enter в поле RNN');
                classifyPokemonRNN();
            }
        });
    }
    
    // Кнопка "Проверить обычность" (Autoencoder)
    if (aeCheckBtn) {
        aeCheckBtn.addEventListener('click', () => {
            console.log('🔘 Нажата кнопка: Проверить обычность (AE)');
            checkPokemonOrdinariness();
        });
    }
    
    // Кнопка "Статистика" (Autoencoder)
    if (aeStatsBtn) {
        aeStatsBtn.addEventListener('click', () => {
            console.log('🔘 Нажата кнопка: Статистика (AE)');
            showAEStatistics();
        });
    }
    
    // Поиск по Enter в поле AE
    if (aeInput) {
        aeInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                console.log('🔘 Нажат Enter в поле AE');
                checkPokemonOrdinariness();
            }
        });
    }
    
    // Кнопка "Рассчитать вероятность" (MLP)
    if (mlpPredictBtn) {
        mlpPredictBtn.addEventListener('click', () => {
            console.log('🔘 Нажата кнопка: Рассчитать вероятность (MLP)');
            predictWinProbability();
        });
    }
    
    // Кнопка обновления таблицы
    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => {
            console.log('🔘 Нажата кнопка: Обновить таблицу');
            loadPokemon();
        });
    }
    
    // ==================== ФУНКЦИИ ====================
    
    /**
     * Проверка работоспособности сервера
     */
    async function checkServerHealth() {
        try {
            const response = await fetch('/api/health');
            const data = await response.json();
            
            if (data.status === 'ok') {
                statusText.textContent = `Подключено | Покемонов: ${data.pokemon_count || 0}`;
                statusDot.classList.add('connected');
                console.log('✅ Сервер работает:', data);
            } else {
                statusText.textContent = 'Ошибка';
                statusDot.classList.remove('connected');
            }
        } catch (error) {
            statusText.textContent = 'Не подключено';
            statusDot.classList.remove('connected');
            console.error('❌ Сервер недоступен:', error);
        }
    }
    
    /**
     * Загрузка покемонов из БД в таблицу
     */
    async function loadPokemon() {
        try {
            const response = await fetch('/api/pokemon');
            const pokemonList = await response.json();
            
            pokemonTableBody.innerHTML = '';
            dbCount.textContent = pokemonList.length;
            
            pokemonList.forEach(pokemon => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${pokemon.id}</td>
                    <td><strong>${pokemon.name}</strong></td>
                    <td>${pokemon.types || 'N/A'}</td>
                    <td>${pokemon.hp || 0}</td>
                    <td>${pokemon.attack || 0}</td>
                    <td>${pokemon.defense || 0}</td>
                    <td>${pokemon.speed || 0}</td>
                    <td class="sprite-cell">
                        <img src="${pokemon.sprite_url || 'https://via.placeholder.com/48?text=?'}" 
                             alt="${pokemon.name}"
                             onerror="this.src='https://via.placeholder.com/48?text=Error'">
                    </td>
                `;
                pokemonTableBody.appendChild(row);
            });
            
            console.log(`✅ Загружено ${pokemonList.length} покемонов`);
        } catch (error) {
            console.error('❌ Ошибка загрузки покемонов:', error);
            dbCount.textContent = '0';
        }
    }
    
    /**
     * Добавить всех покемонов из PokeAPI
     */
    async function addAllPokemon() {
        if (!addAllBtn) return;
        
        addAllBtn.disabled = true;
        showProgress(true);
        updateProgress(0, 'Получение списка...', '');
        
        try {
            const limit = 1302;
            const response = await fetch(`https://pokeapi.co/api/v2/pokemon?limit=${limit}`);
            const data = await response.json();
            
            const total = data.results.length;
            let processed = 0;
            let successCount = 0;
            let failedCount = 0;
            
            updateProgress(5, 'Проверка существующих...', '');
            const existingResponse = await fetch('/api/pokemon');
            const existing = await existingResponse.json();
            const existingIds = existing.map(p => p.id);
            
            updateProgress(10, `Найдено ${total} покемонов. Новые: ${total - existingIds.length}`, '');
            
            const BATCH_SIZE = 10;
            
            for (let i = 0; i < total; i += BATCH_SIZE) {
                const batch = data.results.slice(i, i + BATCH_SIZE);
                
                const promises = batch.map(async (pokemon, idx) => {
                    const id = i + idx + 1;
                    
                    if (existingIds.includes(id)) {
                        processed++;
                        return;
                    }
                    
                    try {
                        const detailResponse = await fetch(`https://pokeapi.co/api/v2/pokemon/${id}`);
                        
                        if (!detailResponse.ok) {
                            failedCount++;
                            processed++;
                            return;
                        }
                        
                        const detail = await detailResponse.json();
                        const spriteUrl = detail.sprites?.front_default || null;
                        
                        await fetch('/api/pokemon', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                id: detail.id,
                                name: detail.name,
                                types: detail.types.map(t => t.type.name).join(', '),
                                height: detail.height,
                                weight: detail.weight,
                                hp: detail.stats.find(s => s.stat.name === 'hp')?.base_stat || 0,
                                attack: detail.stats.find(s => s.stat.name === 'attack')?.base_stat || 0,
                                defense: detail.stats.find(s => s.stat.name === 'defense')?.base_stat || 0,
                                'special-attack': detail.stats.find(s => s.stat.name === 'special-attack')?.base_stat || 0,
                                'special-defense': detail.stats.find(s => s.stat.name === 'special-defense')?.base_stat || 0,
                                speed: detail.stats.find(s => s.stat.name === 'speed')?.base_stat || 0,
                                sprite_url: spriteUrl
                            })
                        });
                        
                        successCount++;
                        
                    } catch (err) {
                        console.warn(`⚠️ Ошибка ${pokemon.name}:`, err);
                        failedCount++;
                    }
                    
                    processed++;
                    const percent = Math.round((processed / total) * 100);
                    updateProgress(
                        percent, 
                        `Обработано: ${processed}/${total}`, 
                        `${percent}% | Успешно: ${successCount} | Ошибки: ${failedCount}`
                    );
                });
                
                await Promise.all(promises);
                
                if (i + BATCH_SIZE < total) {
                    await sleep(200);
                }
            }
            
            updateProgress(
                100, 
                `✅ Готово! Добавлено: ${successCount}`, 
                `Всего в БД: ${existing.length + successCount}`
            );
            
            loadPokemon();
            setTimeout(() => showProgress(false), 3000);
            
        } catch (error) {
            console.error('❌ Ошибка добавления покемонов:', error);
            updateProgress(0, '❌ Ошибка', '0%');
            alert('Ошибка при добавлении покемонов: ' + error.message);
        } finally {
            addAllBtn.disabled = false;
        }
    }
    
    /**
     * Обучить CNN модель
     */
    async function trainCNN() {
        if (!trainCnnBtn) return;
        
        trainCnnBtn.disabled = true;
        trainCnnBtn.textContent = '⏳ Обучение...';
        
        try {
            const response = await fetch('/api/cnn/train', { method: 'POST' });
            const result = await response.json();
            
            if (result.success) {
                alert(
                    `✅ CNN модель обучена!\n\n` +
                    `📊 Покемонов в обучении: ${result.samples}\n` +
                    `🔄 Эпох: ${result.epochs}\n` +
                    `📉 Финальная потеря: ${result.final_loss?.toFixed(4)}\n\n` +
                    `💡 Теперь можно искать визуально похожих покемонов!`
                );
            } else {
                let message = `⚠️ Ошибка обучения CNN: ${result.reason || 'Неизвестная ошибка'}\n\n`;
                if (result.current_count && result.required_count) {
                    message += `📊 Сейчас в БД: ${result.current_count} покемонов со спрайтами\n`;
                    message += `🎯 Нужно минимум: ${result.required_count}\n\n`;
                    message += `💡 Нажмите "Добавить всех покемонов" и дождитесь завершения!`;
                }
                alert(message);
            }
        } catch (error) {
            console.error('❌ Ошибка обучения CNN:', error);
            alert('Ошибка при обучении CNN модели: ' + error.message);
        } finally {
            trainCnnBtn.disabled = false;
            trainCnnBtn.textContent = '🎓 Обучить CNN';
        }
    }
    
    /**
     * Обучить RNN модель
     */
    async function trainRNN() {
        if (!trainRnnBtn) return;
        
        trainRnnBtn.disabled = true;
        trainRnnBtn.textContent = '⏳ Обучение...';
        
        try {
            const response = await fetch('/api/rnn/train', { method: 'POST' });
            const result = await response.json();
            
            if (result.success) {
                alert(
                    `✅ RNN модель обучена!\n\n` +
                    `📊 Покемонов в обучении: ${result.samples}\n` +
                    `🔄 Эпох: ${result.epochs}\n` +
                    `📈 Финальная точность: ${(result.final_accuracy * 100).toFixed(1)}%\n\n` +
                    `💡 Теперь можно классифицировать силу покемонов!`
                );
            } else {
                let message = `⚠️ Ошибка обучения RNN: ${result.reason || 'Неизвестная ошибка'}\n\n`;
                if (result.current_count && result.required_count) {
                    message += `📊 Сейчас в БД: ${result.current_count} покемонов\n`;
                    message += `🎯 Нужно минимум: ${result.required_count}\n\n`;
                    message += `💡 Нажмите "Добавить всех покемонов" и дождитесь завершения!`;
                }
                alert(message);
            }
        } catch (error) {
            console.error('❌ Ошибка обучения RNN:', error);
            alert('Ошибка при обучении RNN модели: ' + error.message);
        } finally {
            trainRnnBtn.disabled = false;
            trainRnnBtn.textContent = '🧠 Обучить RNN';
        }
    }
    
    /**
     * Обучить Autoencoder модель
     */
    async function trainAE() {
        if (!trainAeBtn) return;
        
        trainAeBtn.disabled = true;
        trainAeBtn.textContent = '⏳ Обучение...';
        
        try {
            const response = await fetch('/api/autoencoder/train', { method: 'POST' });
            const result = await response.json();
            
            if (result.success) {
                alert(
                    `✅ Autoencoder модель обучена!\n\n` +
                    `📊 Покемонов в обучении: ${result.samples}\n` +
                    `🔄 Эпох: ${result.epochs}\n` +
                    `📉 Финальная потеря: ${result.final_loss?.toFixed(4)}\n\n` +
                    `💡 Теперь можно проверять "обычность" покемонов!`
                );
            } else {
                let message = `⚠️ Ошибка обучения Autoencoder: ${result.reason || 'Неизвестная ошибка'}\n\n`;
                if (result.current_count && result.required_count) {
                    message += `📊 Сейчас в БД: ${result.current_count} покемонов\n`;
                    message += `🎯 Нужно минимум: ${result.required_count}\n\n`;
                    message += `💡 Нажмите "Добавить всех покемонов" и дождитесь завершения!`;
                }
                alert(message);
            }
        } catch (error) {
            console.error('❌ Ошибка обучения Autoencoder:', error);
            alert('Ошибка при обучении Autoencoder модели: ' + error.message);
        } finally {
            trainAeBtn.disabled = false;
            trainAeBtn.textContent = '🔄 Обучить Autoencoder';
        }
    }
    
    /**
     * Обучить MLP модель
     */
    async function trainMLP() {
        if (!trainMlpBtn) return;
        
        trainMlpBtn.disabled = true;
        trainMlpBtn.textContent = '⏳ Обучение...';
        
        try {
            const response = await fetch('/api/mlp/train', { method: 'POST' });
            const result = await response.json();
            
            if (result.success) {
                alert(
                    `✅ MLP модель обучена!\n\n` +
                    `📊 Симулировано боёв: ${result.samples}\n` +
                    `🔄 Эпох: ${result.epochs}\n` +
                    `📈 Финальная точность: ${(result.final_accuracy * 100).toFixed(1)}%\n\n` +
                    `💡 Теперь можно предсказывать вероятность победы!`
                );
            } else {
                let message = `⚠️ Ошибка обучения MLP: ${result.reason || 'Неизвестная ошибка'}\n\n`;
                if (result.current_count && result.required_count) {
                    message += `📊 Сейчас в БД: ${result.current_count} покемонов\n`;
                    message += `🎯 Нужно минимум: ${result.required_count}\n\n`;
                    message += `💡 Нажмите "Добавить всех покемонов" и дождитесь завершения!`;
                }
                alert(message);
            }
        } catch (error) {
            console.error('❌ Ошибка обучения MLP:', error);
            alert('Ошибка при обучении MLP модели: ' + error.message);
        } finally {
            trainMlpBtn.disabled = false;
            trainMlpBtn.textContent = '⚔️ Обучить MLP';
        }
    }
    
    /**
     * Очистить все данные
     */
    async function clearAllData() {
        try {
            await fetch('/api/pokemon', { method: 'DELETE' });
            loadPokemon();
            alert('✅ Все данные удалены. Модели сброшены.');
        } catch (error) {
            console.error('❌ Ошибка очистки:', error);
            alert('Ошибка при очистке данных: ' + error.message);
        }
    }
    
    /**
     * Найти визуально похожих покемонов (CNN)
     */
    async function findSimilarCNN() {
        const name = cnnInput?.value.trim().toLowerCase();
        
        if (!name) {
            alert('⚠️ Введите имя покемона!');
            return;
        }
        
        cnnLoading?.classList.remove('hidden');
        cnnResults.classList.remove('hidden');
        cnnResults.innerHTML = '<div class="loading-panel"><div class="spinner"></div><span>Поиск...</span></div>';
        
        try {
            const response = await fetch(`/api/cnn/similar?name=${encodeURIComponent(name)}&k=5`);
            const data = await response.json();
            
            if (data.error) {
                cnnResults.innerHTML = `<p class="error">❌ ${data.error}</p>`;
                return;
            }
            
            let html = `
                <div class="result-card target-card">
                    <div class="card-badge">🎯 Цель</div>
                    <img src="${data.target.sprite_url || 'https://via.placeholder.com/96?text=No+Image'}" 
                         alt="${data.target.name}" 
                         class="sprite-img"
                         onerror="this.src='https://via.placeholder.com/96?text=Error'">
                    <div class="pokemon-name">${capitalize(data.target.name)}</div>
                </div>
            `;
            
            if (data.similar.length === 0) {
                html += '<p class="text-center text-muted" style="grid-column: 1/-1;">Похожих не найдено. Обучите CNN модель сначала.</p>';
            } else {
                data.similar.forEach((pokemon, index) => {
                    const similarityPercent = Math.round(pokemon.similarity * 100);
                    const color = similarityPercent > 80 ? '#22c55e' : similarityPercent > 60 ? '#f59e0b' : '#ef4444';
                    
                    html += `
                        <div class="result-card similar-card">
                            <div class="rank-badge">#${index + 1}</div>
                            <img src="${pokemon.sprite_url || 'https://via.placeholder.com/96?text=No+Image'}" 
                                 alt="${pokemon.name}" 
                                 class="sprite-img"
                                 onerror="this.src='https://via.placeholder.com/96?text=Error'">
                            <div class="pokemon-name">${capitalize(pokemon.name)}</div>
                            <div class="similarity-meter">
                                <span class="sim-label">Сходство:</span>
                                <div class="sim-bar">
                                    <div class="sim-fill" style="width: ${similarityPercent}%; background: ${color}"></div>
                                </div>
                                <span class="sim-value">${similarityPercent}%</span>
                            </div>
                        </div>
                    `;
                });
            }
            
            cnnResults.innerHTML = html;
            
        } catch (error) {
            console.error('❌ Ошибка CNN поиска:', error);
            cnnResults.innerHTML = `<p class="error">❌ Ошибка: ${error.message}</p>`;
        } finally {
            cnnLoading?.classList.add('hidden');
        }
    }
    
    /**
     * Классифицировать покемона по категории силы (RNN)
     */
    async function classifyPokemonRNN() {
        const name = rnnInput?.value.trim().toLowerCase();
        
        if (!name) {
            alert('⚠️ Введите имя покемона!');
            return;
        }
        
        rnnLoading?.classList.remove('hidden');
        rnnResults.classList.remove('hidden');
        rnnDistribution.classList.add('hidden');
        rnnResults.innerHTML = '<div class="loading-panel"><div class="spinner"></div><span>Классификация...</span></div>';
        
        try {
            const response = await fetch(`/api/rnn/classify?name=${encodeURIComponent(name)}`);
            const data = await response.json();
            
            if (data.error) {
                rnnResults.innerHTML = `<p class="error">❌ ${data.error}</p>`;
                return;
            }
            
            const strengthClass = data.result.class;
            const confidence = Math.round(data.result.confidence * 100);
            
            let html = `
                <div class="result-card target-card" style="grid-column: 1 / -1;">
                    <div class="card-badge">${data.result.class_ru}</div>
                    <img src="${data.pokemon.sprite_url || 'https://via.placeholder.com/96?text=No+Image'}" 
                         alt="${data.pokemon.name}" 
                         class="sprite-img"
                         onerror="this.src='https://via.placeholder.com/96?text=Error'">
                    <div class="pokemon-name">${capitalize(data.pokemon.name)}</div>
                    <div class="strength-badge ${strengthClass}">${data.result.class_ru}</div>
                    <div class="text-muted">BST: ${data.pokemon.bst}</div>
                    <div class="text-muted">Уверенность: ${confidence}%</div>
                    
                    <div class="probability-bars">
                        <div class="prob-item">
                            <span class="prob-label">🔹 Слабый:</span>
                            <div class="prob-bar"><div class="prob-fill" style="width: ${data.result.probabilities[0]*100}%; background: #3b82f6"></div></div>
                            <span>${Math.round(data.result.probabilities[0]*100)}%</span>
                        </div>
                        <div class="prob-item">
                            <span class="prob-label">🔸 Средний:</span>
                            <div class="prob-bar"><div class="prob-fill" style="width: ${data.result.probabilities[1]*100}%; background: #f59e0b"></div></div>
                            <span>${Math.round(data.result.probabilities[1]*100)}%</span>
                        </div>
                        <div class="prob-item">
                            <span class="prob-label">🔴 Сильный:</span>
                            <div class="prob-bar"><div class="prob-fill" style="width: ${data.result.probabilities[2]*100}%; background: #ef4444"></div></div>
                            <span>${Math.round(data.result.probabilities[2]*100)}%</span>
                        </div>
                    </div>
                </div>
            `;
            
            rnnResults.innerHTML = html;
            
        } catch (error) {
            console.error('❌ Ошибка RNN классификации:', error);
            rnnResults.innerHTML = `<p class="error">❌ Ошибка: ${error.message}</p>`;
        } finally {
            rnnLoading?.classList.add('hidden');
        }
    }
    
    /**
     * Показать распределение покемонов по классам силы (RNN)
     */
    async function showRNNDistribution() {
        rnnLoading?.classList.remove('hidden');
        rnnDistribution.classList.remove('hidden');
        rnnResults.classList.add('hidden');
        
        try {
            const response = await fetch('/api/rnn/distribution');
            const data = await response.json();
            
            const dist = data.distribution;
            const pct = data.percentages;
            
            let html = `
                <h3>📊 Распределение покемонов по силе</h3>
                <p class="text-muted">Всего покемонов: ${data.total}</p>
                <div class="distribution-chart">
                    <div class="distribution-item">
                        <div class="pokemon-name">🔹 Слабый</div>
                        <div class="distribution-bar"><div class="distribution-fill weak" style="width: ${pct.weak}%"></div></div>
                        <div><strong>${dist.weak}</strong> (${pct.weak}%)</div>
                    </div>
                    <div class="distribution-item">
                        <div class="pokemon-name">🔸 Средний</div>
                        <div class="distribution-bar"><div class="distribution-fill medium" style="width: ${pct.medium}%"></div></div>
                        <div><strong>${dist.medium}</strong> (${pct.medium}%)</div>
                    </div>
                    <div class="distribution-item">
                        <div class="pokemon-name">🔴 Сильный</div>
                        <div class="distribution-bar"><div class="distribution-fill strong" style="width: ${pct.strong}%"></div></div>
                        <div><strong>${dist.strong}</strong> (${pct.strong}%)</div>
                    </div>
                </div>
            `;
            
            rnnDistribution.innerHTML = html;
            
        } catch (error) {
            console.error('❌ Ошибка получения распределения:', error);
            rnnDistribution.innerHTML = `<p class="error">❌ Ошибка: ${error.message}</p>`;
        } finally {
            rnnLoading?.classList.add('hidden');
        }
    }
    
    /**
     * Проверить "обычность" покемона через автоэнкодер (AE)
     */
    async function checkPokemonOrdinariness() {
        const name = aeInput?.value.trim().toLowerCase();
        
        if (!name) {
            alert('⚠️ Введите имя покемона!');
            return;
        }
        
        aeLoading?.classList.remove('hidden');
        aeResults.classList.remove('hidden');
        aeStatistics.classList.add('hidden');
        aeResults.innerHTML = '<div class="loading-panel"><div class="spinner"></div><span>Анализ...</span></div>';
        
        try {
            const response = await fetch(`/api/autoencoder/ordinariness?name=${encodeURIComponent(name)}`);
            const data = await response.json();
            
            if (data.error) {
                aeResults.innerHTML = `<p class="error">❌ ${data.error}</p>`;
                return;
            }
            
            const ord = data.result.ordinariness;
            const colorClass = ord >= 80 ? 'ordinariness-high' : ord >= 60 ? 'ordinariness-medium' : 'ordinariness-low';
            
            let html = `
                <div class="result-card ordinariness-card">
                    <img src="${data.pokemon.sprite_url || 'https://via.placeholder.com/96?text=No+Image'}" 
                         alt="${data.pokemon.name}" 
                         class="sprite-img"
                         onerror="this.src='https://via.placeholder.com/96?text=Error'">
                    <div class="pokemon-name">${capitalize(data.pokemon.name)}</div>
                    
                    <div class="ordinariness-circle" style="--ordinariness: ${ord}%">
                        <span class="ordinariness-value ${colorClass}">${ord}%</span>
                        <span class="ordinariness-label">обычности</span>
                    </div>
                    
                    <div class="ordinariness-description">
                        ${data.result.description || ''}
                    </div>
                    
                    <div class="text-muted mt-sm">
                        Ошибка восстановления: ${data.result.reconstruction_error?.toFixed(4) || 'N/A'}
                    </div>
                </div>
            `;
            
            aeResults.innerHTML = html;
            
        } catch (error) {
            console.error('❌ Ошибка AE проверки:', error);
            aeResults.innerHTML = `<p class="error">❌ Ошибка: ${error.message}</p>`;
        } finally {
            aeLoading?.classList.add('hidden');
        }
    }
    
    /**
     * Показать статистику по обычности всех покемонов (AE)
     */
    async function showAEStatistics() {
        aeLoading?.classList.remove('hidden');
        aeStatistics.classList.remove('hidden');
        aeResults.classList.add('hidden');
        
        try {
            const response = await fetch('/api/autoencoder/statistics');
            const data = await response.json();
            
            if (data.message) {
                aeStatistics.innerHTML = `<p class="text-center text-muted">${data.message}</p>`;
                return;
            }
            
            let html = `
                <h3>📊 Статистика обычности покемонов</h3>
                <p class="text-muted">Всего проанализировано: ${data.count}</p>
                
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-value ordinariness-high">${data.very_ordinary || 0}</div>
                        <div class="stat-label">🟢 Очень обычные</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value ordinariness-medium">${data.ordinary || 0}</div>
                        <div class="stat-label">🔵 Обычные</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${data.unusual || 0}</div>
                        <div class="stat-label">🟡 Необычные</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value ordinariness-low">${data.rare || 0}</div>
                        <div class="stat-label">🟠 Редкие</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value ordinariness-low">${data.unique || 0}</div>
                        <div class="stat-label">🔴 Уникальные</div>
                    </div>
                </div>
                
                <div class="mt-md">
                    <p><strong>Средняя обычность:</strong> ${data.mean}%</p>
                    <p><strong>Медиана:</strong> ${data.median}%</p>
                    <p><strong>Диапазон:</strong> ${data.min}% — ${data.max}%</p>
                </div>
            `;
            
            aeStatistics.innerHTML = html;
            
        } catch (error) {
            console.error('❌ Ошибка получения статистики:', error);
            aeStatistics.innerHTML = `<p class="error">❌ Ошибка: ${error.message}</p>`;
        } finally {
            aeLoading?.classList.add('hidden');
        }
    }
    
    /**
     * Предсказать вероятность победы покемона (MLP)
     */
    async function predictWinProbability() {
        const hp = parseFloat(mlpHpInput?.value) || 0;
        const atk = parseFloat(mlpAtkInput?.value) || 0;
        const def_ = parseFloat(mlpDefInput?.value) || 0;
        const spd = parseFloat(mlpSpdInput?.value) || 0;
        
        if (hp <= 0 && atk <= 0 && def_ <= 0 && spd <= 0) {
            alert('⚠️ Введите хотя бы одну характеристику!');
            return;
        }
        
        mlpLoading?.classList.remove('hidden');
        mlpResults.classList.remove('hidden');
        mlpResults.innerHTML = '<div class="loading-panel"><div class="spinner"></div><span>Расчёт...</span></div>';
        
        try {
            const response = await fetch(
                `/api/mlp/predict?hp=${hp}&atk=${atk}&def=${def_}&spd=${spd}`
            );
            const data = await response.json();
            
            if (data.error) {
                mlpResults.innerHTML = `<p class="error">❌ ${data.error}</p>`;
                return;
            }
            
            const prob = data.result.win_probability;
            const classification = data.result.classification || '';
            
            let colorClass = 'low';
            let probabilityColor = '#ef4444';
            if (prob >= 80) {
                colorClass = 'high';
                probabilityColor = '#22c55e';
            } else if (prob >= 60) {
                probabilityColor = '#22c55e';
            } else if (prob >= 40) {
                probabilityColor = '#f59e0b';
            }
            
            let html = `
                <div class="result-card win-probability-card">
                    <div class="probability-circle" style="--probability: ${prob}%">
                        <span class="probability-value" style="color: ${probabilityColor}">${prob}%</span>
                        <span class="probability-label">победы</span>
                    </div>
                    
                    <div class="probability-classification ${colorClass}">
                        ${classification}
                    </div>
                    
                    <div class="probability-description">
                        ${data.result.description || ''}
                    </div>
                    
                    <div class="probability-stats">
                        <div class="prob-stat">
                            <div class="prob-stat-value">${data.result.bst || 0}</div>
                            <div class="prob-stat-label">BST</div>
                        </div>
                        <div class="prob-stat">
                            <div class="prob-stat-value">${hp + atk + def_ + spd}</div>
                            <div class="prob-stat-label">Сумма введённых</div>
                        </div>
                    </div>
                </div>
            `;
            
            mlpResults.innerHTML = html;
            
        } catch (error) {
            console.error('❌ Ошибка MLP предсказания:', error);
            mlpResults.innerHTML = `<p class="error">❌ Ошибка: ${error.message}</p>`;
        } finally {
            mlpLoading?.classList.add('hidden');
        }
    }
    
    // ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================
    
    function showProgress(show) {
        if (!mainProgress) return;
        if (show) {
            mainProgress.classList.remove('hidden');
        } else {
            mainProgress.classList.add('hidden');
        }
    }
    
    function updateProgress(percent, label, percentText) {
        if (!progressFill || !progressLabel || !progressPercent) return;
        progressFill.style.width = `${percent}%`;
        progressLabel.textContent = label;
        progressPercent.textContent = percentText;
        if (progressDetails) {
            progressDetails.textContent = percentText;
        }
    }
    
    function sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    function capitalize(str) {
        return str.charAt(0).toUpperCase() + str.slice(1);
    }
    
    async function loadAutocomplete() {
        try {
            const response = await fetch('/api/pokemon');
            const pokemonList = await response.json();
            
            const oldDatalist = document.getElementById('pokemon-suggestions');
            if (oldDatalist) {
                oldDatalist.remove();
            }
            
            const datalist = document.createElement('datalist');
            datalist.id = 'pokemon-suggestions';
            
            pokemonList.forEach(p => {
                const option = document.createElement('option');
                option.value = p.name;
                datalist.appendChild(option);
            });
            
            document.body.appendChild(datalist);
            
            if (cnnInput) cnnInput.setAttribute('list', 'pokemon-suggestions');
            if (rnnInput) rnnInput.setAttribute('list', 'pokemon-suggestions');
            if (aeInput) aeInput.setAttribute('list', 'pokemon-suggestions');
            
            console.log(`✅ Автокомплит загружен (${pokemonList.length} покемонов)`);
        } catch (e) {
            console.warn('⚠️ Не удалось загрузить автокомплит:', e);
        }
    }
    
    console.log('✅ Все обработчики событий установлены');
    console.log('🎮 Pokémon AI Search готов к работе!');
    console.log('🖼️  CNN: Визуальный поиск похожих покемонов');
    console.log('🧠 RNN: Классификация силы (слабый/средний/сильный)');
    console.log('🔄 AE: Определение "обычности" покемона в %');
    console.log('⚔️  MLP: Предсказание вероятности победы');
});