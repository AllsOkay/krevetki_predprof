/**
 * Pokémon AI - Frontend логика
 * Управление интерфейсом, обучение моделей, предсказания
 */

document.addEventListener('DOMContentLoaded', () => {
    // === DOM элементы ===
    const elements = {
        addAllBtn: document.getElementById('add-all-btn'),
        trainBtns: {
            mlp: document.getElementById('train-mlp-btn'),
            cnn: document.getElementById('train-cnn-btn'),
            rnn: document.getElementById('train-rnn-btn'),
            autoencoder: document.getElementById('train-autoencoder-btn'),
            siamese: document.getElementById('train-siamese-btn')
        },
        predictBtns: {
            mlp: document.getElementById('predict-mlp-btn'),
            cnn: document.getElementById('predict-cnn-btn'),
            rnn: document.getElementById('predict-rnn-btn'),
            autoencoder: document.getElementById('predict-autoencoder-btn'),
            siamese: document.getElementById('predict-siamese-btn')
        },
        clearBtn: document.getElementById('clear-btn'),
        findSimilarBtn: document.getElementById('find-similar-btn'),
        
        pokemonInput: document.getElementById('pokemon-input'),
        similarInput: document.getElementById('similar-input'),
        referenceInput: document.getElementById('reference-input'),
        
        tableBody: document.getElementById('pokemon-table-body'),
        countDb: document.getElementById('count-db'),
        
        progressContainer: document.getElementById('progress-container'),
        progressFill: document.getElementById('progress-fill'),
        progressText: document.getElementById('progress-text'),
        
        similarResults: document.getElementById('similar-results'),
        predictResults: document.getElementById('predict-results'),
        modelStatus: document.getElementById('model-status'),
        
        chartCanvas: document.getElementById('loss-chart')
    };
    
    // === Конфигурация ===
    const CONFIG = {
        POKEAPI_BASE: 'https://pokeapi.co/api/v2',
        BATCH_SIZE: 5,
        REQUEST_DELAY: 100,
        TOTAL_POKEMON: 1302
    };
    
    // === Инициализация ===
    loadPokemon();
    checkModelsStatus();
    
    // === Обработчики событий ===
    elements.addAllBtn.addEventListener('click', addAllPokemon);
    elements.clearBtn.addEventListener('click', clearData);
    elements.findSimilarBtn.addEventListener('click', findSimilar);
    
    // Обработчики обучения моделей
    Object.entries(elements.trainBtns).forEach(([modelType, btn]) => {
        if (btn) {
            btn.addEventListener('click', () => trainModel(modelType));
        }
    });
    
    // Обработчики предсказания
    Object.entries(elements.predictBtns).forEach(([modelType, btn]) => {
        if (btn) {
            btn.addEventListener('click', () => predictWithModel(modelType));
        }
    });
    
    // Поддержка Enter в полях ввода
   elements.pokemonInput?.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        const selectedModel = document.getElementById('model-select').value;
        predictWithModel(selectedModel);
    }
});
    
    // === Основные функции ===
    
    /**
     * Загружает покемонов из БД и отображает в таблице
     */
    async function loadPokemon() {
        try {
            const res = await fetch('/api/pokemon');
            const data = await res.json();
            
            elements.tableBody.innerHTML = '';
            elements.countDb.textContent = data.length;
            
            data.forEach(p => {
                const tr = document.createElement('tr');
                tr.dataset.id = p.id;
                tr.innerHTML = `
                    <td>${p.id}</td>
                    <td><strong>${capitalize(p.name)}</strong></td>
                    <td>${p.types || '-'}</td>
                    <td>${p.hp}</td>
                    <td>${p.attack}</td>
                    <td>${p.defense}</td>
                    <td>${p.speed}</td>
                `;
                elements.tableBody.appendChild(tr);
            });
        } catch (e) {
            console.error('Ошибка загрузки покемонов:', e);
        }
    }
    
    /**
     * Добавляет всех покемонов из PokeAPI
     */
    async function addAllPokemon() {
        setButtonsDisabled(true);
        showProgress(true);
        
        try {
            // Шаг 1: Получаем список всех покемонов
            updateProgress(0, '📡 Получение списка покемонов...');
            const allPokemon = await fetchAllPokemonList();
            
            // Шаг 2: Получаем уже существующих ID
            const existingIds = await getExistingPokemonIds();
            const newPokemon = allPokemon.filter(p => !existingIds.includes(p.id));
            
            if (newPokemon.length === 0) {
                updateProgress(100, '✅ Все покемоны уже добавлены!');
                setTimeout(() => showProgress(false), 2000);
                setButtonsDisabled(false);
                return;
            }
            
            updateProgress(0, `🔄 Обработка ${newPokemon.length} новых покемонов...`);
            
            // Шаг 3: Обрабатываем пакетами
            let processed = 0;
            let errors = 0;
            
            for (let i = 0; i < newPokemon.length; i += CONFIG.BATCH_SIZE) {
                const batch = newPokemon.slice(i, i + CONFIG.BATCH_SIZE);
                
                const promises = batch.map(pokemon =>
                    fetchPokemonDetails(pokemon.url)
                        .then(details => savePokemonToDB(details))
                        .then(() => {
                            addPokemonToTable(details);
                            processed++;
                            const percent = Math.round((processed / newPokemon.length) * 100);
                            updateProgress(percent, `✅ Обработано: ${processed}/${newPokemon.length}`);
                        })
                        .catch(err => {
                            console.warn(`⚠️ Не удалось: ${pokemon.name}`, err);
                            errors++;
                            processed++;
                        })
                );
                
                await Promise.all(promises);
                
                // Задержка между пакетами (уважение к API)
                if (i + CONFIG.BATCH_SIZE < newPokemon.length) {
                    await sleep(CONFIG.REQUEST_DELAY);
                }
            }
            
            const message = errors > 0 
                ? `✅ Готово! Добавлено ${processed - errors}, ошибок: ${errors}`
                : `✅ Готово! Добавлено ${newPokemon.length} покемонов`;
                
            updateProgress(100, message);
            elements.countDb.textContent = parseInt(elements.countDb.textContent) + (processed - errors);
            
            setTimeout(() => showProgress(false), 3000);
            
        } catch (error) {
            console.error('❌ Критическая ошибка:', error);
            updateProgress(0, `❌ Ошибка: ${error.message}`);
            alert('Произошла ошибка при добавлении покемонов');
        } finally {
            setButtonsDisabled(false);
        }
    }
    
    /**
     * Получает список всех покемонов из PokeAPI
     */
    async function fetchAllPokemonList() {
        const response = await fetch(`${CONFIG.POKEAPI_BASE}/pokemon?limit=${CONFIG.TOTAL_POKEMON}`);
        if (!response.ok) throw new Error('Не удалось получить список покемонов');
        
        const data = await response.json();
        return data.results.map((item, index) => ({
            id: index + 1,
            name: item.name,
            url: item.url
        }));
    }
    
    /**
     * Получает детальные данные о покемоне
     */
    async function fetchPokemonDetails(url) {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const data = await response.json();
        
        return {
            id: data.id,
            name: data.name,
            types: data.types.map(t => t.type.name).join(', '),
            height: data.height,
            weight: data.weight,
            hp: getStatValue(data.stats, 'hp'),
            attack: getStatValue(data.stats, 'attack'),
            defense: getStatValue(data.stats, 'defense'),
            'special-attack': getStatValue(data.stats, 'special-attack'),
            'special-defense': getStatValue(data.stats, 'special-defense'),
            speed: getStatValue(data.stats, 'speed')
        };
    }
    
    /**
     * Извлекает значение характеристики
     */
    function getStatValue(stats, statName) {
        const stat = stats.find(s => s.stat.name === statName);
        return stat ? stat.base_stat : 0;
    }
    
    /**
     * Получает список существующих ID
     */
    async function getExistingPokemonIds() {
        try {
            const response = await fetch('/api/pokemon');
            if (response.ok) {
                const data = await response.json();
                return data.map(p => p.id);
            }
        } catch (e) {
            console.warn('Не удалось получить существующие ID:', e);
        }
        return [];
    }
    
    /**
     * Сохраняет покемона в БД
     */
    async function savePokemonToDB(pokemon) {
        const response = await fetch('/api/pokemon', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(pokemon)
        });
        
        if (!response.ok && response.status !== 409) {
            throw new Error('Не удалось сохранить в БД');
        }
        return pokemon;
    }
    
    /**
     * Добавляет строку в таблицу
     */
    function addPokemonToTable(pokemon) {
        const existingRow = elements.tableBody.querySelector(`tr[data-id="${pokemon.id}"]`);
        if (existingRow) return;
        
        const row = document.createElement('tr');
        row.dataset.id = pokemon.id;
        row.style.animation = 'fadeIn 0.3s ease';
        
        row.innerHTML = `
            <td>${pokemon.id}</td>
            <td><strong>${capitalize(pokemon.name)}</strong></td>
            <td>${pokemon.types}</td>
            <td>${pokemon.hp}</td>
            <td>${pokemon.attack}</td>
            <td>${pokemon.defense}</td>
            <td>${pokemon.speed}</td>
        `;
        elements.tableBody.appendChild(row);
    }
    
    // === Обучение моделей ===
    
    /**
     * Запускает обучение указанной модели
     */
    async function trainModel(modelType) {
        const btn = elements.trainBtns[modelType];
        if (!btn) return;
        
        const originalText = btn.textContent;
        btn.disabled = true;
        btn.textContent = '🔄 Обучение...';
        showProgress(true);
        updateProgress(0, `Начало обучения ${modelType.toUpperCase()}...`);
        
        try {
            const res = await fetch(`/api/train/${modelType}`, { method: 'POST' });
            const data = await res.json();
            
            if (data.status === 'trained') {
                updateProgress(100, `✅ ${modelType.toUpperCase()} успешно обучена!`);
                
                // Отображаем график потерь если есть данные
                if (data.history?.train_loss) {
                    renderLossChart(data.history, modelType);
                }
                
                setTimeout(() => {
                    showProgress(false);
                    checkModelsStatus();
                }, 2000);
            } else {
                updateProgress(0, `❌ Ошибка: ${data.error || 'Неизвестная ошибка'}`);
                alert(`⚠️ Ошибка обучения: ${data.error || 'Недостаточно данных'}`);
            }
        } catch (e) {
            updateProgress(0, `❌ Ошибка соединения`);
            alert('❌ Ошибка соединения с сервером');
        } finally {
            btn.disabled = false;
            btn.textContent = originalText;
        }
    }
    
    /**
     * Проверяет статус загруженных моделей
     */
    async function checkModelsStatus() {
        try {
            const res = await fetch('/api/models/status');
            const status = await res.json();
            
            if (elements.modelStatus) {
                elements.modelStatus.innerHTML = `
                    <strong>📊 Статус:</strong><br>
                    Загружено моделей: ${status.loaded.join(', ') || 'нет'}<br>
                    Покемонов в БД: ${status.pokemon_count}<br>
                    Нормализация: ${status.scaler_loaded ? '✅' : '❌'}
                `;
            }
            
            // Обновляем состояние кнопок предсказания
            Object.entries(elements.predictBtns).forEach(([type, btn]) => {
                if (btn) {
                    btn.disabled = !status.loaded.includes(type);
                    btn.title = status.loaded.includes(type) 
                        ? 'Готово к использованию' 
                        : 'Сначала обучите модель';
                }
            });
            
        } catch (e) {
            console.warn('Не удалось проверить статус моделей:', e);
        }
    }
    
    // === Предсказание ===
    
    /**
     * Выполняет предсказание с использованием указанной модели
     */
    async function predictWithModel(modelType) {
        const pokemonName = elements.pokemonInput?.value.trim().toLowerCase();
        if (!pokemonName) {
            alert('Введите имя покемона');
            elements.pokemonInput?.focus();
            return;
        }
        
        const btn = elements.predictBtns[modelType];
        if (btn?.disabled) {
            alert(`Сначала обучите модель ${modelType.toUpperCase()}`);
            return;
        }
        
        elements.predictResults.innerHTML = '<p>🔮 Предсказание...</p>';
        elements.predictResults.classList.remove('hidden');
        
        try {
            const payload = { pokemon_name: pokemonName };
            
            // Для сиамской сети добавляем эталонного покемона
            if (modelType === 'siamese' && elements.referenceInput?.value) {
                payload.reference_pokemon = elements.referenceInput.value.trim().toLowerCase();
            }
            
            const res = await fetch(`/api/predict/${modelType}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            const data = await res.json();
            
            if (data.error) {
                elements.predictResults.innerHTML = `<p style="color:red">❌ ${data.error}</p>`;
                return;
            }
            
            // Формируем вывод в зависимости от типа модели
            let output = `<h3>📋 Результат для ${capitalize(data.pokemon)}</h3>`;
            
            if (modelType === 'mlp' || modelType === 'cnn') {
                output += `
                    <p><strong>Предсказанный тип:</strong> ${capitalize(data.predicted_type)}</p>
                    <p><strong>Уверенность:</strong> ${(data.confidence * 100).toFixed(1)}%</p>
                `;
            }
            else if (modelType === 'rnn') {
                output += `<p><strong>Предсказанные характеристики:</strong></p><ul>`;
                Object.entries(data.predicted_stats).forEach(([key, val]) => {
                    output += `<li>${key}: ${Math.round(val)}</li>`;
                });
                output += `</ul>`;
            }
            else if (modelType === 'autoencoder') {
                const color = data.interpretation === 'нормальный' ? 'green' : 'orange';
                output += `
                    <p><strong>Оценка аномальности:</strong> ${data.anomaly_score.toFixed(4)}</p>
                    <p><strong>Интерпретация:</strong> <span style="color:${color}">${data.interpretation}</span></p>
                `;
            }
            else if (modelType === 'siamese') {
                const similarityPercent = (data.similarity * 100).toFixed(1);
                output += `
                    <p><strong>Эталон:</strong> ${capitalize(data.reference)}</p>
                    <p><strong>Сходство:</strong> ${similarityPercent}%</p>
                    <p><strong>Интерпретация:</strong> ${data.interpretation}</p>
                `;
            }
            
            elements.predictResults.innerHTML = output;
            
        } catch (e) {
            elements.predictResults.innerHTML = '<p style="color:red">❌ Ошибка запроса</p>';
        }
    }
    
    // === Поиск похожих (использует автоэнкодер) ===
    
    /**
     * Находит похожих покемонов
     */
    async function findSimilar() {
        const name = elements.similarInput?.value.trim().toLowerCase();
        if (!name) {
            alert('Введите имя покемона');
            elements.similarInput?.focus();
            return;
        }
        
        elements.similarResults.innerHTML = '<p>🔍 Поиск в латентном пространстве...</p>';
        elements.similarResults.classList.remove('hidden');
        
        try {
            const res = await fetch(`/api/similar/${encodeURIComponent(name)}`);
            const data = await res.json();
            
            if (data.error) {
                elements.similarResults.innerHTML = `<p style="color:red">❌ ${data.error}</p>`;
                return;
            }
            
            if (data.similar.length === 0) {
                elements.similarResults.innerHTML = '<p>📭 Похожие покемоны не найдены</p>';
                return;
            }
            
            elements.similarResults.innerHTML = '';
            
            data.similar.forEach((p, idx) => {
                const card = document.createElement('div');
                card.className = 'result-card';
                card.style.animationDelay = `${idx * 0.1}s`;
                
                const similarityPercent = (p.similarity * 100).toFixed(1);
                const barWidth = Math.min(100, p.similarity * 100);
                
                card.innerHTML = `
                    <h3>${capitalize(p.name)}</h3>
                    <div class="similarity-bar">
                        <div class="similarity-fill" style="width: ${barWidth}%"></div>
                    </div>
                    <div class="score">${similarityPercent}% сходства</div>
                    <small>Тип: ${p.types}</small><br>
                    <small>HP: ${p.stats.hp} | ATK: ${p.stats.attack} | SPD: ${p.stats.speed}</small>
                `;
                elements.similarResults.appendChild(card);
            });
            
        } catch (e) {
            elements.similarResults.innerHTML = '<p style="color:red">❌ Ошибка запроса</p>';
        }
    }
    
    // === Очистка данных ===
    
    /**
     * Очищает все данные из БД
     */
    async function clearData() {
        if (!confirm('⚠️ Вы уверены, что хотите удалить все данные и модели?')) return;
        
        try {
            const res = await fetch('/api/pokemon', { method: 'DELETE' });
            if (res.ok) {
                elements.tableBody.innerHTML = '';
                elements.countDb.textContent = '0';
                elements.similarResults.innerHTML = '';
                elements.predictResults.innerHTML = '';
                
                // Сбрасываем статус моделей
                if (elements.modelStatus) {
                    elements.modelStatus.innerHTML = '<em>Модели сброшены</em>';
                }
                
                alert('✅ Все данные и модели удалены');
                checkModelsStatus();
            } else {
                throw new Error('Не удалось очистить данные');
            }
        } catch (e) {
            alert('❌ Не удалось удалить данные');
        }
    }
    
    // === Визуализация ===
    
    /**
     * Отображает график потерь при обучении
     */
    function renderLossChart(history, modelType) {
        if (!elements.chartCanvas || !history.train_loss) return;
        
        const ctx = elements.chartCanvas.getContext('2d');
        const width = elements.chartCanvas.width;
        const height = elements.chartCanvas.height;
        
        // Очистка
        ctx.clearRect(0, 0, width, height);
        
        const trainLoss = history.train_loss;
        const valLoss = history.val_loss || [];
        const maxLoss = Math.max(...trainLoss, ...(valLoss.length ? valLoss : [1]));
        
        // Отрисовка осей
        ctx.strokeStyle = '#ccc';
        ctx.beginPath();
        ctx.moveTo(40, 10);
        ctx.lineTo(40, height - 30);
        ctx.lineTo(width - 10, height - 30);
        ctx.stroke();
        
        // Функция для преобразования координат
        const toY = (loss) => height - 30 - (loss / maxLoss) * (height - 50);
        const toX = (idx) => 40 + (idx / (trainLoss.length - 1)) * (width - 60);
        
        // График train loss
        ctx.strokeStyle = '#667eea';
        ctx.lineWidth = 2;
        ctx.beginPath();
        trainLoss.forEach((loss, idx) => {
            const x = toX(idx);
            const y = toY(loss);
            if (idx === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();
        
        // График val loss если есть
        if (valLoss.length) {
            ctx.strokeStyle = '#38ef7d';
            ctx.beginPath();
            valLoss.forEach((loss, idx) => {
                const x = toX(idx);
                const y = toY(loss);
                if (idx === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();
        }
        
        // Легенда
        ctx.fillStyle = '#667eea';
        ctx.fillRect(width - 120, 10, 15, 15);
        ctx.fillStyle = '#333';
        ctx.font = '12px sans-serif';
        ctx.fillText('Train', width - 100, 22);
        
        if (valLoss.length) {
            ctx.fillStyle = '#38ef7d';
            ctx.fillRect(width - 120, 30, 15, 15);
            ctx.fillStyle = '#333';
            ctx.fillText('Validation', width - 100, 42);
        }
        
        // Заголовок
        ctx.font = 'bold 14px sans-serif';
        ctx.fillText(`Loss curve: ${modelType.toUpperCase()}`, 50, 25);
    }
    
    // === Вспомогательные функции ===
    
    function showProgress(show) {
        elements.progressContainer?.classList.toggle('hidden', !show);
    }
    
    function updateProgress(percent, text) {
        if (elements.progressFill) elements.progressFill.style.width = `${percent}%`;
        if (elements.progressText) elements.progressText.textContent = text;
    }
    
    function setButtonsDisabled(disabled) {
        Object.values(elements.trainBtns).forEach(btn => {
            if (btn) btn.disabled = disabled;
        });
        if (elements.addAllBtn) elements.addAllBtn.disabled = disabled;
        if (elements.clearBtn) elements.clearBtn.disabled = disabled;
    }
    
    function sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    function capitalize(str) {
        if (!str) return '';
        return str.charAt(0).toUpperCase() + str.slice(1);
    }
});