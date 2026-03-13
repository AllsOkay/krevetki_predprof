document.addEventListener('DOMContentLoaded', () => {
    const addAllBtn = document.getElementById('add-all-btn');
    const trainBtn = document.getElementById('train-btn');
    const clearBtn = document.getElementById('clear-btn');
    const findSimilarBtn = document.getElementById('find-similar-btn');
    const similarInput = document.getElementById('similar-input');
    const similarResults = document.getElementById('similar-results');
    const tableBody = document.getElementById('pokemon-table-body');
    const countDb = document.getElementById('count-db');
    const progressContainer = document.getElementById('progress-container');
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');
    
    // Конфигурация
    const POKEAPI_BASE = 'https://pokeapi.co/api/v2';
    const BATCH_SIZE = 5;
    const REQUEST_DELAY = 100;
    
    // Загрузка существующих данных
    loadPokemon();
    
    // Обработчики кнопок
    addAllBtn.addEventListener('click', addAllPokemon);
    trainBtn.addEventListener('click', trainModel);
    clearBtn.addEventListener('click', clearData);
    findSimilarBtn.addEventListener('click', findSimilar);
    
    similarInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') findSimilar();
    });
    
    /**
     * Загружает покемонов из БД и отображает в таблице
     */
    async function loadPokemon() {
        try {
            const res = await fetch('/api/pokemon');
            const data = await res.json();
            tableBody.innerHTML = '';
            countDb.textContent = data.length;
            
            data.forEach(p => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${p.id}</td>
                    <td><strong>${p.name}</strong></td>
                    <td>${p.types}</td>
                    <td>${p.hp}</td>
                    <td>${p.attack}</td>
                    <td>${p.defense}</td>
                    <td>${p.speed}</td>
                `;
                tableBody.appendChild(tr);
            });
        } catch (e) {
            console.error('Ошибка загрузки:', e);
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
            updateProgress(0, 'Получение списка покемонов...');
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
            
            updateProgress(0, `Обработка ${newPokemon.length} новых покемонов...`);
            
            // Шаг 3: Обрабатываем пакетами
            let processed = 0;
            for (let i = 0; i < newPokemon.length; i += BATCH_SIZE) {
                const batch = newPokemon.slice(i, i + BATCH_SIZE);
                
                const promises = batch.map(pokemon =>
                    fetchPokemonDetails(pokemon.url)
                        .then(details => savePokemonToDB(details))
                        .then(() => {
                            addPokemonToTable(details);
                            processed++;
                            const percent = Math.round((processed / newPokemon.length) * 100);
                            updateProgress(percent, `Обработано: ${processed}/${newPokemon.length}`);
                        })
                        .catch(err => {
                            console.warn(`⚠️ Не удалось: ${pokemon.name}`, err);
                            processed++;
                        })
                );
                
                await Promise.all(promises);
                
                if (i + BATCH_SIZE < newPokemon.length) {
                    await sleep(REQUEST_DELAY);
                }
            }
            
            updateProgress(100, `✅ Готово! Добавлено ${newPokemon.length} покемонов`);
            countDb.textContent = parseInt(countDb.textContent) + newPokemon.length;
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
        const limit = 1302;
        const response = await fetch(`${POKEAPI_BASE}/pokemon?limit=${limit}`);
        
        if (!response.ok) throw new Error('Не удалось получить список');
        
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
            console.warn('Не удалось получить ID:', e);
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
            throw new Error('Не удалось сохранить');
        }
        
        return pokemon;
    }
    
    /**
     * Добавляет строку в таблицу
     */
    function addPokemonToTable(pokemon) {
        const existingRow = tableBody.querySelector(`tr[data-id="${pokemon.id}"]`);
        if (existingRow) return;
        
        const row = document.createElement('tr');
        row.dataset.id = pokemon.id;
        row.style.animation = 'fadeIn 0.3s ease';
        
        row.innerHTML = `
            <td>${pokemon.id}</td>
            <td><strong>${pokemon.name}</strong></td>
            <td>${pokemon.types}</td>
            <td>${pokemon.hp}</td>
            <td>${pokemon.attack}</td>
            <td>${pokemon.defense}</td>
            <td>${pokemon.speed}</td>
        `;
        
        tableBody.appendChild(row);
    }
    
    /**
     * Обучает нейросеть
     */
    async function trainModel() {
        trainBtn.disabled = true;
        trainBtn.textContent = 'Обучение...';
        
        try {
            const res = await fetch('/api/train', { method: 'POST' });
            const data = await res.json();
            
            if (data.status === 'trained') {
                alert('✅ Нейросеть успешно обучена!');
            } else {
                alert('⚠️ Ошибка: ' + (data.reason || 'Недостаточно данных'));
            }
        } catch (e) {
            alert('Ошибка соединения');
        } finally {
            trainBtn.disabled = false;
            trainBtn.textContent = '🎓 Обучить нейросеть';
        }
    }
    
    /**
     * Очищает все данные
     */
    async function clearData() {
        if (!confirm('⚠️ Удалить все данные и модель?')) return;
        
        try {
            await fetch('/api/pokemon', { method: 'DELETE' });
            loadPokemon();
            alert('✅ Данные удалены. Модель сброшена.');
        } catch (e) {
            alert('❌ Не удалось удалить данные');
        }
    }
    
    /**
     * Находит похожих покемонов
     */
    async function findSimilar() {
        const name = similarInput.value.trim();
        
        if (!name) {
            alert('Введите имя покемона');
            return;
        }
        
        similarResults.innerHTML = '<p>Поиск...</p>';
        similarResults.classList.remove('hidden');
        
        try {
            const res = await fetch(`/api/similar?name=${encodeURIComponent(name)}`);
            const data = await res.json();
            
            if (data.error) {
                similarResults.innerHTML = `<p style="color:red">${data.error}</p>`;
                return;
            }
            
            similarResults.innerHTML = '';
            
            if (data.similar.length === 0) {
                similarResults.innerHTML = '<p>Похожих не найдено</p>';
                return;
            }
            
            data.similar.forEach(p => {
                const card = document.createElement('div');
                card.className = 'result-card';
                card.innerHTML = `
                    <h3>${capitalize(p.name)}</h3>
                    <div class="score">Сходство: ${(p.similarity * 100).toFixed(1)}%</div>
                    <small>${p.types}</small>
                `;
                similarResults.appendChild(card);
            });
            
        } catch (e) {
            similarResults.innerHTML = '<p>Ошибка запроса</p>';
        }
    }
    
    /**
     * Вспомогательные функции
     */
    function showProgress(show) {
        progressContainer.classList.toggle('hidden', !show);
    }
    
    function updateProgress(percent, text) {
        progressFill.style.width = `${percent}%`;
        progressText.textContent = text;
    }
    
    function setButtonsDisabled(disabled) {
        addAllBtn.disabled = disabled;
        trainBtn.disabled = disabled;
        clearBtn.disabled = disabled;
    }
    
    function sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    function capitalize(str) {
        return str.charAt(0).toUpperCase() + str.slice(1);
    }
});