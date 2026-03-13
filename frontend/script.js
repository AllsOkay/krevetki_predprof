const API = ''; // Пустая строка, т.к. Flask служит файлы с того же домена

document.addEventListener('DOMContentLoaded', () => {
    loadPokemon();
    
    document.getElementById('add-all-btn').addEventListener('click', addAllPokemon);
    document.getElementById('train-btn').addEventListener('click', trainModel);
    document.getElementById('clear-btn').addEventListener('click', clearData);
    document.getElementById('find-similar-btn').addEventListener('click', findSimilar);
});

async function loadPokemon() {
    const res = await fetch(`${API}/api/pokemon`);
    const data = await res.json();
    const tbody = document.getElementById('pokemon-table-body');
    tbody.innerHTML = '';
    document.getElementById('count-db').textContent = data.length;
    
    data.forEach(p => {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${p.id}</td><td>${p.name}</td><td>${p.types}</td><td>${p.hp}</td><td>${p.attack}</td><td>${p.defense}</td><td>${p.speed}</td>`;
        tbody.appendChild(tr);
    });
}

async function addAllPokemon() {
    // Упрощенная логика добавления (как в предыдущем примере)
    // Для краткости здесь вызывается заглушка, реализуйте логику пакетной загрузки из предыдущего ответа
    alert("Функция добавления всех покемонов (используйте код из предыдущей версии)");
}

async function trainModel() {
    const btn = document.getElementById('train-btn');
    btn.disabled = true;
    btn.textContent = "Обучение...";
    
    try {
        const res = await fetch(`${API}/api/train`, { method: 'POST' });
        const data = await res.json();
        if(data.status === 'trained') {
            alert("✅ Нейросеть успешно обучена!");
        } else {
            alert("⚠️ Ошибка: " + (data.reason || "Недостаточно данных"));
        }
    } catch(e) {
        alert("Ошибка соединения");
    } finally {
        btn.disabled = false;
        btn.textContent = "🎓 Обучить нейросеть";
    }
}

async function clearData() {
    if(!confirm("Удалить всё?")) return;
    await fetch(`${API}/api/pokemon`, { method: 'DELETE' });
    loadPokemon();
    alert("Данные удалены. Модель сброшена.");
}

async function findSimilar() {
    const name = document.getElementById('similar-input').value.trim();
    const resultsDiv = document.getElementById('similar-results');
    
    if(!name) return alert("Введите имя");
    
    resultsDiv.innerHTML = '<p>Поиск...</p>';
    resultsDiv.classList.remove('hidden');
    
    try {
        const res = await fetch(`${API}/api/similar?name=${name}`);
        const data = await res.json();
        
        if(data.error) {
            resultsDiv.innerHTML = `<p style="color:red">${data.error}</p>`;
            return;
        }
        
        resultsDiv.innerHTML = '';
        if(data.similar.length === 0) {
            resultsDiv.innerHTML = '<p>Похожих не найдено</p>';
            return;
        }
        
        data.similar.forEach(p => {
            const card = document.createElement('div');
            card.className = 'result-card';
            card.innerHTML = `
                <h3>${p.name}</h3>
                <div class="score">Сходство: ${(p.similarity * 100).toFixed(1)}%</div>
                <small>${p.types}</small>
            `;
            resultsDiv.appendChild(card);
        });
        
    } catch(e) {
        resultsDiv.innerHTML = '<p>Ошибка запроса</p>';
    }
}