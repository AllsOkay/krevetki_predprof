// frontend/js/upload.js
// Модуль загрузки файлов и обработки результатов классификации

/**
 * Обработчик отправки формы загрузки тестовых данных
 * @param {Event} event - Событие submit формы
 */
async function handleFileUpload(event) {
    event.preventDefault();
    
    const form = event.target;
    const fileInput = document.getElementById('testFile');
    const statusDiv = document.getElementById('uploadStatus');
    const resultsDiv = document.getElementById('predictionResults');
    
    // Валидация
    if (!fileInput.files[0]) {
        alert('Пожалуйста, выберите файл для загрузки');
        return;
    }
    
    const file = fileInput.files[0];
    
    // Проверка типа файла
    if (!file.name.endsWith('.npz')) {
        alert('Пожалуйста, выберите файл в формате .npz');
        return;
    }
    
    // Показываем индикатор загрузки
    statusDiv.classList.remove('hidden');
    statusDiv.innerHTML = '<div class="status-badge status-warning">⏳ Загрузка и обработка...</div>';
    resultsDiv.classList.add('hidden');
    
    // Подготовка FormData для отправки файла
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        // Отправка запроса на сервер
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${window.currentUser?.token}`
                // Content-Type не указываем - браузер установит его с boundary для FormData
            },
            body: formData
        });
        
        const result = await response.json();
        
        if (response.ok) {
            // Отображаем результаты
            displayPredictionResults(result);
            statusDiv.classList.add('hidden');
            resultsDiv.classList.remove('hidden');
            
            // Обновляем графики если есть данные для аналитики
            if (result.analytics) {
                window.Charts?.update(result.analytics);
            }
            
        } else {
            // Ошибка обработки
            statusDiv.innerHTML = `<div class="status-badge status-error">❌ Ошибка: ${result.error}</div>`;
        }
        
    } catch (error) {
        console.error('Upload error:', error);
        statusDiv.innerHTML = `<div class="status-badge status-error">❌ Ошибка сети: ${error.message}</div>`;
        
    } finally {
        // Сбрасываем форму для возможности повторной загрузки
        form.reset();
    }
}

/**
 * Отображение результатов классификации в интерфейсе
 * @param {Object} result - Данные от сервера
 */
function displayPredictionResults(result) {
    // Обновляем метрики
    document.getElementById('resultAccuracy').textContent = 
        result.accuracy ? `${(result.accuracy * 100).toFixed(2)}%` : 'N/A';
    
    document.getElementById('resultLoss').textContent = 
        result.loss ? result.loss.toFixed(4) : 'N/A';
    
    document.getElementById('resultCount').textContent = 
        result.n_samples || result.predictions?.length || 'N/A';
    
    // Если есть per-class точность - можно показать дополнительно
    if (result.per_class_accuracy) {
        console.log('Per-class accuracy:', result.per_class_accuracy);
        // Здесь можно добавить отображение в таблицу если нужно
    }
}

/**
 * Прогресс-бар для загрузки (опционально, для больших файлов)
 * @param {number} loaded - Загружено байт
 * @param {number} total - Всего байт
 */
function updateUploadProgress(loaded, total) {
    const percent = Math.round((loaded / total) * 100);
    const statusDiv = document.getElementById('uploadStatus');
    
    if (statusDiv && !statusDiv.classList.contains('hidden')) {
        statusDiv.innerHTML = `
            <div style="display: flex; align-items: center; gap: 12px;">
                <div class="status-badge status-warning">⏳ Загрузка: ${percent}%</div>
                <div style="flex: 1; height: 8px; background: #eee; border-radius: 4px; overflow: hidden;">
                    <div style="width: ${percent}%; height: 100%; background: linear-gradient(90deg, #FF6B9D, #FF8E53); transition: width 0.2s;"></div>
                </div>
            </div>
        `;
    }
}

// Инициализация обработчиков при загрузке страницы
document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    if (uploadForm) {
        uploadForm.addEventListener('submit', handleFileUpload);
    }
    
    // Опционально: отслеживание прогресса загрузки
    // (требует модификации fetch для доступа к xhr)
});

// Экспорт функций
window.Upload = {
    handleFileUpload,
    displayPredictionResults
};