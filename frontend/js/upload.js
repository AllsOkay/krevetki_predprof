// frontend/js/upload.js
// Модуль загрузки файлов и обработки результатов классификации

/**
 * Обработчик отправки формы загрузки тестовых данных
 */
async function handleFileUpload(event) {
    event.preventDefault();
    
    const form = event.target;
    const fileInput = document.getElementById('testFile');
    const statusDiv = document.getElementById('uploadStatus');
    const resultsDiv = document.getElementById('predictionResults');
    
    if (!fileInput.files[0]) {
        alert('Пожалуйста, выберите файл для загрузки');
        return;
    }
    
    const file = fileInput.files[0];
    
    if (!file.name.endsWith('.npz')) {
        alert('Пожалуйста, выберите файл в формате .npz');
        return;
    }
    
    statusDiv.classList.remove('hidden');
    statusDiv.innerHTML = '<div class="status-badge status-warning">⏳ Загрузка и обработка...</div>';
    resultsDiv.classList.add('hidden');
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${window.currentUser?.token}`
            },
            body: formData
        });
        
        const result = await response.json();
        
        if (response.ok) {
            displayPredictionResults(result);
            statusDiv.classList.add('hidden');
            resultsDiv.classList.remove('hidden');
            
            if (result.analytics) {
                window.Charts?.update(result.analytics);
            }
        } else {
            statusDiv.innerHTML = `<div class="status-badge status-error">❌ Ошибка: ${result.error}</div>`;
        }
    } catch (error) {
        console.error('Upload error:', error);
        statusDiv.innerHTML = `<div class="status-badge status-error">❌ Ошибка сети: ${error.message}</div>`;
    } finally {
        form.reset();
    }
}

/**
 * Отображение результатов классификации в интерфейсе
 */
function displayPredictionResults(result) {
    const accuracyEl = document.getElementById('resultAccuracy');
    const lossEl = document.getElementById('resultLoss');
    const countEl = document.getElementById('resultCount');
    
    if (accuracyEl) {
        accuracyEl.textContent = result.accuracy 
            ? `${(result.accuracy * 100).toFixed(2)}%` 
            : 'N/A';
    }
    
    if (lossEl) {
        lossEl.textContent = result.loss ? result.loss.toFixed(4) : 'N/A';
    }
    
    if (countEl) {
        countEl.textContent = result.n_samples || result.predictions?.length || 'N/A';
    }
    
    if (result.per_class_accuracy) {
        console.log('Per-class accuracy:', result.per_class_accuracy);
    }
}

/**
 * Обновление прогресс-бара загрузки
 */
function updateUploadProgress(loaded, total) {
    const percent = Math.round((loaded / total) * 100);
    const statusDiv = document.getElementById('uploadStatus');
    
    if (statusDiv && !statusDiv.classList.contains('hidden')) {
        statusDiv.innerHTML = `
            <div style="display: flex; align-items: center; gap: 12px;">
                <div class="status-badge status-warning">⏳ Загрузка: ${percent}%</div>
                <div class="progress-bar">
                    <div class="progress-bar-fill" style="width: ${percent}%"></div>
                </div>
            </div>
        `;
    }
}

// Инициализация обработчиков
document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    if (uploadForm) {
        uploadForm.addEventListener('submit', handleFileUpload);
    }
});

window.Upload = {
    handleFileUpload,
    displayPredictionResults,
    updateUploadProgress
};