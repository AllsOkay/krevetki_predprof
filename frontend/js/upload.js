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
    statusDiv.innerHTML = '<div class="status-badge status-warning">Загрузка и обработка...</div>';
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
            
            resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
        } else {
            statusDiv.innerHTML = `<div class="status-badge status-error">Ошибка: ${result.error}</div>`;
        }
    } catch (error) {
        console.error('Upload error:', error);
        statusDiv.innerHTML = `<div class="status-badge status-error">Ошибка сети: ${error.message}</div>`;
    } finally {
        form.reset();
    }
}

function displayPredictionResults(result) {
    const accuracyEl = document.getElementById('resultAccuracy');
    const lossEl = document.getElementById('resultLoss');
    const countEl = document.getElementById('resultCount');
    const correctEl = document.getElementById('resultCorrect');
    
    if (accuracyEl) {
        const accuracy = result.accuracy || 0;
        const accuracyPercent = (accuracy * 100).toFixed(2);
        accuracyEl.textContent = `${accuracyPercent}%`;
        accuracyEl.className = 'metric-value ' + (accuracy >= 0.8 ? 'high' : 'low');
    }
    
    if (lossEl) {
        const loss = result.loss || 0;
        lossEl.textContent = loss.toFixed(4);
        lossEl.className = 'metric-value ' + (loss < 0.5 ? 'high' : 'low');
    }
    
    if (countEl) {
        countEl.textContent = result.n_samples || result.predictions?.length || 'N/A';
    }
    
    if (correctEl && result.n_samples && result.accuracy) {
        const correct = Math.round(result.n_samples * result.accuracy);
        correctEl.textContent = `${correct} из ${result.n_samples}`;
    }
    
    if (result.per_class_accuracy) {
        console.log('Per-class accuracy:', result.per_class_accuracy);
    }
    
    if (result.accuracy && window.Charts) {
        const analytics = {
            accuracy_vs_epochs: {
                epochs: [0],
                accuracy: [],
                val_accuracy: [],
                test_accuracy: [result.accuracy]
            }
        };
        window.Charts.update(analytics);
    }
}

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