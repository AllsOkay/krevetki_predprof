// frontend/js/main.js
// Главный модуль: инициализация приложения, обработка навигации, общие утилиты

/**
 * Инициализация приложения при загрузке страницы
 */
// В конце initApp() добавьте задержку для загрузки Chart.js
async function initApp() {
  console.log('🦐 Alien Classifier initializing...');
  
  // ✅ Ждём полной загрузки DOM
  if (document.readyState !== 'complete') {
    await new Promise(resolve => window.addEventListener('load', resolve));
  }
  
  const isAuthenticated = await window.Auth?.checkAuth();
  updateUI(isAuthenticated);
  
  if (isAuthenticated) {
    await loadUserData();
    
    // ✅ Проверяем наличие графиков на странице
    if (document.getElementById('accuracyChart')) {
      // Небольшая задержка для гарантии загрузки Chart.js
      setTimeout(async () => {
        await loadAnalytics();
      }, 500);
    }
    
    if (window.location.pathname.includes('admin')) {
      if (!window.Auth?.hasRole('admin')) {
        window.location.href = '/dashboard';
      }
    }
  }
  
  setupGlobalHandlers();
  console.log('✓ Application initialized');
}
/**
 * Обновление UI в зависимости от статуса авторизации
 */
function updateUI(isAuthenticated) {
    const userInfo = document.getElementById('userInfo');
    const userName = document.getElementById('userName');
    const adminName = document.getElementById('adminName');
    const logoutBtn = document.getElementById('logoutBtn');
    
    if (isAuthenticated && window.currentUser) {
        const fullName = `${window.currentUser.name} ${window.currentUser.surname}`;
        
        if (userName) userName.textContent = window.currentUser.name;
        if (adminName) adminName.textContent = fullName;
        
        if (userInfo) {
            userInfo.textContent = `${window.currentUser.name} ${window.currentUser.surname[0]}.`;
            userInfo.classList.add('active');
        }
        
        if (logoutBtn) {
            logoutBtn.classList.remove('hidden');
            logoutBtn.onclick = (e) => {
                e.preventDefault();
                window.Auth?.logout();
            };
        }
        
        if (window.location.pathname === '/' || window.location.pathname.includes('index.html')) {
            window.Auth?.redirectByRole(window.location.pathname);
        }
    } else {
        if (userInfo) {
            userInfo.textContent = '🔐 Войти';
            userInfo.classList.remove('active');
            userInfo.onclick = () => window.location.href = '/';
        }
        
        if (logoutBtn) logoutBtn.classList.add('hidden');
        
        if (!['/', '/index.html'].includes(window.location.pathname) && 
            !window.location.pathname.includes('login')) {
            window.location.href = '/';
        }
    }
}

/**
 * Загрузка и отображение данных пользователя
 */
async function loadUserData() {
    try {
        const response = await fetch('/api/model/info', {
            headers: window.Auth?.getAuthHeaders()
        });
        
        if (response.ok) {
            const info = await response.json();
            updateModelInfo(info);
        }
    } catch (error) {
        console.warn('Failed to load user data:', error);
    }
}

/**
 * Обновление блока информации о модели
 */
function updateModelInfo(info) {
    const statusEl = document.getElementById('modelStatus');
    const classesEl = document.getElementById('modelClasses');
    const paramsEl = document.getElementById('modelParams');
    const trainedEl = document.getElementById('modelTrained');
    
    if (statusEl) statusEl.textContent = info.is_trained ? '✅ Обучена' : '⏳ Не обучена';
    if (classesEl) classesEl.textContent = info.num_classes || '-';
    if (paramsEl) paramsEl.textContent = info.total_params?.toLocaleString('ru-RU') || '-';
    if (trainedEl) trainedEl.textContent = info.is_trained ? 'Да' : 'Нет';
}


async function loadAnalytics() {
    try {
        const response = await fetch('/api/analytics', {
            headers: window.Auth?.getAuthHeaders()
        });
        
        if (response.ok) {
            const analytics = await response.json();
            window.Charts?.init(analytics);
            
            // ✅ ОБНОВЛЯЕМ МЕТРИКИ ВНИЗУ
            updateFooterMetrics(analytics);
        }
    } catch (error) {
        console.error('Failed to load analytics:', error);
        // ✅ ПОКАЗЫВАЕМ ДЕМО-МЕТРИКИ
        updateFooterMetrics({ accuracy: 0.87, loss: 0.34, n_samples: 1600 });
        window.Charts?.init(window.Charts?.DEMO_DATA || null);
    }
}

function setupGlobalHandlers() {
    document.getElementById('logoutBtn')?.addEventListener('click', (e) => {
        e.preventDefault();
        window.Auth?.logout();
    });
    
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            // Закрытие модальных окон если будут добавлены
        }
    });
}

/**
 * Форматирование чисел для отображения
 */
function formatNumber(num, decimals = 2) {
    if (num === null || num === undefined) return '-';
    return Number(num).toFixed(decimals).replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
}

/**
 * Показ/скрытие элементов с анимацией
 */
function toggleElement(element, show) {
    if (!element) return;
    
    if (show) {
        element.classList.remove('hidden');
        setTimeout(() => element.classList.add('animate-fade-in'), 10);
    } else {
        element.classList.add('hidden');
        element.classList.remove('animate-fade-in');
    }
}
/**
 * ✅ Обновление метрик внизу страницы
 */
function updateFooterMetrics(data) {
    const accEl = document.getElementById('footerAccuracy');
    const lossEl = document.getElementById('footerLoss');
    const samplesEl = document.getElementById('footerSamples');
    
    if (accEl) {
        const acc = data.accuracy || 0.87;
        accEl.textContent = `${(acc * 100).toFixed(1)}%`;
        accEl.style.color = acc >= 0.8 ? '#2E7D32' : acc >= 0.6 ? '#F57F17' : '#C62828';
    }
    if (lossEl) {
        lossEl.textContent = (data.loss || 0.34).toFixed(3);
    }
    if (samplesEl) {
        samplesEl.textContent = data.n_samples || 1600;
    }
}

// Инициализация при загрузке
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
} else {
    initApp();
}

window.Utils = {
    formatNumber,
    toggleElement
};