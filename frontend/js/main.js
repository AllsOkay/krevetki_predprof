// frontend/js/main.js
// Главный модуль: инициализация приложения, обработка навигации, общие утилиты

/**
 * Инициализация приложения при загрузке страницы
 */
async function initApp() {
    console.log('🦐 Alien Classifier initializing...');
    
    const isAuthenticated = await window.Auth?.checkAuth();
    updateUI(isAuthenticated);
    
    if (isAuthenticated) {
        await loadUserData();
        
        if (document.getElementById('accuracyChart')) {
            await loadAnalytics();
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

/**
 * Загрузка аналитики для графиков
 */
async function loadAnalytics() {
    try {
        const response = await fetch('/api/analytics', {
            headers: window.Auth?.getAuthHeaders()
        });
        
        if (response.ok) {
            const analytics = await response.json();
            window.Charts?.init(analytics);
        }
    } catch (error) {
        console.error('Failed to load analytics:', error);
        
        const chartContainers = document.querySelectorAll('.chart-container');
        chartContainers.forEach(container => {
            if (!container.querySelector('.status-badge')) {
                const errorBadge = document.createElement('div');
                errorBadge.className = 'status-badge status-error';
                errorBadge.textContent = '⚠️ Не удалось загрузить данные';
                container.appendChild(errorBadge);
            }
        });
    }
}

/**
 * Настройка глобальных обработчиков событий
 */
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