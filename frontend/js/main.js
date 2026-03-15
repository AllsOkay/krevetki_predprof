// frontend/js/main.js
// Главный модуль: инициализация приложения, обработка навигации, общие утилиты

/**
 * Инициализация приложения при загрузке страницы
 */
async function initApp() {
    console.log('🦐 Alien Classifier initializing...');
    
    // 1. Проверка авторизации
    const isAuthenticated = await window.Auth?.checkAuth();
    
    // 2. Обновление интерфейса в зависимости от статуса
    updateUI(isAuthenticated);
    
    // 3. Если авторизован - загрузка дополнительных данных
    if (isAuthenticated) {
        await loadUserData();
        
        // Если на странице с графиками - загружаем аналитику
        if (document.getElementById('accuracyChart')) {
            await loadAnalytics();
        }
        
        // Если на админ-странице - проверяем права
        if (window.location.pathname.includes('admin')) {
            if (!window.Auth?.hasRole('admin')) {
                window.location.href = '/dashboard';
            }
        }
    }
    
    // 4. Настройка глобальных обработчиков
    setupGlobalHandlers();
    
    console.log('✓ Application initialized');
}

/**
 * Обновление UI в зависимости от статуса авторизации
 * @param {boolean} isAuthenticated - Авторизован ли пользователь
 */
function updateUI(isAuthenticated) {
    // Элементы которые меняются
    const userInfo = document.getElementById('userInfo');
    const userName = document.getElementById('userName');
    const adminName = document.getElementById('adminName');
    const logoutBtn = document.getElementById('logoutBtn');
    
    if (isAuthenticated && window.currentUser) {
        // Показываем имя пользователя
        const fullName = `${window.currentUser.name} ${window.currentUser.surname}`;
        if (userName) userName.textContent = window.currentUser.name;
        if (adminName) adminName.textContent = fullName;
        if (userInfo) {
            userInfo.textContent = `${window.currentUser.name} ${window.currentUser.surname[0]}.`;
            userInfo.classList.add('active');
        }
        
        // Показываем кнопку выхода
        if (logoutBtn) {
            logoutBtn.classList.remove('hidden');
            logoutBtn.onclick = (e) => {
                e.preventDefault();
                window.Auth?.logout();
            };
        }
        
        // Редирект если на странице входа
        if (window.location.pathname === '/' || window.location.pathname.includes('index.html')) {
            window.Auth?.redirectByRole(window.location.pathname);
        }
        
    } else {
        // Не авторизован - скрываем пользовательские элементы
        if (userInfo) {
            userInfo.textContent = '🔐 Войти';
            userInfo.classList.remove('active');
            userInfo.onclick = () => window.location.href = '/';
        }
        if (logoutBtn) logoutBtn.classList.add('hidden');
        
        // Редирект на вход если пытаемся зайти в защищённую область
        if (!['/', '/index.html'].includes(window.location.pathname) && !window.location.pathname.includes('login')) {
            window.location.href = '/';
        }
    }
}

/**
 * Загрузка и отображение данных пользователя
 */
async function loadUserData() {
    try {
        // Загрузка информации о модели
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
 * @param {Object} info - Данные от /api/model/info
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
        // Показать сообщение об ошибке в интерфейсе
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
    // Обработчик выхода по кнопке (дублирование для надёжности)
    document.getElementById('logoutBtn')?.addEventListener('click', (e) => {
        e.preventDefault();
        window.Auth?.logout();
    });
    
    // Обработчик нажатия клавиш (например, Enter в формах)
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            // Закрытие модальных окон если будут добавлены
        }
    });
    
    // Предотвращение отправки формы при нажатии Enter в некоторых полях
    document.querySelectorAll('input[type="text"], input[type="password"]').forEach(input => {
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.target.form?.querySelector('button[type="submit"]:focus')) {
                // Можно добавить кастомную логику
            }
        });
    });
}

/**
 * Форматирование чисел для отображения
 * @param {number} num - Число для форматирования
 * @param {number} decimals - Количество знаков после запятой
 * @returns {string} Отформатированная строка
 */
function formatNumber(num, decimals = 2) {
    if (num === null || num === undefined) return '-';
    return Number(num).toFixed(decimals).replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
}

/**
 * Показ/скрытие элементов с анимацией
 * @param {HTMLElement} element - Элемент для переключения
 * @param {boolean} show - Показать или скрыть
 */
function toggleElement(element, show) {
    if (!element) return;
    
    if (show) {
        element.classList.remove('hidden');
        // Добавляем класс для анимации если нужно
        setTimeout(() => element.classList.add('animate-fade-in'), 10);
    } else {
        element.classList.add('hidden');
        element.classList.remove('animate-fade-in');
    }
}

// === ИНИЦИАЛИЗАЦИЯ ПРИ ЗАГРУЗКЕ ===

// Ждём загрузки DOM
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
} else {
    // DOM уже готов
    initApp();
}

// Экспорт утилит для использования в других модулях
window.Utils = {
    formatNumber,
    toggleElement
};