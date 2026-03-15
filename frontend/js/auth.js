// frontend/js/auth.js
// Модуль аутентификации: безопасный вход, восстановление сессии, обработка токенов

/**
 * Глобальный объект для хранения состояния пользователя
 */
window.currentUser = null;

/**
 * Авторизация пользователя через API
 * @param {string} username - Логин пользователя
 * @param {string} password - Пароль пользователя
 * @returns {Promise<Object>} Данные пользователя или ошибка
 */
async function login(username, password) {
    try {
        const response = await fetch('/api/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                username: username,
                password: password
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            // ✅ Сохраняем ВСЕ данные пользователя, включая username
            window.currentUser = {
                token: data.token,
                username: data.username,  // ← Было пропущено!
                name: data.name,
                surname: data.surname,
                role: data.role
            };
            
            // Сохраняем в localStorage
            localStorage.setItem('auth_token', data.token);
            localStorage.setItem('user_data', JSON.stringify({
                username: data.username,  // ← Добавлено!
                name: data.name,
                surname: data.surname,
                role: data.role
            }));
            
            return { success: true, user: window.currentUser };
        } else {
            return { success: false, error: data.error || 'Ошибка авторизации' };
        }
    } catch (error) {
        console.error('Login error:', error);
        return { success: false, error: 'Не удалось подключиться к серверу' };
    }
}

/**
 * Проверка валидности токена и восстановление сессии
 */
async function checkAuth() {
    const token = localStorage.getItem('auth_token');
    const userData = localStorage.getItem('user_data');
    
    if (!token || !userData) {
        return false;
    }
    
    try {
        // ✅ Используем endpoint, который проверяет токен (не /api/health!)
        const response = await fetch('/api/model/info', {
            headers: {
                'Authorization': `Bearer ${token}`  // ✅ Добавляем префикс Bearer
            }
        });
        
        // ✅ Обрабатываем 401 — токен истёк или невалиден
        if (response.status === 401) {
            console.warn('⚠️ Токен недействителен, выполняем выход');
            logout();
            return false;
        }
        
        if (response.ok) {
            const parsed = JSON.parse(userData);
            window.currentUser = { 
                token, 
                ...parsed 
            };
            return true;
        }
    } catch (error) {
        console.warn('Auth check failed:', error);
    }
    
    // Если проверка не прошла — очищаем данные
    logout();
    return false;
}

/**
 * Выход из системы
 */
function logout() {
    // ✅ Опционально: можно отправить запрос на аннулирование токена
    // fetch('/api/logout', { method: 'POST', headers: getAuthHeaders() });
    
    window.currentUser = null;
    localStorage.removeItem('auth_token');
    localStorage.removeItem('user_data');
    
    // ✅ Перенаправляем на главную, но не если мы уже там
    if (!window.location.pathname.includes('/index.html') && window.location.pathname !== '/') {
        window.location.href = '/';
    }
}

/**
 * Получение заголовков для авторизованных запросов
 * ✅ Гарантирует правильный формат: "Bearer <token>"
 */
function getAuthHeaders() {
    const token = window.currentUser?.token || localStorage.getItem('auth_token');
    
    if (!token) {
        return { 'Content-Type': 'application/json' };
    }
    
    // ✅ Убираем префикс если он уже есть (защита от дублирования)
    const cleanToken = token.startsWith('Bearer ') ? token.slice(7) : token;
    
    return {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${cleanToken}`
    };
}

/**
 * Проверка роли пользователя
 * @param {string} role - Требуемая роль ('admin' или 'user')
 * @returns {boolean} True если роль совпадает
 */
function hasRole(role) {
    return window.currentUser?.role === role;
}

/**
 * Перенаправление по роли
 * @param {string} currentPath - Текущий путь страницы
 */
function redirectByRole(currentPath) {
    if (!window.currentUser?.role) return;
    
    if (window.currentUser.role === 'admin') {
        // Админ может заходить куда угодно, но если на главной — отправляем в админку
        if (currentPath === '/' || currentPath.includes('index.html')) {
            window.location.href = '/admin';
        }
    } else {
        // Обычный пользователь не должен видеть админку
        if (currentPath.includes('admin')) {
            window.location.href = '/dashboard';
        }
    }
}

/**
 * ✅ НОВЫЙ МЕТОД: Глобальный обработчик 401 ошибок
 * Вызывайте его при инициализации приложения для авто-выхода при истечении токена
 */
function setupAuthInterceptor() {
    // Сохраняем оригинальный fetch
    const originalFetch = window.fetch;
    
    window.fetch = async function(...args) {
        const response = await originalFetch.apply(this, args);
        
        // Если получили 401 — токен истёк
        if (response.status === 401) {
            const url = args[0];
            // Игнорируем сам запрос логина и проверки
            if (!url.includes('/api/login') && !url.includes('/api/health')) {
                console.warn('🔐 Сессия истекла (401), выполняем выход');
                logout();
            }
        }
        
        return response;
    };
}

// ✅ Экспортируем все функции
window.Auth = { 
    login, 
    checkAuth, 
    logout, 
    getAuthHeaders,
    hasRole,
    redirectByRole,
    setupAuthInterceptor  // ← Новый метод для глобальной обработки ошибок
};

// ✅ Авто-инициализация перехватчика (опционально)
// document.addEventListener('DOMContentLoaded', () => {
//     window.Auth?.setupAuthInterceptor();
// });