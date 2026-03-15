// frontend/js/auth.js
// Модуль аутентификации: БЕЗОПАСНЫЙ вход через POST запрос

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
            window.currentUser = {
                token: data.token,
                username: data.username,
                name: data.name,
                surname: data.surname,
                role: data.role
            };
            
            localStorage.setItem('auth_token', data.token);
            localStorage.setItem('user_data', JSON.stringify({
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
        const response = await fetch('/api/health', {
            headers: { 'Authorization': `Bearer ${token}` }
        });
        
        if (response.ok) {
            const parsed = JSON.parse(userData);
            window.currentUser = { token, ...parsed };
            return true;
        }
    } catch (error) {
        console.warn('Auth check failed:', error);
    }
    
    logout();
    return false;
}

/**
 * Выход из системы
 */
function logout() {
    window.currentUser = null;
    localStorage.removeItem('auth_token');
    localStorage.removeItem('user_data');
    window.location.href = '/';
}

/**
 * Получение заголовков для авторизованных запросов
 */
function getAuthHeaders() {
    return {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${window.currentUser?.token}`
    };
}

/**
 * Проверка роли пользователя
 */
function hasRole(role) {
    return window.currentUser?.role === role;
}

/**
 * Перенаправление по роли
 */
function redirectByRole(currentPath) {
    if (window.currentUser?.role === 'admin') {
        if (!currentPath.includes('admin')) {
            window.location.href = '/admin';
        }
    } else {
        if (currentPath.includes('admin')) {
            window.location.href = '/dashboard';
        }
    }
}

// Экспортируем функции
window.Auth = { 
    login, 
    checkAuth, 
    logout, 
    getAuthHeaders,
    hasRole,
    redirectByRole
};