// frontend/js/auth.js
// Модуль аутентификации: логин, проверка токена, выход
// Работает с Flask backend через Fetch API

// Глобальный объект для хранения состояния пользователя
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
            body: JSON.stringify({ username, password })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            // Сохраняем данные пользователя в памяти и localStorage
            window.currentUser = {
                token: data.token,
                username: data.username,
                name: data.name,
                surname: data.surname,
                role: data.role
            };
            
            // Сохраняем токен для восстановления сессии
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
 * @returns {Promise<boolean>} Успешна ли проверка
 */
async function checkAuth() {
    const token = localStorage.getItem('auth_token');
    const userData = localStorage.getItem('user_data');
    
    if (!token || !userData) {
        return false;
    }
    
    try {
        // Проверяем токен через health endpoint или специальный /api/me
        const response = await fetch('/api/health', {
            headers: { 'Authorization': `Bearer ${token}` }
        });
        
        if (response.ok) {
            // Восстанавливаем данные пользователя
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
    
    // Если проверка не прошла - очищаем данные
    logout();
    return false;
}

/**
 * Выход из системы: очистка данных сессии
 */
function logout() {
    // Опционально: отозвать токен на сервере
    if (window.currentUser?.token) {
        fetch('/api/logout', {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${window.currentUser.token}` }
        }).catch(() => {}); // Игнорируем ошибки при выходе
    }
    
    // Очищаем локальные данные
    window.currentUser = null;
    localStorage.removeItem('auth_token');
    localStorage.removeItem('user_data');
    
    // Редирект на страницу входа
    window.location.href = '/';
}

/**
 * Получение заголовков для авторизованных запросов
 * @returns {Object} Заголовки с токеном
 */
function getAuthHeaders() {
    return {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${window.currentUser?.token}`
    };
}

/**
 * Проверка роли пользователя
 * @param {string} requiredRole - Требуемая роль ('admin' или 'user')
 * @returns {boolean} Имеет ли пользователь требуемую роль
 */
function hasRole(requiredRole) {
    return window.currentUser?.role === requiredRole;
}

/**
 * Редирект в зависимости от роли
 * @param {string} currentPath - Текущий путь
 */
function redirectByRole(currentPath) {
    if (!window.currentUser) {
        window.location.href = '/';
        return;
    }
    
    // Если на странице входа и авторизован - редирект по роли
    if (currentPath === '/' || currentPath.includes('index.html')) {
        if (hasRole('admin')) {
            window.location.href = '/admin';
        } else {
            window.location.href = '/dashboard';
        }
        return;
    }
    
    // Если пользователь пытается зайти в админку без прав
    if (currentPath.includes('admin') && !hasRole('admin')) {
        window.location.href = '/dashboard';
    }
    
    // Если админ пытается зайти в пользовательскую панель как обычный пользователь
    // (не блокируем, админ может смотреть и пользовательский интерфейс)
}

// Экспортируем функции для использования в других модулях
window.Auth = {
    login,
    checkAuth,
    logout,
    getAuthHeaders,
    hasRole,
    redirectByRole
};