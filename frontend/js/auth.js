window.currentUser = null;

/**
 * Авторизация пользователя
 * @param {string} username
 * @param {string} password
 * @returns {Promise<Object>}
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
                username: data.username,
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

async function checkAuth() {
    const token = localStorage.getItem('auth_token');
    const userData = localStorage.getItem('user_data');
    
    if (!token || !userData) {
        return false;
    }
    
    try {
        const response = await fetch('/api/model/info', {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (response.status === 401) {
            console.warn('Токен недействителен, выполняем выход');
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
    
    logout();
    return false;
}

function logout() {
    window.currentUser = null;
    localStorage.removeItem('auth_token');
    localStorage.removeItem('user_data');
    
    if (!window.location.pathname.includes('/index.html') && window.location.pathname !== '/') {
        window.location.href = '/';
    }
}

function getAuthHeaders() {
    const token = window.currentUser?.token || localStorage.getItem('auth_token');
    
    if (!token) {
        return { 'Content-Type': 'application/json' };
    }
    
    const cleanToken = token.startsWith('Bearer ') ? token.slice(7) : token;
    
    return {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${cleanToken}`
    };
}

/**
 * Проверка роли пользователя
 * @param {string} role
 * @returns {boolean}
 */
function hasRole(role) {
    return window.currentUser?.role === role;
}

/**
 * Перенаправление по роли
 * @param {string} currentPath
 */
function redirectByRole(currentPath) {
    if (!window.currentUser?.role) return;
    
    if (window.currentUser.role === 'admin') {
        if (currentPath === '/' || currentPath.includes('index.html')) {
            window.location.href = '/admin';
        }
    } else {
        if (currentPath.includes('admin')) {
            window.location.href = '/dashboard';
        }
    }
}

function setupAuthInterceptor() {
    const originalFetch = window.fetch;
    
    window.fetch = async function(...args) {
        const response = await originalFetch.apply(this, args);
        
        if (response.status === 401) {
            const url = args[0];
            if (!url.includes('/api/login') && !url.includes('/api/health')) {
                console.warn('Сессия истекла (401), выполняем выход');
                logout();
            }
        }
        
        return response;
    };
}

window.Auth = { 
    login, 
    checkAuth, 
    logout, 
    getAuthHeaders,
    hasRole,
    redirectByRole,
    setupAuthInterceptor
};