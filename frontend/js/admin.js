const API_BASE = '/api';
const TOKEN_KEY = 'auth_token';
const USER_DATA_KEY = 'user_data';

let redirectInProgress = false;
let adminPanelInitialized = false;

document.addEventListener('DOMContentLoaded', () => {
    setTimeout(initAdminPanel, 50);
});

async function initAdminPanel() {
    if (adminPanelInitialized) return;
    adminPanelInitialized = true;
    
    if (redirectInProgress) return;
    
    try {
        const auth = getStoredAuth();
        if (!auth.token || !auth.userInfo) {
            safeRedirect('/');
            return;
        }
        
        if (auth.userInfo.role !== 'admin') {
            safeRedirect('/dashboard');
            return;
        }
        
        displayAdminName(auth.userInfo);
        setupEventListeners();

        loadUsersList().catch(console.error);
        loadSystemInfo().catch(console.error);
        
    } catch (error) {
        console.error('Admin panel init error:', error);
        showErrorState();
    }
}

function getStoredAuth() {
    try {
        const token = localStorage.getItem(TOKEN_KEY);
        const userDataRaw = localStorage.getItem(USER_DATA_KEY);
        const userInfo = userDataRaw ? JSON.parse(userDataRaw) : null;
        
        if (!userInfo && window.currentUser) {
            return {
                token: window.currentUser.token || token,
                userInfo: {
                    username: window.currentUser.username,
                    name: window.currentUser.name,
                    surname: window.currentUser.surname,
                    role: window.currentUser.role
                }
            };
        }
        
        return { token, userInfo };
    } catch (e) {
        console.warn('Auth data parse error:', e);
        return { token: null, userInfo: null };
    }
}

function safeRedirect(url) {
    if (redirectInProgress) return;
    redirectInProgress = true;
    
    setTimeout(() => {
        const auth = getStoredAuth();
        if (!auth.token || !auth.userInfo || auth.userInfo.role !== 'admin') {
            localStorage.removeItem(TOKEN_KEY);
            localStorage.removeItem(USER_DATA_KEY);
        }
        window.location.href = url;
    }, 150);
}

function displayAdminName(userInfo) {
    const adminNameEl = document.getElementById('adminName');
    if (adminNameEl && userInfo) {
        const name = userInfo.name || userInfo.username || '';
        const surname = userInfo.surname || '';
        adminNameEl.textContent = `👤 ${name} ${surname}`.trim();
    }
}

function showErrorState() {
    const usersListEl = document.getElementById('usersList');
    if (usersListEl) {
        usersListEl.innerHTML = '<div class="status-badge status-error">⚠️ Ошибка загрузки</div>';
    }
}

function displayMessage(elementId, message, type = 'info') {
    const el = document.getElementById(elementId);
    if (!el) return;
    
    el.classList.remove('hidden', 'status-success', 'status-error', 'status-warning');
    el.classList.add(`status-${type}`);
    el.textContent = message;
    el.style.display = 'block';
    
    if (type === 'success') {
        setTimeout(() => {
            el.classList.add('hidden');
            el.style.display = 'none';
        }, 5000);
    }
}

function setupEventListeners() {
    const logoutBtn = document.getElementById('logoutBtn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', (e) => {
            e.preventDefault();
            handleLogout();
        });
    }
    
    const createUserForm = document.getElementById('createUserForm');
    if (createUserForm) {
        createUserForm.addEventListener('submit', handleCreateUser);
    }
}

function handleLogout() {
    redirectInProgress = true;
    if (typeof window.Auth?.logout === 'function') {
        window.Auth.logout();
    } else {
        localStorage.removeItem(TOKEN_KEY);
        localStorage.removeItem(USER_DATA_KEY);
        window.currentUser = null;
        window.location.href = '/';
    }
}

function getAuthHeaders() {
    const token = localStorage.getItem(TOKEN_KEY) || window.currentUser?.token;
    
    if (!token) {
        return { 'Content-Type': 'application/json' };
    }
    
    const cleanToken = token.startsWith('Bearer ') ? token.slice(7).trim() : token.trim();
    
    return {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${cleanToken}`
    };
}

async function handleCreateUser(e) {
    e.preventDefault();
    
    const form = e.target;
    const submitBtn = form.querySelector('button[type="submit"]');
    
    const formData = {
        username: form.querySelector('#newUsername').value.trim(),
        password: form.querySelector('#newPassword').value,
        name: form.querySelector('#newName').value.trim(),
        surname: form.querySelector('#newSurname').value.trim()
    };
    
    if (!formData.username || !formData.password || !formData.name || !formData.surname) {
        displayMessage('createUserMessage', 'Заполните все обязательные поля', 'error');
        return;
    }
    
    if (formData.password.length < 8) {
        displayMessage('createUserMessage', 'Пароль минимум 8 символов', 'error');
        return;
    }
    
    const originalBtnText = submitBtn.innerHTML;
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Создание...';
    displayMessage('createUserMessage', '', 'info');
    
    try {
        const response = await fetch(`${API_BASE}/register`, {
            method: 'POST',
            headers: getAuthHeaders(),
            body: JSON.stringify(formData)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            displayMessage('createUserMessage', `${result.message}`, 'success');
            form.reset();
            await loadUsersList();
        } else {
            displayMessage('createUserMessage', `${result.error || 'Ошибка'}`, 'error');
        }
    } catch (error) {
        console.error('Create user error:', error);
        displayMessage('createUserMessage', 'Ошибка соединения', 'error');
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = originalBtnText;
    }
}

async function loadUsersList() {
    const usersListEl = document.getElementById('usersList');
    if (!usersListEl) return;
    
    usersListEl.innerHTML = '<div class="status-badge status-warning">Загрузка...</div>';
    
    try {
        const response = await fetch(`${API_BASE}/users`, {
            method: 'GET',
            headers: getAuthHeaders()
        });
        
        if (response.status === 401 || response.status === 403) {
            usersListEl.innerHTML = '<div class="status-badge status-error">Доступ запрещён</div>';
            return;
        }
        
        if (!response.ok) throw new Error('Failed to fetch users');
        
        const users = await response.json();
        
        if (users.length === 0) {
            usersListEl.innerHTML = '<div class="status-badge">Нет пользователей</div>';
            return;
        }
        
        usersListEl.innerHTML = `
            <div class="users-table-wrapper">
                <table class="users-table">
                    <thead>
                        <tr>
                            <th>Логин</th>
                            <th>Имя</th>
                            <th>Фамилия</th>
                            <th>Роль</th>
                            <th>Создан</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${users.map(user => `
                            <tr>
                                <td><code>${escapeHtml(user.username)}</code></td>
                                <td>${escapeHtml(user.name)}</td>
                                <td>${escapeHtml(user.surname)}</td>
                                <td><span class="role-badge role-${user.role}">${user.role}</span></td>
                                <td>${formatDate(user.created_at)}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
            <div style="margin-top:12px;color:var(--text-secondary);font-size:0.9em">
                Всего: <strong>${users.length}</strong>
            </div>
        `;
        
    } catch (error) {
        console.error('Load users error:', error);
        usersListEl.innerHTML = '<div class="status-badge status-error">Ошибка загрузки</div>';
    }
}

async function loadSystemInfo() {
    await Promise.allSettled([
        checkServerHealth(),
        loadModelInfo(),
        updateActiveSessions()
    ]);
}

async function checkServerHealth() {
    const statusEl = document.getElementById('serverStatus');
    const dbEl = document.getElementById('dbStatus');
    
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();
        
        if (statusEl) {
            statusEl.textContent = data.status === 'ok' ? 'Онлайн' : 'Оффлайн';
            statusEl.style.color = data.status === 'ok' ? 'var(--success)' : 'var(--error)';
        }
        if (dbEl) {
            dbEl.textContent = data.database === 'connected' ? 'Подключена' : 'Ошибка';
            dbEl.style.color = data.database === 'connected' ? 'var(--success)' : 'var(--error)';
        }
    } catch {
        if (statusEl) { statusEl.textContent = 'Недоступен'; statusEl.style.color = 'var(--error)'; }
        if (dbEl) { dbEl.textContent = 'Ошибка'; dbEl.style.color = 'var(--error)'; }
    }
}

async function loadModelInfo() {
    const modelEl = document.getElementById('adminModelStatus');
    
    try {
        const response = await fetch(`${API_BASE}/model/info`, { headers: getAuthHeaders() });
        
        if (response.status === 401) {
            if (modelEl) modelEl.textContent = 'Требуется вход';
            return;
        }
        if (!response.ok) {
            if (modelEl) modelEl.textContent = 'Не загружена';
            return;
        }
        
        const data = await response.json();
        if (modelEl) {
            modelEl.innerHTML = `Загружена <small style="display:block;color:var(--text-secondary);margin-top:4px">${data.num_classes} классов • ${data.total_params?.toLocaleString() || '?'} параметров</small>`;
        }
    } catch {
        if (modelEl) modelEl.textContent = 'Ошибка';
    }
}

async function updateActiveSessions() {
    const el = document.getElementById('activeSessions');
    if (el) el.textContent = '1';
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatDate(isoString) {
    if (!isoString) return '—';
    try {
        return new Date(isoString).toLocaleDateString('ru-RU', {
            year: 'numeric', month: 'short', day: 'numeric',
            hour: '2-digit', minute: '2-digit'
        });
    } catch { return isoString; }
}