document.addEventListener('DOMContentLoaded', function() {
    const loginForm = document.getElementById('loginForm');
    const loginBtn = document.getElementById('loginBtn');
    const errorDiv = document.getElementById('loginError');
    const errorText = document.getElementById('errorText');
    const usernameInput = document.getElementById('username');
    const passwordInput = document.getElementById('password');
    const passwordToggle = document.getElementById('passwordToggle');
    const toggleIcon = passwordToggle?.querySelector('i');

    checkExistingAuth();

    loginForm?.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const username = usernameInput.value.trim();
        const password = passwordInput.value;
        
        hideError();
        setLoading(true);
        
        try {
            const result = await window.Auth.login(username, password);
            
            if (result.success) {
                console.log('Вход успешен:', result.user);
                if (result.user.role === 'admin') {
                    window.location.href = '/admin';
                } else {
                    window.location.href = '/dashboard';
                }
            } else {
                showError(result.error || 'Неверный логин или пароль');
                setLoading(false);
            }
        } catch (error) {
            console.error('Login error:', error);
            showError('Не удалось подключиться к серверу. Проверьте соединение.');
            setLoading(false);
        }
    });

    passwordToggle?.addEventListener('click', function() {
        const isPassword = passwordInput.type === 'password';
        passwordInput.type = isPassword ? 'text' : 'password';
        toggleIcon.className = isPassword ? 'fas fa-eye-slash' : 'fas fa-eye';
    });

    usernameInput?.addEventListener('input', hideError);
    passwordInput?.addEventListener('input', hideError);

    async function checkExistingAuth() {
        const isAuthenticated = await window.Auth.checkAuth();
        if (isAuthenticated && window.currentUser) {
            console.log('Сессия восстановлена');
            if (window.currentUser.role === 'admin') {
                window.location.href = '/admin';
            } else {
                window.location.href = '/dashboard';
            }
        }
    }

    function showError(message) {
        errorText.textContent = message;
        errorDiv.classList.add('show');
        passwordInput.style.borderColor = '#F44336';
        setTimeout(() => {
            passwordInput.style.borderColor = '#E0E0E0';
        }, 500);
    }

    function hideError() {
        errorDiv.classList.remove('show');
    }

    function setLoading(isLoading) {
        if (isLoading) {
            loginBtn.disabled = true;
            loginBtn.innerHTML = '<span class="loading-spinner"></span> Вход...';
        } else {
            loginBtn.disabled = false;
            loginBtn.innerHTML = '<i class="fas fa-sign-in-alt"></i> Войти в систему';
        }
    }
});