import hashlib
import secrets
import time
from typing import Optional, Dict
from config import Config


def hash_password(password: str, salt: Optional[str] = None) -> str:
    password_hash = hashlib.sha256(password.encode('utf-8')).hexdigest()
    return f"{password_hash}"


def verify_password(password: str, stored_hash: str) -> bool:
    """
    Проверяет пароль против сохранённого хэша.
    
    :param password: Пароль, введённый пользователем
    :param stored_hash: Хэш из базы данных (формат "salt$hash")
    :return: True если пароль верный
    """
    try:
        salt, original_hash = stored_hash.split('$', 1)
        test_hash = hash_password(password, salt)
        return secrets.compare_digest(test_hash, f"{salt}${original_hash}")
        
    except (ValueError, AttributeError):
        return False
_active_tokens: Dict[str, Dict] = {}


def generate_token(username: str, role: str) -> str:
    token = secrets.token_urlsafe(32)
    expires_at = time.time() + Config.TOKEN_EXPIRY_SECONDS
    _active_tokens[token] = {
        'username': username,
        'role': role,
        'expires': expires_at,
        'created': time.time()
    }
    
    _cleanup_expired_tokens()
    
    return token


def verify_token(token: Optional[str], required_role: Optional[str] = None) -> bool:
    if not token:
        return False

    if token.startswith('Bearer '):
        token = token[7:]
    
    if token not in _active_tokens:
        return False
    
    token_data = _active_tokens[token]
    
    if time.time() > token_data['expires']:
        del _active_tokens[token]
        return False
    
    if required_role and token_data['role'] != required_role:
        return False
    
    return True


def get_token_info(token: str) -> Optional[Dict]:
    search_token = token[7:] if token.startswith('Bearer ') else token
    if verify_token(token):
        data = _active_tokens[search_token].copy()
        data.pop('expires', None)
        data.pop('created', None)
        return data
    return None


def revoke_token(token: str) -> bool:
    if token.startswith('Bearer '):
        token = token[7:]
    
    if token in _active_tokens:
        del _active_tokens[token]
        return True
    return False


def _cleanup_expired_tokens() -> None:
    current_time = time.time()
    expired = [t for t, d in _active_tokens.items() if d['expires'] < current_time]
    for token in expired:
        del _active_tokens[token]

def require_auth(f):
    from flask import request, jsonify
    import functools
    
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        
        if not verify_token(token):
            return jsonify({'error': 'Authentication required'}), 401
        
        request.user = get_token_info(token)
        return f(*args, **kwargs)
    
    return decorated_function


def require_role(role: str):
    from flask import request, jsonify
    import functools
    
    def decorator(f):
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            if not hasattr(request, 'user') or request.user.get('role') != role:
                return jsonify({'error': f'{role} access required'}), 403
            return f(*args, **kwargs)
        return decorated_function
    
    return decorator