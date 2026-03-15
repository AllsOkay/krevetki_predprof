from datetime import datetime
from typing import Optional, List, Dict
import hashlib
from auth import hash_password
from sqlalchemy import create_engine, Column, Integer, String, DateTime, CheckConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError

DB_PATH = "database.db"

engine = create_engine(
    f"sqlite:///{DB_PATH}",
    echo=False,
    connect_args={"check_same_thread": False}
)
Base = declarative_base()

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    name = Column(String, nullable=False)
    surname = Column(String, nullable=False)
    role = Column(String, nullable=False, default="user")
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    __table_args__ = (
        CheckConstraint("role IN ('admin', 'user')", name="check_role"),
    )


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_demo_users():
    db = SessionLocal()
    try:
        existing = db.query(User).filter_by(username='admin').first()
        if existing:
            return
        admin = User(
            username='admin',
            password_hash=hash_password('admin123'),
            name='Админ',
            surname='Системы',
            role='admin'
        )
        db.add(admin)
        
        user = User(
            username='user',
            password_hash=hash_password('user123'),
            name='Тестовый',
            surname='Пользователь',
            role='user'
        )
        db.add(user)
        
        db.commit()
        print("Демо-пользователи созданы: admin/admin123, user/user123")
        
    except IntegrityError:
        db.rollback()
    finally:
        db.close()


def create_user(username: str, password_hash: str, name: str,
                surname: str, role: str = "user") -> bool:
    db = SessionLocal()
    try:
        user = User(
            username=username,
            password_hash=password_hash,
            name=name,
            surname=surname,
            role=role
        )
        db.add(user)
        db.commit()
        return True
    except IntegrityError:
        db.rollback()
        return False
    finally:
        db.close()


def get_user(username: str, password_hash: Optional[str] = None) -> Optional[Dict]:
    db = SessionLocal()
    try:
        print(username)
        if password_hash:
            print(password_hash)
            user = db.query(User).filter_by(username=username, password_hash=password_hash).first()
        else:
            user = None

        if user:
            return {
                "id": user.id,
                "username": user.username,
                "password_hash": user.password_hash,
                "name": user.name,
                "surname": user.surname,
                "role": user.role,
                "created_at": user.created_at,
                "last_login": user.last_login
            }
        return None
    finally:
        db.close()


def update_last_login(username: str) -> bool:
    db = SessionLocal()
    try:
        user = db.query(User).filter_by(username=username).first()
        if user:
            user.last_login = datetime.utcnow()
            db.commit()
            return True
        return False
    finally:
        db.close()


def get_all_users() -> List[Dict]:
    db = SessionLocal()
    try:
        users = db.query(User).order_by(User.created_at.desc()).all()
        return [
            {
                "id": u.id,
                "username": u.username,
                "name": u.name,
                "surname": u.surname,
                "role": u.role,
                "created_at": u.created_at,
                "last_login": u.last_login
            }
            for u in users
        ]
    finally:
        db.close()


def delete_user(username: str) -> bool:
    db = SessionLocal()
    try:
        user = db.query(User).filter_by(username=username).first()
        if user:
            db.delete(user)
            db.commit()
            return True
        return False
    finally:
        db.close()


Base.metadata.create_all(bind=engine)
create_demo_users()