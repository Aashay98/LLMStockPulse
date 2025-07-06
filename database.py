import json
from datetime import datetime
from typing import Dict, List

import numpy as np
from passlib.context import CryptContext
from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

import config
from tools import get_sentence_transformer

Base = declarative_base()
engine = create_engine(config.DB_URL)
SessionLocal = sessionmaker(bind=engine)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _embed_text(text: str) -> List[float]:
    """Generate embedding vector for the given text."""
    model = get_sentence_transformer()
    return model.encode([text])[0].tolist()


def _cosine_sim(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    va = np.array(a)
    vb = np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom else 0.0


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    history = relationship("History", back_populates="user")


class History(Base):
    __tablename__ = "history"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="history")


def init_db() -> None:
    """Create database tables and seed default users."""
    Base.metadata.create_all(engine)
    session = SessionLocal()
    try:
        for username, pwd in config.USER_CREDENTIALS.items():
            if not session.query(User).filter_by(username=username).first():
                session.add(
                    User(username=username, password_hash=pwd_context.hash(pwd))
                )
        session.commit()
    finally:
        session.close()


def verify_user(username: str, password: str) -> bool:
    """Verify username and password against stored hash."""
    session = SessionLocal()
    try:
        user = session.query(User).filter_by(username=username).first()
        if user and pwd_context.verify(password, user.password_hash):
            return True
        return False
    finally:
        session.close()


def load_history(user_id: str) -> List[Dict[str, str]]:
    session = SessionLocal()
    try:
        user = session.query(User).filter_by(username=user_id).first()
        if not user:
            return []
        records = (
            session.query(History)
            .filter_by(user_id=user.id)
            .order_by(History.timestamp)
            .all()
        )
        return [{"role": r.role, "content": r.content} for r in records]
    finally:
        session.close()


def load_relevant_history(
    user_id: str, query: str, limit: int = 5
) -> List[Dict[str, str]]:
    """Return the most relevant history entries based on vector similarity."""
    session = SessionLocal()
    try:
        user = session.query(User).filter_by(username=user_id).first()
        if not user:
            return []
        records = session.query(History).filter_by(user_id=user.id).all()
        if not records:
            return []
        query_vec = _embed_text(query)
        scored = []
        for r in records:
            if r.embedding:
                try:
                    emb = json.loads(r.embedding)
                except Exception:
                    emb = []
                score = _cosine_sim(query_vec, emb) if emb else 0.0
                scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"role": r.role, "content": r.content}
            for score, r in scored[:limit]
            if score > 0
        ]
    finally:
        session.close()


def append_history(entries: List[Dict[str, str]], user_id: str) -> None:
    session = SessionLocal()
    try:
        user = session.query(User).filter_by(username=user_id).first()
        if not user:
            print("no user found")
            return
        for e in entries:
            embedding = json.dumps(_embed_text(e["content"]))
            session.add(
                History(
                    user_id=user.id,
                    role=e["role"],
                    content=e["content"],
                    embedding=embedding,
                )
            )
        session.commit()
    finally:
        session.close()


def clear_history(user_id: str) -> None:
    session = SessionLocal()
    try:
        user = session.query(User).filter_by(username=user_id).first()
        if not user:
            return
        session.query(History).filter_by(user_id=user.id).delete()
        session.commit()
    finally:
        session.close()
