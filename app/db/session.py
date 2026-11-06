"""SQLAlchemy engine and session factory."""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


@lru_cache
def get_engine() -> Engine:
    return create_engine(get_settings().database_url, pool_pre_ping=True, pool_recycle=3600)


@lru_cache
def _session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


@contextmanager
def session_scope() -> Iterator[Session]:
    session = _session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def ensure_database() -> None:
    """Create the MySQL database if it does not exist yet (needs CREATE privilege)."""
    url = get_engine().url
    if not url.drivername.startswith("mysql") or not url.database:
        return
    server = create_engine(url._replace(database=None))  # connect to the server, no DB
    try:
        with server.begin() as conn:
            conn.execute(
                text(
                    f"CREATE DATABASE IF NOT EXISTS `{url.database}` "
                    "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )
            )
    finally:
        server.dispose()


def init_db() -> None:
    """Create the database (MySQL) and tables if they do not exist."""
    from app.db import models  # noqa: F401  (registers models on Base)

    ensure_database()
    Base.metadata.create_all(get_engine())
