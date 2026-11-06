"""SQLAlchemy engine and session factory."""

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


def _connect_args() -> dict:
    """TLS settings for hosted MySQL (MYSQL_SSL / MYSQL_SSL_CA_CERT)."""
    s = get_settings()
    if s.database_url_override or not (s.mysql_ssl or s.mysql_ssl_ca_cert):
        return {}
    if s.mysql_ssl_ca_cert:
        ca = Path(tempfile.gettempdir()) / "mysql-ca.pem"
        ca.write_text(s.mysql_ssl_ca_cert.replace("\\n", "\n"), encoding="utf-8")
        return {"ssl": {"ca": str(ca)}}  # encrypted + server certificate verified
    return {"ssl": {"verify_mode": "none"}}  # encrypted, certificate not verified


def _make_engine(url) -> Engine:
    return create_engine(url, pool_pre_ping=True, pool_recycle=3600, connect_args=_connect_args())


@lru_cache
def get_engine() -> Engine:
    return _make_engine(get_settings().database_url)


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
    server = _make_engine(url._replace(database=None))  # connect to the server, no DB
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
