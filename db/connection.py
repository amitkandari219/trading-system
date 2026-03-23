"""
Database connection pool with auto-reconnect.

Replaces raw psycopg2.connect() calls throughout the codebase.
Uses ThreadedConnectionPool for safe concurrent access.

Usage:
    from db.connection import get_db

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1")
"""

import logging
import threading
from contextlib import contextmanager

import psycopg2
from psycopg2 import pool

from config.settings import DATABASE_DSN

logger = logging.getLogger(__name__)

_pool_lock = threading.Lock()
_pool_instance: pool.ThreadedConnectionPool = None


def _parse_dsn(dsn: str) -> dict:
    """Parse DSN string into kwargs for psycopg2."""
    # psycopg2.connect() accepts DSN strings directly
    return {"dsn": dsn}


def _create_pool() -> pool.ThreadedConnectionPool:
    """Create the connection pool (called once)."""
    return pool.ThreadedConnectionPool(
        minconn=2,
        maxconn=10,
        dsn=DATABASE_DSN,
        connect_timeout=5,
    )


def get_pool() -> pool.ThreadedConnectionPool:
    """Get or create the singleton pool."""
    global _pool_instance
    if _pool_instance is None or _pool_instance.closed:
        with _pool_lock:
            if _pool_instance is None or _pool_instance.closed:
                _pool_instance = _create_pool()
                logger.info("DB connection pool created (min=2, max=10)")
    return _pool_instance


def get_conn():
    """
    Get a connection from the pool with health check.
    Caller MUST return it via put_conn() or use get_db() context manager.
    """
    p = get_pool()
    conn = p.getconn()

    # Health check — if connection is dead, close and get a fresh one
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        conn.rollback()  # clear any implicit transaction from health check
    except Exception:
        logger.warning("DB connection failed health check — reconnecting")
        try:
            p.putconn(conn, close=True)
        except Exception:
            pass
        conn = p.getconn()

    return conn


def put_conn(conn):
    """Return a connection to the pool."""
    try:
        get_pool().putconn(conn)
    except Exception:
        pass


@contextmanager
def get_db():
    """
    Context manager: get connection, auto-commit on success, rollback on error.

    Usage:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(...)
    """
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        put_conn(conn)


def close_pool():
    """Shut down the pool (call at process exit)."""
    global _pool_instance
    if _pool_instance is not None and not _pool_instance.closed:
        _pool_instance.closeall()
        logger.info("DB connection pool closed")
        _pool_instance = None
