"""
Database setup helper.
Creates the trading database and runs schema.sql.

Usage:
    python -m db.setup              # Create DB + run schema
    python -m db.setup --schema     # Run schema only (DB already exists)
    python -m db.setup --check      # Check DB connection and tables
"""

import subprocess
import sys
from pathlib import Path


SCHEMA_PATH = Path(__file__).parent / 'schema.sql'


def create_database(db_name='trading'):
    """Create the trading database if it doesn't exist."""
    print(f"Creating database '{db_name}'...")
    try:
        result = subprocess.run(
            ['createdb', db_name],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  Database '{db_name}' created.")
        elif 'already exists' in result.stderr:
            print(f"  Database '{db_name}' already exists.")
        else:
            print(f"  Error: {result.stderr}")
            return False
    except FileNotFoundError:
        print("  ERROR: 'createdb' not found. Is PostgreSQL installed and in PATH?")
        return False
    return True


def run_schema(dsn='postgresql://localhost/trading'):
    """Execute schema.sql against the database."""
    import psycopg2

    print(f"Running schema.sql...")
    schema_sql = SCHEMA_PATH.read_text()

    try:
        conn = psycopg2.connect(dsn)
        conn.autocommit = True
        cur = conn.cursor()

        # Split on semicolons and execute each statement
        # (TimescaleDB functions need autocommit)
        statements = schema_sql.split(';')
        executed = 0
        for stmt in statements:
            stmt = stmt.strip()
            if stmt and not stmt.startswith('--'):
                try:
                    cur.execute(stmt)
                    executed += 1
                except psycopg2.errors.DuplicateObject:
                    conn.rollback() if not conn.autocommit else None
                except psycopg2.errors.DuplicateTable:
                    conn.rollback() if not conn.autocommit else None
                except Exception as e:
                    error_msg = str(e).strip()
                    # Skip non-critical errors (extensions, hypertables on existing tables)
                    if any(skip in error_msg for skip in [
                        'already exists', 'duplicate key',
                        'is already a hypertable'
                    ]):
                        pass
                    else:
                        print(f"  WARNING: {error_msg[:100]}")

        print(f"  Executed {executed} statements.")
        conn.close()
        return True

    except psycopg2.OperationalError as e:
        print(f"  ERROR: Could not connect to database: {e}")
        print(f"  Make sure PostgreSQL is running and database exists.")
        print(f"  Run: createdb trading")
        return False


def check_database(dsn='postgresql://localhost/trading'):
    """Check DB connection and list tables."""
    import psycopg2

    try:
        conn = psycopg2.connect(dsn)
        cur = conn.cursor()

        # List all tables
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        tables = [row[0] for row in cur.fetchall()]

        print(f"Database connection: OK")
        print(f"Tables found: {len(tables)}")
        for t in tables:
            cur.execute(f"SELECT COUNT(*) FROM {t}")
            count = cur.fetchone()[0]
            print(f"  {t:30s} {count:>8d} rows")

        # Check TimescaleDB
        cur.execute("SELECT extname FROM pg_extension WHERE extname = 'timescaledb'")
        if cur.fetchone():
            print(f"\nTimescaleDB: installed")
        else:
            print(f"\nTimescaleDB: NOT installed")
            print(f"  Install: https://docs.timescale.com/install/latest/")
            print(f"  Or remove TimescaleDB lines from schema.sql to proceed without it.")

        conn.close()
        return True

    except psycopg2.OperationalError as e:
        print(f"ERROR: {e}")
        return False


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Database setup')
    parser.add_argument('--schema', action='store_true',
                        help='Run schema.sql only')
    parser.add_argument('--check', action='store_true',
                        help='Check DB connection and tables')
    parser.add_argument('--dsn', default='postgresql://localhost/trading',
                        help='PostgreSQL DSN')
    args = parser.parse_args()

    if args.check:
        check_database(args.dsn)
    elif args.schema:
        run_schema(args.dsn)
    else:
        # Full setup
        if create_database():
            run_schema(args.dsn)
            print("\nVerifying setup...")
            check_database(args.dsn)
