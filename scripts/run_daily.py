#!/usr/bin/env python3
"""
Daily trading session entry point.

Connects to the Docker PostgreSQL database, instantiates the TradingOrchestrator
in paper mode, and runs the full daily signal evaluation + sizing pipeline.

Usage:
    python -m scripts.run_daily
    # or
    venv/bin/python3 scripts/run_daily.py
"""

import logging
import sys

import psycopg2

from core.orchestrator import TradingOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(name)s %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    db = psycopg2.connect('postgresql://trader:trader123@localhost:5450/trading')
    try:
        orch = TradingOrchestrator(mode='paper', db_connection=db)
        trades = orch.run_daily_session()
        logger.info('Daily session finished — %d trades sized', len(trades))
    finally:
        db.close()


if __name__ == '__main__':
    main()
