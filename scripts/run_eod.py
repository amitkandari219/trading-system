#!/usr/bin/env python3
"""
End-of-day processing entry point.

Connects to the Docker PostgreSQL database, instantiates the TradingOrchestrator
in paper mode, and runs EOD reconciliation, P&L logging, and equity updates.

Usage:
    python -m scripts.run_eod
    # or
    venv/bin/python3 scripts/run_eod.py
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
        result = orch.run_eod_processing()
        logger.info('EOD processing finished — status: %s', result.get('status', 'unknown'))
    finally:
        db.close()


if __name__ == '__main__':
    main()
