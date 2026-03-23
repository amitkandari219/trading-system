#!/usr/bin/env python3
"""
Intraday signal check entry point.

Connects to the Docker PostgreSQL database, instantiates the TradingOrchestrator
in paper mode, and runs the intraday signal evaluation pipeline.

Usage:
    python -m scripts.run_intraday
    # or
    venv/bin/python3 scripts/run_intraday.py
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
        result = orch.run_intraday_check()
        logger.info('Intraday check finished — regime: %s',
                     result.get('regime', {}).get('regime', 'UNKNOWN'))
    finally:
        db.close()


if __name__ == '__main__':
    main()
