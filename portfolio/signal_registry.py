"""
Signal registry — thin query layer over the signals table.
Passed into SignalSelector.run() at 8:50 AM each day.
"""


class SignalRegistry:
    """
    Thin query layer over the signals table.
    Passed into SignalSelector.run() at 8:50 AM each day.
    All rule/parameter content writes go through store_approved_signal().
    Status and lifecycle changes go through set_status().
    Never write rule fields (entry_conditions, parameters, etc.) directly
    via ad-hoc UPDATE — use store_approved_signal() so the auto-versioning
    trigger in schema.sql captures the change correctly.
    """

    def __init__(self, db):
        self.db = db

    def get_active_signals(self) -> list:
        """
        Returns all Signal objects with status = 'ACTIVE'.
        Called by SignalSelector to build today's candidate pool.
        Excludes: CANDIDATE, BACKTESTING, WATCH (selector sees WATCH
        separately via get_watch_signals), INACTIVE, ARCHIVED.
        """
        from portfolio.signal_model import Signal
        rows = self.db.execute("""
            SELECT * FROM signals
            WHERE status = 'ACTIVE'
            ORDER BY signal_id
        """).fetchall()
        return [Signal.from_db_row(row) for row in rows]

    def get_watch_signals(self) -> list:
        """Returns signals in WATCH status — reduced size, under review."""
        from portfolio.signal_model import Signal
        rows = self.db.execute(
            "SELECT * FROM signals WHERE status = 'WATCH' ORDER BY signal_id"
        ).fetchall()
        return [Signal.from_db_row(row) for row in rows]

    def set_status(self, signal_id: str, new_status: str,
                   reason: str, changed_by: str = 'SYSTEM'):
        """
        Status transitions: CANDIDATE→BACKTESTING→ACTIVE→WATCH→INACTIVE→ARCHIVED
        Records change via pending_change_by/reason columns so the
        auto-versioning trigger in schema.sql captures the transition.
        """
        VALID_STATUSES = {
            'CANDIDATE', 'BACKTESTING', 'ACTIVE',
            'WATCH', 'INACTIVE', 'ARCHIVED'
        }
        if new_status not in VALID_STATUSES:
            raise ValueError(f"Invalid status: {new_status}")
        self.db.execute("""
            UPDATE signals
            SET status = %s,
                pending_change_by = %s,
                pending_change_reason = %s,
                updated_at = NOW()
            WHERE signal_id = %s
        """, (new_status, changed_by, reason, signal_id))
