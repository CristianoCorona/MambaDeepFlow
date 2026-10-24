"""scripts/run_queue.next_cutoff: HH:MM (next occurrence, unchanged behaviour) and the absolute YYYY-MM-DDTHH:MM form
added on 2026-10-17 for night 9 (launched on Saturday morning with the cutoff on Sunday 13:00).

Run: python -m pytest tests/test_run_queue_cutoff.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from run_queue import next_cutoff  # noqa: E402

START = datetime(2026, 10, 17, 10, 0).astimezone()


def test_hhmm_next_occurrence_unchanged():
    assert next_cutoff("13:00", START) == START.replace(hour=13)  # same day: later than the start
    assert next_cutoff("07:00", START) == START.replace(hour=7) + timedelta(days=1)  # next day


def test_absolute_date_time():
    cut = next_cutoff("2026-10-18T13:00", START)
    assert cut == datetime(2026, 10, 18, 13, 0).astimezone() and cut - START == timedelta(hours=27)
    with pytest.raises(SystemExit):
        next_cutoff("2026-10-17T09:00", START)  # in the past
