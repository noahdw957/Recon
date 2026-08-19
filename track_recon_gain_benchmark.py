#!/usr/bin/env python3
"""
RECON projected-gain benchmark tracker.

Day 0 = Friday 2026-08-14 close.
Rows are U.S. trading sessions, not calendar days:
    Day 0 = Fri 2026-08-14
    Day 1 = Mon 2026-08-17
    Day 2 = Tue 2026-08-18
    ...

Output:
    projected_gain_benchmark.csv

Columns:
    Day,Date,SATL,ONDS,VSEC,LHX,CXW,RTX

Each ticker cell is cumulative percentage gain from that ticker's Day-0 close:
    100 * (close_on_date / close_on_day0 - 1)

The file appends missing trading days only. Re-running on the same day never
duplicates or overwrites an existing Day row.
"""

from __future__ import annotations

import csv
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf


TICKERS = ["SATL", "ONDS", "VSEC", "LHX", "CXW", "RTX"]
DAY0 = date(2026, 8, 14)  # Friday
OUTPUT_FILE = Path("projected_gain_benchmark.csv")

# Manual runs before the U.S. market is actually closed should not write
# an unfinished "closing" value for today. GitHub Actions below runs later.
EASTERN = ZoneInfo("America/New_York")
MARKET_CLOSE_GUARD = time(16, 15)


def last_completed_market_date() -> date:
    now_et = datetime.now(EASTERN)

    # Weekday before 4:15 PM ET: today's daily bar is not a closing price yet.
    if now_et.weekday() < 5 and now_et.time() < MARKET_CLOSE_GUARD:
        return now_et.date() - timedelta(days=1)

    return now_et.date()


def download_daily_closes(symbol: str, end_date: date) -> dict[date, float]:
    """Download raw daily Close values from Day 0 through end_date."""
    # yfinance's end date is exclusive.
    hist = yf.Ticker(symbol).history(
        start=DAY0.isoformat(),
        end=(end_date + timedelta(days=1)).isoformat(),
        interval="1d",
        auto_adjust=False,
        actions=False,
        repair=False,
        timeout=30,
        raise_errors=True,
    )

    if hist is None or hist.empty or "Close" not in hist.columns:
        raise RuntimeError(f"No daily Close history returned for {symbol}")

    closes: dict[date, float] = {}
    for idx, value in hist["Close"].items():
        if pd.isna(value):
            continue
        d = pd.Timestamp(idx).date()
        if DAY0 <= d <= end_date:
            closes[d] = float(value)

    if DAY0 not in closes:
        raise RuntimeError(
            f"{symbol} is missing the Day-0 close for {DAY0.isoformat()}"
        )

    return closes


def read_existing_days() -> set[int]:
    """Read existing Day values and validate the CSV header."""
    if not OUTPUT_FILE.exists():
        return set()

    with OUTPUT_FILE.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        expected = ["Day", "Date", *TICKERS]

        if reader.fieldnames != expected:
            raise RuntimeError(
                f"{OUTPUT_FILE} header is {reader.fieldnames}; expected {expected}"
            )

        days: set[int] = set()
        for row in reader:
            raw = str(row.get("Day", "")).strip()
            if raw:
                days.add(int(raw))

        return days


def fmt_gain(value: float | None) -> str:
    # Numeric percent points, suitable for later charting/regression.
    return "" if value is None else f"{value:.4f}"


def main() -> None:
    cutoff = last_completed_market_date()
    if cutoff < DAY0:
        print("Day 0 has not occurred yet.")
        return

    # SPY is used only as the NYSE trading-session calendar. It is not written
    # to the benchmark file.
    spy_closes = download_daily_closes("SPY", cutoff)
    sessions = sorted(d for d in spy_closes if DAY0 <= d <= cutoff)

    if not sessions or sessions[0] != DAY0:
        raise RuntimeError(
            f"Could not establish Day 0 as a trading session on {DAY0}"
        )

    ticker_closes = {
        ticker: download_daily_closes(ticker, cutoff)
        for ticker in TICKERS
    }

    day0_close = {
        ticker: ticker_closes[ticker][DAY0]
        for ticker in TICKERS
    }

    existing_days = read_existing_days()
    rows_to_add: list[list[str | int]] = []

    for day_number, session_date in enumerate(sessions):
        if day_number in existing_days:
            continue

        # Do not write a partial day. If Yahoo is late on even one ticker,
        # stop here and let the next run append this day cleanly. This keeps
        # the benchmark contiguous and avoids permanent blanks.
        missing = [
            ticker
            for ticker in TICKERS
            if ticker_closes[ticker].get(session_date) is None
        ]
        if missing:
            print(
                f"Day {day_number} {session_date} not appended; "
                f"missing close(s): {', '.join(missing)}"
            )
            break

        row: list[str | int] = [
            day_number,
            session_date.isoformat(),
        ]

        for ticker in TICKERS:
            close = ticker_closes[ticker][session_date]
            gain = 100.0 * (close / day0_close[ticker] - 1.0)
            row.append(fmt_gain(gain))

        rows_to_add.append(row)

    if not rows_to_add:
        print("No new completed trading day to append.")
        return

    write_header = not OUTPUT_FILE.exists()

    with OUTPUT_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow(["Day", "Date", *TICKERS])

        writer.writerows(rows_to_add)

    print(f"Appended {len(rows_to_add)} trading day(s) to {OUTPUT_FILE}:")
    for row in rows_to_add:
        print(f"  Day {row[0]}  {row[1]}")


if __name__ == "__main__":
    main()
