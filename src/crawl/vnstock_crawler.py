"""
vnstock Crawler -- Crawl OHLCV + compute indicators + save CSV

Output format khop chinh xac voi data An Do de dua thang vao main_pipeline.py:
  - Filename : {SYMBOL}_minute_data_with_indicators.csv
  - Columns  : date, close, high, low, open, volume, <54 TA-Lib indicators>
  - Date fmt : YYYY-MM-DD HH:mm:ss+07:00  (Vietnam TZ)
  - Output   : stock/stock-market-data-vn/
"""

import os
import time
import logging

import pandas as pd

from crawl.rate_limiter import RateLimiter, fetch_with_retry
from crawl.indicators import compute_all_indicators, CSV_COLUMN_ORDER, check_talib

log = logging.getLogger("stock_crawl")

# VN30 - 30 co phieu thanh khoan nhat san HOSE
VN30_SYMBOLS = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR",
    "HDB", "HPG", "MBB", "MSN", "MWG", "PLX", "PNJ", "POW",
    "SAB", "SHB", "SSB", "SSI", "STB", "TCB", "TPB", "VCB",
    "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
]

VN_TIMEZONE = "+07:00"
API_SOURCE = "KBS"


def fetch_symbols(source: str = API_SOURCE, rate_limiter: RateLimiter = None):
    """Lay danh sach tat ca symbols tu vnstock.

    Returns:
        list[str] hoac None neu that bai
    """
    from vnstock import Listing

    rl = rate_limiter or RateLimiter()
    listing = Listing(source=source)
    df = fetch_with_retry(listing.all_symbols, rl)
    if df is not None and not df.empty:
        symbols = df["symbol"].tolist()
        log.info("Fetched %d symbols from vnstock (%s)", len(symbols), source)
        return symbols
    log.error("Failed to fetch symbols")
    return None


def fetch_ohlcv(
    symbol: str,
    source: str = API_SOURCE,
    interval: str = "1m",
    rate_limiter: RateLimiter = None,
) -> pd.DataFrame | None:
    """Fetch OHLCV history cho 1 symbol.

    Args:
        symbol: Ma co phieu (e.g. "FPT")
        source: vnstock source (default "KBS")
        interval: "1m" cho minute data
        rate_limiter: RateLimiter instance

    Returns:
        DataFrame voi cot: time, open, high, low, close, volume | None
    """
    from vnstock import Quote

    rl = rate_limiter or RateLimiter()
    quote = Quote(symbol=symbol, source=source)
    df = fetch_with_retry(
        lambda: quote.history(length="1Y", interval=interval, get_all=True),
        rl,
    )
    if df is not None and not df.empty:
        return df
    return None


def _format_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Chuyen cot time cua vnstock thanh format giong Indian data.

    vnstock tra ve cot 'time' dang datetime.
    Indian format: "2015-02-02 09:15:00+05:30"
    VN format:     "2024-01-15 09:15:00+07:00"
    """
    time_col = None
    for candidate in ["time", "Time", "datetime", "Datetime", "date", "Date"]:
        if candidate in df.columns:
            time_col = candidate
            break
    if time_col is None:
        raise ValueError(f"No time column found in DataFrame. Columns: {df.columns.tolist()}")

    df = df.rename(columns={time_col: "date"})

    if pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = df["date"].dt.strftime(f"%Y-%m-%d %H:%M:%S{VN_TIMEZONE}")
    else:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime(f"%Y-%m-%d %H:%M:%S{VN_TIMEZONE}")

    return df


def _normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Chuan hoa ten cot OHLCV tu vnstock ve format Indian."""
    rename_map = {}
    for col_lower in ["open", "high", "low", "close", "volume"]:
        for col_actual in df.columns:
            if col_actual.lower() == col_lower and col_actual != col_lower:
                rename_map[col_actual] = col_lower
    if rename_map:
        df = df.rename(columns=rename_map)

    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")

    return df


def process_stock(
    symbol: str,
    output_dir: str,
    source: str = API_SOURCE,
    interval: str = "1m",
    rate_limiter: RateLimiter = None,
) -> bool:
    """Pipeline hoan chinh cho 1 stock: fetch -> indicators -> save CSV.

    Output file: {output_dir}/{SYMBOL}_minute_data_with_indicators.csv
    Filename convention khop voi regex trong main_pipeline.py step_enrich().

    Returns:
        True neu thanh cong, False neu that bai
    """
    t0 = time.time()

    df = fetch_ohlcv(symbol, source=source, interval=interval, rate_limiter=rate_limiter)
    if df is None:
        log.warning("[%s] Failed to fetch OHLCV", symbol)
        return False

    try:
        df = _format_date_column(df)
        df = _normalize_ohlcv_columns(df)

        for col_name in ["open", "high", "low", "close", "volume"]:
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
        df = df.dropna(subset=["close"])

        if len(df) < 30:
            log.warning("[%s] Too few rows (%d), skipping", symbol, len(df))
            return False

        df = compute_all_indicators(df)

        # Sap xep cot theo dung thu tu cua Indian CSV
        df = df[CSV_COLUMN_ORDER]

        os.makedirs(output_dir, exist_ok=True)
        filename = f"{symbol}_minute_data_with_indicators.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)

        log.info(
            "[%s] OK | %s rows | %.1fs | %s",
            symbol, f"{len(df):,}", time.time() - t0, filepath,
        )
        return True

    except Exception as e:
        log.error("[%s] Processing error: %s", symbol, str(e)[:200])
        return False


def crawl_all(
    symbols: list[str],
    output_dir: str,
    source: str = API_SOURCE,
    interval: str = "1m",
    rate_limiter: RateLimiter = None,
) -> dict:
    """Crawl tat ca symbols.

    Returns:
        dict voi keys: success, failed, total_time
    """
    rl = rate_limiter or RateLimiter()
    check_talib()

    success = []
    failed = []
    t0 = time.time()

    for i, sym in enumerate(symbols):
        ok = process_stock(sym, output_dir, source=source, interval=interval, rate_limiter=rl)
        if ok:
            success.append(sym)
        else:
            failed.append(sym)

        if (i + 1) % 10 == 0 or (i + 1) == len(symbols):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
            remaining = (len(symbols) - i - 1) / max(rate, 0.01)
            log.info(
                "Progress: %d/%d | OK: %d | Fail: %d | %.1f sym/min | ETA: %.1f min",
                i + 1, len(symbols), len(success), len(failed), rate, remaining,
            )

    total_time = time.time() - t0
    log.info(
        "CRAWL DONE | %d/%d success | %.1f min total",
        len(success), len(symbols), total_time / 60,
    )
    if failed:
        log.warning("Failed symbols (%d): %s", len(failed), failed)

    return {"success": success, "failed": failed, "total_time": total_time}
