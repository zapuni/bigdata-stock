"""
Runner for Vietnam Stock Crawl Module

Usage:
    # Crawl VN30 (31 co phieu chinh, ~2h voi Guest rate limit)
    python src/run_crawl_vn.py

    # Crawl danh sach tu dinh (file txt, moi dong 1 ma)
    python src/run_crawl_vn.py --symbols-file my_symbols.txt

    # Crawl mot vai ma cu the
    python src/run_crawl_vn.py --symbols FPT,VNM,VCB

    # Crawl TAT CA ma tren san (~1600 ma, ~6h+ voi Guest)
    python src/run_crawl_vn.py --all

    # Thay doi interval (mac dinh 1m)
    python src/run_crawl_vn.py --interval 1D

Output:
    stock/stock-market-data-vn/{SYMBOL}_minute_data_with_indicators.csv
    Format khop 100% voi data An Do -> chay main_pipeline.py duoc luon.

Luu y:
    - Can cai vnstock:  pip install vnstock
    - Can cai TA-Lib:   conda install -c conda-forge ta-lib
    - Guest rate limit: 18 req/min -> VN30 mat ~2h, all ~6h+
"""

import os
import sys
import argparse
import logging

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_DIR)

from config.settings import VN_RAW_CSV_PATH, LOGS_DIR

os.makedirs(LOGS_DIR, exist_ok=True)

log = logging.getLogger("stock_crawl")
log.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
_fh = logging.FileHandler(os.path.join(LOGS_DIR, "crawl_vn.log"))
_fh.setFormatter(_fmt)
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_fmt)
log.addHandler(_fh)
log.addHandler(_sh)


def _check_dependencies():
    """Kiem tra vnstock va TA-Lib truoc khi chay."""
    errors = []
    try:
        import vnstock  # noqa: F401
    except ImportError:
        errors.append("vnstock not installed. Run: pip install vnstock")

    try:
        import talib  # noqa: F401
    except ImportError:
        errors.append(
            "TA-Lib not installed. Run:\n"
            "  conda install -c conda-forge ta-lib\n"
            "Or install C library manually then: pip install TA-Lib"
        )

    if errors:
        for e in errors:
            log.error(e)
        sys.exit(1)


def _parse_symbols(args) -> list[str]:
    """Xac dinh danh sach symbols tu arguments."""
    from crawl.vnstock_crawler import VN30_SYMBOLS, fetch_symbols
    from crawl.rate_limiter import RateLimiter

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
        log.info("Custom symbols: %d stocks", len(symbols))
        return symbols

    if args.symbols_file:
        with open(args.symbols_file) as f:
            symbols = [line.strip().upper() for line in f if line.strip()]
        log.info("Symbols from file %s: %d stocks", args.symbols_file, len(symbols))
        return symbols

    if args.all:
        rl = RateLimiter(max_per_minute=args.rate_limit)
        symbols = fetch_symbols(source=args.source, rate_limiter=rl)
        if symbols is None:
            log.error("Cannot fetch symbol list from vnstock")
            sys.exit(1)
        log.info("ALL symbols from exchange: %d stocks", len(symbols))
        return symbols

    log.info("Default: VN30 (%d stocks)", len(VN30_SYMBOLS))
    return VN30_SYMBOLS


def main():
    parser = argparse.ArgumentParser(description="Vietnam Stock Crawl -> CSV (Indian format)")
    parser.add_argument("--symbols", default=None,
                        help="Comma-separated symbols (e.g. FPT,VNM,VCB)")
    parser.add_argument("--symbols-file", default=None,
                        help="Text file with one symbol per line")
    parser.add_argument("--all", action="store_true",
                        help="Crawl all symbols from exchange (~1600)")
    parser.add_argument("--interval", default="1m",
                        help="Data interval: 1m, 5m, 15m, 1H, 1D (default: 1m)")
    parser.add_argument("--source", default="KBS",
                        help="vnstock data source (default: KBS)")
    parser.add_argument("--rate-limit", type=int, default=18,
                        help="Max requests per minute (default: 18 for Guest)")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: stock/stock-market-data-vn/)")
    args = parser.parse_args()

    log.info("=" * 65)
    log.info("VIETNAM STOCK CRAWL")
    log.info("=" * 65)

    _check_dependencies()

    from crawl.vnstock_crawler import crawl_all
    from crawl.rate_limiter import RateLimiter

    symbols = _parse_symbols(args)
    output_dir = args.output or VN_RAW_CSV_PATH
    rl = RateLimiter(max_per_minute=args.rate_limit)

    log.info("Symbols  : %d", len(symbols))
    log.info("Interval : %s", args.interval)
    log.info("Source   : %s", args.source)
    log.info("Rate     : %d req/min", args.rate_limit)
    log.info("Output   : %s", output_dir)
    est_minutes = len(symbols) * 60 / args.rate_limit / 60
    log.info("Estimated: ~%.0f min (%.1f hours)", est_minutes, est_minutes / 60)
    log.info("-" * 65)

    result = crawl_all(
        symbols=symbols,
        output_dir=output_dir,
        source=args.source,
        interval=args.interval,
        rate_limiter=rl,
    )

    log.info("=" * 65)
    log.info("CRAWL COMPLETED")
    log.info("  Success: %d | Failed: %d | Time: %.1f min",
             len(result["success"]), len(result["failed"]), result["total_time"] / 60)
    log.info("  Output : %s", output_dir)
    if result["success"]:
        log.info("  Next   : python src/main_pipeline.py  (set RAW_CSV_PATH to %s)", output_dir)
    log.info("=" * 65)


if __name__ == "__main__":
    main()
