"""
Crawl Module - Data acquisition for Vietnam stock market

Modules:
- rate_limiter.py  : API rate limiter for vnstock (Guest: 20 req/min)
- indicators.py    : TA-Lib indicator computation matching Indian schema
- vnstock_crawler.py: Orchestrator - crawl OHLCV + compute indicators + save CSV
"""
