"""
Technical Indicator Computation -- TA-Lib

Tinh du 54 indicator khop chinh xac voi schema cua data An Do
(xem config/settings.py::get_csv_schema va stock/1. Data description.txt).

Requires: pip install TA-Lib  (can cai C library truoc)
"""

import numpy as np
import pandas as pd

try:
    import talib
except ImportError:
    talib = None

# Column order khop chinh xac voi get_csv_schema() trong settings.py
# Indian CSV header: date, close, high, low, open, volume, <54 indicators>
CSV_COLUMN_ORDER = [
    "date", "close", "high", "low", "open", "volume",
    "sma5", "sma10", "sma15", "sma20",
    "ema5", "ema10", "ema15", "ema20",
    "upperband", "middleband", "lowerband",
    "HT_TRENDLINE",
    "KAMA10", "KAMA20", "KAMA30",
    "SAR",
    "TRIMA5", "TRIMA10", "TRIMA20",
    "ADX5", "ADX10", "ADX20",
    "APO",
    "CCI5", "CCI10", "CCI15",
    "macd510", "macd520", "macd1020", "macd1520", "macd1226",
    "MFI",
    "MOM10", "MOM15", "MOM20",
    "ROC5", "ROC10", "ROC20",
    "PPO",
    "RSI14", "RSI8",
    "slowk", "slowd",
    "fastk", "fastd",
    "fastksr", "fastdsr",
    "ULTOSC", "WILLR",
    "ATR", "Trange", "TYPPRICE",
    "HT_DCPERIOD", "BETA",
]


def check_talib():
    """Kiem tra TA-Lib da cai dat chua. Raise ImportError neu thieu."""
    if talib is None:
        raise ImportError(
            "TA-Lib is required but not installed.\n"
            "Install C library first:\n"
            "  conda install -c conda-forge ta-lib\n"
            "Or manually:\n"
            "  wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz\n"
            "  tar -xzf ta-lib-0.4.0-src.tar.gz && cd ta-lib\n"
            "  ./configure --prefix=/usr && make && sudo make install\n"
            "  pip install TA-Lib"
        )


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Tinh du 54 technical indicators tu OHLCV data.

    Args:
        df: DataFrame voi cot: open, high, low, close, volume (float64)

    Returns:
        DataFrame goc + 54 cot indicator moi
    """
    check_talib()

    o = df["open"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    lo = df["low"].values.astype(np.float64)
    c = df["close"].values.astype(np.float64)
    v = df["volume"].values.astype(np.float64)

    # --- Moving Averages (8) ---
    df["sma5"] = talib.SMA(c, timeperiod=5)
    df["sma10"] = talib.SMA(c, timeperiod=10)
    df["sma15"] = talib.SMA(c, timeperiod=15)
    df["sma20"] = talib.SMA(c, timeperiod=20)
    df["ema5"] = talib.EMA(c, timeperiod=5)
    df["ema10"] = talib.EMA(c, timeperiod=10)
    df["ema15"] = talib.EMA(c, timeperiod=15)
    df["ema20"] = talib.EMA(c, timeperiod=20)

    # --- Bollinger Bands (3) ---
    upper, middle, lower = talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df["upperband"] = upper
    df["middleband"] = middle
    df["lowerband"] = lower

    # --- Trend (8) ---
    df["HT_TRENDLINE"] = talib.HT_TRENDLINE(c)
    df["KAMA10"] = talib.KAMA(c, timeperiod=10)
    df["KAMA20"] = talib.KAMA(c, timeperiod=20)
    df["KAMA30"] = talib.KAMA(c, timeperiod=30)
    df["SAR"] = talib.SAR(h, lo, acceleration=0.02, maximum=0.2)
    df["TRIMA5"] = talib.TRIMA(c, timeperiod=5)
    df["TRIMA10"] = talib.TRIMA(c, timeperiod=10)
    df["TRIMA20"] = talib.TRIMA(c, timeperiod=20)

    # --- ADX + APO (4) ---
    df["ADX5"] = talib.ADX(h, lo, c, timeperiod=5)
    df["ADX10"] = talib.ADX(h, lo, c, timeperiod=10)
    df["ADX20"] = talib.ADX(h, lo, c, timeperiod=20)
    df["APO"] = talib.APO(c, fastperiod=12, slowperiod=26, matype=0)

    # --- CCI (3) ---
    df["CCI5"] = talib.CCI(h, lo, c, timeperiod=5)
    df["CCI10"] = talib.CCI(h, lo, c, timeperiod=10)
    df["CCI15"] = talib.CCI(h, lo, c, timeperiod=15)

    # --- MACD (5) - chi lay macd line, bo signal va histogram ---
    df["macd510"], _, _ = talib.MACD(c, fastperiod=5, slowperiod=10, signalperiod=9)
    df["macd520"], _, _ = talib.MACD(c, fastperiod=5, slowperiod=20, signalperiod=9)
    df["macd1020"], _, _ = talib.MACD(c, fastperiod=10, slowperiod=20, signalperiod=9)
    df["macd1520"], _, _ = talib.MACD(c, fastperiod=15, slowperiod=20, signalperiod=9)
    df["macd1226"], _, _ = talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)

    # --- MFI (1) ---
    df["MFI"] = talib.MFI(h, lo, c, v, timeperiod=14)

    # --- Momentum (3) ---
    df["MOM10"] = talib.MOM(c, timeperiod=10)
    df["MOM15"] = talib.MOM(c, timeperiod=15)
    df["MOM20"] = talib.MOM(c, timeperiod=20)

    # --- Rate of Change (3) ---
    df["ROC5"] = talib.ROC(c, timeperiod=5)
    df["ROC10"] = talib.ROC(c, timeperiod=10)
    df["ROC20"] = talib.ROC(c, timeperiod=20)

    # --- PPO (1) ---
    df["PPO"] = talib.PPO(c, fastperiod=12, slowperiod=26, matype=0)

    # --- RSI (2) ---
    df["RSI14"] = talib.RSI(c, timeperiod=14)
    df["RSI8"] = talib.RSI(c, timeperiod=8)

    # --- Stochastic Slow (2) ---
    df["slowk"], df["slowd"] = talib.STOCH(
        h, lo, c, fastk_period=5, slowk_period=3, slowk_matype=0,
        slowd_period=3, slowd_matype=0,
    )

    # --- Stochastic Fast (2) ---
    df["fastk"], df["fastd"] = talib.STOCHF(
        h, lo, c, fastk_period=5, fastd_period=3, fastd_matype=0,
    )

    # --- Stochastic RSI (2) ---
    df["fastksr"], df["fastdsr"] = talib.STOCHRSI(
        c, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0,
    )

    # --- Oscillators (2) ---
    df["ULTOSC"] = talib.ULTOSC(h, lo, c, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    df["WILLR"] = talib.WILLR(h, lo, c, timeperiod=14)

    # --- Volatility (3) ---
    df["ATR"] = talib.ATR(h, lo, c, timeperiod=14)
    df["Trange"] = talib.TRANGE(h, lo, c)
    df["TYPPRICE"] = talib.TYPPRICE(h, lo, c)

    # --- Other (2) ---
    df["HT_DCPERIOD"] = talib.HT_DCPERIOD(c)
    df["BETA"] = talib.BETA(h, lo, timeperiod=5)

    return df
