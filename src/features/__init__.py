"""
Feature Engineering Module
Mining Massive Dataset Project - Stock Market Analysis

Modules:
- trend.py: Trend analysis features (SMA distance, EMA crossovers, ADX strength)
- momentum.py: Momentum & oscillator features (RSI status, MACD, Stochastic)
- volatility.py: Volatility features (Bollinger Band width, ATR percentage)
- phase2_prep.py: Phase 2 preparation features (log_return, zscore, labels)
"""

from .trend import add_trend_features
from .momentum import add_momentum_features
from .volatility import add_volatility_features
from .phase2_prep import add_phase2_features

__all__ = [
    "add_trend_features",
    "add_momentum_features",
    "add_volatility_features",
    "add_phase2_features",
]
