"""
Phương án A — Cảnh báo theo tiền lệ (Similarity Search).

Pipeline (cosine similarity + SimHash LSH):
    feature_vector.py   STEP 2  : pattern 20 ngày → hybrid feature vector
                                   (z-score theo từng thị trường)
    simhash_lsh.py      STEP 3  : SimHash random-hyperplane index cho cosine
    precedent_query.py  STEP 5–6: tìm tiền lệ (Ấn Độ qua LSH, VN qua cosine
                                   exact) + thống kê có base rate & CI
    signal_fusion.py    STEP 7  : hợp nhất hai tín hiệu → risk score 0–100
                                   + nhãn cảnh báo
    sanity_check.py     STEP 4  : so phân phối Ấn Độ vs VN ở cấp pattern
    backtest.py         STEP 8  : backtest walk-forward bằng Spark LSH-join
    memory_engine.py             : engine in-memory (numpy) cho dashboard

Dùng chung 1 cách đo (cosine) cho cả hai thị trường để so sánh công bằng.
Ấn Độ = kho kinh nghiệm (LSH tăng tốc); VN = nơi kiểm chứng (cosine exact).
"""

from .feature_vector import build_feature_vectors
from .simhash_lsh import (
    SimHasher,
    build_simhash_index,
    load_hyperplanes,
    add_band_buckets,
)
from .precedent_query import (
    lookup_pattern_vector,
    query_precedents_lsh,
    query_precedents_exact,
    summarize_precedents,
)
from .signal_fusion import fuse_signals
from .sanity_check import run_sanity_check
from .backtest import run_backtest
from .memory_engine import PrecedentEngine

__all__ = [
    "build_feature_vectors",
    "SimHasher",
    "build_simhash_index",
    "load_hyperplanes",
    "add_band_buckets",
    "lookup_pattern_vector",
    "query_precedents_lsh",
    "query_precedents_exact",
    "summarize_precedents",
    "fuse_signals",
    "run_sanity_check",
    "run_backtest",
    "PrecedentEngine",
]
