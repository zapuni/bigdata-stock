"""
LSH Benchmark Runner -- Module 2 chot bao cao

Muc dich: Xuat benchmark_table.csv + summary cho bao cao.
Tai dung: Kho LSH index da build (stock/lsh-similarity/), khong chay lai ETL.

Usage:
    # Default: 1000 queries, top-K=10
    python src/run_lsh_benchmark.py

    # Smoke test
    python src/run_lsh_benchmark.py --n-queries 50

    # Tang n queries / top-K
    python src/run_lsh_benchmark.py --n-queries 2000 --k 20

Output: reports/lsh-benchmark/
    benchmark_table.csv          - 1 row / query (full data)
    benchmark_summary_table.csv  - 7-row summary cho bao cao
    benchmark_summary.txt        - doan van dinh luong (paste vao bao cao)
    benchmark_speedup_chart.png  - Hinh 4.1
    benchmark_precision_recall.png - Hinh 4.2
"""

import os
import sys
import time
import random
import logging
import argparse
from collections import defaultdict
from typing import Dict, FrozenSet, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_DIR)

from config.settings import (
    LSH_PATH, LSH_BENCHMARK_REPORTS_DIR, LSH_CONFIG, LOGS_DIR,
)
from algorithms.lsh import (
    _hash_shingle, _band_hash, _create_shingle_set,
    SIGNAL_COLS, WINDOW_DAYS, K_SHINGLE,
    N_HASH, N_BANDS, ROWS_PER_BAND,
)

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(LSH_BENCHMARK_REPORTS_DIR, exist_ok=True)

log = logging.getLogger("stock_lsh_bench")
log.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
_fh = logging.FileHandler(os.path.join(LOGS_DIR, "lsh_benchmark.log"))
_fh.setFormatter(_fmt)
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_fmt)
log.addHandler(_fh)
log.addHandler(_sh)


# ---------------------------------------------------------------------------
# STEP 1: LOAD CORPUS (shingles per (stock, date)) + SIGNATURES
# ---------------------------------------------------------------------------

def load_corpus_from_daily(daily_path: str) -> Dict[Tuple[str, str], FrozenSet[int]]:
    """Reconstruct shingle sets cho moi (stock, date) tu daily/ parquet.

    daily_state khong duoc luu (lsh.py drop sau Stage 2), nen ta tinh lai:
      1. concat SIGNAL_COLS thanh daily_state string
      2. window 20 ngay -> list of states
      3. _create_shingle_set(states, k=2) -> frozenset[int]

    Chi giu rows co du WINDOW_DAYS lich su (1:1 voi signatures parquet).
    """
    log.info("LOAD CORPUS  | Reading %s", daily_path)
    t0 = time.time()
    df = pq.read_table(daily_path).to_pandas()
    log.info("LOAD CORPUS  | %s daily rows in %.1fs",
             f"{len(df):,}", time.time() - t0)

    # Build daily_state nhu lsh.py: "rsi_status|macd_cross_signal|..."
    df["daily_state"] = df[SIGNAL_COLS].astype(str).agg("|".join, axis=1)

    # Sort theo (stock, date) de window dung
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["stock_symbol", "trade_date"]).reset_index(drop=True)

    log.info("LOAD CORPUS  | Building shingles (window=%d, k=%d)...",
             WINDOW_DAYS, K_SHINGLE)
    t0 = time.time()

    corpus: Dict[Tuple[str, str], FrozenSet[int]] = {}
    for stock, grp in df.groupby("stock_symbol", sort=False):
        states = grp["daily_state"].tolist()
        dates = grp["trade_date"].tolist()
        # Cua so truot 20 ngay - chi giu rows da co du history
        for i in range(WINDOW_DAYS - 1, len(states)):
            window = states[i - WINDOW_DAYS + 1: i + 1]
            shingles = _create_shingle_set(window, K_SHINGLE)
            key = (stock, dates[i].strftime("%Y-%m-%d"))
            corpus[key] = frozenset(shingles)

    log.info("LOAD CORPUS  | %s docs reconstructed in %.1fs",
             f"{len(corpus):,}", time.time() - t0)
    return corpus


def load_signatures(sig_path: str) -> Dict[Tuple[str, str], np.ndarray]:
    """Load signatures parquet -> dict (stock, date_str) -> np.array[100] uint64."""
    log.info("LOAD SIGS    | Reading %s", sig_path)
    t0 = time.time()
    df = pq.read_table(
        sig_path,
        columns=["stock_symbol", "trade_date", "signature"],
    ).to_pandas()
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    sig_store: Dict[Tuple[str, str], np.ndarray] = {}
    for row in df.itertuples(index=False):
        key = (row.stock_symbol, row.trade_date.strftime("%Y-%m-%d"))
        sig_store[key] = np.array(row.signature, dtype=np.int64)

    log.info("LOAD SIGS    | %s signatures in %.1fs",
             f"{len(sig_store):,}", time.time() - t0)
    return sig_store


# ---------------------------------------------------------------------------
# STEP 2: BUILD IN-MEMORY LSH INDEX
# ---------------------------------------------------------------------------

def build_inmemory_index(
    sig_store: Dict[Tuple[str, str], np.ndarray],
) -> Dict[Tuple[int, int], List[Tuple[str, str]]]:
    """Re-band signatures -> dict (band_idx, bucket_hash) -> list[(stock, date)].

    Su dung cung _band_hash() voi lsh.py de bucket assignment giong het pipeline.
    """
    log.info("BUILD INDEX  | Banding %s signatures (b=%d, r=%d)...",
             f"{len(sig_store):,}", N_BANDS, ROWS_PER_BAND)
    t0 = time.time()

    index: Dict[Tuple[int, int], List[Tuple[str, str]]] = defaultdict(list)
    for key, sig in sig_store.items():
        for b in range(N_BANDS):
            chunk = sig[b * ROWS_PER_BAND: (b + 1) * ROWS_PER_BAND]
            bucket = _band_hash(chunk.tolist())
            index[(b, bucket)].append(key)

    n_buckets = len(index)
    avg_size = sum(len(v) for v in index.values()) / max(n_buckets, 1)
    log.info("BUILD INDEX  | %s buckets, avg=%.1f docs/bucket in %.1fs",
             f"{n_buckets:,}", avg_size, time.time() - t0)
    return index


# ---------------------------------------------------------------------------
# STEP 3: BENCHMARK PRIMITIVES
# ---------------------------------------------------------------------------

def jaccard(a: FrozenSet[int], b: FrozenSet[int]) -> float:
    """Jaccard(A, B) = |A intersect B| / |A union B| -- ground truth similarity."""
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union > 0 else 0.0


def brute_force_top_k(
    query_key: Tuple[str, str],
    corpus: Dict[Tuple[str, str], FrozenSet[int]],
    k: int,
) -> Tuple[List[Tuple[Tuple[str, str], float]], float]:
    """O(N) - tinh Jaccard voi tat ca docs. Ground truth cho precision/recall."""
    query_set = corpus[query_key]
    t0 = time.perf_counter()
    scores = []
    for key, shingles in corpus.items():
        if key == query_key:
            continue
        scores.append((key, jaccard(query_set, shingles)))
    scores.sort(key=lambda x: x[1], reverse=True)
    elapsed = time.perf_counter() - t0
    return scores[:k], elapsed


def lsh_top_k(
    query_key: Tuple[str, str],
    corpus: Dict[Tuple[str, str], FrozenSet[int]],
    sig_store: Dict[Tuple[str, str], np.ndarray],
    index: Dict[Tuple[int, int], List[Tuple[str, str]]],
    k: int,
) -> Tuple[List[Tuple[Tuple[str, str], float]], float, int]:
    """O(b*r + |candidates|) -- collect candidates from band lookups,
    verify Jaccard chinh xac chi tren candidates."""
    query_set = corpus[query_key]
    query_sig = sig_store[query_key]

    t0 = time.perf_counter()

    # Step 1: Collect candidates tu cac bands
    candidates = set()
    for b in range(N_BANDS):
        chunk = query_sig[b * ROWS_PER_BAND: (b + 1) * ROWS_PER_BAND]
        bucket = _band_hash(chunk.tolist())
        bucket_keys = index.get((b, bucket))
        if bucket_keys:
            for ck in bucket_keys:
                if ck != query_key:
                    candidates.add(ck)

    # Step 2: Verify Jaccard chi tren candidates
    scores = []
    for ck in candidates:
        scores.append((ck, jaccard(query_set, corpus[ck])))
    scores.sort(key=lambda x: x[1], reverse=True)

    elapsed = time.perf_counter() - t0
    return scores[:k], elapsed, len(candidates)


# ---------------------------------------------------------------------------
# STEP 4: BENCHMARK LOOP
# ---------------------------------------------------------------------------

def run_benchmark(
    queries: List[Tuple[str, str]],
    corpus: Dict[Tuple[str, str], FrozenSet[int]],
    sig_store: Dict[Tuple[str, str], np.ndarray],
    index: Dict[Tuple[int, int], List[Tuple[str, str]]],
    k: int,
) -> pd.DataFrame:
    """Loop n_queries, cho moi query dung BF (ground truth) va LSH, do thoi gian va metrics."""
    log.info("BENCHMARK    | Running %s queries (top-K=%d)...",
             f"{len(queries):,}", k)
    n_corpus = len(corpus)
    rows = []
    t_total = time.time()

    for i, q in enumerate(queries):
        if i > 0 and i % 100 == 0:
            elapsed = time.time() - t_total
            rate = i / elapsed
            eta = (len(queries) - i) / max(rate, 1e-6)
            log.info(
                "BENCHMARK    | Progress %d/%d (%.1f q/s, ETA %.0fs)",
                i, len(queries), rate, eta,
            )

        bf_results, bf_time = brute_force_top_k(q, corpus, k)
        lsh_results, lsh_time, n_cands = lsh_top_k(q, corpus, sig_store, index, k)

        bf_keys = {x[0] for x in bf_results}
        lsh_keys = {x[0] for x in lsh_results}
        overlap = len(bf_keys & lsh_keys)

        precision = overlap / len(lsh_keys) if lsh_keys else 0.0
        recall = overlap / len(bf_keys) if bf_keys else 0.0
        speedup = bf_time / lsh_time if lsh_time > 0 else float("inf")

        bf_avg_jac = float(np.mean([s for _, s in bf_results])) if bf_results else 0.0
        lsh_avg_jac = float(np.mean([s for _, s in lsh_results])) if lsh_results else 0.0

        rows.append({
            "query_stock": q[0],
            "query_date": q[1],
            "bf_time_ms": round(bf_time * 1000, 3),
            "lsh_time_ms": round(lsh_time * 1000, 3),
            "speedup": round(speedup, 1),
            "n_candidates": n_cands,
            f"precision@{k}": round(precision, 4),
            f"recall@{k}": round(recall, 4),
            "avg_jaccard_bf": round(bf_avg_jac, 4),
            "avg_jaccard_lsh": round(lsh_avg_jac, 4),
        })

    log.info(
        "BENCHMARK    | Done %d queries in %.1fs (corpus=%s)",
        len(queries), time.time() - t_total, f"{n_corpus:,}",
    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# STEP 5: EXPORT TABLE + SUMMARY
# ---------------------------------------------------------------------------

def export_tables(df: pd.DataFrame, k: int, n_corpus: int) -> Dict:
    """Save full CSV + summary CSV. Return stats dict cho summary text."""
    full_path = os.path.join(LSH_BENCHMARK_REPORTS_DIR, "benchmark_table.csv")
    df.to_csv(full_path, index=False)
    log.info("EXPORT       | Full table -> %s (%s rows)",
             full_path, f"{len(df):,}")

    p_col = f"precision@{k}"
    r_col = f"recall@{k}"

    stats = {
        "n_queries": len(df),
        "n_corpus": n_corpus,
        "median_bf_ms": float(df["bf_time_ms"].median()),
        "median_lsh_ms": float(df["lsh_time_ms"].median()),
        "p95_bf_ms": float(df["bf_time_ms"].quantile(0.95)),
        "p95_lsh_ms": float(df["lsh_time_ms"].quantile(0.95)),
        "median_speedup": float(df["speedup"].median()),
        "mean_speedup": float(df["speedup"].mean()),
        "median_cands": float(df["n_candidates"].median()),
        "mean_precision": float(df[p_col].mean()),
        "mean_recall": float(df[r_col].mean()),
        "mean_jac_bf": float(df["avg_jaccard_bf"].mean()),
        "mean_jac_lsh": float(df["avg_jaccard_lsh"].mean()),
        "k": k,
        "b": N_BANDS,
        "r": ROWS_PER_BAND,
    }

    cands_pct = (1 - stats["median_cands"] / n_corpus) * 100

    summary = pd.DataFrame({
        "Metric": [
            "Median query time (ms)",
            "P95 query time (ms)",
            "Speedup vs Brute-Force (median)",
            "Speedup vs Brute-Force (mean)",
            "Median candidates examined",
            "Search space reduction",
            f"Precision@{k}",
            f"Recall@{k}",
            f"Avg Jaccard top-{k}",
            "N queries",
        ],
        "Brute-Force": [
            f"{stats['median_bf_ms']:,.1f}",
            f"{stats['p95_bf_ms']:,.1f}",
            "1x  (baseline)",
            "1x  (baseline)",
            f"{n_corpus:,}",
            "0%",
            "1.000",
            "1.000",
            f"{stats['mean_jac_bf']:.3f}",
            f"{stats['n_queries']:,}",
        ],
        f"LSH (b={N_BANDS}, r={ROWS_PER_BAND})": [
            f"{stats['median_lsh_ms']:,.2f}",
            f"{stats['p95_lsh_ms']:,.2f}",
            f"~{stats['median_speedup']:,.0f}x",
            f"~{stats['mean_speedup']:,.0f}x",
            f"{stats['median_cands']:,.0f}",
            f"{cands_pct:.2f}%",
            f"{stats['mean_precision']:.3f}",
            f"{stats['mean_recall']:.3f}",
            f"{stats['mean_jac_lsh']:.3f}",
            f"{stats['n_queries']:,}",
        ],
    })

    sum_path = os.path.join(LSH_BENCHMARK_REPORTS_DIR, "benchmark_summary_table.csv")
    summary.to_csv(sum_path, index=False)
    log.info("EXPORT       | Summary table -> %s", sum_path)

    return stats


# ---------------------------------------------------------------------------
# STEP 6: VISUALIZATIONS
# ---------------------------------------------------------------------------

def plot_speedup(df: pd.DataFrame) -> None:
    """Box plot query time + histogram speedup."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: box plot query times (log scale)
    bp = axes[0].boxplot(
        [df["bf_time_ms"], df["lsh_time_ms"]],
        labels=["Brute-Force", "LSH"],
        patch_artist=True,
        showfliers=False,
    )
    for patch, color in zip(bp["boxes"], ["#d1e5f0", "#fee08b"]):
        patch.set_facecolor(color)
    axes[0].set_ylabel("Query Time (ms, log scale)")
    axes[0].set_title("Query Time Distribution\n(lower is better)")
    axes[0].set_yscale("log")
    axes[0].grid(True, axis="y", alpha=0.3)

    # Right: speedup histogram
    speedups = df["speedup"].replace([np.inf, -np.inf], np.nan).dropna()
    speedups_clipped = speedups.clip(upper=speedups.quantile(0.99))
    axes[1].hist(speedups_clipped, bins=40, color="#2166ac",
                 alpha=0.85, edgecolor="white")
    median_s = speedups.median()
    axes[1].axvline(median_s, color="red", linestyle="--", linewidth=2,
                    label=f"Median: {median_s:,.0f}x")
    axes[1].set_xlabel("Speedup (BF time / LSH time)")
    axes[1].set_ylabel("Number of Queries")
    axes[1].set_title(f"Speedup Distribution\n(across {len(df):,} queries)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(LSH_BENCHMARK_REPORTS_DIR, "benchmark_speedup_chart.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("PLOT         | %s", out)


def plot_precision_recall(df: pd.DataFrame, k: int) -> None:
    """Scatter n_candidates vs precision + binned PR curve."""
    p_col = f"precision@{k}"
    r_col = f"recall@{k}"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: scatter n_cand vs precision, color by recall
    sc = axes[0].scatter(
        df["n_candidates"], df[p_col],
        alpha=0.4, s=12, c=df[r_col], cmap="RdYlGn",
    )
    axes[0].axhline(df[p_col].mean(), color="red", linestyle="--",
                    linewidth=1.5, label=f"Mean Precision: {df[p_col].mean():.3f}")
    axes[0].set_xlabel("Candidates Examined by LSH")
    axes[0].set_ylabel(f"Precision@{k}")
    axes[0].set_title(f"Candidates vs Precision@{k}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.2)
    plt.colorbar(sc, ax=axes[0], label=f"Recall@{k}")

    # Right: binned mean precision/recall by candidate count
    try:
        bins = pd.qcut(df["n_candidates"], q=8, duplicates="drop")
        grp = df.groupby(bins, observed=True)[[p_col, r_col]].mean().reset_index()
        x = range(len(grp))
        axes[1].plot(x, grp[p_col], marker="o", linewidth=2,
                     label=f"Precision@{k}", color="#2166ac")
        axes[1].plot(x, grp[r_col], marker="s", linewidth=2,
                     label=f"Recall@{k}", color="#d73027")
        axes[1].set_xticks(list(x))
        axes[1].set_xticklabels([str(b) for b in grp["n_candidates"]],
                                rotation=45, ha="right", fontsize=7)
        axes[1].set_ylabel("Score")
        axes[1].set_title(f"Precision/Recall@{k} binned by candidates")
        axes[1].set_ylim(0, 1.05)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    except ValueError:
        # n_candidates qua dong deu de qcut
        axes[1].text(0.5, 0.5, "Insufficient variance in n_candidates",
                     ha="center", va="center", transform=axes[1].transAxes)

    fig.tight_layout()
    out = os.path.join(LSH_BENCHMARK_REPORTS_DIR, "benchmark_precision_recall.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("PLOT         | %s", out)


# ---------------------------------------------------------------------------
# STEP 7: SUMMARY TEXT FOR REPORT
# ---------------------------------------------------------------------------

def write_summary_text(stats: Dict) -> None:
    """Sinh doan van dinh luong de paste vao bao cao."""
    k = stats["k"]
    b = stats["b"]
    r = stats["r"]
    n_corpus = stats["n_corpus"]
    n_queries = stats["n_queries"]
    cands_pct = (1 - stats["median_cands"] / n_corpus) * 100

    # Cong thuc threshold ly thuyet: tai t = (1/b)^(1/r), P(same bucket) = 0.5
    t_threshold = (1 / b) ** (1 / r)

    text = f"""LSH BENCHMARK SUMMARY -- MODULE 2 (b={b}, r={r})
{"=" * 70}

[PASTE 1 -- Muc 4.2 Benchmark hieu nang]

De danh gia hieu qua cua LSH so voi tinh Jaccard truc tiep, chung toi
thuc hien thi nghiem benchmark tren {n_queries:,} truy van duoc chon ngau
nhien tu tap {n_corpus:,} (stock, ngay) -- moi diem ung voi mot chuoi
20 ngay giao dich gan nhat. Voi moi truy van, ca hai phuong phap deu
duoc do bang cung mot tap shingles va cung quy trinh xep hang theo
Jaccard similarity.

Ket qua thuc do:
  - Brute-force median = {stats['median_bf_ms']:,.1f} ms/query (P95 = {stats['p95_bf_ms']:,.1f} ms)
  - LSH        median = {stats['median_lsh_ms']:.2f} ms/query (P95 = {stats['p95_lsh_ms']:.2f} ms)
  - Speedup median  = {stats['median_speedup']:,.0f}x (mean = {stats['mean_speedup']:,.0f}x)
  - Khong gian tim kiem giam: {n_corpus:,} -> trung binh {stats['median_cands']:,.0f} candidates/query
    (giam {cands_pct:.2f}% so voi brute-force).

Phan tich: LSH thu hep khong gian tim kiem bang ban-luot (b={b} bands,
r={r} rows/band) tren MinHash signature 100 chieu. Voi tham so nay,
nguong ly thuyet de hai diem co xac suat 50% vao chung mot bucket la
t = (1/b)^(1/r) = (1/{b})^(1/{r}) = {t_threshold:.3f}, nghia la cap
shingle co Jaccard >= {t_threshold:.2f} co kha nang cao xuat hien
chung bucket trong it nhat 1 trong {b} bands.

[PASTE 2 -- Muc 4.3 Do chinh xac (Precision / Recall)]

De do do chinh xac cua approximation cua LSH, chung toi lay top-{k} ket
qua brute-force (sorted theo Jaccard chinh xac) lam ground truth, sau
do tinh:
  Precision@{k} = |LSH ket qua giao BF ket qua| / |LSH ket qua|
  Recall@{k}    = |LSH ket qua giao BF ket qua| / |BF ket qua|

Ket qua:
  - Precision@{k} = {stats['mean_precision']:.3f}
  - Recall@{k}    = {stats['mean_recall']:.3f}
  - Avg Jaccard top-{k} (BF):  {stats['mean_jac_bf']:.3f}
  - Avg Jaccard top-{k} (LSH): {stats['mean_jac_lsh']:.3f}

LSH tim dung {stats['mean_precision']*100:.1f}% trong top-{k} ket qua thuc su gan nhat
va bat duoc {stats['mean_recall']*100:.1f}% tong so ket qua gan nhat trong corpus.
Sai so ~{(1-stats['mean_recall'])*100:.1f}% la trade-off chap nhan duoc so voi loi
ich toc do {stats['median_speedup']:,.0f}x. Diem nay phu hop voi dac tinh ly thuyet
cua LSH (Rajaraman & Ullman, MMDS Chapter 3): xac suat false negative
duoc kiem soat thong qua tham so b va r.

So sanh chat luong xep hang: avg Jaccard cua top-{k} LSH ({stats['mean_jac_lsh']:.3f})
gan voi avg Jaccard cua top-{k} BF ({stats['mean_jac_bf']:.3f}) -- chenh lech
{abs(stats['mean_jac_bf'] - stats['mean_jac_lsh']):.3f}, cho thay LSH khong chi
tra ve nhanh ma con tra ve cac ket qua co chat luong tuong duong.

[KET LUAN MODULE 2]

LSH trong dataset nay chuyen tu chi phi O(N) ({n_corpus:,} so sanh) ve O(b*r + |C|)
({b * r} hash lookups + ~{stats['median_cands']:,.0f} verifications), giam {stats['median_speedup']:,.0f}x
thoi gian truy van trong khi van giu Precision >= {stats['mean_precision']:.2f} va Recall >= {stats['mean_recall']:.2f}.
Day la giai phap thuc te cho bai toan tim ngay giao dich tuong tu trong
corpus 100K+ diem ma brute-force khong dap ung duoc real-time tu dashboard.
"""

    out = os.path.join(LSH_BENCHMARK_REPORTS_DIR, "benchmark_summary.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    log.info("EXPORT       | Summary text -> %s", out)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LSH Benchmark for Module 2 report")
    parser.add_argument("--n-queries", type=int,
                        default=LSH_CONFIG["benchmark_n_queries"],
                        help="So queries de benchmark (default: 1000)")
    parser.add_argument("--k", type=int,
                        default=LSH_CONFIG["benchmark_top_k"],
                        help="Top-K cho Precision/Recall (default: 10)")
    parser.add_argument("--seed", type=int,
                        default=LSH_CONFIG["benchmark_seed"],
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("LSH BENCHMARK -- MODULE 2 CHOT BAO CAO")
    log.info("=" * 70)
    log.info("LSH path     : %s", LSH_PATH)
    log.info("Reports dir  : %s", LSH_BENCHMARK_REPORTS_DIR)
    log.info("N queries    : %d", args.n_queries)
    log.info("Top-K        : %d", args.k)
    log.info("LSH params   : b=%d, r=%d, n_hash=%d", N_BANDS, ROWS_PER_BAND, N_HASH)
    log.info("-" * 70)

    daily_path = os.path.join(LSH_PATH, "daily")
    sig_path = os.path.join(LSH_PATH, "signatures")

    if not os.path.isdir(daily_path) or not os.path.isdir(sig_path):
        log.error(
            "Missing LSH artifacts. Run `python src/run_lsh.py` first.\n"
            "  Expected: %s and %s", daily_path, sig_path,
        )
        sys.exit(1)

    t_total = time.time()
    try:
        # STEP 1: Load corpus + signatures
        corpus = load_corpus_from_daily(daily_path)
        sig_store = load_signatures(sig_path)

        # Sanity: corpus va sig_store nen co cung keys
        common = set(corpus.keys()) & set(sig_store.keys())
        log.info(
            "SANITY       | corpus=%s, sig=%s, common=%s",
            f"{len(corpus):,}", f"{len(sig_store):,}", f"{len(common):,}",
        )
        if len(common) == 0:
            log.error("No common keys between corpus and signatures!")
            sys.exit(1)
        # Restrict to common keys de tranh KeyError trong loop
        corpus = {k: corpus[k] for k in common}
        sig_store = {k: sig_store[k] for k in common}

        # STEP 2: Build in-memory LSH index
        index = build_inmemory_index(sig_store)

        # STEP 3: Sample queries
        random.seed(args.seed)
        all_keys = list(corpus.keys())
        n = min(args.n_queries, len(all_keys))
        queries = random.sample(all_keys, n)
        log.info("QUERIES      | Sampled %s queries (seed=%d)",
                 f"{n:,}", args.seed)

        # Warning if BF se lau
        bf_per_query_est_ms = len(corpus) / 1000  # rough: ~1us per Jaccard
        total_bf_min = (bf_per_query_est_ms * n) / 1000 / 60
        if total_bf_min > 5:
            log.warning(
                "BF estimated total: ~%.1f min. Use --n-queries 200 to test first.",
                total_bf_min,
            )

        # STEP 4: Run benchmark
        df = run_benchmark(queries, corpus, sig_store, index, k=args.k)

        # STEP 5-7: Export + plots + summary
        stats = export_tables(df, k=args.k, n_corpus=len(corpus))
        plot_speedup(df)
        plot_precision_recall(df, k=args.k)
        write_summary_text(stats)

        elapsed = time.time() - t_total
        log.info("=" * 70)
        log.info("BENCHMARK COMPLETED in %.1f min", elapsed / 60)
        log.info("=" * 70)
        log.info("Median speedup    : %s",
                 f"{stats['median_speedup']:,.0f}x")
        log.info("Mean Precision@%d : %.4f", args.k, stats["mean_precision"])
        log.info("Mean Recall@%d    : %.4f", args.k, stats["mean_recall"])
        log.info("Median BF time    : %s ms", f"{stats['median_bf_ms']:,.1f}")
        log.info("Median LSH time   : %s ms", f"{stats['median_lsh_ms']:,.2f}")
        log.info("=" * 70)

    except Exception:
        log.exception("BENCHMARK FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
