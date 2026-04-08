"""
Graph + PageRank Module -- Phase 2: Mining Massive Datasets

Bai toan: Trong 101 ma co phieu An Do, ma nao la "trung tam anh huong"
cua thi truong? Khi ma A bien dong, ma B co xu huong theo khong?

Pipeline:
  STAGE 1  load_daily_returns      : LSH daily parquet -> daily log_return matrix
  STAGE 2  compute_correlation     : Pearson corr 101x101 (log_return stationary)
  STAGE 3  build_graph              : adjacency voi |corr| >= threshold
  STAGE 4  run_pagerank             : NetworkX power iteration alpha=0.85
  STAGE 5  detect_communities       : Louvain (fallback Label Propagation)

Tai sao log_return:
- Close price co unit root (non-stationary) -> Pearson corr bi spurious
- log_return = log(close_t / close_{t-1}) la stationary -> corr co y nghia
- Reuse end-of-day close tu lsh-similarity/daily/ thay vi tinh lai 63.9M rows

Tai sao |corr| (khong phai corr):
- corr = -0.8 nghia la 2 stocks anh huong manh nguoc chieu -> van la "bonded"
- PageRank can weight duong de toi uu chinh xac
- Da xet do manh, khong xet chieu (option DiGraph cho cuc bo nang cao)
"""

import os
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import networkx as nx

log = logging.getLogger("stock_graph")


# ---------------------------------------------------------------------------
# STAGE 1: LOAD DAILY LOG RETURNS
# ---------------------------------------------------------------------------

def load_daily_returns(
    daily_path: str,
    config: Dict,
) -> pd.DataFrame:
    """Load end-of-day close tu LSH daily parquet -> daily log_return matrix.

    Daily log_return = log(close_t / close_{t-1}) tinh per stock tu EOD close.
    KHONG dung cot 'log_return' san co (no la log_return cua phut cuoi cung,
    khong phai log_return cua ngay).

    Args:
        daily_path : duong toi lsh-similarity/daily/ parquet
        config     : GRAPH_CONFIG dict

    Returns:
        DataFrame shape (n_dates, n_stocks)
            index = trade_date (datetime)
            columns = stock_symbol
            values = daily log_return
    """
    date_col = config["date_col"]
    stock_col = config["stock_col"]

    log.info("STAGE 1 LOAD     | Reading EOD close from %s", daily_path)
    df = pq.read_table(
        daily_path,
        columns=[stock_col, date_col, "close"],
    ).to_pandas()
    log.info("STAGE 1 LOAD     | %s rows, %d stocks",
             f"{len(df):,}", df[stock_col].nunique())

    # Sort de groupby + shift hoat dong dung
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([stock_col, date_col]).reset_index(drop=True)

    # Compute daily log_return per stock tu EOD close
    df["daily_log_return"] = (
        df.groupby(stock_col)["close"]
        .transform(lambda s: np.log(s / s.shift(1)))
    )
    df = df.dropna(subset=["daily_log_return"])

    # Pivot: rows = date, columns = stock
    returns_matrix = df.pivot_table(
        index=date_col,
        columns=stock_col,
        values="daily_log_return",
        aggfunc="mean",
    )

    # Filter ngay co it nhat <pct> stocks co data (tranh ngay le thua thot)
    min_pct = config["min_stocks_per_day_pct"]
    n_stocks = returns_matrix.shape[1]
    threshold_n = int(n_stocks * min_pct)
    returns_matrix = returns_matrix.dropna(thresh=threshold_n)

    log.info("STAGE 1 LOAD     | Returns matrix: %s dates x %d stocks",
             f"{returns_matrix.shape[0]:,}", returns_matrix.shape[1])
    log.info("STAGE 1 LOAD     | Date range: %s -> %s",
             returns_matrix.index.min().strftime("%Y-%m-%d"),
             returns_matrix.index.max().strftime("%Y-%m-%d"))
    return returns_matrix


# ---------------------------------------------------------------------------
# STAGE 2: PEARSON CORRELATION MATRIX
# ---------------------------------------------------------------------------

def compute_correlation(
    returns_matrix: pd.DataFrame,
    config: Dict,
) -> pd.DataFrame:
    """Pearson corr 101x101 tu daily log_return matrix.

    Forward-fill missing values truoc khi tinh corr de tranh
    drop pair khi 1 stock thieu data (vd: BANDHANBNK chi tu 2018).
    min_periods = ngay tinh chung toi thieu de corr co y nghia.
    """
    min_periods = config["min_periods"]
    log.info("STAGE 2 CORR     | Computing Pearson (min_periods=%d)...", min_periods)

    # Forward fill missing (thanh long pause), neu van NaN thi 0
    filled = returns_matrix.ffill().fillna(0.0)

    corr = filled.corr(method="pearson", min_periods=min_periods)

    n = corr.shape[0]
    upper = corr.values[np.triu_indices(n, k=1)]
    upper = upper[~np.isnan(upper)]

    log.info(
        "STAGE 2 CORR     | %d stocks, %s pairs | mean=%.3f std=%.3f min=%.3f max=%.3f",
        n, f"{len(upper):,}",
        upper.mean(), upper.std(), upper.min(), upper.max(),
    )
    return corr


# ---------------------------------------------------------------------------
# STAGE 3: BUILD GRAPH
# ---------------------------------------------------------------------------

def build_graph(
    corr: pd.DataFrame,
    threshold: float,
) -> nx.Graph:
    """Xay dung do thi vo huong co trong so tu correlation matrix.

    Edge: (i, j) khi |corr(i, j)| >= threshold
    Weight: |corr(i, j)| (luon duong, PageRank yeu cau)
    Self-loops: bi loai bo (corr_ii = 1)

    Threshold guideline:
      0.10-0.40 density => OK
      < 0.05   => threshold qua cao, giam threshold
      > 0.60   => threshold qua thap, tang threshold
    """
    stocks = corr.columns.tolist()
    n = len(stocks)

    G = nx.Graph()
    G.add_nodes_from(stocks)

    # Vectorized: lay upper triangle (k=1 de skip diagonal)
    arr = corr.values
    iu = np.triu_indices(n, k=1)
    abs_vals = np.abs(arr[iu])

    # Mask threshold
    mask = (abs_vals >= threshold) & ~np.isnan(abs_vals)
    edges = []
    for idx in np.where(mask)[0]:
        i = iu[0][idx]
        j = iu[1][idx]
        edges.append((stocks[i], stocks[j], float(abs_vals[idx])))

    G.add_weighted_edges_from(edges)

    n_edges = G.number_of_edges()
    n_possible = n * (n - 1) // 2
    density = n_edges / n_possible if n_possible > 0 else 0.0

    log.info("STAGE 3 GRAPH    | %d nodes, %s edges | density=%.4f (%.2f%% of %s pairs)",
             G.number_of_nodes(), f"{n_edges:,}", density,
             density * 100, f"{n_possible:,}")
    log.info("STAGE 3 GRAPH    | Connected components: %d",
             nx.number_connected_components(G))

    isolated = list(nx.isolates(G))
    if isolated:
        log.warning("STAGE 3 GRAPH    | %d isolated stocks: %s",
                    len(isolated), isolated[:10])

    return G


# ---------------------------------------------------------------------------
# STAGE 4: PAGERANK
# ---------------------------------------------------------------------------

def run_pagerank(
    G: nx.Graph,
    alpha: float = 0.85,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """PageRank power iteration tren weighted undirected graph.

    Cong thuc CS246:
        r(j) = alpha * sum_{i->j} r(i) * w(i,j) / strength(i) + (1-alpha)/n

    Returns:
        DataFrame sorted by pagerank desc, columns:
            stock_symbol, pagerank_score, rank, degree, strength
    """
    log.info("STAGE 4 PAGERANK | alpha=%.2f, max_iter=%d, tol=%.0e",
             alpha, max_iter, tol)

    pr_scores = nx.pagerank(
        G, alpha=alpha, max_iter=max_iter, tol=tol, weight="weight",
    )
    degree = dict(G.degree())
    strength = dict(G.degree(weight="weight"))

    rows = []
    for stock, score in pr_scores.items():
        rows.append({
            "stock_symbol": stock,
            "pagerank_score": float(score),
            "degree": int(degree.get(stock, 0)),
            "strength": round(float(strength.get(stock, 0.0)), 4),
        })
    df = pd.DataFrame(rows).sort_values("pagerank_score", ascending=False)
    df = df.reset_index(drop=True)
    df["rank"] = df.index + 1
    df = df[["rank", "stock_symbol", "pagerank_score", "degree", "strength"]]

    log.info("STAGE 4 PAGERANK | Sum=%f (should be ~1.0)",
             float(df["pagerank_score"].sum()))
    log.info("STAGE 4 PAGERANK | Top 5:")
    for _, row in df.head(5).iterrows():
        log.info("                   #%2d %-12s PR=%.5f deg=%3d strength=%.2f",
                 row["rank"], row["stock_symbol"], row["pagerank_score"],
                 row["degree"], row["strength"])
    return df


# ---------------------------------------------------------------------------
# STAGE 5: COMMUNITY DETECTION
# ---------------------------------------------------------------------------

def detect_communities(
    G: nx.Graph,
) -> Tuple[pd.DataFrame, Dict[int, List[str]], Optional[float], str]:
    """Phan cum stocks bang Louvain (fallback Label Propagation).

    Returns:
        comm_df    : DataFrame (stock_symbol, community_id, community_size, method)
        communities: dict community_id -> list[stock]
        modularity : float | None
        method     : "Louvain" | "LabelPropagation"
    """
    method = None
    partition: Dict[str, int] = {}

    # Try Louvain (python-louvain hoac networkx >= 3.x co built-in)
    try:
        import community as community_louvain  # python-louvain
        partition = community_louvain.best_partition(G, weight="weight")
        method = "Louvain"
    except ImportError:
        try:
            from networkx.algorithms.community import louvain_communities
            comms = louvain_communities(G, weight="weight", seed=42)
            for cid, members in enumerate(comms):
                for node in members:
                    partition[node] = cid
            method = "Louvain (nx)"
        except (ImportError, AttributeError):
            log.warning("STAGE 5 COMM     | Louvain unavailable, fallback to Label Propagation")
            comms = list(nx.algorithms.community.label_propagation_communities(G))
            for cid, members in enumerate(comms):
                for node in members:
                    partition[node] = cid
            method = "LabelPropagation"

    communities: Dict[int, List[str]] = {}
    for node, cid in partition.items():
        communities.setdefault(cid, []).append(node)

    # Modularity
    modularity = None
    try:
        modularity = float(nx.algorithms.community.modularity(
            G,
            [set(members) for members in communities.values()],
            weight="weight",
        ))
    except Exception as e:
        log.warning("STAGE 5 COMM     | Modularity error: %s", str(e)[:120])

    n_comms = len(communities)
    log.info("STAGE 5 COMM     | Method=%s | %d communities | modularity=%s",
             method, n_comms,
             f"{modularity:.4f}" if modularity is not None else "N/A")
    for cid in sorted(communities.keys()):
        members = communities[cid]
        sample = ", ".join(sorted(members)[:5])
        log.info("                   Community %d: %d stocks [%s%s]",
                 cid, len(members), sample,
                 ", ..." if len(members) > 5 else "")

    rows = []
    for node, cid in partition.items():
        rows.append({
            "stock_symbol": node,
            "community_id": int(cid),
            "community_size": len(communities[cid]),
            "detection_method": method,
        })
    comm_df = pd.DataFrame(rows).sort_values(["community_id", "stock_symbol"])
    comm_df = comm_df.reset_index(drop=True)
    if modularity is not None:
        comm_df["modularity"] = round(modularity, 4)

    return comm_df, communities, modularity, method
