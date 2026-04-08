"""
Module 3 Runner: Graph Analysis + PageRank

Pipeline:
    1. Load daily log_returns tu lsh-similarity/daily/ (~169K rows EOD close)
    2. Compute Pearson correlation matrix 101x101
    3. Build undirected weighted graph (|corr| >= threshold)
    4. PageRank (alpha=0.85)
    5. Community detection (Louvain hoac Label Propagation)
    6. Export CSV/Parquet + 3 visualizations
    7. Sinh doan VN paste vao bao cao

Usage:
    conda activate stock

    # Default threshold 0.5
    python src/run_graph_pagerank.py

    # Tuy chinh threshold neu graph qua thua hoac qua day
    python src/run_graph_pagerank.py --threshold 0.6

    # Cache correlation matrix de chay nhanh khi tweak threshold
    python src/run_graph_pagerank.py --skip-corr --threshold 0.65

Dependencies:
    pandas, numpy, networkx, scipy, pyarrow, matplotlib (da co)
    python-louvain (optional, conda install -c conda-forge python-louvain)
"""

import os
import sys
import time
import argparse
import logging

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_DIR)

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from config.settings import (
    LSH_PATH, GRAPH_CONFIG, GRAPH_OUTPUT_PATH, GRAPH_REPORTS_DIR,
    LOGS_DIR,
)
from algorithms.graph_pagerank import (
    load_daily_returns,
    compute_correlation,
    build_graph,
    run_pagerank,
    detect_communities,
)

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(GRAPH_OUTPUT_PATH, exist_ok=True)
os.makedirs(GRAPH_REPORTS_DIR, exist_ok=True)

log = logging.getLogger("stock_graph")
log.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
_fh = logging.FileHandler(os.path.join(LOGS_DIR, "graph_pagerank.log"))
_fh.setFormatter(_fmt)
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_fmt)
log.addHandler(_fh)
log.addHandler(_sh)


# ---------------------------------------------------------------------------
# EXPORT
# ---------------------------------------------------------------------------

def export_csvs(pr_df, comm_df, corr_matrix, G):
    pr_path = os.path.join(GRAPH_REPORTS_DIR, "pagerank_scores.csv")
    pr_df.to_csv(pr_path, index=False)
    log.info("EXPORT CSV       | %s (%d rows)", pr_path, len(pr_df))

    comm_path = os.path.join(GRAPH_REPORTS_DIR, "community_assignments.csv")
    comm_df.to_csv(comm_path, index=False)
    log.info("EXPORT CSV       | %s (%d rows)", comm_path, len(comm_df))

    corr_path = os.path.join(GRAPH_REPORTS_DIR, "correlation_matrix.csv")
    corr_matrix.to_csv(corr_path)
    log.info("EXPORT CSV       | %s (%dx%d)", corr_path,
             corr_matrix.shape[0], corr_matrix.shape[1])

    edges = [{"stock_i": u, "stock_j": v, "weight": d["weight"]}
             for u, v, d in G.edges(data=True)]
    edges_df = pd.DataFrame(edges)
    edges_path = os.path.join(GRAPH_REPORTS_DIR, "graph_edges.csv")
    edges_df.to_csv(edges_path, index=False)
    log.info("EXPORT CSV       | %s (%d edges)", edges_path, len(edges_df))


def export_parquets(pr_df, comm_df):
    pr_path = os.path.join(GRAPH_OUTPUT_PATH, "pagerank_scores.parquet")
    pr_df.to_parquet(pr_path, index=False)
    log.info("EXPORT PARQUET   | %s", pr_path)

    comm_path = os.path.join(GRAPH_OUTPUT_PATH, "community_assignments.parquet")
    comm_df.to_parquet(comm_path, index=False)
    log.info("EXPORT PARQUET   | %s", comm_path)


# ---------------------------------------------------------------------------
# VISUALIZATIONS
# ---------------------------------------------------------------------------

def plot_pagerank_bar(pr_df, config):
    """Top N stocks by PageRank score - horizontal bar chart."""
    top_n = config["top_n_display"]
    top_df = pr_df.head(top_n)

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = cm.RdYlGn(np.linspace(0.3, 0.9, top_n))[::-1]
    bars = ax.barh(
        top_df["stock_symbol"][::-1],
        top_df["pagerank_score"][::-1],
        color=colors, edgecolor="white", height=0.75,
    )
    ax.set_xlabel("PageRank Score", fontsize=11)
    ax.set_title(
        f"Top {top_n} Stocks by PageRank\n"
        f"(Correlation graph, threshold={config['corr_threshold']}, "
        f"alpha={config['pagerank_alpha']})",
        fontsize=12,
    )
    ax.grid(axis="x", alpha=0.3)
    for bar, (_, row) in zip(bars[::-1], top_df.iterrows()):
        ax.text(
            bar.get_width() + (0.00005 if bar.get_width() < 0.05 else 0.0001),
            bar.get_y() + bar.get_height() / 2,
            f"deg={row['degree']}", va="center", fontsize=8,
        )
    fig.tight_layout()
    out = os.path.join(GRAPH_REPORTS_DIR, "pagerank_bar_chart.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("PLOT             | %s", out)


def plot_correlation_heatmap(pr_df, corr_matrix, config):
    """Heatmap of top-N stocks by PageRank, ordered by community/PR."""
    top_n = config["heatmap_top_n"]
    top_stocks = pr_df.head(top_n)["stock_symbol"].tolist()
    sub = corr_matrix.loc[top_stocks, top_stocks]

    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(sub.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(top_stocks)))
    ax.set_yticks(range(len(top_stocks)))
    ax.set_xticklabels(top_stocks, rotation=90, fontsize=7)
    ax.set_yticklabels(top_stocks, fontsize=7)
    plt.colorbar(im, ax=ax, label="Pearson Correlation", shrink=0.8)
    ax.set_title(f"Correlation Heatmap - Top {top_n} Stocks by PageRank",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(GRAPH_REPORTS_DIR, "correlation_heatmap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("PLOT             | %s", out)


def plot_network(G, pr_df, comm_df, config):
    """Spring layout: top-N nodes, edges filtered by min weight."""
    top_n = config["network_top_n"]
    edge_min_w = config["network_edge_min_weight"]

    top_stocks = pr_df.head(top_n)["stock_symbol"].tolist()
    sub = G.subgraph(top_stocks).copy()

    # Filter edges by weight de visualization de doc
    sub_filtered = nx.Graph()
    sub_filtered.add_nodes_from(sub.nodes())
    for u, v, d in sub.edges(data=True):
        if d["weight"] >= edge_min_w:
            sub_filtered.add_edge(u, v, weight=d["weight"])

    fig, ax = plt.subplots(figsize=(16, 13))
    pos = nx.spring_layout(sub_filtered, seed=42, k=2.5, iterations=100)

    comm_dict = dict(zip(comm_df["stock_symbol"], comm_df["community_id"]))
    pr_dict = dict(zip(pr_df["stock_symbol"], pr_df["pagerank_score"]))

    node_colors = [comm_dict.get(n, 0) for n in sub_filtered.nodes()]
    pr_max = max(pr_dict.values())
    node_sizes = [
        max(pr_dict.get(n, 0.001) / pr_max * 2500, 200)
        for n in sub_filtered.nodes()
    ]
    edge_weights = [d["weight"] for _, _, d in sub_filtered.edges(data=True)]

    nx.draw_networkx_nodes(
        sub_filtered, pos, node_color=node_colors, node_size=node_sizes,
        cmap=cm.tab10, alpha=0.85, ax=ax, edgecolors="black", linewidths=0.5,
    )
    if edge_weights:
        nx.draw_networkx_edges(
            sub_filtered, pos,
            width=[w * 2.0 for w in edge_weights],
            alpha=0.5, edge_color=edge_weights,
            edge_cmap=cm.Blues, ax=ax,
        )
    nx.draw_networkx_labels(sub_filtered, pos, font_size=7.5, ax=ax)
    ax.set_title(
        f"Stock Correlation Network - Top {top_n} by PageRank\n"
        f"Node size proportional to PageRank | Color = Community | "
        f"Edge width proportional to |Correlation| (filter >= {edge_min_w})",
        fontsize=11,
    )
    ax.axis("off")
    fig.tight_layout()
    out = os.path.join(GRAPH_REPORTS_DIR, "network_graph.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("PLOT             | %s", out)


# ---------------------------------------------------------------------------
# SUMMARY TEXT
# ---------------------------------------------------------------------------

def write_summary_text(
    pr_df, comm_df, G, corr_matrix, modularity, method, config,
):
    threshold = config["corr_threshold"]
    alpha = config["pagerank_alpha"]
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    n_possible = n_nodes * (n_nodes - 1) // 2
    density = n_edges / n_possible if n_possible > 0 else 0.0
    n_comms = comm_df["community_id"].nunique()

    upper = corr_matrix.values[np.triu_indices(corr_matrix.shape[0], k=1)]
    upper = upper[~np.isnan(upper)]
    pct_positive = (upper > 0).mean() * 100
    pct_above_thresh = (np.abs(upper) >= threshold).mean() * 100

    top5 = pr_df.head(5)
    top5_lines = []
    for _, row in top5.iterrows():
        top5_lines.append(
            f"    #{row['rank']:2d} {row['stock_symbol']:<12s} "
            f"PR={row['pagerank_score']:.5f}  degree={row['degree']:3d}  "
            f"strength={row['strength']:.2f}"
        )
    top5_str = "\n".join(top5_lines)

    mod_str = f"{modularity:.4f}" if modularity is not None else "N/A"

    text = f"""GRAPH + PAGERANK SUMMARY -- MODULE 3
{"=" * 70}

[PASTE 1 -- Muc 5.1 Xay dung do thi tuong quan]

De phan tich mang luoi anh huong giua {n_nodes} ma co phieu An Do, chung
toi xay dung do thi vo huong co trong so tu ma tran tuong quan Pearson
cua chuoi log_return hang ngay. Dung log_return = log(close_t / close_(t-1))
thay vi Close price truc tiep nham dam bao chuoi thoi gian stationary,
tranh tuong quan gia (spurious correlation) thuong xuat hien khi tinh
Pearson tren chuoi gia tich luy non-stationary.

Moi cap co phieu (i, j) duoc noi boi mot canh voi trong so |corr(i, j)|
khi |corr| >= {threshold} (nguong tuong quan trung binh-manh). Trong tong so
{n_possible:,} cap co the ({n_nodes} stocks), {pct_above_thresh:.1f}% cap dat
nguong, tao ra do thi voi {n_edges:,} canh (mat do {density:.4f}). Dang
chu y, {pct_positive:.1f}% cap stocks co tuong quan duong -- phan anh xu
huong co-movement chung cua thi truong chung khoan An Do.

[PASTE 2 -- Muc 5.2 PageRank xac dinh systemic importance]

Chung toi ap dung thuat toan PageRank (damping factor alpha={alpha}) tren
do thi nay de do "tam anh huong he thong" (systemic importance) cua tung
ma. Theo ly thuyet CS246 (Rajaraman & Ullman, MMDS Chapter 5), PageRank
diem cao nghia la ma do duoc nhieu ma khac co anh huong manh tro den --
tuc bien dong cua no phan anh hoac dan dat bien dong cua nhieu ma trong
he thong.

Cong thuc PageRank tren do thi co weight:
    r(j) = alpha * Sum_(i->j) [w(i,j) * r(i) / strength(i)] + (1-alpha)/n
Voi: w(i,j) = |corr(i,j)|, strength(i) = sum weight cac canh ke voi i.

Top 5 ma co PageRank cao nhat:

{top5_str}

Cac ma nay co dac diem chung: degree va strength cao -- nghia la chung
tuong quan manh voi nhieu ma khac trong thi truong, dong vai tro
"hub" cua mang luoi anh huong he thong.

[PASTE 3 -- Muc 5.3 Community Detection]

Ben canh PageRank, chung toi ap dung thuat toan {method} de nhom cac
co phieu thanh {n_comms} cong dong (communities) dua tren cau truc
do thi tuong quan. Modularity = {mod_str} (Q > 0.3 = cau truc cong dong
ro rang, Q > 0.5 = phan nhom rat ro net) cho thay thi truong An Do co
su phan nhom theo sector ro ret.

So sanh voi ket qua K-means tu Module 1 (clustering dua tren feature
profile trung binh sau PCA), cac community tu do thi tuong quan cho
phep phan tich "mang luoi anh huong dong" thay vi chi dua tren dac
trung tinh:
  - Module 1: stocks co cung pattern indicator -> cluster (static)
  - Module 3: stocks dong dieu trong 10 nam -> community (dynamic)

[KET LUAN MODULE 3]

PageRank chuyen ma tran tuong quan {n_nodes}x{n_nodes} = {n_possible:,}
gia tri thanh mot vector {n_nodes} chieu xep hang theo systemic
importance. Modularity = {mod_str} cua {n_comms} communities chung
minh thi truong co cau truc network ro rang. Hai metric nay tra loi
truc tiep cau hoi: "Ma nao la trung tam anh huong?" va "Stocks nao
co xu huong co-move trong dai han?" -- noi dung trong tam cua
phan tich rui ro he thong va portfolio diversification.
"""

    out = os.path.join(GRAPH_REPORTS_DIR, "graph_summary.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    log.info("EXPORT TEXT      | %s", out)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Module 3: Graph + PageRank")
    parser.add_argument("--threshold", type=float, default=None,
                        help=f"Correlation threshold (default: "
                             f"{GRAPH_CONFIG['corr_threshold']})")
    parser.add_argument("--alpha", type=float, default=None,
                        help=f"PageRank damping (default: "
                             f"{GRAPH_CONFIG['pagerank_alpha']})")
    parser.add_argument("--skip-corr", action="store_true",
                        help="Load corr_matrix.csv tu cache")
    parser.add_argument("--input-daily", default=None,
                        help=f"Path toi daily parquet (default: "
                             f"{LSH_PATH}/daily)")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("MODULE 3: GRAPH ANALYSIS + PAGERANK")
    log.info("=" * 70)

    config = GRAPH_CONFIG.copy()
    if args.threshold is not None:
        config["corr_threshold"] = args.threshold
        log.info("Override threshold = %.2f", args.threshold)
    if args.alpha is not None:
        config["pagerank_alpha"] = args.alpha
        log.info("Override alpha = %.2f", args.alpha)

    daily_path = args.input_daily or os.path.join(LSH_PATH, "daily")
    if not os.path.isdir(daily_path):
        log.error(
            "Missing daily parquet. Run `python src/run_lsh.py` first to build it.\n"
            "  Expected: %s", daily_path,
        )
        sys.exit(1)

    log.info("Daily input  : %s", daily_path)
    log.info("Reports dir  : %s", GRAPH_REPORTS_DIR)
    log.info("Output dir   : %s", GRAPH_OUTPUT_PATH)
    log.info("Threshold    : %.2f | Alpha: %.2f",
             config["corr_threshold"], config["pagerank_alpha"])
    log.info("-" * 70)

    t_total = time.time()
    try:
        # STEP 1: Load returns
        returns_matrix = load_daily_returns(daily_path, config)

        # STEP 2: Correlation
        corr_path = os.path.join(GRAPH_REPORTS_DIR, "correlation_matrix.csv")
        if args.skip_corr and os.path.exists(corr_path):
            log.info("STAGE 2 CORR     | Loading cached %s", corr_path)
            corr_matrix = pd.read_csv(corr_path, index_col=0)
        else:
            corr_matrix = compute_correlation(returns_matrix, config)

        # STEP 3: Build graph
        G = build_graph(corr_matrix, threshold=config["corr_threshold"])
        if G.number_of_edges() == 0:
            log.error("Graph has 0 edges. Reduce threshold (--threshold 0.4).")
            sys.exit(1)

        # STEP 4: PageRank
        pr_df = run_pagerank(
            G,
            alpha=config["pagerank_alpha"],
            max_iter=config["pagerank_max_iter"],
            tol=config["pagerank_tol"],
        )

        # STEP 5: Community detection
        comm_df, communities, modularity, method = detect_communities(G)

        # STEP 6: Export
        export_csvs(pr_df, comm_df, corr_matrix, G)
        export_parquets(pr_df, comm_df)
        plot_pagerank_bar(pr_df, config)
        plot_correlation_heatmap(pr_df, corr_matrix, config)
        plot_network(G, pr_df, comm_df, config)

        # STEP 7: Summary text
        write_summary_text(
            pr_df, comm_df, G, corr_matrix, modularity, method, config,
        )

        elapsed = time.time() - t_total
        log.info("=" * 70)
        log.info("MODULE 3 COMPLETED in %.1fs", elapsed)
        log.info("=" * 70)
        log.info("Top 5 PageRank: %s",
                 ", ".join(pr_df.head(5)["stock_symbol"].tolist()))
        log.info("Communities   : %d (modularity=%s)",
                 comm_df["community_id"].nunique(),
                 f"{modularity:.4f}" if modularity is not None else "N/A")
        log.info("Edges         : %s (density=%.4f)",
                 f"{G.number_of_edges():,}",
                 G.number_of_edges() / (G.number_of_nodes() *
                                        (G.number_of_nodes() - 1) // 2))
        log.info("=" * 70)

    except Exception:
        log.exception("MODULE 3 FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
