"""
DASHBOARD — Phương án A: Cảnh báo theo tiền lệ.

Streamlit app gồm 4 khu theo đúng tài liệu:
    KHU 1: kết quả cảnh báo (label + risk score)
    KHU 2: tín hiệu Ấn Độ (kho kinh nghiệm qua LSH)
    KHU 3: tín hiệu VN (kiểm chứng nội địa, cosine exact)
    KHU 4: bằng chứng — pattern hiện tại + các tiền lệ giống nhất chồng lên

Engine truy vấn là in-memory (numpy), nạp artifacts đã build qua
``run_precedent_alert.py`` → trả lời mili-giây, không cần Spark khi demo.
Có thêm trang Backtest hiển thị kết quả STEP 8 và Sanity check (STEP 4).
"""

import os
import sys
import json
import logging

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_DIR)

from config.settings import (
    PRECEDENT_PATH, PRECEDENT_REPORTS_DIR, PRECEDENT_CONFIG,
    PCA_REPORTS_DIR, GRAPH_REPORTS_DIR,
)
from similarity import PrecedentEngine
from analysis.anomaly import compute_anomalies, summarize_anomalies
from analysis.portfolio import build_portfolios, portfolio_metrics  # noqa: F401
from similarity.signal_fusion import LABEL_SAFE, LABEL_CAUTION, LABEL_STRONG

logging.basicConfig(level=logging.WARNING)

INDIA_DIR = os.path.join(PRECEDENT_PATH, "india")
VN_DIR = os.path.join(PRECEDENT_PATH, "vn")

st.set_page_config(
    page_title="Cảnh báo theo tiền lệ",
    page_icon="⚠️",
    layout="wide",
)


# ---------------------------------------------------------------------------
# CACHE: nạp engine 1 lần, dùng cho mọi query
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Đang nạp artifacts (Ấn Độ + VN)...")
def _load_engine():
    return PrecedentEngine(INDIA_DIR, VN_DIR, PRECEDENT_CONFIG)


@st.cache_data(show_spinner=False)
def _load_backtest_report():
    p = os.path.join(PRECEDENT_REPORTS_DIR, "backtest.json")
    if os.path.exists(p):
        return json.load(open(p))
    return None


@st.cache_data(show_spinner=False)
def _load_sanity_report():
    p = os.path.join(PRECEDENT_REPORTS_DIR, "sanity_check.json")
    if os.path.exists(p):
        return json.load(open(p))
    return None


@st.cache_data(show_spinner=False)
def _load_daily_close(market: str) -> pd.DataFrame:
    """Doc chuoi daily close tu vectors da build (cho anomaly)."""
    vdir = os.path.join(PRECEDENT_PATH, market, "vectors")
    if not os.path.isdir(vdir):
        return pd.DataFrame()
    return pd.read_parquet(vdir, columns=["stock_symbol", "trade_date", "close"])


@st.cache_data(show_spinner=False)
def _build_portfolio_demo():
    """Dựng danh mục từ cluster + daily close India (cache)."""
    close = _load_daily_close("india")
    clusters = _read_csv(os.path.join(PCA_REPORTS_DIR, "cluster_assignments.csv"))
    if close.empty or clusters is None:
        return None
    return build_portfolios(close, clusters)


@st.cache_data(show_spinner=False)
def _read_csv(path: str):
    return pd.read_csv(path) if os.path.exists(path) else None


@st.cache_data(show_spinner=False)
def _read_text(path: str):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    return None


def _show_image(path: str, caption: str = None):
    if path and os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.caption(f"(chưa có ảnh: {os.path.basename(path) if path else '—'})")


# ---------------------------------------------------------------------------
# UI HELPERS
# ---------------------------------------------------------------------------

# Bảng màu nhãn cảnh báo. Khoá phải khớp giá trị enum trong signal_fusion.
LABEL_COLORS = {
    LABEL_SAFE: "#1a9850",       # xanh
    LABEL_CAUTION: "#fee08b",    # vàng
    LABEL_STRONG: "#d73027",     # đỏ
}


def _verdict_banner(verdict: dict):
    color = LABEL_COLORS.get(verdict["label"], "#888")
    st.markdown(
        f"""
        <div style="background:{color}25;border-left:8px solid {color};
                    padding:18px 22px;border-radius:6px;margin-bottom:6px;">
          <div style="font-size:32px;font-weight:700;color:{color};">
            {verdict['label']}
          </div>
          <div style="font-size:18px;color:#222;margin-top:4px;">
            Risk score <b>{verdict['risk_score']}/100</b> &nbsp;|&nbsp;
            Độ tin cậy <b>{verdict['confidence_label']}</b>
              ({verdict['confidence']:.2f}) &nbsp;|&nbsp;
            <b>{verdict['agreement']}</b>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Chi tiết lý do", expanded=True):
        for line in verdict["reason"]:
            st.markdown(f"- {line}")


def _signal_panel(name: str, sig: dict, color: str):
    """Hiển thị 1 khu tín hiệu (Ấn Độ hoặc VN)."""
    st.markdown(
        f"""<div style="border-left:5px solid {color};padding:6px 12px;
                       margin-bottom:8px;font-size:18px;font-weight:600;">
              {name}
            </div>""",
        unsafe_allow_html=True,
    )
    if sig["n"] == 0:
        st.warning("Không tìm được tiền lệ giống.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Số tiền lệ", sig["n"])
    c2.metric("% giảm 3 ngày", f"{sig['p_down'] * 100:.0f}%",
              delta=f"{sig['excess'] * 100:+.1f}pt vs nền")
    c3.metric("Cosine TB", f"{sig['avg_cosine']:.3f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Lợi suất TB", f"{sig['mean_return'] * 100:+.2f}%")
    c5.metric("Sâu nhất", f"{sig['worst_return'] * 100:+.2f}%")
    c6.metric("Base rate", f"{sig['base_rate'] * 100:.0f}%")

    # Biểu đồ cột: % giảm vs % tăng vs nền
    df = pd.DataFrame({
        "type": ["Tiền lệ: GIẢM", "Tiền lệ: TĂNG", "Nền: GIẢM"],
        "pct": [sig["p_down"], 1 - sig["p_down"], sig["base_rate"]],
    })
    fig = px.bar(df, x="type", y="pct", color="type",
                 color_discrete_sequence=["#d73027", "#1a9850", "#888"],
                 height=220)
    fig.update_layout(
        showlegend=False, yaxis_tickformat=".0%", yaxis_title=None,
        xaxis_title=None, margin=dict(l=8, r=8, t=8, b=8),
    )
    fig.add_hline(y=sig["base_rate"], line_dash="dash", line_color="#444",
                  annotation_text=f"nền {sig['base_rate']:.0%}",
                  annotation_position="top right")
    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"CI95 = [{sig['ci_low']:.0%}, {sig['ci_high']:.0%}]")


def _evidence_section(result: dict, engine):
    """KHU 4: pattern hiện tại + tiền lệ giống (chồng lên)."""
    qvec = np.asarray(result["pattern_vector"])
    window_days = PRECEDENT_CONFIG["window_days"]
    zret = qvec[:window_days]   # phần "hình dạng" (chuỗi lợi suất z-score)

    st.subheader("KHU 4 — Bằng chứng")

    n_show = 5
    days = list(range(1, window_days + 1))

    fig = go.Figure()
    # Các tiền lệ Ấn Độ
    for i, p in enumerate(result["india_precedents"][:n_show]):
        z = engine.get_zret("india", p["stock_symbol"], p["trade_date"])
        if z is None:
            continue
        fig.add_trace(go.Scatter(
            x=days, y=z, mode="lines", opacity=0.55,
            name=f"IN: {p['stock_symbol']} {p['trade_date']} (cos={p['cosine']:.2f})",
            line=dict(color="#2166ac", width=1.5, dash="dot"),
            showlegend=(i == 0),
            legendgroup="india",
            legendgrouptitle_text="Ấn Độ",
        ))
    # Các tiền lệ VN
    for i, p in enumerate(result["vn_precedents"][:n_show]):
        z = engine.get_zret("vn", p["stock_symbol"], p["trade_date"])
        if z is None:
            continue
        fig.add_trace(go.Scatter(
            x=days, y=z, mode="lines", opacity=0.55,
            name=f"VN: {p['stock_symbol']} {p['trade_date']} (cos={p['cosine']:.2f})",
            line=dict(color="#d6604d", width=1.5, dash="dash"),
            showlegend=(i == 0),
            legendgroup="vn",
            legendgrouptitle_text="VN",
        ))
    # Pattern hiện tại (vẽ cuối → nằm trên cùng)
    fig.add_trace(go.Scatter(
        x=days, y=zret, mode="lines+markers",
        name="Pattern hiện tại",
        line=dict(color="#08306b", width=4),
        marker=dict(size=6),
    ))
    fig.update_layout(
        height=420, xaxis_title="Ngày (gần đây nhất ở bên phải)",
        yaxis_title="z-score(daily_return)",
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="v", x=1.02, y=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Top tiền lệ Ấn Độ**")
        df = pd.DataFrame(result["india_precedents"][:10])
        if not df.empty:
            df["fwd_return"] = df["fwd_return"].map(lambda x: f"{x * 100:+.2f}%")
            df["cosine"] = df["cosine"].map(lambda x: f"{x:.3f}")
            df["fwd_down"] = df["fwd_down"].map(lambda x: "GIẢM" if x else "TĂNG")
            st.dataframe(df, use_container_width=True, hide_index=True)
    with cols[1]:
        st.markdown("**Top tiền lệ VN**")
        df = pd.DataFrame(result["vn_precedents"][:10])
        if not df.empty:
            df["fwd_return"] = df["fwd_return"].map(lambda x: f"{x * 100:+.2f}%")
            df["cosine"] = df["cosine"].map(lambda x: f"{x:.3f}")
            df["fwd_down"] = df["fwd_down"].map(lambda x: "GIẢM" if x else "TĂNG")
            st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# TRANG: ALERT
# ---------------------------------------------------------------------------

def page_alert(engine: PrecedentEngine):
    st.title("⚠️ Cảnh báo theo tiền lệ — Phương án A")

    # ---- Form input ----
    stocks = engine.list_stocks()
    c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
    stock = c1.selectbox("Mã VN", stocks,
                         index=stocks.index("VCB") if "VCB" in stocks else 0)
    dates = engine.list_dates(stock)
    if not dates:
        st.error(f"Không có pattern nào cho {stock}.")
        return
    date = c2.selectbox("Ngày phân tích", dates[::-1], index=0)
    top_k = c3.number_input("top_k", min_value=10, max_value=200,
                            value=PRECEDENT_CONFIG["top_k"], step=10)
    go_btn = c4.button("PHÂN TÍCH", type="primary", use_container_width=True)

    if not go_btn:
        st.info("Chọn mã + ngày rồi bấm **PHÂN TÍCH**.")
        return

    result = engine.query(stock, date, top_k=int(top_k))
    if "error" in result:
        st.error(result["error"]); return

    q = result["query"]
    st.write(
        f"**{q['stock']}** @ **{q['pattern_date']}** (close = {q['close']:.2f}) "
        f"&nbsp;|&nbsp; window = {PRECEDENT_CONFIG['window_days']} ngày, top_k = {q['top_k']}"
    )

    # KHU 1 — cảnh báo
    st.subheader("KHU 1 — Kết quả cảnh báo")
    _verdict_banner(result["verdict"])

    # KHU 2 + 3 — hai tín hiệu
    c_in, c_vn = st.columns(2)
    with c_in:
        st.subheader("KHU 2 — Tín hiệu Ấn Độ")
        _signal_panel("Ấn Độ", result["india_signal"], "#2166ac")
    with c_vn:
        st.subheader("KHU 3 — Tín hiệu VN")
        _signal_panel("VN", result["vn_signal"], "#d6604d")

    st.divider()
    _evidence_section(result, engine)


# ---------------------------------------------------------------------------
# TRANG: BACKTEST
# ---------------------------------------------------------------------------

def page_backtest():
    st.title("📊 Backtest walk-forward (STEP 8)")

    bt = _load_backtest_report()
    if bt is None:
        st.warning(
            "Chưa có backtest.json. Chạy: "
            "`python src/run_precedent_alert.py --backtest`"
        )
        return
    if "error" in bt:
        st.error(bt["error"]); return

    c = st.columns(4)
    c[0].metric("Pattern đánh giá", f"{bt['n_evaluated']:,}")
    c[1].metric("Directional acc",
                f"{bt['directional_accuracy'] * 100:.2f}%",
                delta=f"vs base {bt['base_rate_down_vn'] * 100:.2f}%")
    c[2].metric("Precision (down)", f"{bt['precision_down'] * 100:.2f}%")
    c[3].metric("Safe-call winrate", f"{bt['safe_call_winrate'] * 100:.2f}%")

    st.divider()

    # Confusion matrix
    cm = bt["confusion"]
    z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    fig = go.Figure(data=go.Heatmap(
        z=z, x=["pred UP", "pred DOWN"], y=["actual UP", "actual DOWN"],
        text=[[str(v) for v in r] for r in z], texttemplate="%{text}",
        colorscale="Blues", showscale=False,
    ))
    fig.update_layout(height=320, title="Ma trận nhầm lẫn (confusion matrix)")

    # Phân tích theo nhãn
    by_label = bt.get("by_label", {})
    if by_label:
        df = pd.DataFrame([
            {"label": k, "count": v["count"],
             "actual_p_down": v["actual_p_down"],
             "mean_fwd_return": v["mean_fwd_return"]}
            for k, v in by_label.items()
        ])
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=df["label"], y=df["actual_p_down"],
            marker_color=[LABEL_COLORS.get(l, "#888") for l in df["label"]],
            name="actual p_down",
        ))
        fig2.add_hline(y=bt["base_rate_down_vn"], line_dash="dash",
                       annotation_text=f"nền {bt['base_rate_down_vn']:.2%}")
        fig2.update_layout(height=320, yaxis_tickformat=".0%",
                           title="% giảm thực tế theo nhãn cảnh báo")
    else:
        fig2 = None

    cc = st.columns(2)
    cc[0].plotly_chart(fig, use_container_width=True)
    if fig2 is not None:
        cc[1].plotly_chart(fig2, use_container_width=True)
        st.dataframe(df.style.format({"actual_p_down": "{:.2%}",
                                      "mean_fwd_return": "{:+.2%}"}),
                     use_container_width=True, hide_index=True)

    st.subheader("LSH efficiency (quy mô big-data)")
    eff = bt["lsh_efficiency"]
    cc = st.columns(4)
    cc[0].metric("VN patterns", f"{eff['vn_patterns']:,}")
    cc[1].metric("Ấn Độ library", f"{eff['india_library']:,}")
    cc[2].metric("Brute-force pairs", f"{eff['all_pairs_bruteforce']:,}")
    cc[3].metric("LSH reduction", f"{eff['reduction_x']:,.1f}x")
    st.caption(
        f"Tổng cặp brute-force: {eff['all_pairs_bruteforce']:,}. "
        f"LSH chỉ sinh {eff['lsh_candidate_pairs']:,} candidate "
        f"→ giảm {eff['reduction_x']:,.1f}x phép so sánh."
    )


# ---------------------------------------------------------------------------
# TRANG: SANITY
# ---------------------------------------------------------------------------

def page_sanity():
    st.title("🔬 Sanity check Ấn Độ vs VN (STEP 4)")
    rep = _load_sanity_report()
    if rep is None:
        st.warning(
            "Chưa có sanity_check.json. Chạy: "
            "`python src/run_precedent_alert.py --sanity`"
        )
        return

    v = rep["verdict"]
    st.success(f"Kết luận: **{v['level']}** ({v['n_ok']}/{v['n_total']} checks đạt)")

    df = pd.DataFrame(v["checks"])
    df["status"] = df["ok"].map({True: "✅ OK", False: "❌ X"})
    st.dataframe(df[["metric", "india", "vn", "diff", "tol", "status"]]
                 .style.format({"india": "{:+.4f}", "vn": "{:+.4f}",
                                "diff": "{:.4f}", "tol": "{:.3f}"}),
                 use_container_width=True, hide_index=True)

    plot_path = rep.get("plot")
    if plot_path and os.path.exists(plot_path):
        st.image(plot_path,
                 caption="Phân phối cấp pattern (fwd_return) Ấn Độ vs VN")


# ---------------------------------------------------------------------------
# TRANG: CLUSTERING (PCA)  -- Module 1
# ---------------------------------------------------------------------------

def page_clustering():
    st.title("🧬 Clustering theo nhân tố (PCA + K-means/CURE)")

    sil = _read_text(os.path.join(PCA_REPORTS_DIR, "silhouette_comparison.txt"))
    if sil:
        # Trich vai so chinh ra metric
        c = st.columns(3)
        for line in sil.splitlines():
            if "BEFORE PCA" in line:
                c[0].metric("Silhouette TRƯỚC PCA", line.split(":")[-1].strip())
            elif "k=6)" in line and "K-means" in line:
                c[1].metric("Silhouette SAU PCA (K-means)", line.split(":")[-1].strip())
            elif "variance at" in line.lower() or "Explained Variance" in line:
                c[2].metric("Variance giữ lại", line.split(":")[-1].strip())
        with st.expander("Chi tiết silhouette / variance"):
            st.code(sil)

    col1, col2 = st.columns(2)
    with col1:
        _show_image(os.path.join(PCA_REPORTS_DIR, "scree_plot.png"),
                    "Scree plot — chọn số PC giữ ≥90% variance")
    with col2:
        _show_image(os.path.join(PCA_REPORTS_DIR, "pca_scatter_clusters.png"),
                    "Cổ phiếu trên không gian PC1-PC2, tô màu theo cụm")
    _show_image(os.path.join(PCA_REPORTS_DIR, "cluster_profiles.png"),
                "Hồ sơ trung bình từng cụm")

    df = _read_csv(os.path.join(PCA_REPORTS_DIR, "cluster_assignments.csv"))
    if df is not None:
        st.subheader("Phân cụm cổ phiếu")
        cluster_col = next((c for c in df.columns if "cluster" in c.lower()), None)
        if cluster_col:
            counts = df[cluster_col].value_counts().sort_index()
            st.bar_chart(counts)
        st.dataframe(df, use_container_width=True, hide_index=True, height=300)

    # ----- Danh mục từ cụm (Portfolio Optimization) -----
    st.divider()
    st.subheader("💼 Danh mục đầu tư xây từ cụm")
    pf = _build_portfolio_demo()
    if pf is None:
        st.info("Chưa đủ dữ liệu (cần cluster_assignments.csv + vectors Ấn Độ).")
    else:
        reps = pf["representatives"]
        st.caption("Đại diện mỗi cụm (gần tâm cụm nhất): "
                   + ", ".join(f"cụm {c}→{s}" for c, s in sorted(reps.items())))

        # Đường vốn 3 danh mục
        eq = pf["equity"]
        figpf = go.Figure()
        for col in eq.columns:
            figpf.add_trace(go.Scatter(x=eq.index, y=eq[col], mode="lines", name=col))
        figpf.update_layout(height=360, title="Đường vốn (1 đồng ban đầu)",
                            margin=dict(l=8, r=8, t=40, b=8),
                            legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(figpf, use_container_width=True)

        # Bảng metrics
        mt = pd.DataFrame(pf["metrics"]).T
        mt = mt.rename(columns={
            "cum_return": "Tổng lời", "ann_return": "Lời/năm",
            "ann_vol": "Biến động/năm", "sharpe": "Sharpe",
            "max_drawdown": "Sụt sâu nhất"})
        st.dataframe(
            mt.style.format({"Tổng lời": "{:+.1%}", "Lời/năm": "{:+.1%}",
                             "Biến động/năm": "{:.1%}", "Sharpe": "{:.2f}",
                             "Sụt sâu nhất": "{:.1%}"}),
            use_container_width=True)


# ---------------------------------------------------------------------------
# TRANG: MẠNG NGÀNH & PAGERANK  -- Module 3
# ---------------------------------------------------------------------------

def page_pagerank():
    st.title("🕸️ Mạng ngành & PageRank (rủi ro hệ thống)")

    pr = _read_csv(os.path.join(GRAPH_REPORTS_DIR, "pagerank_scores.csv"))
    if pr is not None and not pr.empty:
        st.subheader("Top cổ phiếu theo PageRank (tâm ảnh hưởng hệ thống)")
        score_col = next((c for c in pr.columns if "pagerank" in c.lower()
                          or "score" in c.lower()), None) or pr.columns[-1]
        name_col = "stock_symbol" if "stock_symbol" in pr.columns else pr.columns[0]
        top = pr.sort_values(score_col, ascending=False).head(15)
        fig = px.bar(top, x=score_col, y=name_col, orientation="h", height=440,
                     color=score_col, color_continuous_scale="Reds")
        fig.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=8, r=8, t=8, b=8))
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        _show_image(os.path.join(GRAPH_REPORTS_DIR, "network_graph.png"),
                    "Mạng tương quan — node lớn = hub ảnh hưởng")
    with col2:
        _show_image(os.path.join(GRAPH_REPORTS_DIR, "correlation_heatmap.png"),
                    "Heatmap tương quan top mã")

    summary = _read_text(os.path.join(GRAPH_REPORTS_DIR, "graph_summary.txt"))
    if summary:
        with st.expander("Tóm tắt phân tích mạng (PageRank + cộng đồng)"):
            st.code(summary)


# ---------------------------------------------------------------------------
# TRANG: PHÁT HIỆN BẤT THƯỜNG  -- z-score + CUSUM
# ---------------------------------------------------------------------------

def page_anomaly():
    st.title("🚨 Phát hiện bất thường (Z-score + CUSUM)")

    c1, c2, c3 = st.columns([1, 2, 2])
    market = c1.selectbox("Thị trường", ["vn", "india"], index=0)
    daily_all = _load_daily_close(market)
    if daily_all.empty:
        st.warning(f"Chưa có vectors cho thị trường {market}.")
        return
    stocks = sorted(daily_all["stock_symbol"].unique().tolist())
    default = stocks.index("VCB") if "VCB" in stocks else 0
    stock = c2.selectbox("Cổ phiếu", stocks, index=default)
    z_th = c3.slider("Ngưỡng z-score (|z|>)", 2.0, 4.0, 3.0, step=0.5)

    one = daily_all[daily_all["stock_symbol"] == stock][["trade_date", "close"]]
    res = compute_anomalies(one, z_threshold=z_th)
    summ = summarize_anomalies(res)

    m = st.columns(4)
    m[0].metric("Số ngày", summ["n_days"])
    m[1].metric("Điểm bất thường", summ["n_anomalies"],
                delta=f"{summ['anomaly_rate'] * 100:.1f}%")
    m[2].metric("Lần đổi chế độ (CUSUM)", summ["n_regime_changes"])
    if summ["worst_anomaly"]:
        m[3].metric("Sốc mạnh nhất (z)", f"{summ['worst_anomaly']['zscore']:+.1f}")

    res["trade_date"] = pd.to_datetime(res["trade_date"])

    # Biểu đồ 1: giá + chấm bất thường + đường đổi chế độ
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res["trade_date"], y=res["close"], mode="lines",
                             name="close", line=dict(color="#08306b", width=1.5)))
    anom = res[res["is_anomaly"]]
    fig.add_trace(go.Scatter(x=anom["trade_date"], y=anom["close"], mode="markers",
                             name="bất thường (z)", marker=dict(color="#d73027", size=10,
                                                                 symbol="x")))
    for _, r in res[res["regime_change"] != 0].iterrows():
        fig.add_vline(x=r["trade_date"], line_dash="dash",
                      line_color="#1a9850" if r["regime_change"] > 0 else "#d73027",
                      opacity=0.5)
    fig.update_layout(height=360, title=f"{stock} — giá + điểm bất thường + đổi chế độ",
                      margin=dict(l=8, r=8, t=40, b=8))
    st.plotly_chart(fig, use_container_width=True)

    # Biểu đồ 2: z-score và CUSUM
    col1, col2 = st.columns(2)
    with col1:
        figz = go.Figure()
        figz.add_trace(go.Scatter(x=res["trade_date"], y=res["zscore"], mode="lines",
                                  line=dict(color="#542788")))
        figz.add_hline(y=z_th, line_dash="dash", line_color="#d73027")
        figz.add_hline(y=-z_th, line_dash="dash", line_color="#d73027")
        figz.update_layout(height=280, title="Z-score cuốn chiếu", margin=dict(l=8, r=8, t=40, b=8))
        st.plotly_chart(figz, use_container_width=True)
    with col2:
        figc = go.Figure()
        figc.add_trace(go.Scatter(x=res["trade_date"], y=res["cusum_pos"], mode="lines",
                                  name="CUSUM+", line=dict(color="#1a9850")))
        figc.add_trace(go.Scatter(x=res["trade_date"], y=res["cusum_neg"], mode="lines",
                                  name="CUSUM−", line=dict(color="#d73027")))
        figc.update_layout(height=280, title="CUSUM (tích lũy lệch → đổi chế độ)",
                           margin=dict(l=8, r=8, t=40, b=8))
        st.plotly_chart(figc, use_container_width=True)

    if summ["regime_dates"]:
        st.caption("Ngày phát hiện đổi chế độ: " + ", ".join(summ["regime_dates"]))


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    st.sidebar.markdown("### Stock BigData — Demo")
    page = st.sidebar.radio(
        "Trang",
        ["⚠️ Cảnh báo (Phương án A)", "📊 Backtest", "🔬 Sanity check",
         "🧬 Clustering (PCA)", "🕸️ Mạng ngành & PageRank", "🚨 Phát hiện bất thường"],
    )

    if page.startswith("⚠️"):
        engine = _load_engine()
        page_alert(engine)
    elif page.startswith("📊"):
        page_backtest()
    elif page.startswith("🔬"):
        page_sanity()
    elif page.startswith("🧬"):
        page_clustering()
    elif page.startswith("🕸️"):
        page_pagerank()
    elif page.startswith("🚨"):
        page_anomaly()


if __name__ == "__main__":
    main()
