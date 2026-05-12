"""
Portfolio Optimization tu ket qua Clustering (PCA + K-means).

Vai tro (tra loi "clustering de lam gi?"):
  Cluster gom cac co phieu CUNG HANH VI. De giam rui ro, chon DAI DIEN tu MOI
  cum -> danh muc da dang hoa, tranh dom het tien vao nhom co phieu di cung nhau.
  So sanh 3 danh muc:
    - DA DANG (diversified)  : 1 dai dien / cum
    - TAP TRUNG (concentrated): N ma trong CUNG 1 cum lon nhat
    - THI TRUONG (market)    : equal-weight tat ca ma (nen so sanh)
  Da dang co volatility thap hon / Sharpe cao hon => clustering tao gia tri.
"""

import numpy as np
import pandas as pd


_TRADING_DAYS = 252
_PC_COLS = ["pc1", "pc2", "pc3", "pc4", "pc5"]


def daily_returns_matrix(close_df: pd.DataFrame) -> pd.DataFrame:
    """close_df[stock_symbol, trade_date, close] -> ma tran return (date x stock)."""
    piv = close_df.pivot_table(index="trade_date", columns="stock_symbol",
                               values="close", aggfunc="last").sort_index()
    return piv.pct_change(fill_method=None)


def pick_representatives(clusters: pd.DataFrame, valid_stocks: set) -> dict:
    """Moi cum -> co phieu gan tam cum nhat (trong khong gian PC).

    Bo cac ma chi so (NIFTY...) va ma khong co du lieu gia.
    """
    pc_cols = [c for c in _PC_COLS if c in clusters.columns]
    reps = {}
    df = clusters[clusters["stock_symbol"].isin(valid_stocks)].copy()
    df = df[~df["stock_symbol"].str.contains("NIFTY", case=False, na=False)]
    for cid, grp in df.groupby("cluster_id"):
        centroid = grp[pc_cols].mean().to_numpy()
        dist = np.linalg.norm(grp[pc_cols].to_numpy() - centroid, axis=1)
        reps[int(cid)] = grp.iloc[int(np.argmin(dist))]["stock_symbol"]
    return reps


def _largest_cluster_members(clusters: pd.DataFrame, valid_stocks: set, n: int) -> list:
    """N co phieu thanh khoan cao nhat trong cum DONG nhat (de tao danh muc tap trung)."""
    df = clusters[clusters["stock_symbol"].isin(valid_stocks)]
    df = df[~df["stock_symbol"].str.contains("NIFTY", case=False, na=False)]
    biggest = df["cluster_id"].value_counts().idxmax()
    grp = df[df["cluster_id"] == biggest]
    sort_col = "trading_days" if "trading_days" in grp.columns else grp.columns[0]
    return grp.sort_values(sort_col, ascending=False)["stock_symbol"].head(n).tolist()


def _portfolio_series(returns: pd.DataFrame, members: list) -> pd.Series:
    """Loi suat danh muc equal-weight theo ngay (trung binh cac thanh vien co mat)."""
    cols = [m for m in members if m in returns.columns]
    if not cols:
        return pd.Series(dtype=float)
    return returns[cols].mean(axis=1, skipna=True).dropna()


def portfolio_metrics(ret: pd.Series) -> dict:
    """Cac thuoc do hieu nang/rui ro chuan."""
    if ret is None or len(ret) < 2:
        return {"cum_return": 0.0, "ann_return": 0.0, "ann_vol": 0.0,
                "sharpe": 0.0, "max_drawdown": 0.0}
    equity = (1 + ret).cumprod()
    cum = float(equity.iloc[-1] - 1)
    ann_ret = float(ret.mean() * _TRADING_DAYS)
    ann_vol = float(ret.std() * np.sqrt(_TRADING_DAYS))
    sharpe = float(ann_ret / ann_vol) if ann_vol > 1e-9 else 0.0
    drawdown = equity / equity.cummax() - 1
    return {"cum_return": cum, "ann_return": ann_ret, "ann_vol": ann_vol,
            "sharpe": sharpe, "max_drawdown": float(drawdown.min())}


def build_portfolios(close_df: pd.DataFrame, clusters: pd.DataFrame) -> dict:
    """Dung 3 danh muc + metrics + duong von tu cluster va daily close.

    Returns dict: representatives, members, equity (DataFrame), metrics (dict).
    """
    returns = daily_returns_matrix(close_df)
    valid = set(returns.columns)

    reps = pick_representatives(clusters, valid)
    div_members = sorted(reps.values())
    k = len(div_members)
    conc_members = _largest_cluster_members(clusters, valid, k)
    market_members = [s for s in valid if "NIFTY" not in s.upper()]

    series = {
        "Đa dạng (1 đại diện/cụm)": _portfolio_series(returns, div_members),
        "Tập trung (cùng 1 cụm)": _portfolio_series(returns, conc_members),
        "Thị trường (toàn bộ)": _portfolio_series(returns, market_members),
    }
    equity = pd.DataFrame({k_: (1 + s).cumprod() for k_, s in series.items() if len(s)})
    metrics = {k_: portfolio_metrics(s) for k_, s in series.items()}
    return {
        "representatives": reps,
        "members": {"diversified": div_members, "concentrated": conc_members,
                    "market_size": len(market_members)},
        "equity": equity,
        "metrics": metrics,
    }
