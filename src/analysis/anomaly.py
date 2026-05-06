"""
Anomaly Detection -- Z-score (diem bat thuong) + CUSUM (doi che do / drift).

Vai tro trong he thong (tra loi cau hoi cua thay "bo di thi sao?"):
  - Z-score bat CU SOC DIEM: mot ngay loi suat lech > 3 do lech chuan so voi
    hanh vi gan day -> sap san, tin sock, dot bien khoi luong. Neu BO -> he
    thong khong co canh bao tuc thoi cho su kien cuc doan rieng le.
  - CUSUM bat DOI CHE DO (regime change): tich luy lech nho keo dai -> thi
    truong chuyen tu tang sang giam (hoac nguoc lai). Z-score KHONG bat duoc
    vi moi ngay deu "binh thuong", chi co XU HUONG tich luy moi lo ra. Neu BO
    -> mat kha nang phat hien chuyen pha cham (vd dau gau bat dau).

Hai phuong phap BO SUNG nhau: diem (z-score) vs xu huong (CUSUM). Day la ly
do ton tai cua module -- khong the thay the bang module khac.

Tat ca thuan pandas/numpy (du lieu daily nho) -> chay tuc thi, khong can Spark.
"""

import numpy as np
import pandas as pd


def _rolling_zscore(log_return: pd.Series, window: int) -> pd.Series:
    """Z-score cuon chieu: lech bao nhieu std so voi `window` ngay gan nhat."""
    mean = log_return.rolling(window, min_periods=max(5, window // 2)).mean()
    std = log_return.rolling(window, min_periods=max(5, window // 2)).std()
    return (log_return - mean) / std.replace(0.0, np.nan)


def _cusum(z: np.ndarray, k: float, h: float):
    """Tabular CUSUM hai phia tren chuoi da chuan hoa.

    S_pos tich luy lech DUONG, S_neg tich luy lech AM. Khi vuot nguong h ->
    bao 'doi che do' va reset. Tra ve (s_pos, s_neg, regime_change).
    """
    n = len(z)
    s_pos = np.zeros(n)
    s_neg = np.zeros(n)
    regime = np.zeros(n, dtype=int)
    sp = sn = 0.0
    for i in range(n):
        x = z[i] if np.isfinite(z[i]) else 0.0
        sp = max(0.0, sp + x - k)
        sn = max(0.0, sn - x - k)
        if sp > h:
            regime[i] = 1   # drift LEN (chuyen sang tang)
            sp = 0.0
        elif sn > h:
            regime[i] = -1  # drift XUONG (chuyen sang giam)
            sn = 0.0
        s_pos[i] = sp
        s_neg[i] = sn
    return s_pos, s_neg, regime


def compute_anomalies(
    daily: pd.DataFrame,
    return_window: int = 20,
    z_threshold: float = 3.0,
    cusum_k: float = 0.5,
    cusum_h: float = 5.0,
) -> pd.DataFrame:
    """Tinh z-score + CUSUM cho 1 chuoi daily cua 1 co phieu.

    Args:
        daily: DataFrame co cot ['trade_date', 'close'] da sort tang dan.
        return_window: cua so cho rolling z-score.
        z_threshold: |z| > nguong nay => diem bat thuong.
        cusum_k, cusum_h: tham so CUSUM (slack & nguong quyet dinh, don vi std).

    Returns:
        DataFrame them cot: log_return, zscore, is_anomaly, cusum_pos,
        cusum_neg, regime_change.
    """
    df = daily.sort_values("trade_date").reset_index(drop=True).copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    df["zscore"] = _rolling_zscore(df["log_return"], return_window)
    df["is_anomaly"] = df["zscore"].abs() > z_threshold

    # Chuan hoa toan chuoi cho CUSUM (mean 0, std 1)
    lr = df["log_return"].to_numpy()
    mu = np.nanmean(lr)
    sd = np.nanstd(lr)
    z_std = (lr - mu) / sd if sd > 1e-12 else np.zeros_like(lr)
    s_pos, s_neg, regime = _cusum(z_std, cusum_k, cusum_h)
    df["cusum_pos"] = s_pos
    df["cusum_neg"] = s_neg
    df["regime_change"] = regime
    return df


def summarize_anomalies(df: pd.DataFrame) -> dict:
    """Thong ke ngan cho dashboard."""
    n = len(df)
    n_anom = int(df["is_anomaly"].sum())
    regimes = df[df["regime_change"] != 0]
    return {
        "n_days": n,
        "n_anomalies": n_anom,
        "anomaly_rate": (n_anom / n) if n else 0.0,
        "n_regime_changes": int((df["regime_change"] != 0).sum()),
        "regime_dates": regimes["trade_date"].astype(str).tolist(),
        "worst_anomaly": (
            df.loc[df["zscore"].abs().idxmax(), ["trade_date", "zscore", "log_return"]].to_dict()
            if n_anom else None
        ),
    }
