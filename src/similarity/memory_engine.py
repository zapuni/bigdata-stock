"""
Engine truy vấn IN-MEMORY (numpy) cho Dashboard.

Tinh thần big data nằm ở BUILD (Spark ETL 63M dòng + dựng SimHash index 169K
pattern). Sau khi index đã có, việc phục vụ 1 query là nearest-neighbor lookup
— giống cách FAISS/ScaNN nạp index vào RAM. Engine này:

  - Nạp 1 lần: Ấn Độ index (~35MB vectors) + VN vectors + hyperplanes.
  - Sinh candidate qua SimHash buckets (MIRROR pipeline Spark), rồi cosine exact.
  - Tái dùng y nguyên SimHasher + summarize_precedents + fuse_signals.

⇒ Dashboard trả lời tức thì, không cần khởi động Spark.
"""

import os
import json
import logging
from collections import defaultdict

import numpy as np
import pandas as pd

from .simhash_lsh import SimHasher, load_hyperplanes
from .precedent_query import summarize_precedents, _wilson_interval  # noqa: F401
from .signal_fusion import fuse_signals

log = logging.getLogger("precedent")


class PrecedentEngine:
    """Nap artifacts da build va tra loi query canh bao (khong dung Spark)."""

    def __init__(self, india_dir: str, vn_dir: str, config: dict):
        self.config = config
        self.india_index_dir = os.path.join(india_dir, "simhash-index")
        self.india_vectors_dir = os.path.join(india_dir, "vectors")
        self.vn_vectors_dir = os.path.join(vn_dir, "vectors")

        self.hasher = SimHasher(
            load_hyperplanes(india_dir),
            config["simhash_band_bits"],
            config["simhash_n_bands"],
        )
        self._load_india()
        self._load_vn()

    # ----- loading -----
    def _load_india(self):
        df = pd.read_parquet(self.india_index_dir)
        self.in_vectors = np.vstack(df["vector"].apply(np.asarray).to_numpy())
        self.in_stock = df["stock_symbol"].to_numpy()
        self.in_date = df["trade_date"].astype(str).to_numpy()
        self.in_down = df["fwd_down"].to_numpy(dtype=float)
        self.in_ret = df["fwd_return"].to_numpy(dtype=float)

        # Inverted index: bucket -> list[row idx] (sinh candidate kieu LSH)
        self.bucket_index = defaultdict(list)
        for i, buckets in enumerate(df["band_buckets"].to_numpy()):
            for b in buckets:
                self.bucket_index[b].append(i)

        self.india_base = {
            "n": int(np.sum(~np.isnan(self.in_down))),
            "p_down": float(np.nanmean(self.in_down)),
            "mean_return": float(np.nanmean(self.in_ret)),
        }
        log.info("ENGINE | India index: %d patterns, %d buckets",
                 len(self.in_stock), len(self.bucket_index))

    def _load_vn(self):
        df = pd.read_parquet(self.vn_vectors_dir).sort_values(["stock_symbol", "trade_date"])
        self.vn_vectors = np.vstack(df["vector"].apply(np.asarray).to_numpy())
        self.vn_stock = df["stock_symbol"].to_numpy()
        self.vn_date = df["trade_date"].astype(str).to_numpy()
        self.vn_close = df["close"].to_numpy(dtype=float)
        self.vn_down = df["fwd_down"].to_numpy(dtype=float)
        self.vn_ret = df["fwd_return"].to_numpy(dtype=float)
        self.vn_base = {
            "n": int(np.sum(~np.isnan(self.vn_down))),
            "p_down": float(np.nanmean(self.vn_down)),
            "mean_return": float(np.nanmean(self.vn_ret)),
        }
        log.info("ENGINE | VN vectors: %d patterns", len(self.vn_stock))

    # ----- helpers -----
    def list_stocks(self) -> list:
        return sorted(set(self.vn_stock.tolist()))

    def list_dates(self, stock: str) -> list:
        return sorted(self.vn_date[self.vn_stock == stock].tolist())

    def get_zret(self, market: str, stock: str, date: str):
        """Lay phan z-score(daily_return) 20 ngay cua 1 (stock, date)."""
        wd = self.config["window_days"]
        if market == "india":
            mask = (self.in_stock == stock) & (self.in_date == date)
            if not mask.any():
                return None
            i = int(np.where(mask)[0][0])
            return self.in_vectors[i, :wd].tolist()
        if market == "vn":
            mask = (self.vn_stock == stock) & (self.vn_date == date)
            if not mask.any():
                return None
            i = int(np.where(mask)[0][0])
            return self.vn_vectors[i, :wd].tolist()
        return None

    def _lookup_vn_vector(self, stock: str, date: str):
        mask = self.vn_stock == stock
        if not mask.any():
            return None
        dates = self.vn_date[mask]
        idxs = np.where(mask)[0]
        exact = idxs[dates == date]
        if len(exact):
            i = int(exact[0])
        else:
            prior = idxs[dates <= date]
            if not len(prior):
                return None
            i = int(prior[-1])
        return i

    # ----- query -----
    def _india_precedents(self, qvec: np.ndarray, top_k: int) -> list:
        buckets = self.hasher.buckets(qvec)
        cand = set()
        for b in buckets:
            cand.update(self.bucket_index.get(b, ()))
        if not cand:
            return []
        cand = np.fromiter(cand, dtype=int)
        valid = cand[~np.isnan(self.in_down[cand])]
        if not len(valid):
            return []
        cos = self.in_vectors[valid] @ qvec
        order = np.argsort(-cos)[:top_k]
        sel = valid[order]
        return [
            {"stock_symbol": str(self.in_stock[j]), "trade_date": str(self.in_date[j]),
             "cosine": float(self.in_vectors[j] @ qvec),
             "fwd_return": float(self.in_ret[j]), "fwd_down": int(self.in_down[j])}
            for j in sel
        ]

    def _vn_precedents(self, qvec: np.ndarray, top_k: int, before_date: str) -> list:
        mask = (self.vn_date < before_date) & (~np.isnan(self.vn_down))
        idxs = np.where(mask)[0]
        if not len(idxs):
            return []
        cos = self.vn_vectors[idxs] @ qvec
        order = np.argsort(-cos)[:top_k]
        sel = idxs[order]
        return [
            {"stock_symbol": str(self.vn_stock[j]), "trade_date": str(self.vn_date[j]),
             "cosine": float(self.vn_vectors[j] @ qvec),
             "fwd_return": float(self.vn_ret[j]), "fwd_down": int(self.vn_down[j])}
            for j in sel
        ]

    def query(self, stock: str, date: str, top_k: int = None) -> dict:
        """Tra ve ket qua canh bao day du cho 1 (stock, date)."""
        top_k = top_k or self.config["top_k"]
        i = self._lookup_vn_vector(stock, date)
        if i is None:
            return {"error": f"Khong co pattern cho {stock} <= {date}"}
        qvec = self.vn_vectors[i]
        actual_date = str(self.vn_date[i])

        india_prec = self._india_precedents(qvec, top_k)
        vn_prec = self._vn_precedents(qvec, top_k, actual_date)

        ci_z = self.config["ci_z"]
        india_sum = summarize_precedents(india_prec, self.india_base, ci_z)
        vn_sum = summarize_precedents(vn_prec, self.vn_base, ci_z)
        verdict = fuse_signals(india_sum, vn_sum, self.config)

        return {
            "query": {"stock": stock, "requested_date": date,
                      "pattern_date": actual_date, "close": float(self.vn_close[i]),
                      "top_k": top_k},
            "pattern_vector": qvec.tolist(),
            "india_signal": india_sum,
            "vn_signal": vn_sum,
            "verdict": verdict,
            "india_precedents": india_prec,
            "vn_precedents": vn_prec,
        }
