"""
STEP 8 — Backtest walk-forward toàn hệ thống (Ấn Độ + VN) bằng Spark LSH-join.

Cách làm (đúng tinh thần big data, KHÔNG gọi query N lần):
  - Gắn band_buckets cho mỗi pattern VN (cùng hyperplanes với index Ấn Độ).
  - MỘT join phân tán trên `bucket` chung → candidate pairs (LSH at scale):
      * VN × Ấn Độ : tín hiệu "kho kinh nghiệm" cho từng pattern VN.
      * VN × VN     : tín hiệu nội địa, CHỈ dùng tiền lệ có trade_date < ngày
                      xét → chống look-ahead (walk-forward tự nhiên).
  - Với mỗi pattern VN: lấy top_k theo cosine, thống kê → hợp nhất → risk score.
  - So sánh dự đoán vs kết quả thực (fwd_down) → directional accuracy,
    precision@k, win-rate, và so với base rate.

Ấn Độ (2015–2022) nằm hoàn toàn TRƯỚC VN (2025–2026) nên không rò rỉ tương lai.
"""

import os
import json
import time
import logging

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyspark.sql import functions as F
from pyspark.sql.functions import col, lit, concat_ws, explode, row_number
from pyspark.sql.types import DoubleType
from pyspark.sql.window import Window

from .simhash_lsh import load_hyperplanes, add_band_buckets
from .precedent_query import _wilson_interval
from .signal_fusion import (
    fuse_signals, LABEL_SAFE, LABEL_CAUTION, LABEL_STRONG,
)

log = logging.getLogger("precedent")


@F.udf(returnType=DoubleType())
def _dot_udf(a, b):
    if not a or not b or len(a) != len(b):
        return None
    return float(sum(x * y for x, y in zip(a, b)))


def _topk_candidates(pairs, key_col: str, top_k: int):
    """Xếp hạng candidate theo cosine, giữ top_k mỗi key, rồi thống kê."""
    w = Window.partitionBy(key_col).orderBy(col("cosine").desc())
    ranked = pairs.withColumn("_rk", row_number().over(w)).filter(col("_rk") <= top_k)
    return ranked.groupBy(key_col).agg(
        F.count("*").alias("n"),
        F.sum("cand_down").alias("n_down"),
        F.avg("cand_ret").alias("mean_return"),
        F.min("cand_ret").alias("worst_return"),
        F.avg("cosine").alias("avg_cosine"),
    )


def _summary_from_agg(row, base_p: float, ci_z: float) -> dict:
    n = int(row["n"]) if row and row["n"] else 0
    if n == 0:
        return {"n": 0, "p_down": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                "mean_return": 0.0, "worst_return": 0.0, "excess": 0.0,
                "avg_cosine": 0.0, "base_rate": base_p}
    n_down = int(row["n_down"] or 0)
    p_down = n_down / n
    ci_low, ci_high = _wilson_interval(n_down, n, ci_z)
    return {"n": n, "p_down": p_down, "ci_low": ci_low, "ci_high": ci_high,
            "mean_return": float(row["mean_return"] or 0.0),
            "worst_return": float(row["worst_return"] or 0.0),
            "excess": p_down - base_p, "avg_cosine": float(row["avg_cosine"] or 0.0),
            "base_rate": base_p}


def run_backtest(spark, india_dir: str, vn_dir: str, config: dict,
                 reports_dir: str) -> dict:
    os.makedirs(reports_dir, exist_ok=True)
    t0 = time.time()
    top_k = config["top_k"]
    ci_z = config["ci_z"]
    band_bits = config["simhash_band_bits"]
    n_bands = config["simhash_n_bands"]

    india_index = os.path.join(india_dir, "simhash-index")
    vn_vectors = os.path.join(vn_dir, "vectors")

    india = spark.read.parquet(india_index).filter(col("fwd_down").isNotNull())
    india = india.withColumn("in_id", concat_ws("|", col("stock_symbol"), col("trade_date")))

    planes = load_hyperplanes(india_dir)
    vn = spark.read.parquet(vn_vectors).filter(col("fwd_down").isNotNull())
    vn = add_band_buckets(spark, vn, planes, band_bits, n_bands)
    vn = vn.withColumn("vn_id", concat_ws("|", col("stock_symbol"), col("trade_date")))
    vn = vn.cache()

    n_vn = vn.count()
    n_india = india.count()
    log.info("BACKTEST | VN patterns=%s  India library=%s  top_k=%d",
             f"{n_vn:,}", f"{n_india:,}", top_k)

    # ---- Bung buckets thanh dong rieng le ----
    vn_b = vn.select("vn_id", explode("band_buckets").alias("bucket"))
    in_b = india.select("in_id", explode("band_buckets").alias("bucket"))

    # ============ VN x India (tin hieu An Do) ============
    pairs_in = (
        vn_b.join(in_b, "bucket")
        .select("vn_id", "in_id").distinct()
        .join(vn.select("vn_id", col("vector").alias("vn_vec")), "vn_id")
        .join(india.select("in_id", col("vector").alias("in_vec"),
                           col("fwd_down").alias("cand_down"),
                           col("fwd_return").alias("cand_ret")), "in_id")
        .withColumn("cosine", _dot_udf(col("vn_vec"), col("in_vec")))
    )
    n_cand_in = pairs_in.count()
    india_agg = {r["vn_id"]: r for r in _topk_candidates(pairs_in, "vn_id", top_k).collect()}

    # ============ VN x VN (tin hieu noi dia, walk-forward) ============
    a = vn.select(col("vn_id").alias("a_id"), col("trade_date").alias("a_date"),
                  col("vector").alias("a_vec"))
    a_b = vn.select(col("vn_id").alias("a_id"), explode("band_buckets").alias("bucket"))
    b_b = vn.select(col("vn_id").alias("b_id"), explode("band_buckets").alias("bucket"))
    pairs_vn = (
        a_b.join(b_b, "bucket")
        .select("a_id", "b_id").distinct()
        .filter(col("a_id") != col("b_id"))
        .join(a, "a_id")
        .join(vn.select(col("vn_id").alias("b_id"), col("trade_date").alias("b_date"),
                        col("vector").alias("b_vec"),
                        col("fwd_down").alias("cand_down"),
                        col("fwd_return").alias("cand_ret")), "b_id")
        .filter(col("b_date") < col("a_date"))   # chỉ dùng quá khứ
        .withColumn("cosine", _dot_udf(col("a_vec"), col("b_vec")))
    )
    vn_agg = {r["a_id"]: r for r in _topk_candidates(pairs_vn, "a_id", top_k).collect()}

    # ---- base rate ----
    india_base = india.agg(F.avg("fwd_down")).collect()[0][0] or 0.0
    vn_base = vn.agg(F.avg("fwd_down")).collect()[0][0] or 0.0

    # ---- ground truth cho mỗi pattern VN ----
    truth = vn.select("vn_id", "stock_symbol", "trade_date",
                      "fwd_down", "fwd_return").collect()

    # ---- fuse + đánh giá (driver-side, n_vn dòng) ----
    records = []
    for r in truth:
        vid = r["vn_id"]
        isum = _summary_from_agg(india_agg.get(vid), india_base, ci_z)
        vsum = _summary_from_agg(vn_agg.get(vid), vn_base, ci_z)
        verdict = fuse_signals(isum, vsum, config)
        pred_down = 1 if verdict["risk_score"] >= 50 else 0
        records.append({
            "stock": r["stock_symbol"], "date": str(r["trade_date"]),
            "risk_score": verdict["risk_score"], "label": verdict["label"],
            "pred_down": pred_down, "actual_down": int(r["fwd_down"]),
            "fwd_return": float(r["fwd_return"]),
            "india_n": isum["n"], "vn_n": vsum["n"],
        })

    metrics = _evaluate(records, india_base, vn_base, n_vn, n_india, n_cand_in, top_k)
    _plot_backtest(records, metrics, reports_dir)

    out = os.path.join(reports_dir, "backtest.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    log.info("BACKTEST | done in %.1fs -> %s", time.time() - t0, out)
    _log_metrics(metrics)
    vn.unpersist()
    return metrics


def _evaluate(records, india_base, vn_base, n_vn, n_india, n_cand_in, top_k) -> dict:
    evaluable = [r for r in records if r["india_n"] > 0 or r["vn_n"] > 0]
    n = len(evaluable)
    if n == 0:
        return {"error": "khong co pattern nao co tien le de danh gia"}

    actual_down = np.array([r["actual_down"] for r in evaluable])
    pred_down = np.array([r["pred_down"] for r in evaluable])
    rets = np.array([r["fwd_return"] for r in evaluable])

    acc = float(np.mean(pred_down == actual_down))
    tp = int(np.sum((pred_down == 1) & (actual_down == 1)))
    fp = int(np.sum((pred_down == 1) & (actual_down == 0)))
    fn = int(np.sum((pred_down == 0) & (actual_down == 1)))
    tn = int(np.sum((pred_down == 0) & (actual_down == 0)))
    precision_down = tp / (tp + fp) if (tp + fp) else 0.0
    recall_down = tp / (tp + fn) if (tp + fn) else 0.0

    # Win-rate của cảnh báo "AN TOÀN" (kỳ vọng tăng): % thực sự tăng
    safe = [r for r in evaluable if r["label"] == LABEL_SAFE]
    safe_winrate = float(np.mean([1 - r["actual_down"] for r in safe])) if safe else 0.0

    # Breakdown theo nhãn
    by_label = {}
    for lab in (LABEL_SAFE, LABEL_CAUTION, LABEL_STRONG):
        grp = [r for r in evaluable if r["label"] == lab]
        if grp:
            by_label[lab] = {
                "count": len(grp),
                "actual_p_down": float(np.mean([r["actual_down"] for r in grp])),
                "mean_fwd_return": float(np.mean([r["fwd_return"] for r in grp])),
            }

    all_pairs = n_vn * n_india
    return {
        "n_evaluated": n,
        "base_rate_down_vn": float(np.mean(actual_down)),
        "directional_accuracy": acc,
        "precision_down": precision_down,
        "recall_down": recall_down,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "safe_call_winrate": safe_winrate,
        "mean_return_overall": float(np.mean(rets)),
        "by_label": by_label,
        "lsh_efficiency": {
            "vn_patterns": n_vn, "india_library": n_india,
            "all_pairs_bruteforce": all_pairs,
            "lsh_candidate_pairs": n_cand_in,
            "reduction_x": round(all_pairs / max(1, n_cand_in), 1),
        },
        "india_base_rate": india_base, "vn_base_rate": vn_base, "top_k": top_k,
    }


def _plot_backtest(records, metrics, reports_dir):
    """Vẽ biểu đồ tổng hợp. Tiêu đề ASCII để tránh font matplotlib mặc định."""
    ev = [r for r in records if r["india_n"] > 0 or r["vn_n"] > 0]
    if not ev:
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) risk score vs ket qua thuc te
    rs_down = [r["risk_score"] for r in ev if r["actual_down"] == 1]
    rs_up = [r["risk_score"] for r in ev if r["actual_down"] == 0]
    axes[0].hist(rs_up, bins=30, alpha=0.6, label="thuc te TANG", color="#1a9850")
    axes[0].hist(rs_down, bins=30, alpha=0.6, label="thuc te GIAM", color="#d73027")
    axes[0].axvline(50, color="k", ls="--", lw=1)
    axes[0].set_title("Risk score theo ket qua thuc")
    axes[0].set_xlabel("risk score"); axes[0].legend(); axes[0].grid(alpha=0.3)

    # 2) confusion matrix
    c = metrics["confusion"]
    cm = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    im = axes[1].imshow(cm, cmap="Blues")
    axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["pred UP", "pred DOWN"])
    axes[1].set_yticks([0, 1]); axes[1].set_yticklabels(["actual UP", "actual DOWN"])
    for (i, j), val in np.ndenumerate(cm):
        axes[1].text(j, i, int(val), ha="center", va="center",
                     color="white" if val > cm.max() / 2 else "black", fontsize=13)
    axes[1].set_title(f"Confusion (acc={metrics['directional_accuracy']:.2%})")
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    # 3) actual p_down theo nhãn (chỉ dùng key gốc, không quan trọng)
    labs = list(metrics["by_label"].keys())
    pdown = [metrics["by_label"][l]["actual_p_down"] for l in labs]
    axes[2].bar(labs, pdown, color=["#1a9850", "#fee08b", "#d73027"][:len(labs)])
    axes[2].axhline(metrics["base_rate_down_vn"], color="k", ls="--",
                    label=f"base rate {metrics['base_rate_down_vn']:.2f}")
    axes[2].set_title("% giam thuc te theo nhan canh bao")
    axes[2].set_ylabel("actual p_down"); axes[2].legend(); axes[2].grid(axis="y", alpha=0.3)
    plt.setp(axes[2].get_xticklabels(), rotation=15, ha="right")

    fig.suptitle("STEP 8 -- Backtest walk-forward (India + VN)", fontsize=14)
    fig.tight_layout()
    out = os.path.join(reports_dir, "backtest.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log.info("BACKTEST | plot -> %s", out)


def _log_metrics(m):
    if "error" in m:
        log.warning("BACKTEST | %s", m["error"]); return
    log.info("BACKTEST | n=%d | acc=%.2f%% | precision_down=%.2f%% | base=%.2f%%",
             m["n_evaluated"], m["directional_accuracy"] * 100,
             m["precision_down"] * 100, m["base_rate_down_vn"] * 100)
    log.info("BACKTEST | safe-call winrate=%.2f%% | LSH reduction=%sx",
             m["safe_call_winrate"] * 100, f"{m['lsh_efficiency']['reduction_x']:,}")
    for lab, v in m["by_label"].items():
        log.info("BACKTEST |   %-15s n=%d actual_down=%.2f%% mean_ret=%+.2f%%",
                 lab, v["count"], v["actual_p_down"] * 100, v["mean_fwd_return"] * 100)
