"""
STEP 4 — Sanity check: hai thị trường có đủ GIỐNG để so sánh không?

Trước khi tin vào tiền lệ Ấn Độ, ta kiểm tra phân phối của đại lượng đầu ra
(fwd_return 3 ngày) giữa Ấn Độ và VN có tương đồng không. Đây là cấp PATTERN
— chính là đơn vị mà cosine đang so sánh, nên phép đo này sát và minh bạch.

Đọc thẳng vectors_dir đã build:
  - n_pattern, base_rate_down (% pattern giảm 3 ngày sau)
  - fwd_return: mean, std, % pattern thực tế tăng
  - sample fwd_return để vẽ phân phối chồng

Không dùng lại daily-agg cấp minute (vì VN có nhiều phút bid-only khác Ấn Độ
khiến phân phối minute lệch, KHÔNG phản ánh đúng độ giống cấp pattern).
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
from pyspark.sql.functions import col, when

log = logging.getLogger("precedent")

_PLOT_SAMPLE_LIMIT = 50000

# Mức kết luận sanity. Là enum, dùng chung dashboard.
LEVEL_GOOD = "GIỐNG (chấp nhận được)"
LEVEL_FAIR = "GIỐNG TƯƠNG ĐỐI (lưu ý vài chỉ số lệch)"
LEVEL_POOR = "LỆCH ĐÁNG KỂ (cẩn thận khi diễn giải)"


def _stats_from_vectors(spark, vectors_dir: str, market: str):
    df = spark.read.parquet(vectors_dir).filter(col("fwd_down").isNotNull())
    df = df.withColumn("is_up", when(col("fwd_return") > 0, 1.0).otherwise(0.0))
    df = df.withColumn("abs_ret", F.abs(col("fwd_return")))
    row = df.agg(
        F.count("*").alias("n_pattern"),
        F.avg("fwd_down").alias("base_rate_down"),
        F.avg("is_up").alias("p_up"),
        F.mean("fwd_return").alias("mean_fwd_return"),
        F.stddev("fwd_return").alias("std_fwd_return"),
        F.mean("abs_ret").alias("volatility_fwd"),
        F.expr("percentile(fwd_return, array(0.05, 0.5, 0.95))").alias("pcts"),
    ).collect()[0].asDict()
    pcts = row["pcts"] or [0.0, 0.0, 0.0]
    stats = {
        "n_pattern": int(row["n_pattern"] or 0),
        "base_rate_down": float(row["base_rate_down"] or 0.0),
        "p_up": float(row["p_up"] or 0.0),
        "mean_fwd_return": float(row["mean_fwd_return"] or 0.0),
        "std_fwd_return": float(row["std_fwd_return"] or 0.0),
        "volatility_fwd": float(row["volatility_fwd"] or 0.0),
        "fwd_return_p05": float(pcts[0]),
        "fwd_return_p50": float(pcts[1]),
        "fwd_return_p95": float(pcts[2]),
    }
    sample = (
        df.select("fwd_return").limit(_PLOT_SAMPLE_LIMIT).toPandas()
    )
    log.info(
        "SANITY [%s] | n=%s base_down=%.3f p_up=%.3f mean=%+.4f std=%.4f "
        "p05=%+.3f p50=%+.3f p95=%+.3f",
        market, f"{stats['n_pattern']:,}", stats["base_rate_down"],
        stats["p_up"], stats["mean_fwd_return"], stats["std_fwd_return"],
        stats["fwd_return_p05"], stats["fwd_return_p50"], stats["fwd_return_p95"],
    )
    return stats, sample


def _verdict(india: dict, vn: dict) -> dict:
    checks = []

    def _check(name, a, b, tol, kind="abs"):
        if kind == "rel":
            diff = abs(a - b) / (abs(a) + 1e-9)
        else:
            diff = abs(a - b)
        ok = diff <= tol
        checks.append({"metric": name, "india": a, "vn": b, "diff": diff,
                       "tol": tol, "ok": bool(ok)})

    _check("base_rate_down (3 ngày)",       india["base_rate_down"],   vn["base_rate_down"],   0.10)
    _check("p_up (% pattern tăng)",          india["p_up"],             vn["p_up"],             0.10)
    _check("mean_fwd_return",                india["mean_fwd_return"],  vn["mean_fwd_return"],  0.01)
    _check("std_fwd_return",                 india["std_fwd_return"],   vn["std_fwd_return"],   0.5, kind="rel")
    _check("volatility_fwd (|ret| TB)",      india["volatility_fwd"],   vn["volatility_fwd"],   0.5, kind="rel")
    _check("fwd_return_p95",                 india["fwd_return_p95"],   vn["fwd_return_p95"],   0.5, kind="rel")
    _check("fwd_return_p05",                 india["fwd_return_p05"],   vn["fwd_return_p05"],   0.5, kind="rel")

    n_ok = sum(c["ok"] for c in checks)
    if n_ok >= len(checks) - 1:
        level = LEVEL_GOOD
    elif n_ok >= len(checks) - 2:
        level = LEVEL_FAIR
    else:
        level = LEVEL_POOR
    return {"level": level, "n_ok": n_ok, "n_total": len(checks), "checks": checks}


def _plot(india_s, vn_s, india_stat, vn_stat, reports_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) Phân phối fwd_return
    lo, hi = -0.08, 0.08
    axes[0].hist(np.clip(india_s["fwd_return"], lo, hi), bins=80, alpha=0.55,
                 density=True,
                 label=f"An Do n={india_stat['n_pattern']:,}",
                 color="#2166ac")
    axes[0].hist(np.clip(vn_s["fwd_return"], lo, hi), bins=60, alpha=0.55,
                 density=True,
                 label=f"VN n={vn_stat['n_pattern']:,}",
                 color="#d6604d")
    axes[0].axvline(0, color="k", lw=0.7)
    axes[0].set_title("Phan phoi fwd_return 3-day (clip ±8%)")
    axes[0].set_xlabel("fwd_return"); axes[0].legend(); axes[0].grid(alpha=0.3)

    # 2) Bar so sánh chỉ số chính
    metrics = ["base_down", "p_up", "vol×100", "std×100"]
    iv = [india_stat["base_rate_down"], india_stat["p_up"],
          india_stat["volatility_fwd"] * 100, india_stat["std_fwd_return"] * 100]
    vv = [vn_stat["base_rate_down"], vn_stat["p_up"],
          vn_stat["volatility_fwd"] * 100, vn_stat["std_fwd_return"] * 100]
    x = np.arange(len(metrics)); w = 0.38
    axes[1].bar(x - w / 2, iv, w, label="An Do", color="#2166ac")
    axes[1].bar(x + w / 2, vv, w, label="VN", color="#d6604d")
    axes[1].set_xticks(x); axes[1].set_xticklabels(metrics, rotation=15)
    axes[1].set_title("So sanh chi so chinh (cap pattern)")
    axes[1].legend(); axes[1].grid(axis="y", alpha=0.3)

    # 3) Percentile so sánh
    pct_x = np.array([5, 50, 95])
    iv_p = [india_stat["fwd_return_p05"], india_stat["fwd_return_p50"],
            india_stat["fwd_return_p95"]]
    vv_p = [vn_stat["fwd_return_p05"], vn_stat["fwd_return_p50"],
            vn_stat["fwd_return_p95"]]
    axes[2].plot(pct_x, iv_p, "o-", color="#2166ac", lw=2, ms=8, label="An Do")
    axes[2].plot(pct_x, vv_p, "s-", color="#d6604d", lw=2, ms=8, label="VN")
    axes[2].axhline(0, color="k", lw=0.7)
    axes[2].set_xticks(pct_x); axes[2].set_xlabel("Percentile")
    axes[2].set_ylabel("fwd_return")
    axes[2].set_title("Percentile fwd_return (5/50/95)")
    axes[2].legend(); axes[2].grid(alpha=0.3)

    fig.suptitle("STEP 4 -- Sanity check cap pattern: An Do vs VN", fontsize=14)
    fig.tight_layout()
    out = os.path.join(reports_dir, "sanity_check.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log.info("SANITY | plot -> %s", out)
    return out


def run_sanity_check(spark, india_final: str, vn_final: str,
                     india_vectors: str, vn_vectors: str,
                     config: dict, reports_dir: str) -> dict:
    """STEP 4 — đọc thẳng vectors_dir, so sánh phân phối cấp pattern."""
    os.makedirs(reports_dir, exist_ok=True)
    t0 = time.time()

    india_stat, india_s = _stats_from_vectors(spark, india_vectors, "india")
    vn_stat, vn_s = _stats_from_vectors(spark, vn_vectors, "vn")

    verdict = _verdict(india_stat, vn_stat)
    plot_path = _plot(india_s, vn_s, india_stat, vn_stat, reports_dir)

    report = {"india": india_stat, "vn": vn_stat, "verdict": verdict, "plot": plot_path}
    out = os.path.join(reports_dir, "sanity_check.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    log.info("SANITY | verdict: %s (%d/%d checks) in %.1fs",
             verdict["level"], verdict["n_ok"], verdict["n_total"], time.time() - t0)
    for c in verdict["checks"]:
        log.info("SANITY |   %-25s India=%+.4f VN=%+.4f diff=%.4f tol=%.3f %s",
                 c["metric"], c["india"], c["vn"], c["diff"], c["tol"],
                 "OK" if c["ok"] else "X")
    return report
