"""
STEP 7 — Hợp nhất tín hiệu Ấn Độ + Việt Nam thành risk score 0–100.

Nguyên tắc (theo tài liệu phương án A):
  - Mỗi thị trường cho một "độ nghiêng giảm" so với base rate (excess).
  - Trọng số dựa trên SỐ MẪU + ĐỘ TIN CẬY (mẫu nhiều, khoảng tin cậy chặt thì
    tin hơn), không cào bằng.
  - Khi hai thị trường MÂU THUẪN → ưu tiên VN (nhân trọng số cao hơn).
  - Quy ra risk score 0–100 và gắn nhãn AN TOÀN / THẬN TRỌNG / CẢNH BÁO MẠNH.

Quy ước risk score (neo theo base rate, minh bạch):
  - 50  = trung tính: tiền lệ giống nghiêng giảm xấp xỉ tỷ lệ nền.
  - >50 = nghiêng giảm NHIỀU HƠN nền  → rủi ro cao hơn bình thường.
  - <50 = nghiêng giảm ÍT HƠN nền      → an toàn hơn bình thường.
"""

import logging

log = logging.getLogger("precedent")

# Các nhãn enum dưới đây được tham chiếu chéo ở dashboard (LABEL_COLORS) và
# backtest.by_label. Nếu đổi giá trị, phải đổi đồng bộ ở mọi nơi.
LABEL_SAFE = "AN TOÀN"
LABEL_CAUTION = "THẬN TRỌNG"
LABEL_STRONG = "CẢNH BÁO MẠNH"

AGREE_ALIGNED = "ĐỒNG THUẬN"
AGREE_CONFLICT = "MÂU THUẪN"
AGREE_NEUTRAL = "TRUNG TÍNH"

CONF_HIGH = "CAO"
CONF_MEDIUM = "TRUNG BÌNH"
CONF_LOW = "THẤP"


def _market_risk(summary: dict) -> float:
    """Đổi excess (p_down − base_rate) thành điểm rủi ro 0–100, neo 50 = trung tính."""
    risk = 50.0 + 100.0 * summary.get("excess", 0.0)
    return max(0.0, min(100.0, risk))


def _market_confidence(summary: dict) -> float:
    """Độ tin cậy 0–1: nhiều mẫu + khoảng tin cậy chặt → cao."""
    n = summary.get("n", 0)
    if n == 0:
        return 0.0
    width = max(0.0, summary.get("ci_high", 0.0) - summary.get("ci_low", 0.0))
    sample_w = n / (n + 10.0)          # bão hoà khi n lớn
    tightness = max(0.0, 1.0 - width)  # CI càng hẹp càng tin
    return sample_w * tightness


def _confidence_label(conf: float) -> str:
    if conf >= 0.6:
        return CONF_HIGH
    if conf >= 0.3:
        return CONF_MEDIUM
    return CONF_LOW


def fuse_signals(india: dict, vn: dict, config: dict) -> dict:
    """Hợp nhất 2 tín hiệu → kết luận cảnh báo.

    Args:
        india: kết quả summarize_precedents() cho Ấn Độ.
        vn:    kết quả summarize_precedents() cho VN.
        config: PRECEDENT_CONFIG.

    Returns:
        dict gồm: risk_score, label, confidence, confidence_label, weights,
        agreement, reason (list[str]).
    """
    vn_priority = config.get("vn_priority_weight", 1.5)
    caution_th = config.get("risk_caution_threshold", 40)
    strong_th = config.get("risk_strong_threshold", 65)

    risk_in, risk_vn = _market_risk(india), _market_risk(vn)
    conf_in, conf_vn = _market_confidence(india), _market_confidence(vn)

    w_in = conf_in
    w_vn = conf_vn * vn_priority  # ưu tiên VN khi mâu thuẫn

    if (w_in + w_vn) <= 1e-9:
        risk_score = 50.0
        combined_conf = 0.0
    else:
        risk_score = (w_in * risk_in + w_vn * risk_vn) / (w_in + w_vn)
        combined_conf = (conf_in * india["n"] + conf_vn * vn["n"]) / max(
            1, india["n"] + vn["n"]
        )

    # Đồng thuận / mâu thuẫn về hướng (so với base rate)
    dir_in = india.get("excess", 0.0)
    dir_vn = vn.get("excess", 0.0)
    same_dir = (dir_in > 0 and dir_vn > 0) or (dir_in < 0 and dir_vn < 0)
    if abs(dir_in) < 1e-9 or abs(dir_vn) < 1e-9:
        agreement = AGREE_NEUTRAL
    elif same_dir:
        agreement = AGREE_ALIGNED
        # Cả hai cùng chiều → tăng nhẹ độ tin cậy
        combined_conf = min(1.0, combined_conf * 1.15)
    else:
        agreement = AGREE_CONFLICT
        combined_conf *= 0.8  # mâu thuẫn → bớt tin

    if risk_score >= strong_th:
        label = LABEL_STRONG
    elif risk_score >= caution_th:
        label = LABEL_CAUTION
    else:
        label = LABEL_SAFE

    reason = _build_reason(india, vn, agreement, risk_score, label)

    return {
        "risk_score": round(risk_score, 1),
        "label": label,
        "confidence": round(combined_conf, 3),
        "confidence_label": _confidence_label(combined_conf),
        "agreement": agreement,
        "weights": {"india": round(w_in, 3), "vn": round(w_vn, 3)},
        "market_risk": {"india": round(risk_in, 1), "vn": round(risk_vn, 1)},
        "reason": reason,
    }


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _build_reason(india: dict, vn: dict, agreement: str,
                  risk_score: float, label: str) -> list:
    lines = []
    if india["n"] > 0:
        lines.append(
            f"Ấn Độ: {india['n']} tiền lệ giống → {_pct(india['p_down'])} giảm "
            f"(nền {_pct(india['base_rate'])}); lợi suất TB {india['mean_return'] * 100:+.1f}%, "
            f"sâu nhất {india['worst_return'] * 100:+.1f}%; "
            f"CI95 [{_pct(india['ci_low'])}, {_pct(india['ci_high'])}]."
        )
    else:
        lines.append("Ấn Độ: không tìm được tiền lệ giống (LSH không có candidate).")

    if vn["n"] > 0:
        lines.append(
            f"VN: {vn['n']} tiền lệ giống → {_pct(vn['p_down'])} giảm "
            f"(nền {_pct(vn['base_rate'])}); lợi suất TB {vn['mean_return'] * 100:+.1f}%, "
            f"sâu nhất {vn['worst_return'] * 100:+.1f}%; "
            f"CI95 [{_pct(vn['ci_low'])}, {_pct(vn['ci_high'])}]."
        )
    else:
        lines.append("VN: không tìm được tiền lệ giống trong lịch sử.")

    if agreement == AGREE_ALIGNED:
        lines.append("Hai thị trường ĐỒNG THUẬN về hướng → cảnh báo đáng tin hơn.")
    elif agreement == AGREE_CONFLICT:
        lines.append("Hai thị trường MÂU THUẪN → ưu tiên dữ liệu VN, giảm độ tin cậy.")
    else:
        lines.append("Tín hiệu gần mức nền → chưa có bằng chứng lệch rõ ràng.")

    lines.append(f"=> Risk score {risk_score:.0f}/100 → {label}.")
    return lines
