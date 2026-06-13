# data/split_confidence.py
# Tách độ tự tin theo ngưỡng động

import pandas as pd
import numpy as np


def print_confidence_histogram(df, bins=10):
    """In histogram confidence để chọn threshold phù hợp"""
    conf = df['snorkel_confidence']
    print("\n📊 PHÂN PHỐI CONFIDENCE:")
    print(f"  Min: {conf.min():.4f}, Max: {conf.max():.4f}")
    print(f"  Mean: {conf.mean():.4f}, Median: {conf.median():.4f}")
    print(f"  Percentiles: 25%={conf.quantile(0.25):.4f}, 75%={conf.quantile(0.75):.4f}")

    # Chia thành các khoảng và in histogram
    thresholds = np.linspace(0, 1, bins + 1)
    print(f"\n  Histogram ({bins} bins):")
    for i in range(bins):
        lo, hi = thresholds[i], thresholds[i + 1]
        count = ((conf >= lo) & (conf < hi)).sum()
        bar = "█" * max(1, count // 5) if count > 0 else ""
        print(f"  [{lo:.1f}-{hi:.1f}): {count:4d} {bar}")

    # Gợi ý threshold dựa trên phân phối
    suggested_75 = conf.quantile(0.75)
    suggested_90 = conf.quantile(0.90)
    print(f"\n  💡 Gợi ý threshold: 75th percentile={suggested_75:.4f}, 90th percentile={suggested_90:.4f}")


def split_by_confidence(threshold=None):
    print("[INFO] Đang phân luồng dữ liệu theo độ tự tin...")

    # 1. Đọc file đã được Snorkel gán nhãn
    df = pd.read_csv('data/labeled_chat_logs.csv')
    df.columns = [c.strip() for c in df.columns]

    print(f"\n📊 Tổng số dữ liệu: {len(df)} rows")

    # 2. In histogram để debug
    print_confidence_histogram(df)

    # 3. Tự động chọn threshold nếu không được chỉ định
    if threshold is None:
        # Lấy 75th percentile, làm tròn xuống 0.05 gần nhất
        threshold = round(df['snorkel_confidence'].quantile(0.75) * 20) / 20
        threshold = max(threshold, 0.3)  # không thấp hơn 0.3
        print(f"\n  🔧 Threshold tự động: {threshold:.2f}")

    # 4. Tách làm 2 tập
    df_auto = df[df['snorkel_confidence'] >= threshold]
    df_review = df[df['snorkel_confidence'] < threshold]

    # 5. Xuất ra 2 file riêng biệt
    df_auto.to_csv('data/high_confidence_auto_labeled.csv', index=False)
    df_review.to_csv('data/needs_human_review.csv', index=False)

    print("\n" + "-" * 40)
    print(f"📊 KẾT QUẢ PHÂN LUỒNG (Ngưỡng: >= {threshold:.2f}):")
    print(f"✅ Auto-Labeled (Máy tự chốt) : {len(df_auto)} câu ({len(df_auto)/len(df)*100:.1f}%)")
    print(f"⚠️ Human Review (Cần xem lại) : {len(df_review)} câu ({len(df_review)/len(df)*100:.1f}%)")
    print("-" * 40)

    # 6. In phân phối intent của từng tập
    if not df_auto.empty:
        print("\n📌 Phân phối intent (Auto-Labeled):")
        print(df_auto['snorkel_intent'].value_counts().to_string())

    return threshold


if __name__ == "__main__":
    split_by_confidence()