# data/split_confidence.py
# Tách độ tự tin theo 1 ngưỡng

import pandas as pd

def split_by_confidence(threshold=0.7):
    print("[INFO] Đang phân luồng dữ liệu theo độ tự tin...")
    
    # 1. Đọc file đã được Snorkel gán nhãn
    df = pd.read_csv('data/labeled_chat_logs.csv')
    
    # 2. Tách làm 2 tập
    df_auto = df[df['snorkel_confidence'] >= threshold]
    df_review = df[df['snorkel_confidence'] < threshold]
    
    # 3. Xuất ra 2 file riêng biệt
    df_auto.to_csv('data/high_confidence_auto_labeled.csv', index=False)
    df_review.to_csv('data/needs_human_review.csv', index=False)
    
    print("-" * 40)
    print("📊 KẾT QUẢ PHÂN LUỒNG (Ngưỡng: >= {}):".format(threshold))
    print(f"✅ Auto-Labeled (Máy tự chốt) : {len(df_auto)} câu")
    print(f"⚠️ Human Review (Cần xem lại) : {len(df_review)} câu")
    print("-" * 40)
    print("💾 Đã lưu thành 2 file riêng biệt trong thư mục data/.")

if __name__ == "__main__":
    split_by_confidence()