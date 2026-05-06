# data/validate_cleanlab.py
# Validate auto label

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from cleanlab.filter import find_label_issues
import warnings

warnings.filterwarnings('ignore')

def run_cleanlab_validation():
    print("[INFO] Đang khởi động Cleanlab AI (Confident Learning)...")
    
    # 1. Đọc dữ liệu đã gán nhãn từ Snorkel (Lấy tập auto-labeled có độ tự tin cao)
    try:
        df = pd.read_csv('data/high_confidence_auto_labeled.csv')
    except FileNotFoundError:
        print("❌ Lỗi: Hãy chạy file split_confidence.py trước nhé!")
        return

    # Chuyển đổi nhãn dạng Text sang ID số nguyên để Machine Learning hiểu được
    intent_to_id = {name: idx for idx, name in enumerate(df['snorkel_intent'].unique())}
    id_to_intent = {idx: name for name, idx in intent_to_id.items()}
    labels = np.array([intent_to_id[label] for label in df['snorkel_intent']])

    # 2. Chuyển văn bản thành Vector (TF-IDF)
    print("🔢 Đang vector hóa văn bản (TF-IDF)...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['cleaned_text'])

    # 3. Huấn luyện mô hình và lấy ma trận xác suất (Cross-validation)
    print("🤖 Đang chạy Cross-validation để ước lượng xác suất...")
    # FIX 1: Thêm class_weight='balanced' để mô hình không bị thiên vị nhãn đông dân
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    
    pred_probs = cross_val_predict(
        estimator=model, X=X, y=labels, cv=5, method="predict_proba"
    )

    # 4. TRIỆU HỒI CLEANLAB (CHẾ ĐỘ NGHIÊM NGẶT)
    print("🔍 Cleanlab đang rà soát các nhãn đáng ngờ (Strict Mode)...")
    ordered_label_issues = find_label_issues(
        labels=labels,
        pred_probs=pred_probs,
        # FIX 2: Ép Cleanlab bắt lỗi nếu dự đoán top 1 của Model khác với Nhãn của Snorkel
        filter_by='predicted_neq_given', 
        return_indices_ranked_by='self_confidence'
    )

    # 5. Báo cáo kết quả
    print("-" * 50)
    print(f"🚨 PHÁT HIỆN {len(ordered_label_issues)} CÂU CÓ KHẢ NĂNG BỊ SNORKEL GÁN NHÃN SAI!")
    print("-" * 50)

    if len(ordered_label_issues) > 0:
        print("\n👀 TOP 5 CÂU CÓ VẤN ĐỀ NHẤT (Cần sửa ngay):")
        for issue_idx in ordered_label_issues[:5]:
            row = df.iloc[issue_idx]
            predicted_by_ml = id_to_intent[np.argmax(pred_probs[issue_idx])]
            print(f"  • Câu Chat   : '{row['cleaned_text']}'")
            print(f"  • Snorkel Gán: [{row['snorkel_intent']}]")
            print(f"  • ML Gợi ý   : [{predicted_by_ml}]")
            print(f"  • Xác suất ML: {np.max(pred_probs[issue_idx]):.2f}\n")
            
        # Xuất danh sách lỗi ra file để review tay
        df_issues = df.iloc[ordered_label_issues]
        df_issues.to_csv('data/cleanlab_detected_issues.csv', index=False)
        print("💾 Đã xuất toàn bộ danh sách lỗi ra file: data/cleanlab_detected_issues.csv")
    else:
        print("🎉 Dữ liệu quá sạch! Cleanlab không tìm thấy lỗi nào đáng kể.")

if __name__ == "__main__":
    run_cleanlab_validation()