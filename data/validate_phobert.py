import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from cleanlab.filter import find_label_issues
import warnings

warnings.filterwarnings('ignore')

def get_phobert_embeddings(texts):
    """Sử dụng PhoBERT để biến văn bản thành ma trận ngữ nghĩa (Embeddings)"""
    print("⏳ Đang tải mô hình PhoBERT-base (Chỉ tải 1 lần đầu tiên, nặng khoảng 500MB)...")
    # Sử dụng phiên bản v2 mới nhất của VinAI
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    model = AutoModel.from_pretrained("vinai/phobert-base-v2")
    
    # Đóng băng mô hình, chỉ dùng để trích xuất đặc trưng (Evaluation Mode)
    model.eval()
    
    embeddings = []
    print(f"🧠 Đang trích xuất vector ngữ nghĩa cho {len(texts)} câu chat...")
    
    # Tắt tính toán đạo hàm để tiết kiệm RAM
    with torch.no_grad():
        for i, text in enumerate(texts):
            if i > 0 and i % 500 == 0:
                print(f"   ... Đã xử lý {i}/{len(texts)} câu")
                
            # Cắt từ và padding cho đều
            inputs = tokenizer(text, padding=True, truncation=True, max_length=256, return_tensors="pt")
            outputs = model(**inputs)
            
            # Lấy vector ở vị trí [CLS] - Vector này đại diện cho toàn bộ ý nghĩa của câu
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            embeddings.append(cls_embedding)
            
    return np.array(embeddings)

def run_cleanlab_phobert():
    print("[INFO] Đang khởi động Cleanlab với 'Mắt thần' PhoBERT...")
    
    try:
        # Tải tập dữ liệu mà Snorkel đã gán nhãn
        df = pd.read_csv('data/high_confidence_auto_labeled.csv')
    except FileNotFoundError:
        print("❌ Lỗi: Hãy chạy file split_confidence.py trước để có dữ liệu.")
        return

    intent_to_id = {name: idx for idx, name in enumerate(df['snorkel_intent'].unique())}
    id_to_intent = {idx: name for name, idx in intent_to_id.items()}
    labels = np.array([intent_to_id[label] for label in df['snorkel_intent']])

    # 1. BIẾN ĐỔI NGỮ NGHĨA VỚI PHO-BERT
    X = get_phobert_embeddings(df['cleaned_text'].tolist())

    # 2. HUẤN LUYỆN LOGISTIC REGRESSION (Trên nền Vector PhoBERT)
    print("🤖 Đang chạy Cross-validation (Logistic Regression + PhoBERT)...")
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    
    pred_probs = cross_val_predict(
        estimator=clf, X=X, y=labels, cv=5, method="predict_proba"
    )

    # 3. TRIỆU HỒI CLEANLAB
    print("🔍 Cleanlab đang rà soát các nhãn đáng ngờ (Strict Mode)...")
    ordered_label_issues = find_label_issues(
        labels=labels,
        pred_probs=pred_probs,
        filter_by='predicted_neq_given', 
        return_indices_ranked_by='self_confidence'
    )

    # 4. IN BÁO CÁO
    print("-" * 50)
    print(f"🚨 PHÁT HIỆN {len(ordered_label_issues)} CÂU CÓ KHẢ NĂNG BỊ LỖI NHÃN!")
    print("-" * 50)

    if len(ordered_label_issues) > 0:
        print("\n👀 TOP CÁC CÂU LỖI NẶNG NHẤT:")
        for issue_idx in ordered_label_issues[:10]: # In ra 10 câu để dễ kiểm tra
            row = df.iloc[issue_idx]
            predicted_by_ml = id_to_intent[np.argmax(pred_probs[issue_idx])]
            print(f"  • Câu Chat   : '{row['cleaned_text']}'")
            print(f"  • Snorkel Gán: [{row['snorkel_intent']}]")
            print(f"  • PhoBERT Gợi ý: [{predicted_by_ml}]")
            print(f"  • Xác suất ML: {np.max(pred_probs[issue_idx]):.2f}\n")
            
        df_issues = df.iloc[ordered_label_issues]
        df_issues.to_csv('data/cleanlab_phobert_issues.csv', index=False)
        print("💾 Đã xuất toàn bộ danh sách lỗi ra: data/cleanlab_phobert_issues.csv")
    else:
        print("🎉 Tuyệt vời! Bộ dữ liệu của bạn đã đạt cảnh giới hoàn hảo.")

if __name__ == "__main__":
    run_cleanlab_phobert()