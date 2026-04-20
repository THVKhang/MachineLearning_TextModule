import os
import pandas as pd

def run_critic(
        result_dir: str = "results",
        csv_filename: str = "tfidf_model_comparison.csv",
        expected_models: list = ["logistic_regression", "svm", "naive_bayes"],
        f1_threshold: float = 0.75 # Ngưỡng chất lượng
):
    
    print("BẮT ĐẦU CHẠY CRITIC ĐÁNH GIÁ")

    csv_path = os.path.join(result_dir, csv_filename)

    print("\n[1/4] Kiểm tra Artifacts (CSV, Confusion Matrices)...")
    missing_artifacts = []

    if not os.path.exists(csv_path):
        missing_artifacts.append(csv_filename)

    for model in expected_models:
        cm_file = f"cm_{model}.png"
        if not os.path.exists(os.path.join(result_dir, cm_file)):
            missing_artifacts.append(cm_file)

    if missing_artifacts:
        print(f"CRITIC FAILED: Thiếu các file Artifacts: {missing_artifacts}")
        return
    print("✅ PASS: Đã tìm thấy đủ file báo cáo CSV và 3 file ảnh Confusion Matrix.")

    # Đọc CSV và chuẩn hóa tên cột về chữ thường
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.lower().str.strip()
    except Exception as e:
        print(f"CRITIC FAILED: Lỗi khi đọc file CSV: {e}")
        return
    
    # Tìm đúng cột chứa tên mô hình và F1-weighted
    model_col = next((col for col in df.columns if 'model' in col), None)
    f1_col = next((col for col in df.columns if 'f1' in col and 'weight' in col), None)

    if not model_col or not f1_col:
        print(f"CRITIC FAILED: File CSV không có cột 'model' hoặc 'f1_weighted'. Cột hiện tại: {df.columns.tolist()}")
        return
    
    # Kiểm tra Scale Benchmark
    scale_cols = ["train_time", "predict_time", "memory", "time", "latency"]
    has_scale_benchmark = any(any(s in col for s in scale_cols) for col in df.columns)
    if not has_scale_benchmark:
        print("CẢNH BÁO: Bảng kết quả THIẾU các chỉ số Scale Benchmark (thời gian train/predict).")
    else:
        print("PASS: Đã tìm thấy dữ liệu Scale Benchmark.")

    print(f"\n[2/4] Xếp hạng theo {f1_col.upper()}...")

    # Chỉ xét 3 model quy định
    df[model_col] = df[model_col].str.lower()
    df_filtered = df[df[model_col].isin(expected_models)].copy()

    df_sorted = df_filtered.sort_values(by=f1_col, ascending=False).reset_index(drop=True)

    for index, row in df_sorted.iterrows():
        print(f"  {index + 1}. {row[model_col].upper()}: {row[f1_col]:.4f}")

    print(f"\n[3/4] Kiểm tra Quality Threshold ({f1_col} >= {f1_threshold})...")
    passed_models = df_sorted[df_sorted[f1_col] >= f1_threshold]

    print("\n[4/4] KẾT LUẬN CỦA CRITIC:")
    if passed_models.empty:
        best_model = df_sorted.iloc[0]
        print(f"FAILED: Không model nào đạt target (Ngưỡng: {f1_threshold}).")
        print(f"-> Mặc dù '{best_model[model_col].upper()}' đứng đầu, nhưng cần EDA hoặc Tuning thêm.")
    else:
        best_model = passed_models.iloc[0]
        print(f"PASSED: Có {len(passed_models)} model đạt ngưỡng chất lượng.")
        print(f"BEST MODEL: '{best_model[model_col].upper()}' ({f1_col} = {best_model[f1_col]:.4f}).")
        print(f"\nĐỀ XUẤT CHO FINAL COMPARISON:")
        print(f"1. GIỮ '{best_model[model_col].upper()}' làm đại diện ML truyền thống để so sánh chéo.")
        if len(passed_models) > 1:
            print(f"2. Back-up: '{passed_models.iloc[1][model_col].upper()}'.")

if __name__ == "__main__":
    run_critic()