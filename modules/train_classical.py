# modules/train_classical.py
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from modules.metrics import calculate_metrics, EvalResult, print_result
from typing import Dict, Any
def get_model(model_type: str, C: float = 1.0, max_iter: int = 200) -> BaseEstimator:
    """
    Factory pattern để người làm Pipeline/Config dễ dàng đổi mô hình.
    """
    if model_type == "logistic_regression":
        return LogisticRegression(C=C, max_iter=max_iter)
    elif model_type == "naive_bayes":
        return MultinomialNB()
    elif model_type == "svm":
        return LinearSVC(C=C, max_iter=max_iter, dual=False)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
def train_eval(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> EvalResult:
    # 1. Huấn luyện dựa trên model được truyền vào
    model.fit(X_train, y_train)

    # 2. Dự đoán
    y_pred = model.predict(X_test)

    # 3. Dùng hàm calculate_metrics của bạn để tính đủ 4 chỉ số
    return calculate_metrics(y_test, y_pred)

def train_eval_logreg(X_train, y_train, X_test, y_test, *, max_iter: int = 200) -> EvalResult:
    # dùng đúng factory hiện có
    model = get_model("logistic_regression", C=1.0, max_iter=max_iter)
    return train_eval(model, X_train, y_train, X_test, y_test)

def pretty_print_result(result: EvalResult) -> Dict[str, Any]:
    # dùng hàm print_result bạn đã có trong modules.metrics
    return print_result(result)
# --- PHẦN TEST VỚI DUMMY FEATURES ---
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    print("--- Bắt đầu test khung Train/Eval với Dummy Features ---")
    
    # 1. Tạo ma trận dummy (mô phỏng TF-IDF)
    X_dummy, y_dummy = make_classification(
        n_samples=500, n_features=100, n_classes=4, n_informative=10, random_state=42
    )

    X_dummy = np.abs(X_dummy)

    X_train, X_test, y_train, y_test = train_test_split(
        X_dummy, y_dummy, test_size=0.2, random_state=42
    )

    # 2. Chạy thử với Logistic Regression
    print("\n[Test 1] Khởi tạo Logistic Regression...")
    model_lr = get_model("logistic_regression")
    result_lr = train_eval(model_lr, X_train, y_train, X_test, y_test)
    print("Kết quả:", print_result(result_lr))

    # 3. Chạy thử với Naive Bayes (Chứng minh khung nhận mọi model)
    print("\n[Test 2] Khởi tạo Naive Bayes...")
    model_nb = get_model("naive_bayes")
    result_nb = train_eval(model_nb, X_train, y_train, X_test, y_test)
    print("Kết quả:", print_result(result_nb))
