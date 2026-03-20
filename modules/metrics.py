#module/metrics.py
from dataclasses import dataclass
from typing import Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

@dataclass
class EvalResult:
    """I/O Contract cho kết quả đánh giá mô hình."""
    accuracy: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float

def calculate_metrics(y_true, y_pred) -> EvalResult:
    """
    Tính toán các metrics đánh giá mô hình phân loại đa lớp.
    Sử dụng average="weighted" để xử lý an toàn nếu dữ liệu mất cân bằng.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1w = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    return EvalResult(
        accuracy=acc,
        precision_weighted=prec,
        recall_weighted=rec,
        f1_weighted=f1w
    )

def print_result(result: EvalResult) -> Dict[str, Any]:
    """Chuyển đổi EvalResult sang dict để dễ in ấn/log cho người làm Pipeline."""
    return {
        "accuracy": result.accuracy,
        "precision_weighted": result.precision_weighted,
        "recall_weighted": result.recall_weighted,
        "f1_weighted": result.f1_weighted
    }