import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class TFIDFExtractor:
    def __init__(self, max_features=None, ngram_range=(1, 1)):
        # Khởi tạo TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        self.feature_names = None

    def fit_transform(self, corpus):
        """Huấn luyện vectorizer và biến đổi corpus thành ma trận TF-IDF."""
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.feature_names = self.vectorizer.get_feature_names_out()
        # Chuyển đổi ma trận thưa (sparse matrix) sang mảng numpy đặc (dense array) để dễ lưu trữ
        return tfidf_matrix.toarray()

    def transform(self, corpus):
        """Biến đổi tập dữ liệu mới (test set) dựa trên vectorizer đã huấn luyện."""
        return self.vectorizer.transform(corpus).toarray()

    def save_features(self, features_array, filename="tfidf_features.npy"):
        """Lưu mảng đặc trưng thành file .npy theo yêu cầu của bài tập."""
        np.save(filename, features_array)
        print(f"Đã lưu ma trận đặc trưng vào {filename}")
