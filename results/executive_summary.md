
# Executive Summary – Text Classification on AG News

## Best Performing Model (Final Results)

- **Model:** SVM (Linear SVC)
- **Feature extraction:** TF-IDF (`tfidf_uni_bi_5k`: unigram+bigram, max_features=5000, min_df=3)
- **Mode:** full (120,000 train / 7,600 test)
- **Primary metric (f1_weighted):** 0.9044
- **Accuracy:** 0.9046
- **Precision (weighted):** 0.9044
- **Recall (weighted):** 0.9046

## Full Comparison Table

| Branch | Scale | Model | F1-weighted |
|---|---|---|---|
| **TF-IDF** | **full (120k)** | **SVM** | **0.9044** |
| TF-IDF | full (120k) | Logistic Regression | 0.9036 |
| TF-IDF | full (120k) | Naive Bayes | 0.8867 |
| SBERT Embedding | 20k_2k | SVM | 0.8954 |
| SBERT Embedding | 20k_2k | Logistic Regression | 0.8929 |
| SBERT Embedding | 5k_2k | Logistic Regression | 0.8736 |

## Key Findings

- TF-IDF (full data) outperforms SBERT Embedding by ~+1% F1 on this task.
- SVM is the best classifier in both branches.
- Increasing embedding training data from 5k → 20k improves SBERT F1 by +2.2%.
- Primary metric: **F1-weighted** (suitable for balanced 4-class AG News dataset).
- Train/test split: official AG News split (no validation set used).

## Artifact Locations

- TF-IDF model comparison: `results/tables/tfidf_model_comparison.csv`
- Embedding benchmark: `results/tables/bert_benchmark_results.csv`
- Feature family comparison: `results/tables/feature_family_comparison.csv`
- Confusion matrices: `results/figures/cm_{svm,logistic_regression,naive_bayes}.png`
- Agency workflow summary: `results/reports/agency_summary.md`
