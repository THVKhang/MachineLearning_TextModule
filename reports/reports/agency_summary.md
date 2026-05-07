# Agency Workflow Summary

## Plan
- Objective: fast
- Mode: demo
- Feature Family: tfidf
- Recommended Model: naive_bayes
- Primary Metric: f1_weighted
- Benchmark Scales: 5k_2k

## Runner Outputs
- TF-IDF log: C:\Users\Admin\Documents\GitHub\MachineLearning_TextModule\results\logs\tfidf_runner_last.json
- TF-IDF best model: naive_bayes (F1=0.8321)
- Embedding log: not-run
- Embedding best model: n/a @ n/a (F1=n/a)

## Critic Verdict
- Status: pass
- Best Family: embedding
- Best Model: svm
- Best f1_weighted: 0.8954
- Threshold: 0.8000

## Recommendation
- Use current best model for final report.

## EDA Highlights
- Total texts analyzed: 7600
- Short texts (< 20 chars): 0
- Long texts (> 1000 chars): 0
- Duplicate instances: 0
- URL-heavy texts: 1
- Estimated noise %: 0.0%
- Error analysis report: C:\Users\Admin\Documents\GitHub\MachineLearning_TextModule\results\eda\error_analysis_report.json
