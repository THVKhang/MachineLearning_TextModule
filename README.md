# MACHINE LEARNING COURSE PROJECT (TEXT DATA)

## 1. Course Information

- Course: Machine Learning
- Course Code: [CO3117]
- Semester: Semester I
- Academic Year: 2025-2026
- Department: Department of Computer Science, Ho Chi Minh City University of Technology, VNU-HCM
- Supervisor: Dr. Le Thanh Sach
- Referenced project specification version: v1.1 (01/09/2025)

## 2. Team Information

| Full Name | Student ID | Email | Main Responsibility | Contribution (%) |
|---|---|---|---|---|
| Tran Hoang Vy Khang (Leader) | 2352502 | N/A | Embedding benchmark, planner workflow, model comparison, final integration | 30 |
| Le Tan Minh Khoa | 2352563  | N/A | Text cleaning and TF-IDF feature engineering | 20 |
| Tao Nguyen Quang Khang | 2352499 | N/A | Classical ML training/evaluation on TF-IDF | 20 |
| Tran Gia Huy | 2252264  | N/A | Pipeline architecture, configuration, reproducibility support | 15 |
| Vo Le Hai Dang | 2352257 | N/A | Data loading, EDA notebook, exploratory analysis artifacts | 15 |

Note: Individual grades follow the project rule:

Individual Grade = Group Grade x Contribution (%)

## 3. Project Objectives

This repository implements Topic 2 (Text Data) and follows the required course workflow:

- Build a complete traditional machine learning pipeline: EDA -> preprocessing -> feature extraction -> training -> evaluation.
- Implement at least two text feature extraction approaches:
    - Traditional: TF-IDF / n-gram.
    - Modern: contextual embeddings with SBERT (Transformer-based).
- Save extracted features in .npy format for reusable experiments.
- Compare multiple classifiers: Logistic Regression, SVM, and Naive Bayes.
- Report performance with Accuracy, Precision, Recall, and F1-weighted.

## 4. Implemented Scope

### 4.1 Dataset and Task

- Dataset: ag_news (from Hugging Face Datasets).
- Task: multi-class text classification.

### 4.2 Required Traditional Pipeline

1. Load train/test data.
2. Text preprocessing (configurable): lowercase, stopword removal, punctuation removal, number removal.
3. TF-IDF extraction with configurable max_features, ngram_range, and min_df.
4. Train traditional classifiers.
5. Evaluate with Accuracy/Precision/Recall/F1 and confusion matrix.

### 4.3 Bonus Modern Embedding Extension

1. Extract SBERT embeddings (all-MiniLM-L6-v2).
2. Save embeddings as .npy files.
3. Run benchmarking on multiple dataset scales (for example: 5k_2k, 20k_2k).
4. Compare 3 models on embeddings and export an aggregated CSV report.

## 5. Project Structure

```
MachineLearning_TextModule/
|-- README.md
|-- requirements.txt
|-- run_experiments.py
|-- bert_benchmark.py
|-- modules/
|   |-- __init__.py
|   |-- config.py
|   |-- data_loader.py
|   |-- text_preprocess.py
|   |-- tfidf_features.py
|   |-- bert_embed.py
|   |-- train_classical.py
|   |-- metrics.py
|   |-- pipeline.py
|-- notebooks/
|   |-- colab_submission.ipynb
|-- features/
|   |-- tfidf_train.npy
|   |-- tfidf_test.npy
|   |-- bert/
|-- results/
|   |-- (figures/tables/logs are generated during execution)
|-- tests/
|   |-- test_modules.py
|-- reports/
|   |-- report.tex
|   |-- report.pdf
```

## 6. Environment Setup

### 6.1 Local Setup (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 6.2 Local Setup (Linux/macOS)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 7. How to Run

### 7.1 Run the Traditional Pipeline (TF-IDF + Classical ML)

```bash
python run_experiments.py
```

Expected outputs:

- Feature files in the features directory.
- Confusion matrix images (if enabled) in results.

### 7.2 Run the SBERT Benchmark (Week 3)

```bash
python bert_benchmark.py
```

This benchmark will:

- Extract SBERT embeddings for benchmark scales.
- Save embeddings and labels in features/bert/<scale>/.
- Train and evaluate 3 models: logistic_regression, svm, naive_bayes.
- Save the summary table to results/tables/bert_benchmark_results.csv.

### 7.3 Run the End-to-End Notebook UI

Main notebook: notebooks/final.ipynb

- Interactive UI for selecting mode/feature/model.
- Week 3 artifact validation checklist cells.
- Benchmark trigger cell for Week 3.

### 7.4 Run the One-Entry Agency Workflow

```bash
python scripts/run_agency_workflow.py fast
python scripts/run_agency_workflow.py balanced
python scripts/run_agency_workflow.py best
```

This workflow executes Planner -> TF-IDF/Embedding Runners -> Critic -> Reporter and exports report-ready tables and logs.

## 8. Google Colab Instructions (Course Requirement)

Important requirements from the assignment:

- The submitted notebook must run successfully with Runtime -> Run all.
- Do not mount personal cloud storage (Google Drive, Dropbox, etc.).
- Data must be downloaded from public sources and prepared automatically inside the notebook.

Suggested Colab bootstrap:

```python
!git clone https://github.com/THVKhang/MachineLearning_TextModule.git
%cd MachineLearning_TextModule
!python -m pip install -r requirements.txt
```

Then open notebooks/colab_submission.ipynb and run all cells top-to-bottom.

## 9. Tests

Run tests:

```bash
python -m pytest -q
```

Current test coverage includes:

- Valid metric ranges (0..1).
- TF-IDF feature shape and dtype checks.
- Save/load .npy round-trip checks.
- SBERT cache/rebuild behavior in the pipeline.

## 10. Mapping to Evaluation Criteria

- Traditional pipeline completeness (40%): EDA + preprocessing + feature extraction + training + evaluation implemented.
- Experiment quality (25%): multiple models/configurations and benchmark scales compared.
- Report quality (20%): documented in report materials.
- Submission completeness (10%): clear notebook/modules/features/results organization.
- Teamwork evidence (5%): module ownership by member, commit history by feature scope, and integrated workflow logs in results/logs.

Bonus opportunities:

- Modern embedding extension compared against the traditional pipeline.
- Public GitHub repository with a complete README and reproducible instructions.

## 11. Submission Links

- GitHub repository: https://github.com/THVKhang/MachineLearning_TextModule
- Colab notebook (submission): https://colab.research.google.com/github/THVKhang/MachineLearning_TextModule/blob/main/notebooks/colab_submission.ipynb
- PDF report: reports/report.pdf
- Feature files (.npy/.h5) if hosted separately: generated locally via workflow and stored under features/ and results/.

## 12. Pre-Submission Checklist

- [x] Colab notebook runs successfully with Run all.
- [x] No personal cloud mounting is used.
- [x] Library installation and data download are automated.
- [x] EDA, pipeline, experiments, comparison tables, and analysis are included.
- [x] Feature files are saved in .npy or .h5 format.
- [x] PDF report and detailed contribution table are included.
- [x] README includes full course/team/run/link information.

## 13. Academic Use and Citation

- This project is developed for academic learning in the Machine Learning course.
- Please cite all data sources and external libraries in the final report.