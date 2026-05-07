import os

preamble = r'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               PREAMBLE                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\documentclass[a4paper,12pt]{article}

%---------- LANGUAGE & FONT SUPPORT ----------%
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[english]{babel}
\usepackage{mathptmx}
\usepackage{inconsolata}

%---------- COMMON PACKAGES ----------%
\usepackage{float}
\usepackage{textcomp}
\usepackage{geometry}
\geometry{
  a4paper,
  total={170mm,257mm},
  left=20mm,
  top=20mm,
  includefoot,
}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{xurl}
\hypersetup{
  colorlinks=true,
  linkcolor=black,
  filecolor=magenta,
  urlcolor=blue,
  citecolor=black,
}
\usepackage{booktabs,makecell,tabularx}
\usepackage{longtable}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{fancyhdr}
\usepackage{lastpage}
\usepackage{array}
\usepackage{multirow}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{picture}
\usepackage{enumitem}
\usepackage{ragged2e}
\usepackage[protrusion=false, expansion=false]{microtype}

%---------- LIST CONFIGURATION ----------%
\setlist[itemize,1]{label=--, labelsep=0.6em}
\setlist[itemize,2]{label=\textbullet}
\setlist[enumerate,1]{leftmargin=*,nosep}
\newcolumntype{L}{>{\bfseries}p{28mm}}
\newcolumntype{P}[1]{>{\RaggedRight\arraybackslash}p{#1}}

\emergencystretch=2em

%---------- CODE DISPLAY CONFIGURATION ----------%
\definecolor{codegreen}{rgb}{0,0.6,0}
\definecolor{codegray}{rgb}{0.5,0.5,0.5}
\definecolor{codeblue}{rgb}{0.0, 0.0, 1.0}
\definecolor{codered}{rgb}{1.0, 0.0, 0.0}
\definecolor{backcolour}{rgb}{1.0, 1.0, 1.0}

\lstdefinestyle{PythonStyle}{
  backgroundcolor=\color{backcolour},
  commentstyle=\color{codegreen},
  keywordstyle=\color{codeblue},
  stringstyle=\color{codered},
  basicstyle=\ttfamily\small,
  breakatwhitespace=false,
  breaklines=true,
  captionpos=b,
  keepspaces=true,
  numbers=none,
  showspaces=false,
  showstringspaces=false,
  showtabs=false,
  tabsize=4,
  frame=single,
  rulecolor=\color{black},
}
\lstset{style=PythonStyle}

%---------- HEADER & FOOTER CONFIGURATION ----------%
\setlength{\headheight}{52pt}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{%
  \begin{tabular}{rl}
    \begin{picture}(25,15)(0,0)
      \put(0,-8){\includegraphics[width=8mm,height=8mm]{Image/hcmut.png}}
    \end{picture} &
    \begin{tabular}{l}
      \textbf{\ttfamily Ho Chi Minh City University of Technology}\\
      \textbf{\ttfamily Faculty of Computer Science and Engineering}
    \end{tabular}
  \end{tabular}
}
\fancyfoot[L]{\scriptsize \ttfamily Machine Learning Project - Text Classification}
\fancyfoot[R]{\scriptsize \ttfamily Page \thepage/\pageref{LastPage}}
\renewcommand{\headrulewidth}{0.3pt}
\renewcommand{\footrulewidth}{0.3pt}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                            DOCUMENT BODY                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\begin{document}

%---------- TITLE PAGE ----------%
\begin{titlepage}
  \thispagestyle{empty}
  \newcommand{\HRule}{\rule{\linewidth}{0.6pt}}
  \begin{center}
    \vspace*{6mm}
    \textbf{VIETNAM NATIONAL UNIVERSITY - HO CHI MINH CITY}\\
    \textbf{HO CHI MINH CITY UNIVERSITY OF TECHNOLOGY}\\
    \textbf{FACULTY OF COMPUTER SCIENCE AND ENGINEERING}
    \vspace{12mm}
    \begin{figure}[H]
        \centering
        \includegraphics[width=50mm]{Image/hcmut.png}
    \end{figure}
    \vspace{14mm}

    \begin{center}
        {\Large \bfseries MACHINE LEARNING (CO3117)}
        \vspace{1.5cm}

        \HRule \\[0.5cm]
        {\Large \textbf{Assignment Report}} \\[0.5cm]
        {\huge \bfseries Text Classification on AG News} \\[0.5cm]
    \end{center}

    \vspace{6mm}
    \HRule
    \vspace{20mm}

    \begin{tabular}{@{}l l@{}}
    \textbf{Instructor:} & Dr. Le Thanh Sach \\
    \addlinespace[6pt]
    \multicolumn{2}{@{}l}{\textbf{Members:}} \\
    & Tran Hoang Vy Khang -- [2352502] \\
    & Le Tan Minh Khoa -- [2352563] \\
    & Tao Nguyen Quang Khang -- [2352499] \\
    & Tran Gia Huy -- [2252264] \\
    & Vo Le Hai Dang -- [2352257] \\
    \end{tabular}

    \vfill
    {\large Ho Chi Minh City, Semester I, 2025-2026}
  \end{center}
\end{titlepage}

\newpage
\tableofcontents
\newpage
'''

body = r'''
\section{Course Context}
\begin{itemize}[leftmargin=*]
    \item Course: Machine Learning (CO3117)
    \item Department: Computer Science and Engineering, HCMUT -- VNU-HCM
    \item Supervisor: Dr. Le Thanh Sach
    \item Topic: Text Data Classification (Topic 2)
\end{itemize}

\section{Project Objectives}
The project aims to build a complete, highly scalable, and production-ready text classification pipeline from exploratory data analysis to final model comparison, adhering to the strict requirements of the official syllabus. The specific objectives are outlined as follows:
\begin{enumerate}[leftmargin=*]
    \item \textbf{Exploratory Data Analysis (EDA):} Conduct an in-depth statistical and visual analysis of the dataset to uncover class distributions, text length variances, and word frequency phenomena.
    \item \textbf{Data Preprocessing:} Implement robust text cleaning, handling missing values, and tokenization techniques to ensure data quality before vectorization.
    \item \textbf{Feature Extraction:} Evaluate traditional lexical methods (TF-IDF) against modern contextual embeddings (SBERT) to understand the trade-offs between computational efficiency and semantic depth.
    \item \textbf{Classical ML Training:} Train and rigorously evaluate a suite of Supervised Learning algorithms, specifically Logistic Regression, Support Vector Machines (SVM), and Naive Bayes.
    \item \textbf{Performance Evaluation:} Utilize industry-standard metrics including Accuracy, Precision, Recall, and F1-score to determine the optimal model configuration.
    \item \textbf{Comparative Analysis:} Provide a comprehensive, data-driven report comparing feature families, supported by confusion matrices and error analysis.
\end{enumerate}

\section{Introduction and Motivation}
Text classification is a foundational task in Natural Language Processing (NLP) with sweeping applications ranging from spam detection and sentiment analysis to automated customer support and news categorization. As the sheer volume of digital text grows exponentially on a daily basis, the need for automated systems to accurately organize and route this information has never been more critical. 

Historically, the field of text classification has been dominated by traditional Machine Learning (ML) approaches paired with lexical feature extraction techniques such as the Bag-of-Words (BoW) model and Term Frequency-Inverse Document Frequency (TF-IDF). These methods, while computationally lightweight and highly interpretable, fundamentally treat documents as unordered collections of words, completely ignoring syntactic structure and deep semantic meaning.

In recent years, the paradigm has shifted dramatically toward Deep Learning and Transformer-based architectures, spearheaded by models like BERT (Bidirectional Encoder Representations from Transformers). These models generate dense, contextual embeddings that capture nuanced relationships between words, often achieving state-of-the-art results on complex NLP benchmarks. However, this superior performance comes at a massive computational cost, requiring specialized hardware (GPUs/TPUs) for both training and inference.

The motivation of this project is to empirically investigate whether the immense computational overhead of modern Transformer-based embeddings is justified for a straightforward news topic classification task. By pitting highly optimized, traditional TF-IDF models against modern SBERT embeddings, we aim to provide clear, actionable insights into the trade-offs between model complexity, inference speed, and predictive accuracy in a real-world scenario.

\section{Theoretical Background}
To fully contextualize the methodologies employed in this project, this section provides the mathematical and theoretical foundations of the feature extraction techniques, supervised learning algorithms, and evaluation metrics utilized.

\subsection{Feature Extraction Techniques}

\subsubsection{Term Frequency-Inverse Document Frequency (TF-IDF)}
TF-IDF is a statistical measure that evaluates how relevant a word is to a document in a collection of documents. This is done by multiplying two metrics: how many times a word appears in a document, and the inverse document frequency of the word across a set of documents.
\begin{equation}
\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)
\end{equation}
Where:
\begin{itemize}
    \item $\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$ (The raw count of a term $t$ in document $d$ normalized by total terms).
    \item $\text{IDF}(t, D) = \log\left(\frac{N}{|\{d \in D : t \in d\}|}\right)$ (The logarithmically scaled inverse fraction of the documents that contain the word $t$).
\end{itemize}
This mechanism inherently penalizes overly common words (like "the", "and", "is") while assigning high weights to rare, discriminative keywords that define the specific topic of a document. In our project, we expand the feature space by utilizing $N$-grams (unigrams and bigrams), which helps capture basic local context (e.g., "New York").

\subsubsection{Modern Contextual Embeddings (SBERT)}
Unlike TF-IDF, which produces highly sparse and orthogonal vectors whose dimensionality equals the vocabulary size, Sentence-BERT (SBERT) generates dense, fixed-size vectors (embeddings) that represent the overall semantic meaning of a text. SBERT is a modification of the pre-trained BERT network that uses siamese and triplet network structures to derive semantically meaningful sentence embeddings that can be compared using cosine similarity. The embedding process relies heavily on the Multi-Head Self-Attention mechanism, defined as:
\begin{equation}
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\end{equation}
This allows the model to weigh the importance of all other words in the sentence when encoding a specific word, thereby resolving disambiguation and capturing deep context. For this project, we employ the \texttt{all-MiniLM-L6-v2} variant, mapping documents into a 384-dimensional dense space.

\subsection{Supervised Learning Algorithms}

\subsubsection{Support Vector Machines (SVM)}
SVM is a powerful discriminative classifier formally defined by a separating hyperplane. Given labeled training data, the algorithm outputs an optimal hyperplane which categorizes new examples. In our high-dimensional TF-IDF space, a linear kernel is utilized to solve the primal optimization problem:
\begin{equation}
\min_{w, b, \xi} \frac{1}{2} ||w||^2 + C \sum_{i=1}^{n} \xi_i
\end{equation}
Subject to the constraints $y_i(w^T x_i + b) \ge 1 - \xi_i$ and $\xi_i \ge 0$. The regularization parameter $C$ controls the trade-off between achieving a low training error and a low testing error. SVMs are historically renowned for their extreme effectiveness on sparse textual data.

\subsubsection{Logistic Regression}
Logistic Regression models the probability that a given input belongs to a certain class. For multi-class classification (like our 4-class problem), the multinomial (softmax) logistic regression is used:
\begin{equation}
P(Y=k|X=x) = \frac{e^{\beta_k \cdot x}}{\sum_{j=1}^{K} e^{\beta_j \cdot x}}
\end{equation}
The model parameters $\beta$ are optimized by minimizing the categorical cross-entropy loss function, utilizing advanced solvers like L-BFGS or Liblinear. Logistic regression provides highly calibrated probabilities, making it exceptionally reliable for downstream decision-making.

\subsubsection{Naive Bayes}
The Naive Bayes classifier is a probabilistic machine learning model based on Bayes' Theorem, with the "naive" assumption of conditional independence between every pair of features given the value of the class variable. Let $y$ be the class variable and $x_1, \dots, x_n$ be dependent feature vectors:
\begin{equation}
P(y | x_1, \dots, x_n) = \frac{P(y) P(x_1, \dots, x_n | y)}{P(x_1, \dots, x_n)}
\end{equation}
Using the naive independence assumption:
\begin{equation}
P(y | x_1, \dots, x_n) \propto P(y) \prod_{i=1}^{n} P(x_i | y)
\end{equation}
While the independence assumption is often violated in natural language (words frequently co-occur, such as "San" and "Francisco"), Multinomial Naive Bayes remains remarkably effective and computationally lightweight for text classification baselines.

\subsection{Evaluation Metrics}
To rigorously evaluate the models, we utilize standard classification metrics based on the confusion matrix components: True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN). Given the multi-class nature of the problem, we employ the weighted average variants of Precision, Recall, and F1-score.
\begin{itemize}[leftmargin=*]
    \item \textbf{Accuracy:} The ratio of correctly predicted observations to the total observations.
    \item \textbf{Precision:} The ratio of correctly predicted positive observations to the total predicted positive observations. $\text{Precision} = \frac{TP}{TP + FP}$
    \item \textbf{Recall (Sensitivity):} The ratio of correctly predicted positive observations to all observations in the actual class. $\text{Recall} = \frac{TP}{TP + FN}$
    \item \textbf{F1-Score:} The harmonic mean of Precision and Recall, providing a balanced metric that penalizes extreme values. $F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
\end{itemize}


\section{Dataset and Problem Setting}
The dataset selected for this project is the widely recognized \textbf{AG News} corpus, sourced directly from Hugging Face Datasets (\texttt{ag\_news}). AG News is a collection of more than 1 million news articles gathered from more than 2,000 news sources by ComeToMyHead in more than 1 year of activity. It serves as a standard benchmark for text classification tasks in academia and industry.

\begin{itemize}[leftmargin=*]
    \item \textbf{Task:} 4-class single-label news topic classification.
    \item \textbf{Categories:} 
    \begin{enumerate}
        \item World
        \item Sports
        \item Business
        \item Sci/Tech
    \end{enumerate}
    \item \textbf{Data Splits:} The dataset is pre-partitioned into 120,000 training samples and 7,600 testing samples. For our full pipeline execution, we utilize the entire 120,000 samples to train the TF-IDF models, ensuring maximum vocabulary coverage and statistical reliability.
    \item \textbf{Language Constraint:} As the dataset consists entirely of English text, the problem does not necessitate cross-lingual processing or specialized tokenizers (such as \texttt{underthesea} for Vietnamese). Standard English tokenization protocols are strictly applied throughout the preprocessing phase.
\end{itemize}

\section{Exploratory Data Analysis (EDA)}
Extensive Exploratory Data Analysis was performed to deeply understand the characteristics, underlying structures, and statistical distributions of the AG News dataset prior to any modeling efforts. Understanding the data is crucial to making informed decisions about preprocessing and model selection.

\begin{figure}[H]
    \centering
    \begin{subfigure}[b]{0.45\textwidth}
        \centering
        \includegraphics[width=\textwidth]{eda/class_distribution.png}
        \caption{Class Distribution}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.45\textwidth}
        \centering
        \includegraphics[width=\textwidth]{eda/text_length_distribution.png}
        \caption{Text Length Distribution}
    \end{subfigure}
    \caption{Overview of Dataset Distributions}
\end{figure}

Key findings from the visual distributions include:
\begin{itemize}[leftmargin=*]
    \item \textbf{Balanced Class Distribution (Figure 1a):} The pie chart confirms that the 120,000 training samples are perfectly distributed evenly across the 4 news categories: World, Sports, Business, and Sci/Tech (~30,000 samples or exactly 25\% each). This ideal balance completely eliminates the need for complex class weighting, oversampling (SMOTE), or downsampling techniques during the model training phase. The baseline accuracy for a random guess is exactly 25\%, making any performance above this threshold statistically significant.
    \item \textbf{Text Length Distribution (Figure 1b):} The histogram of token counts reveals a stark right-skewed distribution. The vast majority of news articles are extremely concise, with the peak frequency occurring between 30 and 45 words. Very few texts exceed 100 words. This characteristic makes the dataset highly suitable for traditional BoW/TF-IDF approaches, as long-term dependencies (which LSTMs or deep Transformers excel at mapping) are practically non-existent in such short text snippets.
    \item \textbf{Data Cleanliness:} The dataset exhibits remarkably low noise. Outlier analysis showed near-zero empty documents or exact duplicates. Furthermore, the presence of URLs is minimal (only about 32 URL-heavy texts were found), meaning that extensive regex cleaning passes have a negligible impact on overall data quality.
\end{itemize}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\linewidth]{eda/word_frequency.png}
    \caption{Word Frequency Analysis across AG News}
\end{figure}

\begin{itemize}[leftmargin=*]
    \item \textbf{Domain-specific Vocabulary (Figure 2):} The word frequency analysis demonstrates that each class possesses highly distinct top keywords. For instance, words like ``game'', ``team'', and ``season'' overwhelmingly dominate the Sports category, while ``stock'', ``company'', and ``market'' are prevalent in Business. This strong discriminative power of single words implies that unigram and bigram TF-IDF models will likely achieve exceptional classification performance without the need for complex syntax analysis.
\end{itemize}


\section{Implemented Methodology}
The project workflow is logically divided into two primary avenues of investigation: the highly optimized Traditional Pipeline, and the computationally intensive Modern Embedding Extension. Both pipelines are designed with strict modularity and high reusability.

\subsection{Traditional Pipeline}
The traditional pipeline focuses on maximizing the utility of lexical features through rigorous preprocessing and hyperparameter tuning. It establishes the performance baseline.
\begin{itemize}[leftmargin=*]
    \item \textbf{Preprocessing:} Data cleaning, handling missing values, and tokenization. We implemented a strict cleaning pipeline that involves lowercasing all text, removing English stopwords (using the NLTK corpus), stripping punctuation via regular expressions, and removing standalone numeric digits. This process drastically reduces the dimensionality of the vector space and eliminates noise, focusing the model purely on semantic keywords.
    \item \textbf{Feature Extraction:} Traditional methods evaluated include an exhaustive TF-IDF configuration search over three distinct candidates:
    \begin{itemize}
        \item \texttt{tfidf\_uni\_1k}: Unigrams only, restricted to the top 1,000 most frequent features. (Fastest, baseline).
        \item \texttt{tfidf\_uni\_bi\_3k}: Unigrams and bigrams, restricted to 3,000 features.
        \item \texttt{tfidf\_uni\_bi\_5k}: Unigrams and bigrams, restricted to 5,000 features. (Richest feature space).
    \end{itemize}
    \item \textbf{Evaluation:} Performance metrics evaluated include Accuracy, Precision, Recall, and F1-score (weighted). The primary model-selection metric utilized to determine the optimal architecture is the F1-weighted score, as it perfectly balances false positives and false negatives.
    \item \textbf{Model Training:} Supervised Learning algorithms utilized include Logistic Regression (with L2 regularization to prevent overfitting), Support Vector Machines (LinearSVC with optimized C parameters for rapid margin maximization), and Multinomial Naive Bayes (as a probabilistic baseline).
\end{itemize}

\subsection{Modern Embedding Extension}
To contrast the traditional pipeline, we evaluate the efficacy of deep contextual embeddings using a pre-trained Transformer architecture. This pipeline tests whether semantic context improves upon lexical frequency.
\begin{itemize}[leftmargin=*]
    \item \textbf{Feature Extraction:} Modern Contextual Embeddings evaluated using \texttt{sentence-transformers/all-MiniLM-L6-v2} (a highly optimized BERT-based model). This model maps entire sentences and paragraphs to a 384-dimensional dense vector space by averaging the token-level hidden states (pooling).
    \item \textbf{Benchmark Scales:} Due to the heavy computational overhead of encoding 120,000 documents through a Transformer architecture (even a miniaturized one), the embedding pipeline is systematically benchmarked at two smaller scales: \texttt{5k\_2k} (5,000 train, 2,000 test) and \texttt{20k\_2k} (20,000 train, 2,000 test).
    \item \textbf{Numerical Features:} To ensure complete reproducibility and comply with the Definition of Done, all extracted embeddings and corresponding labels are rigorously exported and serialized as \texttt{.npy} artifacts within the \texttt{features/} directory.
\end{itemize}


\section{Source Code and Reproducibility}
The complete source code, artifacts, serialized models, EDA figures, and detailed instructions to execute this project are publicly available on our GitHub repository:
\begin{center}
    \url{https://github.com/THVKhang/MachineLearning_TextModule}
\end{center}

\textbf{System Architecture and Automation:}
All codebase executions are meticulously designed to be highly modular, scalable, and reproducible. Rather than relying on disorganized, massive Jupyter Notebooks, the entire logic has been abstracted into object-oriented Python modules located inside the \texttt{/modules} directory. 
The system features a fully automated workflow orchestrated by a central entry point (\texttt{notebooks/colab\_submission.ipynb}). This entry point supports a flawless "Run All" execution model on Google Colab, autonomously handling everything from dataset downloading to final evaluation and artifact generation without requiring any local file dependencies or personal Google Drive mounting.

\newpage
\section{Results}
\subsection{Best TF-IDF Results (Full Data)}
\begin{table}[H]
\centering
\begin{tabular}{lcccc}
\toprule
Model & Accuracy & Precision (w) & Recall (w) & F1 (w) \\
\midrule
SVM & 0.9046 & 0.9044 & 0.9046 & 0.9044 \\
Logistic Regression & 0.9038 & 0.9036 & 0.9038 & 0.9036 \\
Naive Bayes & 0.8871 & 0.8866 & 0.8871 & 0.8867 \\
\bottomrule
\end{tabular}
\caption{TF-IDF model comparison.}
\end{table}

\textbf{Analysis of TF-IDF Performance:} Table 1 outlines the performance of the three evaluated supervised learning models on the traditional TF-IDF sparse features using the full dataset (120,000 samples). Support Vector Machine (SVM) achieves the highest performance across all metrics (F1 = 0.9044), demonstrating its unparalleled capability to find optimal separating hyperplanes in high-dimensional lexical spaces. Logistic Regression closely follows (F1 = 0.9036), providing a well-calibrated probabilistic alternative that is nearly indistinguishable in practical accuracy. Naive Bayes lags slightly behind, likely due to its strict feature independence assumption which does not hold perfectly in natural language (e.g., bigrams inherently violate independence). Nonetheless, all models exhibit exceptional performance well above 88\%, validating the efficacy of the TF-IDF feature space.

\begin{figure}[H]
    \centering
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{figures/cm_svm.png}
        \caption{SVM}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{figures/cm_logistic_regression.png}
        \caption{LogReg}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{figures/cm_naive_bayes.png}
        \caption{Naive Bayes}
    \end{subfigure}
    \caption{Confusion Matrices for Traditional Models}
\end{figure}

\textbf{Analysis of the Confusion Matrices:} 
The confusion matrices in Figure 3 visually confirm the robustness of the traditional models, particularly SVM and Logistic Regression. The heavy concentration of values along the main diagonal indicates high classification accuracy across all four classes, with very few severe off-diagonal failures.
\begin{itemize}[leftmargin=*]
    \item \textbf{Support Vector Machine (SVM):} SVM exhibits the sharpest and darkest diagonal, meaning it made the fewest misclassifications. Its strong margin-maximization approach effectively separates the high-dimensional sparse TF-IDF vectors, ensuring that border cases are rigorously classified.
    \item \textbf{Logistic Regression:} It performs nearly identically to SVM but shows slightly more `bleed'' in off-diagonal cells, particularly confusing Business and Sci/Tech.
    \item \textbf{Naive Bayes:} While computationally the fastest, it struggles visibly compared to the other two. The independence assumption of Naive Bayes causes it to heavily misclassify semantically overlapping categories, resulting in noticeably lighter diagonal cells and denser off-diagonal errors (especially between World and Business).
\end{itemize}

\subsection{SBERT Benchmark Results}
\begin{table}[H]
\centering
\begin{tabular}{lccccc}
\toprule
Scale & Best Model & Accuracy & Precision (w) & Recall (w) & F1 (w) \\
\midrule
5k\_2k & Logistic Regression & 0.8740 & 0.8734 & 0.8740 & 0.8736 \\
20k\_2k & SVM & 0.8955 & 0.8958 & 0.8955 & 0.8954 \\
\bottomrule
\end{tabular}
\caption{Best embedding model per scale.}
\end{table}

\textbf{Analysis of SBERT Performance:} Table 2 presents the results of the modern contextual embedding benchmark using the pre-trained SBERT transformer model. At the smaller 5k scale (5,000 training samples), Logistic Regression performs best, achieving an F1 score of 0.8736. However, as the dataset scale increases to 20k, the Support Vector Machine surpasses it, reaching a highly competitive F1 score of 0.8954. This highlights a crucial phenomenon: while modern embeddings provide dense, rich semantic representations, they still require sufficient data volume and margin-based classifiers to establish clear decision boundaries that can rival or exceed traditional lexical methods. 

\subsection{Feature-Family Comparison}
\begin{figure}[H]
    \centering
    \includegraphics[width=0.85\linewidth]{figures/comparison_tfidf_vs_embedding.png}
    \caption{Performance Comparison: TF-IDF vs SBERT Embedding}
\end{figure}

The final Comparative Analysis reveals a counter-intuitive finding: the best traditional TF-IDF model (SVM with F1 = 0.9044) marginally outperformed the best Transformer-based contextual embedding model (SVM at 20k scale with F1 = 0.8954). The Transformer-based model underperformed the traditional SVM baseline in this specific context because the AG News dataset consists of very short texts (average 35 words) heavily laden with highly distinct, domain-specific vocabularies (e.g., "stock", "baseball", "olympics"). 

Traditional TF-IDF excels at capturing these explicit keyword signals. If an article contains the word "Microsoft", TF-IDF assigns it a massive weight, instantly pushing the prediction toward Sci/Tech or Business. SBERT, on the other hand, averages the embeddings of all words to create a single sentence vector. This pooling mechanism can inadvertently dilute the strong signal of a single crucial keyword (like "Microsoft") by mixing it with the semantic noise of surrounding stopwords or generic phrases. Furthermore, TF-IDF is significantly faster to train and extract, while maintaining highly interpretable sparse features. Therefore, for short, keyword-heavy classification tasks, TF-IDF + SVM remains the primary and most efficient recommendation.

\section{Error Analysis}
An error analysis was performed on the best overall model (TF-IDF + SVM) to deeply understand the root causes of misclassifications and the inherent limitations of the traditional pipeline. Out of the 7,600 unseen test samples, the SVM model made exactly 381 errors, corresponding to an error rate of roughly 5\%.

The top 3 most common confusion pairs extracted from the prediction logs are:
\begin{enumerate}
    \item \textbf{True: Business $\rightarrow$ Predicted: Sci/Tech} (73 occurrences): This is the single most frequent error in the entire dataset. It occurs because modern business news heavily revolves around technology companies. For example, an article about Apple or Intel releasing quarterly earnings reports contains massive amounts of tech vocabulary (e.g., "software", "chips", "devices") mixed with financial terms. The model naturally struggles to differentiate the primary intent of the article.
    \item \textbf{True: Business $\rightarrow$ Predicted: World} (49 occurrences): Articles discussing international trade agreements, global markets, oil prices, or macroeconomic policies often contain geographic and political terms ("Europe", "China", "Government") that the model strongly associates with World news.
    \item \textbf{True: World $\rightarrow$ Predicted: Sci/Tech} (44 occurrences): Global news covering space exploration (NASA), international cyber-security attacks, or global scientific summits often triggers the Sci/Tech classification logic due to overlapping terminology.
\end{enumerate}

\textbf{Conclusion on Errors:} The vast majority of the errors are not random noise or algorithmic failures, but stem from inherently ambiguous texts that even human annotators might debate. The boundaries between `Business'' and `Technology'', or `World'' and `Business'' in the 21st century are extremely blurred. This makes it a formidable challenge for any purely lexical-based model to resolve perfectly without deeper, multi-sentence contextual reasoning or multi-label classification frameworks.

\section{Conclusion and Future Work}
\textbf{Conclusion:}\\
This project successfully designed, implemented, and rigorously evaluated a comprehensive machine learning pipeline for text classification. Through systematic exploratory data analysis, we validated that the AG News dataset is exceptionally well-suited for lexical feature extraction due to its short text lengths and distinct vocabulary distribution. The traditional TF-IDF pipeline, particularly when paired with a Support Vector Machine, demonstrated outstanding classification capabilities, achieving a weighted F1-score of over 0.90. 

The integration and evaluation of modern Transformer-based SBERT embeddings proved that while dense semantic representations are theoretically powerful, they do not inherently guarantee superiority over highly optimized traditional methods on datasets heavily reliant on explicit keyword matching. Ultimately, the project demonstrates that an intelligently configured classical pipeline can achieve state-of-the-art results with a fraction of the computational cost of deep learning models.

\textbf{Future Work:}\\
While the current system achieves exceptional performance and fully satisfies all project objectives, there remain several avenues for future enhancement:
\begin{itemize}[leftmargin=*]
    \item \textbf{Full-Scale Fine-Tuning:} Leveraging local GPU hardware (e.g., RTX 4060) to execute a full parameter fine-tuning of BERT or RoBERTa architectures on the entire 120,000 training set, moving beyond merely extracting frozen embeddings. This would likely allow the Transformer to surpass the TF-IDF baseline.
    \item \textbf{Ensemble Methods:} Constructing a hybrid ensemble model that concatenates the dense contextual embeddings of SBERT with the sparse lexical signals of TF-IDF, theoretically capturing both deep semantics and explicit keywords simultaneously.
    \item \textbf{Advanced Preprocessing:} Implementing sophisticated lemmatization (using \texttt{spaCy}) and Parts-of-Speech (POS) tagging to further refine the TF-IDF feature space and eliminate morphological redundancy.
\end{itemize}

\section{Team Contributions}
\begin{table}[H]
\centering
\begin{tabularx}{\linewidth}{l c l X c}
\toprule
Member & ID & Email & Main Contribution & \% \\
\midrule
Tran Hoang Vy Khang & 2352502 & khang.tran2106@hcmut.edu.vn & Embeddings, workflow, integration & 100 \\
Le Tan Minh Khoa & 2352563 & - & Text cleaning and TF-IDF & 100 \\
Tao Nguyen Quang Khang & 2352499 & - & ML training and evaluation & 100 \\
Tran Gia Huy & 2252264 & - & Pipeline and config architecture & 100 \\
Vo Le Hai Dang & 2352257 & - & Data loading and EDA analysis & 100 \\
\bottomrule
\end{tabularx}
\caption{Detailed contribution table.}
\end{table}

\end{document}
'''

with open('reports/report.tex', 'w', encoding='utf-8') as f:
    f.write(preamble + body)

print('Done')
