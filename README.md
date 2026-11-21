# 🛡️ Day 22: SMS Spam Classifier & Classical NLP Pipeline

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![100 Days of ML](https://img.shields.io/badge/100%20Days%20of%20ML-Day%2022-blueviolet)](#)

An end-to-end Natural Language Processing (NLP) text classification system that filters SMS spam and phishing attempts with **99.22% precision** using custom entity normalization, Porter stemming, sublinear **TF-IDF vectorization**, and classical machine learning algorithms (**Multinomial Naive Bayes**, **Logistic Regression**, and **Random Forest**).

Includes an interactive **Streamlit dashboard** featuring live message classification, word-level explainability/token attribution, model benchmark comparisons, top predictive feature exploration, and an educational step-by-step NLP pipeline visualizer.

---

![Project Banner](assets/banner.jpg)

---

## 📌 Project Overview & Pedagogical Milestone

In **Day 22 of 100 Days of Machine Learning**, we transition into **Natural Language Processing (NLP)**. Before deploying deep neural networks or large language models, mastering classical feature extraction (Bag of Words, N-grams, TF-IDF) and probabilistic/linear classifiers is essential.

### Why Classical NLP Excels in Spam Filtering
1. **Ultra-Low Latency:** Inferences execute in sub-millisecond speeds (< 2 ms) without requiring GPUs.
2. **Interpretability:** Decisions can be directly traced to specific word log-odds ratios and linear coefficients.
3. **Data Efficiency:** High performance achieved on modest tabular corpora (5,574 messages).
4. **Precision Optimization:** Fine-tuned decision thresholds prevent catastrophic False Positives (legitimate emails marked as spam).

---

## 🔬 System Architecture & NLP Pipeline

```mermaid
flowchart TD
    A[Raw Unstructured SMS Text] --> B[Entity Normalization]
    B -->|URLs -> [URL], Phone -> [PHONE], Money -> [MONEY]| C[Punctuation & Case Folding]
    C --> D[Stopword Filtering]
    D --> E[Porter Stemming]
    E --> F[TF-IDF Vectorizer 1-2 N-grams, 3000 Features]
    F --> G1[Multinomial Naive Bayes Laplace alpha=1.0]
    F --> G2[Regularized Logistic Regression C=2.0]
    F --> G3[Random Forest Classifier 100 Trees]
    G1 --> H[Token Attribution & Probability Verdict]
    G2 --> H
    G3 --> H
    H --> I[Streamlit Interactive Dashboard]
```

---

## 📐 Mathematical Formulation

### 1. Term Frequency - Inverse Document Frequency (TF-IDF)
While simple Count Vectorization (Bag of Words) merely counts word frequencies, TF-IDF discounts ubiquitously common words while amplifying distinctive, content-rich keywords:

$$\text{TF}(t, d) = \frac{f_{t, d}}{\sum_{t' \in d} f_{t', d}}$$

$$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + \text{DF}(t)}\right) + 1$$

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

We utilize **sublinear TF scaling** ($1 + \log(\text{TF})$) to prevent words repeated multiple times in a single text message from artificially skewing the feature space.

### 2. Multinomial Naive Bayes with Laplace Smoothing
Applying Bayes' theorem under the conditional independence assumption:

$$P(y = c \mid \mathbf{x}) \propto P(y = c) \prod_{i=1}^{n} P(w_i \mid y = c)$$

To eliminate zero probabilities for out-of-vocabulary or unseen terms, we apply **Laplace Smoothing** ($\alpha = 1.0$):

$$P(w_i \mid y = c) = \frac{N_{ci} + \alpha}{N_c + \alpha |V|}$$

Where:
- $N_{ci}$: Total occurrences of word $w_i$ in class $c$.
- $N_c$: Total word tokens across all documents in class $c$.
- $|V|$: Vocabulary dimension ($3,000$ features).

### 3. The Precision Asymmetry in Spam Classification
In spam classification:
- **False Positive (FP):** A critical personal or business email is incorrectly diverted to the Spam folder.
- **False Negative (FN):** An unwanted spam advertisement arrives in the Inbox.

Because missing a legitimate message carries far higher cost than manually deleting a spam message, **Precision** ($\frac{TP}{TP + FP}$) is the principal optimization metric.

---

## 📊 Benchmark Results (Test Set: 1,114 Messages)

Data split: 80% Train ($4,458$ messages) / 20% Stratified Test ($1,114$ messages).

| Model Algorithm | Accuracy | Precision (Spam) | Recall (Spam) | F1-Score | False Positives (FP) | False Negatives (FN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | 97.94% | **99.22%** | 85.23% | 0.9170 | **1** | 22 |
| **Logistic Regression** | **98.65%** | 98.55% | **91.28%** | **0.9477** | 2 | 13 |
| **Random Forest** | 97.85% | **100.00%** | 83.89% | 0.9124 | **0** | 24 |

*Note: Random Forest achieved zero false positives on the test set, while Logistic Regression achieved the highest overall accuracy (98.65%) and balanced F1-score.*

---

## 🔤 Most Predictive Keywords (Log-Odds Ratios)

### 🚨 Top Spam Triggers
1. `claim`
2. `prize`
3. `urgent`
4. `moneynum` (normalized £/$ values)
5. `urladdr` (normalized phishing links)
6. `phonenum` (normalized shortcodes / premium numbers)
7. `won` / `winner`
8. `txt` / `text`
9. `cash`
10. `guaranteed`

### ✅ Top Ham (Legitimate) Triggers
1. `ok` / `lor` / `da`
2. `later`
3. `home`
4. `come`
5. `ill`
6. `love`
7. `meet`
8. `sorry`
9. `work`
10. `night`

---

## 🖥️ Streamlit Interactive Application Features

1. **Live Spam Detector & Token Attribution:**
   - Real-time classification on arbitrary user text or 5 built-in presets.
   - Interactive token-level explainability chip cards (Red for spam indicators, Green for ham indicators, Gray for neutral).
   - Dynamic confidence meter and probability gauge.
2. **Model Benchmarks & Metrics:**
   - Interactive confusion matrix heatmaps.
   - Comprehensive comparative metrics table.
3. **TF-IDF Keyword Explorer:**
   - Top 20 horizontal bar charts comparing Spam vs. Ham log-probability advantages.
4. **Step-by-Step NLP Pipeline Sandbox:**
   - Visual inspection of each text transformation phase: Raw Text $\rightarrow$ Entity Normalization $\rightarrow$ Punctuation Cleaning $\rightarrow$ Stopword Removal $\rightarrow$ Stemming $\rightarrow$ TF-IDF Vectorization.

---

## 🚀 Quickstart & Local Installation

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/Syed-Muhammad-Qamran/sms-spam-classifier.git
cd sms-spam-classifier
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Test Suite
```bash
python -m unittest test_classifier.py
```

### 5. Launch the Streamlit App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📂 Project Structure

```
sms-spam-classifier/
├── assets/
│   └── banner.jpg               # High-res project banner
├── data/
│   └── sms_spam_collection.csv  # 5,574 labeled SMS messages
├── models/
│   ├── tfidf_vectorizer.joblib  # Trained TF-IDF vectorizer (3000 n-grams)
│   ├── nb_model.joblib          # Trained Multinomial Naive Bayes
│   ├── lr_model.joblib          # Trained Logistic Regression
│   ├── rf_model.joblib          # Trained Random Forest
│   ├── metrics_summary.json     # Test evaluation results
│   └── top_features.json        # Top predictive n-grams & weights
├── src/
│   ├── __init__.py
│   ├── preprocessor.py          # Entity normalization, regex, stemmer
│   └── model_trainer.py         # Pipeline training & inference engine
├── .gitignore
├── app.py                       # Streamlit web application
├── requirements.txt             # Python dependencies
├── test_classifier.py           # Unit tests & verification
└── README.md                    # Detailed documentation
```

---

## 👤 Author

**Syed Muhammad Qamran**  
- **GitHub:** [@Syed-Muhammad-Qamran](https://github.com/Syed-Muhammad-Qamran)  
- **LinkedIn:** [syed-muhammad-qamran](https://www.linkedin.com/in/syed-muhammad-qamran/)  
- **Series:** Part of the **#100DaysOfMachineLearning** Challenge (Day 22)

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
