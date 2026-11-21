"""
Script to generate the complete Day 22 NLP Jupyter Notebook
sms_spam_classifier_walkthrough.ipynb
"""

import json
import os

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

def add_md(source):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    })

def add_code(source):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    })

# -------------------------------------------------------------
# Cell 1: Header
# -------------------------------------------------------------
add_md("""# 🛡️ Day 22: SMS Spam Classifier & Classical NLP Pipeline
### 100 Days of Machine Learning | Syed Muhammad Qamran
**Topic:** Natural Language Processing (NLP) from Scratch  
**Techniques:** Text Normalization, Suffix Stemming, TF-IDF Vectorization, Multinomial Naive Bayes, Logistic Regression, Feature Attribution

---

### Notebook Roadmap:
1. **Setup & Imports** - Load required libraries for NLP, numerical computing, and machine learning.
2. **Exploratory Data Analysis (EDA)** - Analyze class distributions, text lengths, and word frequencies.
3. **Text Preprocessing Pipeline** - Regex entity normalization, punctuation stripping, stopword filtering, and Porter stemming.
4. **Feature Extraction with TF-IDF** - Transforming text into sparse mathematical vector spaces ($TF \\times IDF$).
5. **Model Training** - Training Multinomial Naive Bayes, Logistic Regression, and Random Forest.
6. **Performance Evaluation & Confusion Matrices** - Accuracy, Precision, Recall, F1, and the Precision-first spam philosophy.
7. **Interpretability & Top Predictive Features** - Identifying the highest-weighted terms that signal spam vs. legitimate communication.
8. **Live Prediction with Token Attribution** - Testing unseen messages with word-by-word influence scores.
""")

# -------------------------------------------------------------
# Cell 2 & 3: Imports
# -------------------------------------------------------------
add_md("""## 1. Setup & Imports
We import standard libraries:
- `pandas` & `numpy` for data handling
- `re` & `string` for regular expressions and text cleaning
- `matplotlib` & `seaborn` for visualizations
- `sklearn` for TF-IDF vectorization, classifiers, and evaluation metrics
""")

add_code("""import os
import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-Learn tools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Visual styling
sns.set_theme(style="whitegrid")
%matplotlib inline

print("All libraries successfully imported!")
""")

# -------------------------------------------------------------
# Cell 4 & 5: Data Loading & EDA
# -------------------------------------------------------------
add_md("""## 2. Data Loading & Exploratory Data Analysis (EDA)
We load the SMS Spam Collection dataset containing 5,574 messages labeled as either:
- **`ham`**: Legitimate, normal communication.
- **`spam`**: Unsolicited promotional messages, phishing links, lottery scams, or fraud attempts.
""")

add_code("""# Load dataset
data_path = os.path.join("..", "data", "sms_spam_collection.csv")
if not os.path.exists(data_path):
    data_path = os.path.join("data", "sms_spam_collection.csv")

df = pd.read_csv(data_path)

print(f"Total SMS Records: {len(df)}")
display(df.head())

# Class distribution
print("\\nClass Distribution:")
counts = df['label'].value_counts()
pcts = df['label'].value_counts(normalize=True) * 100
summary_df = pd.DataFrame({'Count': counts, 'Percentage (%)': pcts.round(2)})
display(summary_df)

# Plot class balance
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(x=counts.index, y=counts.values, palette=['#10B981', '#EF4444'], ax=ax)
ax.set_title("SMS Class Distribution (Ham vs. Spam)", fontsize=13, fontweight='bold')
ax.set_ylabel("Message Count")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                ha='center', va='center', color='white', fontweight='bold', fontsize=12)
plt.show()
""")

# -------------------------------------------------------------
# Cell 6: Message Length Analysis
# -------------------------------------------------------------
add_md("""### Analyzing Message Character & Word Lengths
Notice how spam messages tend to be significantly longer than casual conversations because spammers pack in urgent calls to action, fake legal disclaimers, phone numbers, and URLs.
""")

add_code("""# Compute character lengths and word counts
df['char_length'] = df['message'].apply(len)
df['word_count'] = df['message'].apply(lambda m: len(str(m).split()))

# Average statistics per class
print("Average Metrics by Class:")
display(df.groupby('label')[['char_length', 'word_count']].mean().round(1))

# Plot distributions
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
sns.histplot(data=df, x='char_length', hue='label', bins=50, kde=True, palette=['#10B981', '#EF4444'], ax=axes[0])
axes[0].set_title("Character Length Distribution")
axes[0].set_xlim(0, 300)

sns.histplot(data=df, x='word_count', hue='label', bins=30, kde=True, palette=['#10B981', '#EF4444'], ax=axes[1])
axes[1].set_title("Word Count Distribution")
axes[1].set_xlim(0, 60)

plt.tight_layout()
plt.show()
""")

# -------------------------------------------------------------
# Cell 7 & 8: Preprocessing Pipeline
# -------------------------------------------------------------
add_md("""## 3. Text Preprocessing Pipeline
Raw text messages contain noise: varied capitalization, phone numbers, currencies, URLs, and punctuation.
If we do not clean the text, `WIN`, `win!`, and `winning` would be treated as three unrelated tokens.

### Our 5-Stage NLP Preprocessing Pipeline:
1. **Lowercasing:** Converts all characters to lowercase (`Win` == `win`).
2. **Regex Entity Normalization:**
   - URLs (`https://...` or `www...`) $\\rightarrow$ `urladdr`
   - Currencies (`$500`, `£1000`) $\\rightarrow$ `moneynum`
   - Phone numbers & shortcodes $\\rightarrow$ `phonenum`
   - Generic numbers $\\rightarrow$ `nummark`
3. **Punctuation Stripping:** Removes punctuation characters (`!`, `?`, `#`, etc.).
4. **Stopword Removal:** Eliminates common English filler words (`is`, `at`, `which`, `the`).
5. **Porter Stemming:** Strips inflections to recover base word stems (`running` $\\rightarrow$ `run`, `congratulations` $\\rightarrow$ `congratul`).
""")

add_code("""# Self-contained curated English stopwords
STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'aren',
    \"aren't\", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but',
    'by', 'can', 'cannot', 'could', 'couldn', \"couldn't\", 'd', 'did', 'didn', \"didn't\", 'do', 'does',
    'doesn', \"doesn't\", 'doing', 'don', \"don't\", 'down', 'during', 'each', 'few', 'for', 'from', 'further',
    'had', 'hadn', \"hadn't\", 'has', 'hasn', \"hasn't\", 'have', 'haven', \"haven't\", 'having', 'he', 'her',
    'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'isn', \"isn't\",
    'it', \"it's\", 'its', 'itself', 'just', 'll', 'm', 'me', 'mightn', \"mightn't\", 'more', 'most', 'mustn',
    \"mustn't\", 'my', 'myself', 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or',
    'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', \"shan't\", 'she',
    \"she's\", 'should', 'shouldn', \"shouldn't\", 'so', 'some', 'such', 't', 'than', 'that', \"that'll\", 'the',
    'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through',
    'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', \"wasn't\", 'we', 'were', 'weren',
    \"weren't\", 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won',
    \"won't\", 'wouldn', \"wouldn't\", 'y', 'you', \"you'd\", \"you'll\", \"you're\", \"you've\", 'your', 'yours',
    'yourself', 'yourselves'
}

class SimplePorterStemmer:
    \"\"\"
    Deterministic suffix stripper inspired by Porter's algorithm.
    Reduces inflected words to their root stem.
    \"\"\"
    @staticmethod
    def stem(word: str) -> str:
        word = word.lower().strip()
        if len(word) <= 3:
            return word
        
        # Plural and past tense
        if word.endswith('sses'):
            word = word[:-2]
        elif word.endswith('ies'):
            word = word[:-2]
        elif word.endswith('ss'):
            pass
        elif word.endswith('s') and not word.endswith('us') and not word.endswith('is'):
            word = word[:-1]
            
        if word.endswith('eed'):
            if len(word) > 4:
                word = word[:-1]
        elif word.endswith('ed') and len(word) > 4:
            word = word[:-2]
        elif word.endswith('ing') and len(word) > 5:
            word = word[:-3]

        # Double consonant reduction (e.g. runn -> run, winn -> win)
        if len(word) >= 3 and word[-1] == word[-2] and word[-1] in 'bdfgmnprt':
            word = word[:-1]

        # Suffix reductions
        suffixes = (
            ('ational', 'ate'), ('tional', 'tion'), ('ization', 'ize'),
            ('fulness', 'ful'), ('ousness', 'ous'), ('iveness', 'ive'),
            ('ment', ''), ('able', ''), ('ible', ''), ('ful', ''),
            ('ous', ''), ('ive', ''), ('ize', ''), ('ly', ''), ('e', '')
        )
        for old, new in suffixes:
            if word.endswith(old) and len(word) - len(old) >= 3:
                word = word[:-len(old)] + new
                break
                
        return word

def clean_text(text: str, stem: bool = True) -> str:
    \"\"\"Complete text cleaning and normalization pipeline.\"\"\"
    if not isinstance(text, str):
        return \"\"

    # 1. Lowercase
    text = text.lower()

    # 2. Entity Normalization
    text = re.sub(r'https?://\\S+|www\\.\\S+', ' urladdr ', text)
    text = re.sub(r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b', ' emailaddr ', text)
    text = re.sub(r'[£$€]\\s*\\d+(?:\\.\\d+)?|\\d+(?:\\.\\d+)?\\s*[£$€]', ' moneynum ', text)
    text = re.sub(r'\\b(?:\\+?\\d{1,3}[-.\\s]?)?\\(?\\d{3,5}\\)?[-.\\s]?\\d{3,5}[-.\\s]?\\d{3,5}\\b', ' phonenum ', text)
    text = re.sub(r'\\b\\d+\\b', ' nummark ', text)

    # 3. Punctuation removal
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    # 4. Tokenization & Stopwords removal
    tokens = text.split()
    stemmer = SimplePorterStemmer()
    
    cleaned_tokens = []
    for token in tokens:
        if token not in STOPWORDS and len(token) > 1:
            if stem:
                cleaned_tokens.append(stemmer.stem(token))
            else:
                cleaned_tokens.append(token)

    return " ".join(cleaned_tokens)

# Demonstration on raw text
sample_raw = "CONGRATULATIONS! You have won £1,000 cash! Call 0800123456 or visit http://win.com to claim now!"
sample_cleaned = clean_text(sample_raw)

print("--- Preprocessing Demonstration ---")
print(f"Raw Input:      {sample_raw}")
print(f"Cleaned Output: {sample_cleaned}")
""")

# -------------------------------------------------------------
# Cell 9: Applying Preprocessing
# -------------------------------------------------------------
add_md("""### Applying Preprocessing Across the Dataset
We encode the target variable (`ham` $\\rightarrow 0$, `spam` $\\rightarrow 1$) and clean all 5,574 messages.
""")

add_code("""# Target encoding
df['target'] = df['label'].str.lower().map({'ham': 0, 'spam': 1})

# Clean all messages
print("Applying cleaning pipeline across all messages...")
df['cleaned_message'] = df['message'].apply(clean_text)

# Drop any messages that became empty after cleaning
df_clean = df[df['cleaned_message'].str.len() > 0].copy().reset_index(drop=True)
print(f"Preprocessing complete! Usable messages: {len(df_clean)}")
display(df_clean[['label', 'message', 'cleaned_message']].head())
""")

# -------------------------------------------------------------
# Cell 10 & 11: Feature Extraction via TF-IDF
# -------------------------------------------------------------
add_md("""## 4. Feature Extraction via TF-IDF Vectorization
Machine learning models cannot read raw strings; they require a numerical feature matrix.

### The TF-IDF Formulation:
$$\\text{TF}(t, d) = \\frac{\\text{Frequency of term } t \\text{ in document } d}{\\text{Total terms in document } d}$$

$$\\text{IDF}(t, D) = \\log\\left(\\frac{1 + |D|}{1 + \\text{DF}(t)}\\right) + 1$$

$$\\text{TF-IDF}(t, d, D) = \\text{TF}(t, d) \\times \\text{IDF}(t, D)$$

- **Term Frequency (TF):** Measures how central a word is to an individual message.
- **Inverse Document Frequency (IDF):** Discounts terms that appear in nearly all messages (e.g. conversational greetings) and elevates distinctive keywords (e.g. `claim`, `prize`, `urgent`).
- **N-Grams:** We extract both single words (`claim`) and two-word phrases (`claim prize`, `cash reward`) via `ngram_range=(1, 2)`.
""")

add_code("""# Stratified 80/20 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df_clean['cleaned_message'],
    df_clean['target'],
    test_size=0.20,
    random_state=42,
    stratify=df_clean['target']
)

print(f"Train samples: {len(X_train)} ({y_train.value_counts()[0]} ham, {y_train.value_counts()[1]} spam)")
print(f"Test samples:  {len(X_test)} ({y_test.value_counts()[0]} ham, {y_test.value_counts()[1]} spam)")

# Initialize TfidfVectorizer
vectorizer = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2
)

# Fit on training data ONLY, then transform both sets
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

feature_names = np.array(vectorizer.get_feature_names_out())
print(f"TF-IDF Feature Matrix Shape: {X_train_vec.shape}")
print(f"Vocabulary Dimension: {len(feature_names)} features")
print(f"Matrix Sparsity: {(1.0 - (X_train_vec.nnz / (X_train_vec.shape[0] * X_train_vec.shape[1]))) * 100:.2f}%")
""")

# -------------------------------------------------------------
# Cell 12 & 13: Model Training
# -------------------------------------------------------------
add_md("""## 5. Model Training & Comparison
We train three complementary machine learning classifiers:
1. **Multinomial Naive Bayes (`MultinomialNB`)**:
   Computes word conditional probabilities $P(w_i | c)$ with Laplace smoothing ($\\alpha = 1.0$) to avoid zero probabilities for unseen test words.
2. **Regularized Logistic Regression (`LogisticRegression`)**:
   A linear classifier ($L_2$ penalty, $C=2.0$) finding the optimal hyper-plane separating Spam from Ham.
3. **Random Forest (`RandomForestClassifier`)**:
   An ensemble of 100 decision trees testing non-linear feature interactions.
""")

add_code("""models = {
    "Multinomial Naive Bayes": MultinomialNB(alpha=1.0),
    "Logistic Regression": LogisticRegression(C=2.0, max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=25, random_state=42)
}

trained_models = {}
eval_results = {}

for name, clf in models.items():
    print(f"Training {name}...")
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    eval_results[name] = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "Confusion Matrix": cm
    }
    trained_models[name] = clf

# Comparison summary
summary_df = pd.DataFrame([
    {
        "Model": k,
        "Accuracy": f"{v['Accuracy']*100:.2f}%",
        "Precision (Spam)": f"{v['Precision']*100:.2f}%",
        "Recall (Spam)": f"{v['Recall']*100:.2f}%",
        "F1-Score": f"{v['F1-Score']:.4f}",
        "False Positives (FP)": v['Confusion Matrix'][0][1],
        "False Negatives (FN)": v['Confusion Matrix'][1][0]
    }
    for k, v in eval_results.items()
])
display(summary_df)
""")

# -------------------------------------------------------------
# Cell 14: Evaluation & Confusion Matrices
# -------------------------------------------------------------
add_md("""## 6. Performance Evaluation & Confusion Matrices

### The Core Spam Metric: Precision vs. Recall
$$\\text{Precision} = \\frac{\\text{TP}}{\\text{TP} + \\text{FP}} \\quad \\text{vs.} \\quad \\text{Recall} = \\frac{\\text{TP}}{\\text{TP} + \\text{FN}}$$

- **False Positive (FP):** Legitimate email sent to spam $\\rightarrow$ **Catastrophic!** User misses a bank security code or flight confirmation.
- **False Negative (FN):** Spam email sent to inbox $\\rightarrow$ **Minor inconvenience.** User simply swipes delete.

Notice how **Multinomial Naive Bayes** produced only **1 False Positive** out of 965 legitimate test messages, while **Random Forest** produced **0 False Positives**!
""")

add_code("""fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

for idx, (m_name, res) in enumerate(eval_results.items()):
    cm = res["Confusion Matrix"]
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Predicted Ham", "Predicted Spam"],
        yticklabels=["Actual Ham", "Actual Spam"],
        ax=axes[idx], annot_kws={"size": 13, "weight": "bold"}
    )
    axes[idx].set_title(m_name, fontsize=12, fontweight="bold", pad=10)

plt.tight_layout()
plt.show()
""")

# -------------------------------------------------------------
# Cell 15: Interpretability & Top Features
# -------------------------------------------------------------
add_md("""## 7. Model Interpretability: Top Predictive Keywords
In Naive Bayes, each word feature $w$ has conditional log-probabilities $\\log P(w | \\text{spam})$ and $\\log P(w | \\text{ham})$.
The difference:
$$\\Delta(w) = \\log P(w \\mid \\text{spam}) - \\log P(w \\mid \\text{ham})$$
quantifies the log-odds advantage of that word for spam vs. ham.
""")

add_code("""nb = trained_models["Multinomial Naive Bayes"]
nb_diff = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]

# Top 20 spam features
top_spam_idx = np.argsort(nb_diff)[-20:][::-1]
df_top_spam = pd.DataFrame({
    'Keyword': feature_names[top_spam_idx],
    'Log-Odds Advantage': nb_diff[top_spam_idx]
})

# Top 20 ham features
top_ham_idx = np.argsort(nb_diff)[:20]
df_top_ham = pd.DataFrame({
    'Keyword': feature_names[top_ham_idx],
    'Log-Odds Advantage': np.abs(nb_diff[top_ham_idx])
})

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.barplot(data=df_top_spam, x='Log-Odds Advantage', y='Keyword', color='#EF4444', ax=axes[0])
axes[0].set_title("Top 20 Spam-Predictive Keywords", fontsize=13, fontweight='bold', color='#EF4444')

sns.barplot(data=df_top_ham, x='Log-Odds Advantage', y='Keyword', color='#10B981', ax=axes[1])
axes[1].set_title("Top 20 Ham-Predictive Keywords", fontsize=13, fontweight='bold', color='#10B981')

plt.tight_layout()
plt.show()
""")

# -------------------------------------------------------------
# Cell 16 & 17: Live Prediction with Token Attribution
# -------------------------------------------------------------
add_md("""## 8. Live Interactive Inference & Word-Level Attribution
This function takes any raw message, cleans it through our pipeline, computes the prediction probability, and breaks down which tokens pushed the score toward Spam or Ham.
""")

add_code("""def test_message_classification(message: str, model_name: str = "Multinomial Naive Bayes"):
    \"\"\"Predicts whether a message is Ham or Spam and prints token-level attribution.\"\"\"
    model = trained_models[model_name]
    
    # 1. Clean message
    cleaned = clean_text(message)
    if not cleaned.strip():
        print("Message is empty after preprocessing!")
        return
        
    # 2. Vectorize
    vec = vectorizer.transform([cleaned])
    
    # 3. Predict probabilities
    probs = model.predict_proba(vec)[0]
    p_ham, p_spam = probs[0], probs[1]
    verdict = "🚨 SPAM" if p_spam >= 0.5 else "✅ HAM"
    
    print("=" * 65)
    print(f"INPUT:   {message}")
    print(f"CLEANED: {cleaned}")
    print(f"VERDICT: {verdict} (Spam Prob: {p_spam*100:.2f}%, Ham Prob: {p_ham*100:.2f}%)")
    
    # 4. Token Attribution
    vocab_lookup = {name: idx for idx, name in enumerate(feature_names)}
    print("\\nTOKEN ATTRIBUTION BREAKDOWN:")
    for word in cleaned.split():
        if word in vocab_lookup:
            f_idx = vocab_lookup[word]
            score = nb_diff[f_idx]
            tag = "[SPAM SIGNAL]" if score > 0.5 else ("[HAM SIGNAL]" if score < -0.5 else "[NEUTRAL]")
            print(f"  • {word:<15} score: {score:+.3f}  {tag}")
        else:
            print(f"  • {word:<15} [OUT OF VOCABULARY]")
    print("=" * 65 + "\\n")

# Test Case 1: Urgent lottery scam
test_message_classification(
    "URGENT! You have won a £2,000 cash prize. Call 0800123456 now or text CLAIM to 88888!"
)

# Test Case 2: Normal conversational chat
test_message_classification(
    "Hey dad, are we still having dinner at home tonight? Let me know when you leave the office."
)

# Test Case 3: Testing the ambiguous word 'free'
test_message_classification(
    "Are you free tonight? Let's grab some coffee and chat about the project."
)
""")

# -------------------------------------------------------------
# Save to disk
# -------------------------------------------------------------
output_path = os.path.join(
    os.path.dirname(__file__), "notebooks", "sms_spam_classifier_walkthrough.ipynb"
)
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Successfully generated Jupyter Notebook at: {output_path}")
print(f"Total cells: {len(notebook['cells'])}")
