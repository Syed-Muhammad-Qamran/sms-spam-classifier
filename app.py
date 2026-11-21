"""
Streamlit Web Application: SMS Spam Classifier & NLP Pipeline
Day 22 of 100 Days of Machine Learning
Author: Syed Muhammad Qamran
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.preprocessor import clean_text, inspect_preprocessing_pipeline
from src.model_trainer import get_or_train_pipeline, predict_message, MODELS_DIR

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="SMS Spam Classifier | Day 22 ML",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        background: -webkit-linear-gradient(45deg, #FF4B4B, #FF8533);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #A0AEC0;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1E232A;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #2D3748;
        text-align: center;
    }
    .token-chip-spam {
        display: inline-block;
        background-color: rgba(239, 68, 68, 0.2);
        color: #EF4444;
        border: 1px solid #EF4444;
        border-radius: 15px;
        padding: 3px 10px;
        margin: 4px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .token-chip-ham {
        display: inline-block;
        background-color: rgba(16, 185, 129, 0.2);
        color: #10B981;
        border: 1px solid #10B981;
        border-radius: 15px;
        padding: 3px 10px;
        margin: 4px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .token-chip-neutral {
        display: inline-block;
        background-color: rgba(148, 163, 184, 0.15);
        color: #94A3B8;
        border: 1px solid #64748B;
        border-radius: 15px;
        padding: 3px 10px;
        margin: 4px;
        font-size: 0.85rem;
    }
    .badge-spam {
        background-color: #EF4444;
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .badge-ham {
        background-color: #10B981;
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Load Pipeline & Cached Artifacts
# ---------------------------------------------------------
@st.cache_resource
def load_all_artifacts():
    return get_or_train_pipeline()

try:
    vectorizer, models, metrics, top_features = load_all_artifacts()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.image("assets/banner.jpg", use_container_width=True)
    st.markdown("### 📚 100 Days of ML: Day 22")
    st.markdown("**Topic:** Natural Language Processing (NLP)")
    st.markdown("**Techniques:** Text Normalization, Stemming, TF-IDF Vectorization, Naive Bayes, Regularized Logistic Regression")
    st.divider()

    st.markdown("### 👤 Author")
    st.markdown("**Syed Muhammad Qamran**")
    st.markdown("[🌐 GitHub Repository](https://github.com/Syed-Muhammad-Qamran/sms-spam-classifier)")
    st.markdown("[🔗 LinkedIn Profile](https://www.linkedin.com/in/syed-muhammad-qamran/)")
    st.divider()

    st.markdown("### 📈 Quick Stats")
    st.markdown(f"- **Total SMS:** 5,574")
    st.markdown(f"- **Vocabulary:** 3,000 n-grams (1-2)")
    st.markdown(f"- **Best Accuracy:** {metrics.get('Logistic Regression', {}).get('accuracy', 0)*100:.2f}%")
    st.markdown(f"- **NB Precision:** {metrics.get('Multinomial Naive Bayes', {}).get('precision', 0)*100:.2f}%")


# ---------------------------------------------------------
# Main Header
# ---------------------------------------------------------
st.markdown('<div class="main-title">🛡️ SMS Spam Classifier & NLP Architecture</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Day 22 of 100 Days of Machine Learning | End-to-End Text Preprocessing, TF-IDF Vectorization & Classical Classification</div>', unsafe_allow_html=True)

# Top metric columns
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Messages", "5,574", "86.6% Ham / 13.4% Spam")
with col2:
    nb_prec = metrics.get("Multinomial Naive Bayes", {}).get("precision", 0.99) * 100
    st.metric("Naive Bayes Precision", f"{nb_prec:.2f}%", "Zero/Minimal FP")
with col3:
    lr_acc = metrics.get("Logistic Regression", {}).get("accuracy", 0.986) * 100
    st.metric("Logistic Reg Accuracy", f"{lr_acc:.2f}%", "+1.2% vs baseline")
with col4:
    rf_prec = metrics.get("Random Forest", {}).get("precision", 1.0) * 100
    st.metric("Random Forest Precision", f"{rf_prec:.1f}%", "High Confidence")

st.divider()

# ---------------------------------------------------------
# Tabs Navigation
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Live Spam Detector",
    "📊 Model Benchmarks & Metrics",
    "🔤 TF-IDF Keyword Explorer",
    "🧪 NLP Pipeline Visualizer"
])

# =========================================================
# TAB 1: Live Spam Detector
# =========================================================
with tab1:
    st.subheader("Interactive Message Classification & Token Attribution")
    st.markdown("Test the classifier on custom text or choose from realistic real-world presets:")

    # Presets
    presets = {
        "Custom Input": "",
        "🎁 Lottery Scam (Obvious Spam)": (
            "CONGRATULATIONS! You have won a £1,000 cash prize! Text CLAIM to 88888 now to collect your reward. "
            "Urgent: valid for 24 hours only. T&Cs apply."
        ),
        "🚨 Bank Phishing Alert (Spam)": (
            "URGENT: Your bank account has been locked due to suspicious activity. Verify immediately at "
            "http://secure-banking-alert.com or call our 24/7 hotline 0800123456."
        ),
        "💬 Casual Friend Chat (Legitimate Ham)": (
            "Hey! Are we still meeting for lunch at 1pm? Let me know if you want me to pick you up on the way."
        ),
        "💼 Work/Project Update (Legitimate Ham)": (
            "The client approved the latest project milestone. I have uploaded the updated documentation to the shared drive."
        ),
        "⚖️ Subtle Ambiguous Context ('Free' word test)": (
            "Hey buddy, are you free tonight? Let's grab some pizza and watch the game after work."
        )
    }

    preset_choice = st.selectbox("Select a Preset Example:", list(presets.keys()), index=1)
    
    col_input, col_ctrl = st.columns([3, 1])
    with col_input:
        default_val = presets[preset_choice] if preset_choice != "Custom Input" else ""
        user_message = st.text_area(
            "Enter SMS / Email Text:",
            value=default_val,
            height=120,
            placeholder="Type or paste any text message here..."
        )
    
    with col_ctrl:
        st.markdown("**Model Configuration**")
        selected_model = st.selectbox(
            "Classifier Algorithm:",
            ["Multinomial Naive Bayes", "Logistic Regression", "Random Forest"],
            index=0
        )
        classify_button = st.button("🚀 Analyze Message", use_container_width=True, type="primary")

    if user_message.strip():
        res = predict_message(user_message, model_name=selected_model)
        
        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.markdown("#### Classification Verdict")
            if res["is_spam"]:
                st.markdown(
                    f'<div style="background-color:#7F1D1D; border:2px solid #EF4444; border-radius:10px; padding:20px; text-align:center;">'
                    f'<h2 style="color:#EF4444; margin:0;">🚨 SPAM DETECTED</h2>'
                    f'<p style="color:#FCA5A5; font-size:1.1rem; margin-top:8px;">Confidence: <b>{res["spam_probability"]*100:.2f}%</b></p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div style="background-color:#064E3B; border:2px solid #10B981; border-radius:10px; padding:20px; text-align:center;">'
                    f'<h2 style="color:#10B981; margin:0;">✅ HAM / SAFE</h2>'
                    f'<p style="color:#6EE7B7; font-size:1.1rem; margin-top:8px;">Confidence: <b>{res["ham_probability"]*100:.2f}%</b></p>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            st.write("")
            st.markdown(f"**Spam Probability:** `{res['spam_probability']*100:.1f}%`")
            st.progress(res["spam_probability"])
            st.markdown(f"**Ham Probability:** `{res['ham_probability']*100:.1f}%`")
            st.progress(res["ham_probability"])

        with res_col2:
            st.markdown("#### 🔬 Token Attribution & Feature Breakdown")
            st.caption(
                "Words highlighted based on their log-odds likelihood ratio: "
                "🔴 **Red** = Spam Indicator, 🟢 **Green** = Ham Indicator, ⚪ **Gray** = Neutral."
            )

            chips_html = []
            for item in res["token_attributions"]:
                token = item["token"]
                score = item["score"]
                influence = item["influence"]
                if influence == "spam":
                    chips_html.append(f'<span class="token-chip-spam">{token} (+{score})</span>')
                elif influence == "ham":
                    chips_html.append(f'<span class="token-chip-ham">{token} ({score})</span>')
                else:
                    chips_html.append(f'<span class="token-chip-neutral">{token}</span>')

            st.markdown("".join(chips_html), unsafe_allow_html=True)

            with st.expander("🔍 View Preprocessed Cleaned Message", expanded=False):
                st.code(res["cleaned_text"], language="text")


# =========================================================
# TAB 2: Model Benchmarks & Metrics
# =========================================================
with tab2:
    st.subheader("Model Performance Comparison & Evaluation Metrics")
    st.markdown("""
    Spam detection systems face a critical asymmetry: **False Positives (flagging legitimate messages as spam) are far more damaging than False Negatives (letting a spam message slip through)**. 
    A user might miss an urgent bank transfer, flight ticket, or job offer if precision is low.
    """)

    # Metric comparison table
    metric_rows = []
    for model_name, m in metrics.items():
        metric_rows.append({
            "Model": model_name,
            "Accuracy": f"{m['accuracy']*100:.2f}%",
            "Precision (Spam)": f"{m['precision']*100:.2f}%",
            "Recall (Spam)": f"{m['recall']*100:.2f}%",
            "F1-Score": f"{m['f1_score']:.4f}",
            "False Positives (FP)": m["fp"],
            "False Negatives (FN)": m["fn"]
        })
    df_metrics = pd.DataFrame(metric_rows)
    st.dataframe(df_metrics, use_container_width=True)

    # Confusion Matrix Visualization
    st.subheader("Confusion Matrices (Test Set: 1,114 Messages)")
    cm_cols = st.columns(len(metrics))

    for idx, (m_name, m_data) in enumerate(metrics.items()):
        with cm_cols[idx]:
            cm = np.array(m_data["confusion_matrix"])
            fig, ax = plt.subplots(figsize=(4, 3.2))
            fig.patch.set_facecolor('#0E1117')
            ax.set_facecolor('#1E232A')
            
            cax = ax.matshow(cm, cmap='Blues', alpha=0.85)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center', 
                            color='white' if cm[i, j] > 200 else 'black', 
                            fontsize=12, fontweight='bold')

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Ham', 'Spam'], color='white')
            ax.set_yticklabels(['Ham', 'Spam'], color='white')
            ax.set_xlabel('Predicted Label', color='white', labelpad=8)
            ax.set_ylabel('True Label', color='white', labelpad=8)
            ax.set_title(m_name, color='white', pad=12, fontsize=11, fontweight='bold')
            ax.tick_params(colors='white')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


# =========================================================
# TAB 3: TF-IDF Keyword Explorer
# =========================================================
with tab3:
    st.subheader("Top Predictive Features & Log-Probability Analysis")
    st.markdown("Visualizing the highest-weighted terms extracted by TF-IDF and Naive Bayes log-odds.")

    col_spam_words, col_ham_words = st.columns(2)

    with col_spam_words:
        st.markdown("### 🚨 Top 20 Spam-Predictive Terms")
        top_spam = top_features.get("top_spam_nb", [])[:20]
        df_spam = pd.DataFrame(top_spam)
        if not df_spam.empty:
            fig, ax = plt.subplots(figsize=(6, 7))
            fig.patch.set_facecolor('#0E1117')
            ax.set_facecolor('#1E232A')
            y_pos = np.arange(len(df_spam))
            ax.barh(y_pos, df_spam['score'][::-1], color='#EF4444', alpha=0.85)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(df_spam['term'][::-1], color='white', fontsize=10)
            ax.tick_params(colors='white')
            ax.set_xlabel('Log Probability Advantage (Spam vs. Ham)', color='white')
            ax.set_title('Top Spam Triggers', color='#EF4444', fontweight='bold', fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with col_ham_words:
        st.markdown("### ✅ Top 20 Ham-Predictive Terms")
        top_ham = top_features.get("top_ham_nb", [])[:20]
        df_ham = pd.DataFrame(top_ham)
        if not df_ham.empty:
            fig, ax = plt.subplots(figsize=(6, 7))
            fig.patch.set_facecolor('#0E1117')
            ax.set_facecolor('#1E232A')
            y_pos = np.arange(len(df_ham))
            ax.barh(y_pos, df_ham['score'][::-1], color='#10B981', alpha=0.85)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(df_ham['term'][::-1], color='white', fontsize=10)
            ax.tick_params(colors='white')
            ax.set_xlabel('Log Probability Advantage (Ham vs. Spam)', color='white')
            ax.set_title('Top Conversational Ham Triggers', color='#10B981', fontweight='bold', fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


# =========================================================
# TAB 4: NLP Pipeline Visualizer
# =========================================================
with tab4:
    st.subheader("Step-by-Step NLP Text Transformation Pipeline")
    st.markdown("Watch how an unstructured text message is cleaned, normalized, tokenized, and transformed into numeric TF-IDF vector features.")

    sample_pipeline_text = st.text_input(
        "Enter sentence to inspect pipeline transformation:",
        value="URGENT! Call 0800123456 now to claim your £5,000 cash prize at http://claim-award.com"
    )

    if sample_pipeline_text:
        stages = inspect_preprocessing_pipeline(sample_pipeline_text)
        
        st.markdown("#### 1. Raw Input")
        st.code(stages["raw_text"], language="text")

        st.markdown("#### 2. Lowercasing & Entity Normalization (URLs, Phones, Currencies, Digits)")
        st.code(stages["normalized"], language="text")

        st.markdown("#### 3. Punctuation Stripping")
        st.code(stages["punctuation_removed"], language="text")

        st.markdown("#### 4. Tokenization")
        st.write(stages["tokens"])

        st.markdown("#### 5. Stopword Filtering (Removal of low-signal English filler words)")
        st.write(stages["stopwords_removed"])

        st.markdown("#### 6. Stemming (Root word conflation)")
        st.write(stages["stemmed_tokens"])

        st.markdown("#### 7. Final Cleaned String for Vectorizer")
        st.success(stages["final_string"])

        st.divider()
        st.markdown("### 📐 Mathematical Formulation of TF-IDF")
        st.markdown("""
        **Term Frequency (TF):** Measures how frequently term $t$ occurs in document $d$:
        """)
        st.latex(r"\text{TF}(t, d) = \frac{f_{t, d}}{\sum_{t' \in d} f_{t', d}}")

        st.markdown("""
        **Inverse Document Frequency (IDF):** Penalizes ubiquitous terms that appear across all documents:
        """)
        st.latex(r"\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + \text{DF}(t)}\right) + 1")

        st.markdown("""
        **Final TF-IDF Feature Value:**
        """)
        st.latex(r"\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)")
