"""
Text Preprocessing Module for NLP & SMS Spam Classification
Day 22 of 100 Days of Machine Learning

Implements text normalization (URLs, currencies, phones, numbers),
tokenization, stopword removal, and Porter stemming.
"""

import re
import string
from typing import List, Dict, Any

# Curated high-frequency English stopwords (standard NLTK/scikit-learn subset)
# Self-contained without needing external corpus downloads
STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'aren',
    "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but',
    'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does',
    'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further',
    'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', 'her',
    'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'isn', "isn't",
    'it', "it's", 'its', 'itself', 'just', 'll', 'm', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn',
    "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or',
    'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 'she',
    "she's", 'should', 'shouldn', "shouldn't", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the',
    'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through',
    'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', 'were', 'weren',
    "weren't", 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won',
    "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours',
    'yourself', 'yourselves'
}


class SimplePorterStemmer:
    """
    Lightweight, deterministic suffix stripper inspired by Porter's algorithm.
    Strips common inflections (-ing, -ed, -es, -s, -ly, -ment, -tion) and collapses
    double consonant endings (e.g., 'running' -> 'run', 'winner' -> 'win').
    """
    @staticmethod
    def stem(word: str) -> str:
        word = word.lower().strip()
        if len(word) <= 3:
            return word
        
        # Step 1: Plurals and past tense
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

        # Step 1b: Double consonant reduction (e.g. runn -> run, winn -> win)
        if len(word) >= 3 and word[-1] == word[-2] and word[-1] in 'bdfgmnprt':
            word = word[:-1]

        # Step 2: Derivational suffixes
        suffixes = (
            ('ational', 'ate'),
            ('tional', 'tion'),
            ('ization', 'ize'),
            ('fulness', 'ful'),
            ('ousness', 'ous'),
            ('iveness', 'ive'),
            ('ment', ''),
            ('able', ''),
            ('ible', ''),
            ('ful', ''),
            ('ous', ''),
            ('ive', ''),
            ('ize', ''),
            ('ly', ''),
            ('e', ''),  # trailing e reduction (simple -> simpl, free remains free because <=4)
        )
        for old, new in suffixes:
            if word.endswith(old) and len(word) - len(old) >= 3:
                word = word[:-len(old)] + new
                break
                
        return word


def clean_text(text: str, stem: bool = True) -> str:
    """
    Full text preprocessing pipeline for SMS spam classification:
    1. Lowercasing
    2. Normalize URLs, phone numbers, currencies, digits
    3. Punctuation removal
    4. Stopwords removal
    5. Stemming (optional)
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # 1. Normalize URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' urladdr ', text)

    # 2. Normalize email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' emailaddr ', text)

    # 3. Normalize currencies (e.g. £100, $500, 5000$)
    text = re.sub(r'[£$€]\s*\d+(?:\.\d+)?', ' moneynum ', text)
    text = re.sub(r'\d+(?:\.\d+)?\s*[£$€]', ' moneynum ', text)

    # 4. Normalize phone/shortcode numbers (e.g. 0800, 88888, 11-digit numbers)
    text = re.sub(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3,5}\)?[-.\s]?\d{3,5}[-.\s]?\d{3,5}\b', ' phonenum ', text)

    # 5. Normalize generic numbers
    text = re.sub(r'\b\d+\b', ' nummark ', text)

    # 6. Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    # 7. Tokenize and filter
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


def inspect_preprocessing_pipeline(text: str) -> Dict[str, Any]:
    """
    Returns step-by-step intermediate representations of text
    for educational inspection and Streamlit pipeline visualizer.
    """
    # Step 1: Lowercase
    step1_lower = text.lower()

    # Step 2: Normalization
    norm_urls = re.sub(r'https?://\S+|www\.\S+', ' [URL] ', step1_lower)
    norm_emails = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' [EMAIL] ', norm_urls)
    norm_money = re.sub(r'[£$€]\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?\s*[£$€]', ' [MONEY] ', norm_emails)
    norm_phone = re.sub(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3,5}\)?[-.\s]?\d{3,5}[-.\s]?\d{3,5}\b', ' [PHONE] ', norm_money)
    norm_nums = re.sub(r'\b\d+\b', ' [NUM] ', norm_phone)

    # Step 3: Remove punctuation
    translator = str.maketrans('', '', string.punctuation.replace('[', '').replace(']', ''))
    step3_punct = norm_nums.translate(translator)

    # Step 4: Raw tokens
    raw_tokens = step3_punct.split()

    # Step 5: Stopwords removed
    filtered_tokens = [t for t in raw_tokens if t not in STOPWORDS and len(t) > 1]

    # Step 6: Stemmed
    stemmer = SimplePorterStemmer()
    stemmed_tokens = [stemmer.stem(t) if not (t.startswith('[') and t.endswith(']')) else t for t in filtered_tokens]

    return {
        "raw_text": text,
        "lowercased": step1_lower,
        "normalized": norm_nums,
        "punctuation_removed": step3_punct,
        "tokens": raw_tokens,
        "stopwords_removed": filtered_tokens,
        "stemmed_tokens": stemmed_tokens,
        "final_string": " ".join(stemmed_tokens)
    }
