"""
Test Suite for SMS Spam Classifier & Preprocessing Pipeline
Day 22 of 100 Days of Machine Learning
"""

import os
import unittest
from src.preprocessor import clean_text, SimplePorterStemmer, inspect_preprocessing_pipeline
from src.model_trainer import get_or_train_pipeline, predict_message


class TestNLPPreprocessor(unittest.TestCase):
    def test_url_normalization(self):
        raw = "Check our website at https://win-free-cash.com for details!"
        cleaned = clean_text(raw, stem=False)
        self.assertIn("urladdr", cleaned)
        self.assertNotIn("https", cleaned)

    def test_currency_normalization(self):
        raw = "You have won £1000 cash or $500 bonus!"
        cleaned = clean_text(raw, stem=False)
        self.assertIn("moneynum", cleaned)

    def test_phone_normalization(self):
        raw = "Call 08001234567 or text to 88888 now!"
        cleaned = clean_text(raw, stem=False)
        self.assertTrue("phonenum" in cleaned or "nummark" in cleaned)

    def test_stopword_removal(self):
        raw = "This is a simple message about the test"
        cleaned = clean_text(raw, stem=False)
        self.assertNotIn("this", cleaned.split())
        self.assertNotIn("is", cleaned.split())
        self.assertNotIn("about", cleaned.split())
        self.assertIn("simpl", clean_text(raw, stem=True).split())

    def test_porter_stemmer(self):
        stemmer = SimplePorterStemmer()
        self.assertEqual(stemmer.stem("running"), "run")
        self.assertTrue(stemmer.stem("congratulations").startswith("congratul"))

    def test_pipeline_inspector(self):
        raw = "URGENT! Call 0800123456 now to claim your £5000 prize at http://claim.com"
        stages = inspect_preprocessing_pipeline(raw)
        self.assertEqual(stages["raw_text"], raw)
        self.assertTrue(len(stages["tokens"]) > 0)
        self.assertTrue(len(stages["final_string"]) > 0)


class TestSpamClassifierInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure models are trained and ready
        get_or_train_pipeline()

    def test_obvious_spam_detection(self):
        spam_msg = (
            "CONGRATULATIONS! You have won a £1,000 cash prize! "
            "Text CLAIM to 88888 now to collect your reward. Urgent valid 24h only."
        )
        res = predict_message(spam_msg, model_name="Multinomial Naive Bayes")
        self.assertEqual(res["label"], "SPAM")
        self.assertGreater(res["spam_probability"], 0.70)
        self.assertTrue(len(res["token_attributions"]) > 0)

    def test_obvious_ham_detection(self):
        ham_msg = "Hey, are you free to grab coffee after work? Let me know if you need anything."
        res = predict_message(ham_msg, model_name="Multinomial Naive Bayes")
        self.assertEqual(res["label"], "HAM")
        self.assertLess(res["spam_probability"], 0.30)

    def test_logistic_regression_prediction(self):
        msg = "URGENT! Your mobile account has £500 balance bonus expiring today. Call hotline now."
        res = predict_message(msg, model_name="Logistic Regression")
        self.assertEqual(res["label"], "SPAM")


if __name__ == "__main__":
    unittest.main()
