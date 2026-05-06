# AI model for SMS Spam Detection & Classification Pipeline

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-blue?style=for-the-badge)

## 📋 Project Overview

Unsolicited and malicious SMS payloads (Smishing) present a critical threat vector for social engineering and credential harvesting. Traditional static blocklists fail to intercept dynamically generated phishing lures. 

This project engineers a probabilistic machine learning pipeline designed to classify SMS messages as benign (Ham) or malicious (Spam). Built with Python and Scikit-Learn, the architecture focuses on robust feature extraction, rigorous hyperparameter tuning, and CI/CD-ready artifact serialization.

## Architecture

The system is built on a streamlined data-to-deployment workflow ensuring mathematical consistency and preventing data leakage.

* **Dataset:** 2011 UCI SMS Spam Collection (5,572 annotated samples).
* **NLP Preprocessing:** NLTK-driven pipeline executing tokenization, stop-word removal, and Porter Stemming.
* **Feature Extraction:** `CountVectorizer` mapping text to sparse matrices utilizing unigrams and bigrams (`ngram_range=(1,2)`).
* **Classification Engine:** `MultinomialNB` (Naive Bayes) optimized for discrete text frequency data.
* **Serialization:** Model exported via `joblib` for immediate API integration.

## Key Engineering Decisions

To elevate this model from a basic script to a production-ready security tool, several strict engineering constraints were applied:

1. **Preserving High-Signal Indicators:** Standard NLP pipelines strip all punctuation. This pipeline utilizes custom Regular Expressions to explicitly preserve currency (`$`) and exclamation (`!`) symbols. In an AppSec context, these characters act as high-value heuristic flags for urgent financial phishing.
2. **Preventing Data Leakage:** The vectorizer and classifier are bound together using an `sklearn.pipeline.Pipeline`. This ensures the exact same transformation logic applied during training is identically enforced during testing and production inference.
3. **Hyperparameter Optimization:** Default configurations are rarely optimal. The model employs `GridSearchCV` with 5-fold cross-validation to mathematically isolate the optimal Laplace Smoothing parameter (`alpha: 0.25`), balancing the bias-variance tradeoff.
4. **Deployment Readiness:** The final pipeline is serialized into a standalone `.joblib` binary. This decouples the training logic from the application logic, allowing the model to be instantly loaded into a DevSecOps environment or a FastAPI endpoint without retraining.

## Performance & Proof of Concept

The model was evaluated against a synthetic set of unseen, real-world inputs to test baseline accuracy and predictive confidence.

*(Insert Screenshot: Terminal output showing the optimal alpha and the test message probability scores)*

**Inference Results:**
* Successfully classified aggressive phishing attempts (Walmart gift card lures) with **1.00 Spam Probability**.
* Accurately filtered routine benign communications (appointment reminders) with **1.00 Not-Spam Probability**.
* Demonstrated nuanced context awareness by flagging simulated account compromise alerts as Spam with a **0.96 Probability**.

## ⚙️ Quick Start & Reproducibility

**1. Clone the repository and install dependencies:**
```bash
git clone [https://github.com/yourusername/sms-spam-detector.git](https://github.com/yourusername/sms-spam-detector.git)
cd sms-spam-detector
pip install -r requirements.txt
