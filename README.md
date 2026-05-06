# AI model for SMS Spam Detection & Classification Pipeline

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-blue?style=for-the-badge)

##  Summary

**The Threat Landscape:** SMS Phishing (Smishing) and social engineering payloads represent a critical vulnerability in modern enterprise perimeters. Attackers leverage dynamically generated texts to bypass static blocklists and harvest credentials.

**Project Objective:** This project builds a probabilistic, automated machine learning defense mechanism designed to classify inbound SMS traffic as benign or malicious with high mathematical confidence.

## Technical Architecture

**Core Stack:** Python 3.10+, Scikit-Learn, NLTK, Joblib.

**Workflow Diagram:**
`Data Ingestion -> NLP Normalization -> Feature Vectorization -> Naive Bayes Classification -> Artifact Serialization`

## Data Engineering & Natural Language Processing

**Dataset Procurement:** The pipeline ingests the 2011 UCI SMS Spam Collection, establishing a foundational baseline of 5,572 annotated messages.

**Security-Focused Heuristics:** Standard NLP cleaning strips all non-alphabetic characters. This pipeline implements custom Regular Expressions to explicitly preserve currency symbols ($) and exclamation marks (!). These characters serve as high-signal indicators of compromise in financial phishing lures.

**Noise Reduction:** The text undergoes rigorous normalization, including tokenization, stop-word removal, and Porter Stemming, isolating the root linguistic patterns required for accurate modeling.

## Mathematical Foundation & Classification

**Feature Extraction:** Unstructured text is mathematically mapped into sparse matrices using a Bag-of-Words approach. The `CountVectorizer` utilizes both unigrams and bigrams to capture local linguistic context.

**The Naive Bayes Engine:** The core classification relies on Bayes' Theorem, calculating the posterior probability of a message being spam based on historical word frequency distributions.

**Hyperparameter Tuning:** A `GridSearchCV` implementation with 5-fold cross-validation mathematically isolates the optimal Laplace Smoothing parameter. This prevents zero-probability crashes when the model encounters unseen vocabulary in production.


## DevSecOps & Deployment Logistics

**Pipeline Binding:** Scikit-Learn's `Pipeline` architecture binds the vectorizer and classifier. This strict sequencing prevents data leakage between the training and inference phases.

**Artifact Serialization:** The fully trained pipeline is decoupled from the development environment using `joblib`. This creates a binary artifact ready for rapid CI/CD integration.

**Remote Validation:** To simulate production deployment, the serialized artifact was successfully transmitted via an HTTP POST request to a remote REST API evaluation server.

## Proof of Concept & Efficacy

The model was validated against a synthetic holdout set, demonstrating exceptional accuracy.

* **High-Confidence Spam Detection:** Correctly classified blatant phishing lures (e.g., fraudulent gift card links) with a 1.00 probability score.
* **Nuanced Threat Recognition:** Successfully flagged simulated account compromise alerts as malicious with a 0.96 probability score.
* **Benign Traffic Filtering:** Accurately identified routine, safe communications with a 1.00 probability of being non-spam.

## Quick Start & Reproducibility

**Prerequisites:** Python 3.10+ and pip.

**Installation:**
```bash
git clone [https://github.com/aashraymt2/sms-spam-detector.git](https://github.com/aashraymt2/sms-spam-detector.git)
cd sms-spam-detector
pip install -r requirements.txt
