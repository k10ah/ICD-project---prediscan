## Project Evolution

### Version 1: Disease Category Prediction

Objective:
Predict broad disease categories from clinical descriptions.

Classes:
78 Categories

Examples:

* Cardiovascular Disease
* Cancer
* Respiratory Disease
* Digestive Disease
* Infectious Disease

This task is a coarse-grained classification problem and achieved higher accuracy due to fewer target classes.

---

### Version 2: Exact ICD Code Prediction

Objective:
Predict the exact ICD code used during medical insurance claim processing.

Dataset:

* Records Used: 14,492
* ICD Classes: 878

Examples:

Clinical Text:
"Persistent hyperglycemia with elevated HbA1c"

Possible ICD Codes:

* E11.9
* E11.65
* E10.9
* R73.9

Unlike category prediction, the model must identify the precise ICD code among 878 possible classes.

---

## Current Best Results

Model:
LinearSVC + TF-IDF + Structured Features

Performance:

* Accuracy: 39.46%
* Top-3 Accuracy: 55.47%
* Top-5 Accuracy: 63.61%

---

## Why Accuracy Appears Lower

The previous category prediction model classified a small number of broad disease groups.

The current system performs exact ICD prediction across 878 classes, making the problem significantly more difficult.

Therefore, a 39.46% exact ICD prediction accuracy represents a substantially more challenging task than category classification.

---

## Future Work

* PubMedBERT
* ClinicalBERT
* Fine-Tuned Transformer Models
* Top-K ICD Recommendation System
* Real-Time Insurance Claim Assistance
