# AI-Based Multi-Label ICD Code Prediction System for Healthcare Insurance Claim Processing

## Project Overview

This project is an AI-powered healthcare coding system developed as a Final Year Biomedical Engineering project in collaboration with Prediscan.

The system predicts multiple ICD-10 diagnosis codes from clinical notes using Microsoft's **PubMedBERT** model and assists medical coders by providing Top-5 ICD recommendations, confidence scores, and clinically relevant ICD combinations for insurance claim processing.

---

## Features

- Multi-label ICD-10 code prediction
- PubMedBERT-based Biomedical NLP
- Top-3 & Top-5 ICD Recommendations
- ICD Combination Recommendation Engine
- Insurance Claim Workflow
- Confidence Score for every prediction
- Multi-label Classification (878 ICD Labels)
- Google Colab compatible
- Resume checkpoint support
- Mixed Precision Training
- Layer-wise Learning Rate Decay (LLRD)
- Exponential Moving Average (EMA)
- Early Stopping
- Automatic Best Model Saving

---

## Dataset

- Clinical Records : 23,792
- ICD Labels : 878
- Task : Multi-label ICD Prediction

---

## Technologies Used

### Programming Language
- Python

### Deep Learning
- PyTorch
- Hugging Face Transformers
- PubMedBERT

### Machine Learning
- Scikit-learn
- Pandas
- NumPy

### Development Tools
- Google Colab
- Google Drive
- GitHub

---

## Project Structure

```
ICD-project---prediscan/
│
├── README.md
├── requirements.txt
├── demo.py
├── icdmf.py
├── sample_input.txt
├── model_download.txt
│
├── tokenizer/
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   └── vocab.txt
│
├── best_model/
│   └── config.json
│
└── model_files/
    ├── mlb.pkl
    └── combo_engine.pkl
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/Sasivarman1108/ICD-project---prediscan.git
```

Move into the project

```bash
cd ICD-project---prediscan
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Download Trained Model

The trained model exceeds GitHub's file size limit.

Please download the model files from the Google Drive link provided below.

Google Drive:

https://drive.google.com/drive/folders/1buyxwKAhGArglBD-Z_ay83nCw89tAC4s?usp=drive_link

Download the following files:

- best_model_state.pt
- pytorch_model.bin

Place them as follows:

```
best_model_state.pt
↓
model_files/

pytorch_model.bin
↓
best_model/
```

---

## Running the Demo

Run

```bash
python demo.py
```

or in Google Colab

```python
!python demo.py
```

---

## Example Clinical Note

```
Patient is a 60-year-old male with Type 2 Diabetes Mellitus, Hypertension, Chronic Kidney Disease Stage 3, and Hyperlipidemia. He complains of fatigue, increased thirst, frequent urination, and bilateral pedal edema.
```

---

## Example Output

```
Predicted ICD Codes

E11
N18
I10

Top-5 Recommendations

E11
N18
I10
I12
E78

Combination Recommendation

E11 + N18

Insurance Claim Workflow

Medical Coder Review
```

---

## Model Performance

| Metric | Value |
|---------|------:|
| ICD Labels | 878 |
| Dataset Size | 23,792 |
| Micro F1 | 0.165 |
| Macro F1 | 0.223 |
| Top-3 Accuracy | 59.9% |
| Top-5 Accuracy | 62.6% |
| Hamming Loss | 0.015 |

---

## AI Workflow

```
Clinical Note
        │
        ▼
PubMedBERT
        │
        ▼
Multi-label ICD Prediction
        │
        ▼
Top-5 ICD Recommendation
        │
        ▼
Combination Recommendation
        │
        ▼
Medical Coder Review
        │
        ▼
Insurance Claim Processing
