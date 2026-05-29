# ==========================================================
# ICD HEALTHCARE CLASSIFICATION PIPELINE v3
# Optimized for accuracy
# ==========================================================

import pandas as pd
import numpy as np
import scipy.sparse
import re
import warnings
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

print("="*70)
print("ICD HEALTHCARE NLP PIPELINE STARTED")
print("="*70)

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv("icdfinal.csv")

print("Dataset Shape:", df.shape)

# ==========================================================
# PREPROCESSING
# ==========================================================

critical_cols = [
    'Description',
    'ICD_Code',
    'Category',
    'Risk_Level',
    'Age_Group',
    'Age_Midpoint',
    'Sex'
]

df = df.dropna(subset=critical_cols).reset_index(drop=True)

# ==========================================================
# BETTER COMBINED TEXT
# ==========================================================

df['Combined_Text'] = (
    'Diagnosis ' + df['Description'].astype(str) +
    ' ICD ' + df['ICD_Code'].astype(str) +
    ' Risk ' + df['Risk_Level'].astype(str) +
    ' Age ' + df['Age_Group'].astype(str) +
    ' Sex ' + df['Sex'].astype(str)
)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['Combined_Text_Clean'] = df['Combined_Text'].apply(clean_text)

# ==========================================================
# FILTER RARE CLASSES
# ==========================================================

vc = df['Category'].value_counts()

valid_categories = vc[vc >= 50].index

df = df[df['Category'].isin(valid_categories)].reset_index(drop=True)

print("Rows after filtering:", len(df))
print("Classes remaining:", df['Category'].nunique())

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

cat_cols = ['Risk_Level', 'Age_Group', 'Sex']
num_cols = ['Age_Midpoint']

if 'Risk_Score' in df.columns:
    num_cols.append('Risk_Score')

le_dict = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

le_target = LabelEncoder()
y = le_target.fit_transform(df['Category'])

# ==========================================================
# SCALE NUMERICAL FEATURES
# ==========================================================

scaler = StandardScaler()

df[num_cols] = scaler.fit_transform(df[num_cols])

# ==========================================================
# TF-IDF
# ==========================================================

tfidf = TfidfVectorizer(
    max_features=20000,
    stop_words='english',
    ngram_range=(1,3),
    min_df=2,
    max_df=0.90,
    sublinear_tf=True
)

X_text = tfidf.fit_transform(df['Combined_Text_Clean'])

# ==========================================================
# STRUCTURED FEATURES
# ==========================================================

struct_cols = [c + '_enc' for c in cat_cols] + num_cols

X_struct = scipy.sparse.csr_matrix(
    df[struct_cols].values.astype(float)
)

X = scipy.sparse.hstack([X_text, X_struct])

print("Final Feature Matrix:", X.shape)

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================================
# LOGISTIC REGRESSION
# ==========================================================

lr = LogisticRegression(
    max_iter=2000,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

print("Training Logistic Regression...")
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

lr_acc = accuracy_score(y_test, y_pred_lr)

# ==========================================================
# XGBOOST
# ==========================================================

xgb = XGBClassifier(
    n_estimators=800,
    max_depth=10,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=2,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=2,
    objective='multi:softprob',
    eval_metric='mlogloss',
    tree_method='hist',
    random_state=42,
    verbosity=0
)

print("Training XGBoost...")
xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)

xgb_acc = accuracy_score(y_test, y_pred_xgb)

# ==========================================================
# MODEL COMPARISON
# ==========================================================

print("\nLogistic Regression Accuracy :", round(lr_acc*100,2), "%")
print("XGBoost Accuracy            :", round(xgb_acc*100,2), "%")

best_model = xgb if xgb_acc >= lr_acc else lr
best_pred = y_pred_xgb if xgb_acc >= lr_acc else y_pred_lr

# ==========================================================
# REPORT
# ==========================================================

print("\nClassification Report:\n")
print(classification_report(
    y_test,
    best_pred,
    zero_division=0
))

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(y_test, best_pred)

plt.figure(figsize=(12,10))
sns.heatmap(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# ==========================================================
# SAVE FILES
# ==========================================================

joblib.dump(best_model, 'icd_best_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(le_target, 'label_encoder.pkl')
joblib.dump(le_dict, 'categorical_encoders.pkl')
joblib.dump(scaler, 'numerical_scaler.pkl')

print("\nSaved:")
print("icd_best_model.pkl")
print("tfidf_vectorizer.pkl")
print("label_encoder.pkl")

# ==========================================================
# TEST INFERENCE WITH CONFIDENCE
# ==========================================================

sample_text = "malignant neoplasm of colon high both age 60"

sample_clean = clean_text(sample_text)

sample_vec = tfidf.transform([sample_clean])

sample_struct = scipy.sparse.csr_matrix(
    np.zeros((1, len(struct_cols)))
)

sample_final = scipy.sparse.hstack([
    sample_vec,
    sample_struct
])

pred = best_model.predict(sample_final)[0]

predicted_category = le_target.inverse_transform([pred])[0]

probs = best_model.predict_proba(sample_final)[0]

confidence = np.max(probs) * 100

print("\nSample Prediction:")
print("Input      :", sample_text)
print("Category   :", predicted_category)
print("Confidence :", round(confidence,2), "%")

print("\nPIPELINE COMPLETE")
