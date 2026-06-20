"""
=============================================================================
 ICD CODE PREDICTION — PRODUCTION TRAINING PIPELINE
=============================================================================
Multiclass classification of ICD codes from clinical text + structured
metadata. Designed to run end-to-end on Google Colab (CPU or T4 runtime)
in well under 20 minutes for ~14,500 records / 878 classes.

Pipeline stages:
  1. Load + filter rare classes (>= MIN_SAMPLES_PER_CLASS)
  2. Text engineering -> Final_Text (clean, normalized)
  3. TF-IDF vectorization (word n-grams, 1-3)
  4. Structured feature encoding (LabelEncoder + StandardScaler)
  5. chi2 feature selection on the TF-IDF block (50,000 -> 20,000)
  6. Sparse fusion of selected TF-IDF block + scaled structured block
  7. Stratified train/test split
  8. LinearSVC training (class_weight='balanced')
  9. Evaluation: accuracy, precision, recall, F1, Top-3, Top-5 accuracy
 10. Artifact persistence (model, vectorizer, selector, encoders, scaler)
 11. Feature importance analysis
 12. Inference helper + artifact loader

NOTE ON CHI2 + STANDARDSCALER:
chi2 requires non-negative input. TF-IDF values are already non-negative,
but StandardScaler output is not (it's zero-mean). Running chi2 across a
matrix that mixes both would crash or silently corrupt the selection.
This pipeline therefore applies SelectKBest(chi2, k=20000) ONLY to the
TF-IDF block (50,000 -> 20,000 features), then concatenates the 5
(scaled) structured features afterwards untouched. This preserves the
intent of the spec (20k features selected from the dominant text
signal) while keeping the structured signal (which is small and already
informative) fully intact. Final feature space = 20,005 columns.

Author: generated pipeline for ICD_Code prediction (878 classes)
=============================================================================
"""

import os
import re
import json
import time
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    top_k_accuracy_score,
    classification_report,
)

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================

class Config:
    # I/O
    DATA_PATH = "icdf5.csv"
    ARTIFACT_DIR = "artifacts"

    # Filtering
    MIN_SAMPLES_PER_CLASS = 15

    # Text columns combined into Final_Text
    TEXT_COLUMNS = ["Description", "Cleaned_Description", "Combined_Text",
                     "Category", "Risk_Level"]

    # TF-IDF
    TFIDF_MAX_FEATURES = 50_000
    TFIDF_NGRAM_RANGE = (1, 3)
    TFIDF_MIN_DF = 2
    TFIDF_MAX_DF = 0.95
    TFIDF_SUBLINEAR_TF = True
    TFIDF_STOP_WORDS = "english"

    # Structured features
    NUMERIC_FEATURES = ["Risk_Score", "Age_Midpoint", "Is_Combination"]
    CATEGORICAL_FEATURES = ["Age_Group", "Sex"]

    # Feature selection (applied to TF-IDF block only — see module docstring)
    SELECT_K = 20_000

    # Split
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    # Model
    SVC_C = 2.0
    SVC_CLASS_WEIGHT = "balanced"

    # Eval
    TOP_K_VALUES = (3, 5)


# =============================================================================
# 1. LOAD + FILTER
# =============================================================================

def load_and_filter(path: str, min_samples: int) -> pd.DataFrame:
    """Load the raw CSV and drop ICD codes with too few samples."""
    df = pd.read_csv(path)

    required = {"ICD_Code", "Description", "Cleaned_Description", "Category",
                "Risk_Level", "Risk_Score", "Age_Group", "Age_Midpoint",
                "Sex", "Combined_Text", "Is_Combination"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    class_counts = df["ICD_Code"].value_counts()
    keep_codes = class_counts[class_counts >= min_samples].index
    df = df[df["ICD_Code"].isin(keep_codes)].reset_index(drop=True)

    print(f"[load_and_filter] Kept {df['ICD_Code'].nunique()} ICD classes "
          f"({len(df)} rows) with >= {min_samples} samples each.")
    return df


# =============================================================================
# 2. TEXT ENGINEERING
# =============================================================================

_PUNCT_RE = re.compile(r"[^\w\s]")
_SPACE_RE = re.compile(r"\s+")


def clean_text(text) -> str:
    """Lowercase, strip punctuation, collapse whitespace, handle NaN."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    return text


def build_final_text(df: pd.DataFrame, text_columns) -> pd.DataFrame:
    """Concatenate the configured text columns into a single Final_Text field."""
    df = df.copy()
    for col in text_columns:
        df[col] = df[col].fillna("")

    df["Final_Text"] = (
        df[text_columns]
        .astype(str)
        .agg(" ".join, axis=1)
        .apply(clean_text)
    )
    empty = (df["Final_Text"].str.len() == 0).sum()
    if empty:
        print(f"[build_final_text] WARNING: {empty} rows produced empty text.")
    return df


# =============================================================================
# 3. STRUCTURED FEATURES
# =============================================================================

def build_structured_features(df: pd.DataFrame, config: Config,
                                fit: bool, encoders=None, scaler=None):
    """
    Encode categorical structured columns with LabelEncoder and scale
    numeric structured columns with StandardScaler.

    Returns: dense np.ndarray (n_samples, n_structured_features), encoders dict, scaler
    """
    df = df.copy()

    # ---- numeric ----
    for col in config.NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["Is_Combination"] = df["Is_Combination"].astype(int)
    numeric_block = df[config.NUMERIC_FEATURES].values.astype(float)

    if fit:
        scaler = StandardScaler()
        numeric_scaled = scaler.fit_transform(numeric_block)
    else:
        numeric_scaled = scaler.transform(numeric_block)

    # ---- categorical ----
    if fit:
        encoders = {}
        cat_blocks = []
        for col in config.CATEGORICAL_FEATURES:
            df[col] = df[col].fillna("Unknown").astype(str)
            le = LabelEncoder()
            encoded = le.fit_transform(df[col])
            encoders[col] = le
            cat_blocks.append(encoded.reshape(-1, 1))
    else:
        cat_blocks = []
        for col in config.CATEGORICAL_FEATURES:
            df[col] = df[col].fillna("Unknown").astype(str)
            le = encoders[col]
            # Safely map unseen categories to a fallback rather than crashing
            known = set(le.classes_)
            safe_vals = df[col].apply(lambda v: v if v in known else le.classes_[0])
            encoded = le.transform(safe_vals)
            cat_blocks.append(encoded.reshape(-1, 1))

    cat_block = np.hstack(cat_blocks).astype(float)
    structured = np.hstack([numeric_scaled, cat_block])

    feature_names = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES
    return structured, encoders, scaler, feature_names


# =============================================================================
# 4. TF-IDF
# =============================================================================

def build_tfidf(texts, config: Config, fit: bool, vectorizer=None):
    if fit:
        vectorizer = TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            ngram_range=config.TFIDF_NGRAM_RANGE,
            min_df=config.TFIDF_MIN_DF,
            max_df=config.TFIDF_MAX_DF,
            sublinear_tf=config.TFIDF_SUBLINEAR_TF,
            stop_words=config.TFIDF_STOP_WORDS,
        )
        X = vectorizer.fit_transform(texts)
    else:
        X = vectorizer.transform(texts)
    return X, vectorizer


# =============================================================================
# 5. FEATURE SELECTION (chi2 on TF-IDF block only)
# =============================================================================

def select_tfidf_features(X_tfidf, y, k, fit: bool, selector=None):
    k = min(k, X_tfidf.shape[1])
    if fit:
        selector = SelectKBest(chi2, k=k)
        X_sel = selector.fit_transform(X_tfidf, y)
    else:
        X_sel = selector.transform(X_tfidf)
    return X_sel, selector


# =============================================================================
# 6. FEATURE FUSION
# =============================================================================

def fuse_features(X_tfidf_selected, structured_array):
    """Combine sparse TF-IDF block with dense structured block into one sparse matrix."""
    structured_sparse = sp.csr_matrix(structured_array)
    return sp.hstack([X_tfidf_selected, structured_sparse], format="csr")


# =============================================================================
# 7-8. SPLIT + TRAIN
# =============================================================================

def train_model(X_train, y_train, config: Config) -> LinearSVC:
    model = LinearSVC(
        C=config.SVC_C,
        class_weight=config.SVC_CLASS_WEIGHT,
        random_state=config.RANDOM_STATE,
        max_iter=5000,
        dual="auto",
    )
    model.fit(X_train, y_train)
    return model


# =============================================================================
# 9. EVALUATION
# =============================================================================

def evaluate(model, X_test, y_test, label_encoder, config: Config) -> dict:
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    # Top-K accuracy using decision_function scores (monotonic w.r.t. confidence,
    # does not require calibrated probabilities).
    decision_scores = model.decision_function(X_test)
    all_labels = np.arange(len(label_encoder.classes_))
    for k in config.TOP_K_VALUES:
        try:
            metrics[f"top_{k}_accuracy"] = top_k_accuracy_score(
                y_test, decision_scores, k=k, labels=all_labels
            )
        except Exception as e:
            metrics[f"top_{k}_accuracy"] = None
            print(f"[evaluate] Could not compute top-{k} accuracy: {e}")

    return metrics, y_pred


# =============================================================================
# 10. ARTIFACT PERSISTENCE
# =============================================================================

def save_artifacts(out_dir, model, vectorizer, selector, label_encoder,
                    scaler, encoders, structured_feature_names, config_dict):
    os.makedirs(out_dir, exist_ok=True)

    artifacts = {
        "model.pkl": model,
        "tfidf.pkl": vectorizer,
        "selector.pkl": selector,
        "label_encoder.pkl": label_encoder,
        "scaler.pkl": scaler,
        "encoders.pkl": encoders,
    }
    for fname, obj in artifacts.items():
        with open(os.path.join(out_dir, fname), "wb") as f:
            pickle.dump(obj, f)

    meta = {
        "structured_feature_names": structured_feature_names,
        "config": config_dict,
        "saved_at": datetime.now().isoformat(),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[save_artifacts] Saved {len(artifacts)} artifacts + meta.json to '{out_dir}/'")


def load_artifacts(in_dir):
    """Load all pickled artifacts needed for inference."""
    names = ["model.pkl", "tfidf.pkl", "selector.pkl", "label_encoder.pkl",
              "scaler.pkl", "encoders.pkl"]
    loaded = {}
    for name in names:
        with open(os.path.join(in_dir, name), "rb") as f:
            loaded[name.replace(".pkl", "")] = pickle.load(f)
    with open(os.path.join(in_dir, "meta.json")) as f:
        loaded["meta"] = json.load(f)
    return loaded


# =============================================================================
# 11. FEATURE IMPORTANCE ANALYSIS
# =============================================================================

def feature_importance_analysis(model, vectorizer, selector, structured_feature_names,
                                  label_encoder, top_n=15, n_classes_to_show=5):
    """
    LinearSVC is a linear model: model.coef_ has shape (n_classes, n_features)
    for multiclass (one-vs-rest). We report:
      (a) global top features by mean absolute coefficient across classes
      (b) per-class top positive features for a handful of sample classes
    """
    tfidf_vocab = np.array(vectorizer.get_feature_names_out())
    selected_mask = selector.get_support()
    selected_tfidf_names = tfidf_vocab[selected_mask]
    all_feature_names = np.concatenate([selected_tfidf_names, structured_feature_names])

    coef = model.coef_
    if sp.issparse(coef):
        coef = coef.toarray()

    print("\n--- GLOBAL TOP FEATURES (mean |coefficient| across all classes) ---")
    mean_abs_coef = np.abs(coef).mean(axis=0)
    top_idx = np.argsort(mean_abs_coef)[::-1][:top_n]
    for rank, idx in enumerate(top_idx, 1):
        print(f"{rank:2d}. {all_feature_names[idx]:30s}  weight={mean_abs_coef[idx]:.4f}")

    print(f"\n--- PER-CLASS TOP POSITIVE FEATURES (sample of {n_classes_to_show} classes) ---")
    rng = np.random.default_rng(42)
    sample_class_idx = rng.choice(coef.shape[0], size=min(n_classes_to_show, coef.shape[0]),
                                   replace=False)
    for class_idx in sample_class_idx:
        class_label = label_encoder.inverse_transform([class_idx])[0]
        class_coef = coef[class_idx]
        top_feat_idx = np.argsort(class_coef)[::-1][:top_n // 2]
        feats = ", ".join(all_feature_names[i] for i in top_feat_idx)
        print(f"  {class_label}: {feats}")

    return {
        "global_top_features": [(all_feature_names[i], float(mean_abs_coef[i])) for i in top_idx],
        "all_feature_names": all_feature_names,
    }


# =============================================================================
# 12. INFERENCE
# =============================================================================

def predict_icd(raw_record: dict, artifacts: dict, config: Config, top_n: int = 5):
    """
    Run inference on a single new record.

    raw_record must contain the same raw fields used at train time:
      Description, Cleaned_Description, Combined_Text, Category, Risk_Level,
      Risk_Score, Age_Midpoint, Age_Group, Sex, Is_Combination

    Returns the top_n predicted ICD codes ranked by decision-function confidence.
    """
    model = artifacts["model"]
    vectorizer = artifacts["tfidf"]
    selector = artifacts["selector"]
    label_encoder = artifacts["label_encoder"]
    scaler = artifacts["scaler"]
    encoders = artifacts["encoders"]

    df_row = pd.DataFrame([raw_record])

    # Text
    df_row = build_final_text(df_row, config.TEXT_COLUMNS)
    X_text, _ = build_tfidf(df_row["Final_Text"], config, fit=False, vectorizer=vectorizer)
    X_text_sel, _ = select_tfidf_features(X_text, None, config.SELECT_K, fit=False, selector=selector)

    # Structured
    structured, _, _, _ = build_structured_features(
        df_row, config, fit=False, encoders=encoders, scaler=scaler
    )

    X = fuse_features(X_text_sel, structured)

    decision_scores = model.decision_function(X)[0]
    top_idx = np.argsort(decision_scores)[::-1][:top_n]
    top_codes = label_encoder.inverse_transform(top_idx)
    top_scores = decision_scores[top_idx]

    return list(zip(top_codes, top_scores.tolist()))


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    overall_start = time.time()
    config = Config()

    # ---------- 1. Load + filter ----------
    t0 = time.time()
    df = load_and_filter(config.DATA_PATH, config.MIN_SAMPLES_PER_CLASS)
    print(f"  -> {time.time() - t0:.1f}s")

    # ---------- 2. Text engineering ----------
    t0 = time.time()
    df = build_final_text(df, config.TEXT_COLUMNS)
    print(f"[build_final_text] done -> {time.time() - t0:.1f}s")

    # ---------- target encoding ----------
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["ICD_Code"])

    # ---------- 3. TF-IDF ----------
    t0 = time.time()
    X_tfidf, vectorizer = build_tfidf(df["Final_Text"], config, fit=True)
    print(f"[build_tfidf] shape={X_tfidf.shape} -> {time.time() - t0:.1f}s")

    # ---------- 4. Structured features ----------
    t0 = time.time()
    structured, encoders, scaler, structured_names = build_structured_features(
        df, config, fit=True
    )
    print(f"[build_structured_features] shape={structured.shape} -> {time.time() - t0:.1f}s")

    # ---------- 5. chi2 feature selection on TF-IDF block ----------
    t0 = time.time()
    X_tfidf_sel, selector = select_tfidf_features(X_tfidf, y, config.SELECT_K, fit=True)
    print(f"[select_tfidf_features] {X_tfidf.shape[1]} -> {X_tfidf_sel.shape[1]} "
          f"-> {time.time() - t0:.1f}s")

    # ---------- 6. Fuse ----------
    t0 = time.time()
    X = fuse_features(X_tfidf_sel, structured)
    print(f"[fuse_features] final shape={X.shape} -> {time.time() - t0:.1f}s")

    # ---------- 7. Split ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )
    print(f"[split] train={X_train.shape[0]}  test={X_test.shape[0]}")

    # ---------- 8. Train ----------
    t0 = time.time()
    model = train_model(X_train, y_train, config)
    train_time = time.time() - t0
    print(f"[train_model] LinearSVC trained -> {train_time:.1f}s")

    # ---------- 9. Evaluate ----------
    t0 = time.time()
    metrics, y_pred = evaluate(model, X_test, y_test, label_encoder, config)
    print(f"[evaluate] -> {time.time() - t0:.1f}s")

    # ---------- 10. Save ----------
    config_dict = {k: v for k, v in vars(Config).items() if not k.startswith("_")}
    # decision_function / coef_ are large arrays, but pickle handles ndarray fine
    config_dict = {k: (v if isinstance(v, (str, int, float, bool, list, tuple)) else str(v))
                   for k, v in config_dict.items()}
    save_artifacts(config.ARTIFACT_DIR, model, vectorizer, selector, label_encoder,
                    scaler, encoders, structured_names, config_dict)

    # ---------- 11. Feature importance ----------
    importance = feature_importance_analysis(
        model, vectorizer, selector, structured_names, label_encoder
    )

    total_time = time.time() - overall_start

    # ---------- TRAINING SUMMARY ----------
    print("\n" + "=" * 70)
    print(" TRAINING SUMMARY")
    print("=" * 70)
    print(f" Rows used               : {len(df)}")
    print(f" ICD classes             : {df['ICD_Code'].nunique()}")
    print(f" TF-IDF vocab size       : {X_tfidf.shape[1]}")
    print(f" Features after chi2     : {X_tfidf_sel.shape[1]}")
    print(f" Structured features     : {structured.shape[1]}")
    print(f" Final feature dimension : {X.shape[1]}")
    print(f" Train / Test rows       : {X_train.shape[0]} / {X_test.shape[0]}")
    print(f" Model training time     : {train_time:.1f}s")
    print(f" Total pipeline time     : {total_time:.1f}s "
          f"({total_time/60:.1f} min, budget: 20 min)")
    print("-" * 70)
    for k, v in metrics.items():
        if v is not None:
            print(f" {k:22s}: {v:.4f}")
    print("=" * 70)

    if total_time > 20 * 60:
        print("WARNING: pipeline exceeded the 20-minute target budget.")

    return {
        "model": model, "vectorizer": vectorizer, "selector": selector,
        "label_encoder": label_encoder, "scaler": scaler, "encoders": encoders,
        "metrics": metrics, "importance": importance, "total_time": total_time,
    }


if __name__ == "__main__":
    main()
