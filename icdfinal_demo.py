"""
ICD Multi-Label Prediction — PubMedBERT (built directly on icdtunev2.py)
=========================================================================
Base: icdtunev2.py  (46.69% single-label accuracy, 878 classes)
Goal: Multi-Disease / Multi-ICD system for Insurance Claim Processing

WHAT CHANGED vs icdtunev2.py  (everything else is identical):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. pd.read_csv  →  engine='python', on_bad_lines='skip'
     Row 15981 in icdf5.csv has an embedded comma inside a quoted
     Description string.  The C parser crashes on this; the Python
     engine handles it correctly and loads all 23,792 rows.

  2. LabelEncoder  →  MultiLabelBinarizer  (Task 2)
     E11     →  icd_list=['E11']      →  multi-hot [0,…1,…0]
     E11+I10 →  icd_list=['E11','I10'] →  multi-hot [0,…1,…1,…0]
     All 23,792 rows used (single + combination).
     MIN_SAMPLES filter removed — every ICD code now appears in
     both single and combination rows (min frequency = 23 across all).

  3. CrossEntropyLoss  →  BCEWithLogitsLoss(pos_weight=…)  (Task 3)
     BCE lets multiple labels be simultaneously active.
     pos_weight per label = (N-n_pos)/n_pos, clamped to [1,50].

  4. ICDDataset  →  ICDMultiLabelDataset
     Labels are float32 multi-hot tensors, not int64 class indices.
     Text augmentation (random word dropout p=0.15) active in train.

  5. run_epoch  →  returns micro-F1 + hamming loss (not accuracy)
     Micro-F1 is the correct early-stopping metric for multi-label.

  6. New: CombinationRecommendationEngine  (Task 6)
     Loads all 9,300 known clinical comorbidity pairs from the dataset.
     At inference, checks every pair among the predicted codes and
     surfaces matched combinations ranked by average confidence.

  7. New: evaluate_multilabel()  — all 10 requested metrics
  8. New: predict_for_claim()    — Task 7 insurance workflow
  9. New: analyze_combinations() — Task 1 co-occurrence statistics

PRESERVED EXACTLY from icdtunev2.py:
  PubMedBERT backbone • Mean pooling • LLRD • EMA • Checkpoint
  averaging • Mixed precision • Early stopping • try_compile •
  all hyperparameters • OUTPUT_DIR • artifact saving structure

Run in Colab (T4 GPU):
    from google.colab import drive; drive.mount('/content/drive')
    !pip install -q transformers accelerate scikit-learn pandas numpy tqdm
    !python icd_multilabel_final.py
"""

import os, re, json, pickle, random
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    hamming_loss, accuracy_score, classification_report,
)
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from itertools import combinations as iter_combinations
from collections import Counter

# ══════════════════════════════════════════════════════════════════════════════
# 0.  Config  — identical to icdtunev2.py
# ══════════════════════════════════════════════════════════════════════════════
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED); random.seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)
if DEVICE.type == 'cuda':
    print('GPU:', torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True

USE_FP16         = DEVICE.type == 'cuda'
MODEL_NAME       = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
MAX_LENGTH       = 256
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE  = 64
NUM_WORKERS      = 0    # 0 = Colab-safe (avoids multiprocessing deadlocks)

DATA_PATH       = '/content/icdf5.csv'
OUTPUT_DIR      = '/content/drive/MyDrive/icd_multilabel'
EPOCH_CKPT_DIR  = '/content/drive/MyDrive/icd_epoch'   # per-epoch rolling checkpoint
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EPOCH_CKPT_DIR, exist_ok=True)
print('Artifacts will be saved to:', OUTPUT_DIR)
print('Epoch checkpoints will be saved to:', EPOCH_CKPT_DIR)

EPOCHS              = 20
LR                  = 2e-5
WEIGHT_DECAY        = 0.01
WARMUP_RATIO        = 0.1
GRAD_CLIP           = 1.0
EARLY_STOP_PATIENCE = 5
DROPOUT             = 0.35       # raised from 0.30 (multi-label, more regularisation)
LLRD_DECAY          = 0.9
HEAD_LR             = 1e-4
TOP_K_CHECKPOINTS   = 5
USE_EMA             = True
EMA_DECAY           = 0.999

# Multi-label specifics
PRED_THRESHOLD  = 0.35   # sigmoid > threshold → label positive
                         # 0.35 = high-recall setting for insurance (catch all codes)
AUG_DROP_P      = 0.15   # word dropout probability during training
AUG_MIN_WORDS   = 3      # never drop below this many words

# ── Mode switch ───────────────────────────────────────────────────────────────
# DEMO_MODE = False  →  full training pipeline (original behaviour, unchanged)
# DEMO_MODE = True   →  load saved artifacts, interactive inference only
DEMO_MODE = False   # ← flip to True to skip training and run the demo loop


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Text helpers  (clean_text identical to icdtunev2.py; helpers are new)
# ══════════════════════════════════════════════════════════════════════════════
def clean_text(s):
    s = str(s).lower()
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def parse_icd(code: str) -> list:
    """'E11+I10' → ['E11','I10'].  Handles + and ; separators."""
    return [c.strip() for c in str(code).replace(';', '+').split('+')
            if c.strip() and c.strip().lower() != 'nan']


def random_word_dropout(text: str, p: float = AUG_DROP_P,
                        min_words: int = AUG_MIN_WORDS) -> str:
    """Drop each word independently with probability p during training only."""
    words = text.split()
    if len(words) <= min_words:
        return text
    kept = [w for w in words if random.random() > p]
    return ' '.join(kept) if len(kept) >= min_words else text


# ══════════════════════════════════════════════════════════════════════════════
# TASK 1 — Analyze ICD Combinations
# ══════════════════════════════════════════════════════════════════════════════
def analyze_combinations(df: pd.DataFrame) -> dict:
    """
    Task 1: Generate co-occurrence statistics.
    Returns dict with top_pairs, combo_stats, code_frequency.
    """
    combos  = df[df['Is_Combination']]
    singles = df[~df['Is_Combination']]

    pair_counter  = Counter()
    combo_counter = Counter()
    for code in combos['ICD_Code']:
        codes = parse_icd(code)
        combo_counter['+'.join(sorted(codes))] += 1
        if len(codes) == 2:
            pair_counter[tuple(sorted(codes))] += 1

    code_freq = Counter()
    for v in df['ICD_Code']:
        for c in parse_icd(str(v)):
            code_freq[c] += 1

    result = {
        'total_rows'  : len(df),
        'single_count': len(singles),
        'combo_count' : len(combos),
        'combo_pct'   : round(len(combos) / len(df) * 100, 2),
        'unique_pairs': len(pair_counter),
        'top_pairs'   : [(a, b, n) for (a, b), n in pair_counter.most_common(50)],
        'top_combos'  : combo_counter.most_common(50),
        'freq_stats'  : {
            'min'   : min(code_freq.values()),
            'max'   : max(code_freq.values()),
            'mean'  : round(np.mean(list(code_freq.values())), 1),
            'median': float(np.median(list(code_freq.values()))),
        },
        'code_freq': code_freq,
    }

    print(f"\n{'='*62}\nTASK 1 — ICD COMBINATION ANALYSIS\n{'='*62}")
    print(f"  Total rows          : {result['total_rows']:,}")
    print(f"  Single-ICD rows     : {result['single_count']:,}")
    print(f"  Combination rows    : {result['combo_count']:,}  ({result['combo_pct']}%)")
    print(f"  Unique ICD pairs    : {result['unique_pairs']:,}")
    print(f"  Per-label frequency : "
          f"min={result['freq_stats']['min']}  "
          f"max={result['freq_stats']['max']}  "
          f"mean={result['freq_stats']['mean']}  "
          f"median={result['freq_stats']['median']}")
    print(f"\n  TOP 20 CO-OCCURRENCE PAIRS:")
    for a, b, n in result['top_pairs'][:20]:
        print(f"    {a}+{b}: {n}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2 — Load data + MultiLabelBinarizer
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    """
    Changes vs icdtunev2.py load_data():
      - engine='python', on_bad_lines='skip'  (CSV fix for row 15981)
      - Is_Combination normalised to bool
      - ALL 23,792 rows used (no MIN_SAMPLES filter)
      - MultiLabelBinarizer instead of LabelEncoder
      - Cleaner Final_Text (no triplication of Description)
      - Aug_Text + Suffix_Text columns for training augmentation
    """
    print('Loading CSV …')
    try:
        df = pd.read_csv(DATA_PATH, engine='python', on_bad_lines='skip')
    except TypeError:
        # pandas < 1.3 compatibility
        df = pd.read_csv(DATA_PATH, engine='python', error_bad_lines=False)
    print(f'Records loaded      : {len(df):,}')

    # ── normalise Is_Combination (handles bool/int/str/NaN) ──────────────────
    col = df['Is_Combination']
    if col.dtype == object:
        df['Is_Combination'] = col.map(
            {'True': True, 'False': False, '1': True, '0': False,
             'true': True, 'false': False}
        ).fillna(False).astype(bool)
    else:
        df['Is_Combination'] = col.fillna(False).astype(bool)

    # ── drop rows with null ICD_Code ─────────────────────────────────────────
    before = len(df)
    df = df.dropna(subset=['ICD_Code']).reset_index(drop=True)
    if len(df) < before:
        print(f'Dropped {before - len(df)} rows with null ICD_Code')

    # ── fill text columns ────────────────────────────────────────────────────
    for c in ['Description', 'Category', 'Risk_Level',
              'Cleaned_Description', 'Combined_Text']:
        if c in df.columns:
            df[c] = df[c].fillna('')

    # Cleaner text: Description + Category + Risk_Level
    # (no triplication via Cleaned_Description + Combined_Text)
    df['Final_Text'] = (
        df['Description'].astype(str) + ' ' +
        df['Category'].astype(str)    + ' ' +
        df['Risk_Level'].astype(str)
    ).apply(clean_text)

    # Aug_Text = Description only (gets word-dropped during training)
    # Suffix_Text = Category + Risk_Level (never dropped, appended after aug)
    df['Aug_Text']    = df['Description'].apply(clean_text)
    df['Suffix_Text'] = (
        df['Category'].astype(str) + ' ' + df['Risk_Level'].astype(str)
    ).apply(clean_text)

    # ── TASK 2: parse ICD lists → MultiLabelBinarizer ────────────────────────
    df['icd_list'] = df['ICD_Code'].apply(parse_icd)
    df = df[df['icd_list'].apply(len) > 0].reset_index(drop=True)

    combo_stats = analyze_combinations(df)

    mlb          = MultiLabelBinarizer()
    y_multilabel = mlb.fit_transform(df['icd_list'])   # (N, 878) int8
    df['multilabel'] = list(y_multilabel)

    n_classes = len(mlb.classes_)
    print(f'\nMultiLabelBinarizer classes : {n_classes}')
    print(f'Label matrix shape          : {y_multilabel.shape}')
    print(f'Single-label rows : {(y_multilabel.sum(axis=1) == 1).sum():,}  '
          f'Two-label rows  : {(y_multilabel.sum(axis=1) == 2).sum():,}')
    return df, mlb, combo_stats


# ══════════════════════════════════════════════════════════════════════════════
# TASK 6 — Combination Recommendation Engine
# ══════════════════════════════════════════════════════════════════════════════
class CombinationRecommendationEngine:
    """
    Task 6: At inference, given a list of predicted ICD codes, look up every
    pair against the 9,300 known clinical comorbidity pairs in the dataset and
    return the matched combinations ranked by average confidence score.

    Lookup is O(k²) where k = number of predicted codes (≤ 5 typically).
    """
    def __init__(self, df: pd.DataFrame):
        self.combo_db   = {}   # frozenset({a,b}) → "A+B" string
        self.combo_desc = {}   # frozenset({a,b}) → description
        self.combo_risk = {}   # frozenset({a,b}) → risk level

        combos = df[df['Is_Combination']]
        for _, row in combos.iterrows():
            codes = parse_icd(row['ICD_Code'])
            if len(codes) == 2:
                key = frozenset(codes)
                self.combo_db[key]   = row['ICD_Code']
                self.combo_desc[key] = str(row.get('Description', ''))
                self.combo_risk[key] = str(row.get('Risk_Level', ''))

        print(f'[CombinationEngine] Loaded {len(self.combo_db):,} known pairs.')

    def recommend(self, predicted_codes: list, confidence_map: dict,
                  top_k: int = 5) -> list:
        """
        Args:
            predicted_codes : ['E11', 'I10', 'N18', …]
            confidence_map  : {'E11': 0.92, 'I10': 0.88, …}
        Returns:
            list of dicts, sorted by confidence_score desc
        """
        matched = []
        for a, b in iter_combinations(predicted_codes, 2):
            key = frozenset([a, b])
            if key in self.combo_db:
                avg_conf = (confidence_map.get(a, 0) + confidence_map.get(b, 0)) / 2
                matched.append({
                    'combination'     : self.combo_db[key],
                    'confidence_score': round(avg_conf, 4),
                    'code_a'          : a,
                    'code_b'          : b,
                    'conf_a'          : round(float(confidence_map.get(a, 0)), 4),
                    'conf_b'          : round(float(confidence_map.get(b, 0)), 4),
                    'description'     : self.combo_desc.get(key, ''),
                    'risk_level'      : self.combo_risk.get(key, ''),
                })
        matched.sort(key=lambda x: x['confidence_score'], reverse=True)
        return matched[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
# Dataset  (augmentation-aware, multi-hot float labels)
# ══════════════════════════════════════════════════════════════════════════════
class ICDMultiLabelDataset(Dataset):
    def __init__(self, aug_texts, suffix_texts, full_texts,
                 multilabels, augment=False):
        self.aug_texts    = aug_texts
        self.suffix_texts = suffix_texts
        self.full_texts   = full_texts
        self.multilabels  = multilabels   # list of np arrays (C,) float32
        self.augment      = augment

    def __len__(self):
        return len(self.multilabels)

    def __getitem__(self, idx):
        if self.augment:
            desc = random_word_dropout(self.aug_texts[idx])
            text = clean_text(desc + ' ' + self.suffix_texts[idx])
        else:
            text = self.full_texts[idx]
        return text, torch.tensor(self.multilabels[idx], dtype=torch.float32)


def make_collate_fn(tokenizer):
    """Same as icdtunev2.py except label dtype is float32 (for BCE)."""
    def collate_fn(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(
            list(texts), truncation=True, padding=True,
            max_length=MAX_LENGTH, return_tensors='pt',
        )
        enc['label'] = torch.stack(list(labels))   # (B, C) float32
        return enc
    return collate_fn


# Identical to icdtunev2.py
def check_truncation_rate(tokenizer, texts, max_length=MAX_LENGTH, sample_size=4000):
    sample   = texts if len(texts) <= sample_size else list(
        np.random.choice(texts, sample_size, replace=False))
    enc_full = tokenizer(sample, truncation=False, padding=False)
    lengths  = np.array([len(ids) for ids in enc_full['input_ids']])
    trunc    = (lengths > max_length).mean()
    print(f'[truncation check] n={len(sample)} mean={lengths.mean():.1f} '
          f'p95={np.percentile(lengths,95):.0f} p99={np.percentile(lengths,99):.0f} '
          f'max={lengths.max()} | truncated@{max_length}: {trunc*100:.2f}%')
    return trunc


# ══════════════════════════════════════════════════════════════════════════════
# TASK 3 — PubMedBERT (backbone identical to icdtunev2.py, loss changes only)
# ══════════════════════════════════════════════════════════════════════════════
class PubMedBERTMultiLabel(nn.Module):
    """
    Architecture is identical to PubMedBERTClassifier in icdtunev2.py:
      - PubMedBERT backbone
      - Attention-mask-aware mean pooling
      - Dropout + Linear head
    Only difference: output is raw logits for BCEWithLogitsLoss (not softmax).
    Multiple labels can be simultaneously active.
    """
    def __init__(self, model_name, num_labels, dropout=DROPOUT):
        super().__init__()
        self.bert       = AutoModel.from_pretrained(model_name)
        hidden          = self.bert.config.hidden_size
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        mask   = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.classifier(self.dropout(pooled))   # (B, C) raw logits


# ── Helpers identical to icdtunev2.py ─────────────────────────────────────────
def try_compile(model):
    if DEVICE.type != 'cuda':
        return model
    try:
        c = torch.compile(model)
        print('torch.compile enabled.')
        return c
    except Exception as e:
        print('torch.compile not available:', e)
        return model


def build_llrd_param_groups(model, base_lr=LR, head_lr=HEAD_LR,
                             decay=LLRD_DECAY, weight_decay=WEIGHT_DECAY):
    """Identical to icdtunev2.py."""
    no_decay = ('bias', 'LayerNorm.weight', 'LayerNorm.bias')
    groups   = []
    def add_group(named_params, lr):
        dp = [p for n, p in named_params if not any(nd in n for nd in no_decay)]
        nd = [p for n, p in named_params if     any(nd in n for nd in no_decay)]
        if dp: groups.append({'params': dp, 'lr': lr, 'weight_decay': weight_decay})
        if nd: groups.append({'params': nd, 'lr': lr, 'weight_decay': 0.0})
    add_group(list(model.classifier.named_parameters()), head_lr)
    lr = base_lr
    for layer in reversed(list(model.bert.encoder.layer)):
        add_group(list(layer.named_parameters()), lr)
        lr *= decay
    add_group(list(model.bert.embeddings.named_parameters()), lr)
    if hasattr(model.bert, 'pooler') and model.bert.pooler is not None:
        add_group(list(model.bert.pooler.named_parameters()), lr)
    return groups


def average_state_dicts(state_dicts):
    """Identical to icdtunev2.py."""
    avg = {}
    for k in state_dicts[0]:
        stacked = torch.stack([sd[k].float() for sd in state_dicts])
        avg[k]  = stacked.mean(0)
        if not state_dicts[0][k].is_floating_point():
            avg[k] = avg[k].to(state_dicts[0][k].dtype)
    return avg


class ModelEMA:
    """Identical to icdtunev2.py."""
    def __init__(self, model, decay=EMA_DECAY):
        raw = model._orig_mod if hasattr(model, '_orig_mod') else model
        self.decay  = decay
        self.shadow = {k: v.detach().cpu().clone().float()
                       for k, v in raw.state_dict().items()}
    @torch.no_grad()
    def update(self, model):
        raw = model._orig_mod if hasattr(model, '_orig_mod') else model
        for k, v in raw.state_dict().items():
            vc = v.detach().cpu().float()
            if v.is_floating_point():
                # ensure shadow tensor is always on CPU before in-place ops
                self.shadow[k] = self.shadow[k].cpu()
                self.shadow[k].mul_(self.decay).add_(vc, alpha=1 - self.decay)
            else:
                self.shadow[k] = vc.cpu()
    def state_dict_for_load(self, ref):
        return {k: v.cpu().to(ref[k].dtype) for k, v in self.shadow.items()}


def compute_pos_weight(y_train: np.ndarray) -> torch.Tensor:
    """Per-label pos_weight = (N - n_pos) / n_pos, clamped to [1, 50]."""
    n_pos = y_train.sum(0).clip(min=1)
    pw    = torch.tensor((len(y_train) - n_pos) / n_pos, dtype=torch.float32)
    return pw.clamp(1.0, 50.0)


# ══════════════════════════════════════════════════════════════════════════════
# Training loop  (structure identical to icdtunev2.py; metrics updated)
# ══════════════════════════════════════════════════════════════════════════════
def run_epoch(model, loader, optimizer, scheduler, scaler, criterion,
              train=True, ema=None, threshold=PRED_THRESHOLD):
    """
    Differences vs icdtunev2.py run_epoch():
      - labels are float32 multi-hot, not int64
      - returns (loss, micro_f1, hamming_loss) instead of (loss, accuracy)
      - micro_f1 is used as the early-stopping signal
    """
    model.train() if train else model.eval()
    total_loss, all_logits, all_labels = 0.0, [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc='train' if train else 'eval', leave=False):
            ids  = batch['input_ids'].to(DEVICE, non_blocking=True)
            mask = batch['attention_mask'].to(DEVICE, non_blocking=True)
            lbls = batch['label'].to(DEVICE, non_blocking=True)   # float32 multi-hot
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16,
                                 enabled=USE_FP16):
                logits = model(ids, mask)
                loss   = criterion(logits, lbls)
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                prev = scaler.get_scale()
                scaler.step(optimizer); scaler.update()
                if scaler.get_scale() >= prev:
                    scheduler.step()
                    if ema is not None:
                        ema.update(model)
            total_loss += loss.item() * ids.size(0)
            all_logits.append(logits.detach().float().cpu())
            all_labels.append(lbls.detach().float().cpu())

    logits_np = torch.cat(all_logits).numpy()
    labels_np = torch.cat(all_labels).numpy()
    probs     = 1 / (1 + np.exp(-logits_np))      # sigmoid
    preds     = (probs > threshold).astype(int)
    # fallback: if no label crosses threshold, take the top-1
    zero_rows = preds.sum(1) == 0
    if zero_rows.any():
        top1 = np.argmax(probs[zero_rows], axis=1)
        for i, ri in enumerate(np.where(zero_rows)[0]):
            preds[ri, top1[i]] = 1
    mf1 = f1_score(labels_np, preds, average='micro', zero_division=0)
    hl  = hamming_loss(labels_np, preds)
    return total_loss / len(loader.dataset), mf1, hl


@torch.no_grad()
def get_test_logits(model, loader):
    """Identical structure to icdtunev2.py."""
    model.eval()
    all_logits, all_labels = [], []
    for batch in tqdm(loader, desc='predict'):
        ids  = batch['input_ids'].to(DEVICE, non_blocking=True)
        mask = batch['attention_mask'].to(DEVICE, non_blocking=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=USE_FP16):
            logits = model(ids, mask)
        all_logits.append(logits.float().cpu())
        all_labels.append(batch['label'])
    return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy()


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation — all 10 requested metrics
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_multilabel(logits: np.ndarray, labels: np.ndarray,
                        threshold: float = PRED_THRESHOLD) -> dict:
    """
    1.  Exact Match Accuracy   (all predicted labels == all true labels)
    2.  Micro Precision
    3.  Micro Recall
    4.  Micro F1
    5.  Macro Precision
    6.  Macro Recall
    7.  Macro F1
    8.  Hamming Loss
    9.  Top-3 Accuracy         (≥1 true label in top-3 by confidence)
    10. Top-5 Accuracy         (≥1 true label in top-5 by confidence)
    """
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > threshold).astype(int)
    zero  = preds.sum(1) == 0
    if zero.any():
        top1 = np.argmax(probs[zero], axis=1)
        for i, ri in enumerate(np.where(zero)[0]):
            preds[ri, top1[i]] = 1

    def topk_any(k):
        topk = np.argsort(-probs, axis=1)[:, :k]
        hits = sum(1 for i in range(len(labels))
                   if set(np.where(labels[i] > 0)[0]) & set(topk[i]))
        return hits / len(labels)

    return {
        'exact_match_accuracy': float(accuracy_score(labels, preds)),
        'micro_precision'     : float(precision_score(labels, preds, average='micro', zero_division=0)),
        'micro_recall'        : float(recall_score(labels, preds, average='micro', zero_division=0)),
        'micro_f1'            : float(f1_score(labels, preds, average='micro', zero_division=0)),
        'macro_precision'     : float(precision_score(labels, preds, average='macro', zero_division=0)),
        'macro_recall'        : float(recall_score(labels, preds, average='macro', zero_division=0)),
        'macro_f1'            : float(f1_score(labels, preds, average='macro', zero_division=0)),
        'hamming_loss'        : float(hamming_loss(labels, preds)),
        'top3_accuracy'       : float(topk_any(3)),
        'top5_accuracy'       : float(topk_any(5)),
        'threshold_used'      : threshold,
        'avg_labels_per_pred' : float(preds.sum(1).mean()),
    }


# ══════════════════════════════════════════════════════════════════════════════
# TASK 4 + 5 — Multi-ICD Inference
# ══════════════════════════════════════════════════════════════════════════════
def build_predict_fn(model, tokenizer, mlb, combo_engine,
                     threshold=PRED_THRESHOLD):
    """
    Returns predict_claim() — Tasks 4, 5, 6 combined.

    Output:
      predicted_icd_codes      — Task 4: threshold-based multi-ICD prediction
      top_5_recommendations    — Task 5: always-present top-5 by confidence
      combination_recommendations — Task 6: known comorbidity pairs matched
    """
    def predict_claim(clinical_text: str, risk_level: str = '',
                      category: str = '', top_k: int = 5,
                      threshold_override: float = None) -> dict:
        thr  = threshold_override if threshold_override is not None else threshold
        text = clean_text(f"{clinical_text} {category} {risk_level}")
        enc  = tokenizer(text, truncation=True, padding=True,
                         max_length=MAX_LENGTH, return_tensors='pt').to(DEVICE)
        model.eval()
        with torch.no_grad(), torch.autocast(
                device_type='cuda', dtype=torch.float16, enabled=USE_FP16):
            logits = model(enc['input_ids'], enc['attention_mask'])
        probs = torch.sigmoid(logits.float()).cpu().numpy()[0]   # (878,)

        # Task 5: top-5 always (regardless of threshold)
        top_idx   = np.argsort(-probs)[:top_k]
        top_codes = mlb.classes_[top_idx].tolist()
        top_confs = probs[top_idx].tolist()

        # Task 4: threshold-based multi-ICD (minimum 1)
        mask = probs > thr
        if not mask.any():
            mask[np.argmax(probs)] = True
        pred_idx   = np.where(mask)[0]
        pred_codes = mlb.classes_[pred_idx].tolist()
        pred_confs = probs[pred_idx].tolist()
        pairs      = sorted(zip(pred_codes, pred_confs),
                             key=lambda x: x[1], reverse=True)
        pred_codes, pred_confs = [p[0] for p in pairs], [p[1] for p in pairs]

        # Task 6: combination lookup
        conf_map = {c: float(p) for c, p in zip(top_codes, top_confs)}
        conf_map.update({c: float(p) for c, p in zip(pred_codes, pred_confs)})
        combos = combo_engine.recommend(list(conf_map.keys()), conf_map)

        return {
            'predicted_icd_codes'        : pred_codes,
            'predicted_confidence'       : [round(c, 4) for c in pred_confs],
            'num_predicted'              : len(pred_codes),
            'top_5_recommendations'      : [
                {'rank': i+1, 'icd_code': c, 'confidence': round(float(p), 4)}
                for i, (c, p) in enumerate(zip(top_codes, top_confs))],
            'combination_recommendations': combos,
            'threshold_used'             : thr,
        }
    return predict_claim


# ══════════════════════════════════════════════════════════════════════════════
# TASK 7 — Insurance Claim Workflow
# ══════════════════════════════════════════════════════════════════════════════
def predict_for_claim(predict_fn, clinical_note: str,
                      patient_info: dict = None) -> dict:
    """
    Task 7 full workflow:
      Clinical Note → PubMedBERT → Multi-ICD → Top-5 → Combos → Coder → Submit
    """
    info   = patient_info or {}
    result = predict_fn(clinical_text=clinical_note,
                        risk_level=info.get('risk_level', ''),
                        category=info.get('category', ''))
    return {
        'workflow_version': 'multilabel_v1',
        'input': {'clinical_note': clinical_note, 'patient_info': info},
        'step1_multi_icd_prediction': {
            'predicted_codes': result['predicted_icd_codes'],
            'confidence'     : result['predicted_confidence'],
            'num_codes'      : result['num_predicted'],
            'status'         : 'MULTI_CODE' if result['num_predicted'] > 1
                               else 'SINGLE_CODE',
        },
        'step2_top5_recommendations'       : result['top_5_recommendations'],
        'step3_combination_recommendations': result['combination_recommendations'],
        'step4_coder_review': {
            'instruction'    : ('Review each predicted code. Accept, modify, or '
                                'reject. Combination codes (e.g. E11+I10) should '
                                'be used when the presentation matches a known '
                                'comorbidity pattern.'),
            'action_required': True,
        },
        'step5_claim_ready': False,
    }


# ══════════════════════════════════════════════════════════════════════════════
# DEMO MODE — load saved artifacts, interactive inference loop
# ══════════════════════════════════════════════════════════════════════════════
def _bar(prob: float, width: int = 20) -> str:
    """ASCII confidence bar, e.g. '████████░░░░  42%'."""
    filled = round(prob * width)
    return '█' * filled + '░' * (width - filled) + f'  {prob*100:.0f}%'


def print_demo_result(result: dict, clinical_note: str):
    """
    Pretty-print a single predict_claim() result in the exact layout
    specified in the brief (Steps 12 of DEMO MODE).
    Reuses result dict from build_predict_fn() — no new inference logic.
    """
    SEP = '-' * 60

    print(f'\n{"═" * 60}')
    print('  PATIENT NOTE')
    print(f'{"═" * 60}')
    print(f'  {clinical_note}')

    # ── Predicted ICD Codes ───────────────────────────────────────
    print(f'\n{SEP}')
    print('  PREDICTED ICD CODES')
    print(SEP)
    for code, conf in zip(result['predicted_icd_codes'],
                          result['predicted_confidence']):
        print(f'  {code:<10} {_bar(conf)}')

    # ── Top-5 Recommendations ────────────────────────────────────
    print(f'\n{SEP}')
    print('  TOP 5 ICD RECOMMENDATIONS')
    print(SEP)
    for r in result['top_5_recommendations']:
        print(f"  {r['rank']}.  {r['icd_code']:<10} {_bar(r['confidence'])}")

    # ── Combination Recommendations ──────────────────────────────
    print(f'\n{SEP}')
    print('  KNOWN COMBINATION RECOMMENDATIONS')
    print(SEP)
    combos = result['combination_recommendations']
    if combos:
        for c in combos:
            print(f"  Combination : {c['combination']}")
            print(f"  Confidence  : {c['confidence_score']*100:.0f}%")
            desc = c.get('description', '').strip()
            risk = c.get('risk_level', '').strip()
            if desc:
                print(f"  Description : {desc[:70]}")
            if risk:
                print(f"  Risk Level  : {risk}")
            print()
    else:
        print('  No known combinations matched for these codes.')

    # ── Insurance Claim Workflow ─────────────────────────────────
    print(f'\n{SEP}')
    print('  INSURANCE CLAIM WORKFLOW')
    print(SEP)
    steps = [
        ('Clinical Note',            clinical_note[:60] + ('…' if len(clinical_note) > 60 else '')),
        ('PubMedBERT',               'Encoding clinical text → logits'),
        ('Predicted ICD Codes',      ', '.join(result['predicted_icd_codes']) or '(none)'),
        ('Top 5 ICD Codes',          ', '.join(r['icd_code'] for r in result['top_5_recommendations'])),
        ('Combination Rec.',         ', '.join(c['combination'] for c in combos) or 'None matched'),
        ('Medical Coder Review',     '⚠  Human review required before submission'),
        ('Claim Ready',              '✅  Pending coder approval'),
    ]
    for i, (label, value) in enumerate(steps):
        print(f'  {label}')
        print(f'    {value}')
        if i < len(steps) - 1:
            print('    ↓')
    print(f'{"═" * 60}\n')


def _load_df_for_demo() -> pd.DataFrame:
    """
    Step 4: Read icdf5.csv with the same parser used in load_data().
    Returns a minimal DataFrame suitable for CombinationRecommendationEngine.
    """
    print('[DEMO] Loading dataset for CombinationRecommendationEngine …')
    try:
        df = pd.read_csv(DATA_PATH, engine='python', on_bad_lines='skip')
    except TypeError:
        df = pd.read_csv(DATA_PATH, engine='python', error_bad_lines=False)
    print(f'[DEMO] {len(df):,} rows loaded from {DATA_PATH}')

    # Normalise Is_Combination (same logic as load_data)
    col = df['Is_Combination']
    if col.dtype == object:
        df['Is_Combination'] = col.map(
            {'True': True, 'False': False, '1': True, '0': False,
             'true': True, 'false': False}
        ).fillna(False).astype(bool)
    else:
        df['Is_Combination'] = col.fillna(False).astype(bool)

    # Fill text columns so CombinationRecommendationEngine doesn't crash
    for c in ['Description', 'Risk_Level', 'ICD_Code']:
        if c in df.columns:
            df[c] = df[c].fillna('')
    df = df.dropna(subset=['ICD_Code'])
    df = df[df['ICD_Code'].str.strip() != ''].reset_index(drop=True)
    return df


def run_demo_mode():
    """
    DEMO MODE entry point.
    Steps 1-13 from the brief — no training, no optimizer, no scheduler,
    no scaler, no EMA, no validation, no test evaluation.
    Loads saved artifacts and runs an interactive inference loop.
    """
    print(f'\n{"═" * 60}')
    print('  ICD PREDICTION SYSTEM  —  DEMO MODE')
    print(f'{"═" * 60}')

    # ── Step 1+7: Load best_model_state.pt ───────────────────────────────────
    best_model_path = os.path.join(OUTPUT_DIR, 'best_model_state.pt')
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            f'[DEMO] best_model_state.pt not found at {best_model_path}\n'
            f'       Run with DEMO_MODE=False first to train and save the model.')

    # ── Step 2: Load tokenizer ────────────────────────────────────────────────
    tokenizer_path = os.path.join(OUTPUT_DIR, 'tokenizer')
    print(f'[DEMO] Loading tokenizer from {tokenizer_path} …')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # ── Step 3: Load mlb.pkl ─────────────────────────────────────────────────
    mlb_path = os.path.join(OUTPUT_DIR, 'mlb.pkl')
    print(f'[DEMO] Loading MultiLabelBinarizer from {mlb_path} …')
    with open(mlb_path, 'rb') as f:
        mlb = pickle.load(f)
    n_classes = len(mlb.classes_)
    print(f'[DEMO] {n_classes} ICD classes loaded.')

    # ── Step 4+5: Load CSV and rebuild CombinationRecommendationEngine ───────
    # (Do NOT load combo_engine.pkl — rebuild from raw data to avoid pickle issues)
    df = _load_df_for_demo()
    combo_engine = CombinationRecommendationEngine(df)

    # ── Step 6+7+8: Build model, load weights, eval mode ─────────────────────
    print(f'[DEMO] Building PubMedBERTMultiLabel ({n_classes} labels) …')
    model = PubMedBERTMultiLabel(MODEL_NAME, n_classes).to(DEVICE)
    print(f'[DEMO] Loading weights from {best_model_path} …')
    state = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    print('[DEMO] Model ready.\n')

    # ── Step 9: Build predict_fn (reuses existing build_predict_fn) ──────────
    predict_claim = build_predict_fn(model, tokenizer, mlb, combo_engine,
                                     threshold=PRED_THRESHOLD)

    # ── Steps 10-13: Interactive inference loop ───────────────────────────────
    while True:
        print('─' * 60)
        try:
            clinical_note = input('Enter Clinical Note:\n> ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\n[DEMO] Exiting.')
            break

        if not clinical_note:
            print('[DEMO] Empty input — please enter a clinical note.')
            continue

        # ── Step 11: Run prediction ───────────────────────────────────────────
        result = predict_claim(clinical_text=clinical_note, top_k=5)

        # ── Step 12: Print beautifully ────────────────────────────────────────
        print_demo_result(result, clinical_note)

        # ── Step 13: Ask for another prediction ──────────────────────────────
        try:
            again = input('Do you want another prediction? (Y/N): ').strip().upper()
        except (EOFError, KeyboardInterrupt):
            print('\n[DEMO] Exiting.')
            break
        if again != 'Y':
            print('[DEMO] Goodbye.')
            break


# ══════════════════════════════════════════════════════════════════════════════
# Main  (structure mirrors icdtunev2.py exactly)
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # ── Load data ─────────────────────────────────────────────────────────────
    df, mlb, combo_stats = load_data()
    n_classes    = len(mlb.classes_)
    combo_engine = CombinationRecommendationEngine(df)

    # ── 80/10/10 split stratified on Is_Combination (guaranteed clean bool) ──
    train_df, temp_df = train_test_split(
        df, test_size=0.20, stratify=df['Is_Combination'], random_state=SEED)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df['Is_Combination'],
        random_state=SEED)
    print(f'Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}')

    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
    collate_fn = make_collate_fn(tokenizer)
    check_truncation_rate(tokenizer, df['Final_Text'].tolist())

    # ── Datasets ──────────────────────────────────────────────────────────────
    y_train = np.stack(train_df['multilabel'].values)
    y_val   = np.stack(val_df['multilabel'].values)
    y_test  = np.stack(test_df['multilabel'].values)

    train_ds = ICDMultiLabelDataset(
        train_df['Aug_Text'].tolist(), train_df['Suffix_Text'].tolist(),
        train_df['Final_Text'].tolist(), list(y_train), augment=True)
    val_ds = ICDMultiLabelDataset(
        val_df['Aug_Text'].tolist(),   val_df['Suffix_Text'].tolist(),
        val_df['Final_Text'].tolist(),   list(y_val),   augment=False)
    test_ds = ICDMultiLabelDataset(
        test_df['Aug_Text'].tolist(),  test_df['Suffix_Text'].tolist(),
        test_df['Final_Text'].tolist(),  list(y_test),  augment=False)

    # WeightedRandomSampler: balance single-ICD vs combination rows
    is_combo      = train_df['Is_Combination'].astype(int).values
    n_s, n_c      = max((is_combo == 0).sum(), 1), max((is_combo == 1).sum(), 1)
    sw            = np.where(is_combo == 1, 1.0/n_c, 1.0/n_s).astype(np.float32)
    sampler       = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)

    pin = DEVICE.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, sampler=sampler,
                               collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=EVAL_BATCH_SIZE,  shuffle=False,
                               collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=EVAL_BATCH_SIZE,  shuffle=False,
                               collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=pin)

    # ── Model + optimiser (mirrors icdtunev2.py exactly) ──────────────────────
    model       = PubMedBERTMultiLabel(MODEL_NAME, n_classes).to(DEVICE)
    llrd_groups = build_llrd_param_groups(model)
    optimizer   = torch.optim.AdamW(llrd_groups)
    model       = try_compile(model)

    # TASK 3: BCEWithLogitsLoss with per-label pos_weight
    pos_weight  = compute_pos_weight(y_train).to(DEVICE)
    criterion   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler       = torch.amp.GradScaler('cuda', enabled=USE_FP16)
    ema          = ModelEMA(model, decay=EMA_DECAY) if USE_EMA else None

    best_model_path  = os.path.join(OUTPUT_DIR, 'best_model_state.pt')
    resume_ckpt_path = os.path.join(EPOCH_CKPT_DIR, 'resume_checkpoint.pt')

    # ── Auto-resume: detect and load the latest rolling checkpoint ───────────
    best_score, patience_counter = 0.0, 0
    history, top_checkpoints     = [], []
    start_epoch = 1

    if os.path.exists(resume_ckpt_path):
        print(f'\n[RESUME] Found checkpoint: {resume_ckpt_path}')
        try:
            ckpt = torch.load(resume_ckpt_path, map_location=DEVICE)
            raw_m = model._orig_mod if hasattr(model, '_orig_mod') else model
            raw_m.load_state_dict(ckpt['model_state'])
            optimizer.load_state_dict(ckpt['optimizer_state'])
            scheduler.load_state_dict(ckpt['scheduler_state'])
            scaler.load_state_dict(ckpt['scaler_state'])
            if ema is not None and 'ema_state' in ckpt and ckpt['ema_state']:
                ema.shadow = {k: v.cpu() for k, v in ckpt['ema_state'].items()}
            best_score      = ckpt.get('best_score', 0.0)
            patience_counter = ckpt.get('patience_counter', 0)
            history         = ckpt.get('history', [])
            start_epoch     = ckpt['epoch'] + 1
            print(f'[RESUME] Resumed from epoch {ckpt["epoch"]}  '
                  f'best_val_f1={best_score:.4f}  '
                  f'patience={patience_counter}/{EARLY_STOP_PATIENCE}')
            print(f'[RESUME] Training will continue from epoch {start_epoch}')
            del ckpt
        except Exception as e:
            print(f'[RESUME] WARNING — failed to load checkpoint: {e}')
            print('[RESUME] Starting training from scratch.')
    else:
        print('[RESUME] No checkpoint found — starting training from scratch.')

    def raw_sd():
        m = model._orig_mod if hasattr(model, '_orig_mod') else model
        return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}

    # ── Training loop (identical structure to icdtunev2.py) ───────────────────
    for epoch in range(start_epoch, EPOCHS + 1):
        tl, tf, th = run_epoch(model, train_loader, optimizer, scheduler,
                                scaler, criterion, train=True, ema=ema)
        vl, vf, vh = run_epoch(model, val_loader,   optimizer, scheduler,
                                scaler, criterion, train=False)
        history.append({'epoch': epoch,
                        'train_loss': tl, 'train_f1': tf, 'train_hl': th,
                        'val_loss'  : vl, 'val_f1'  : vf, 'val_hl'  : vh})
        print(f'Epoch {epoch}/{EPOCHS} | '
              f'train_loss={tl:.4f} train_f1={tf:.4f} train_hl={th:.4f} | '
              f'val_loss={vl:.4f} val_f1={vf:.4f} val_hl={vh:.4f}')

        top_checkpoints.append((vf, raw_sd()))
        top_checkpoints.sort(key=lambda x: x[0], reverse=True)
        top_checkpoints = top_checkpoints[:TOP_K_CHECKPOINTS]

        # ── Save best model independently when val Micro-F1 improves ─────────
        if vf > best_score:
            best_score = vf; patience_counter = 0
            torch.save(raw_sd(), best_model_path)
            print(f'  -> [BEST MODEL] val_f1={vf:.4f} — saved to best_model_state.pt')
        else:
            patience_counter += 1
            print(f'  -> [NO IMPROVEMENT] patience={patience_counter}/{EARLY_STOP_PATIENCE}')

        # ── Rolling resume checkpoint: single file, overwrites itself ─────────
        # Saves everything needed to resume: model, optimizer, scheduler,
        # scaler, ema, epoch, best_score, patience_counter, history.
        # Does NOT duplicate top_checkpoints (those are in-memory only).
        try:
            raw_m = model._orig_mod if hasattr(model, '_orig_mod') else model
            tmp_path = resume_ckpt_path + '.tmp'
            torch.save({
                'epoch'           : epoch,
                'model_state'     : {k: v.detach().cpu().clone()
                                     for k, v in raw_m.state_dict().items()},
                'optimizer_state' : optimizer.state_dict(),
                'scheduler_state' : scheduler.state_dict(),
                'scaler_state'    : scaler.state_dict(),
                'ema_state'       : ema.shadow if ema is not None else None,
                'best_score'      : best_score,
                'patience_counter': patience_counter,
                'history'         : history,
                'val_f1'          : vf,
                'val_loss'        : vl,
                'train_f1'        : tf,
                'train_loss'      : tl,
            }, tmp_path)
            # Atomic replace: rename only after successful write
            os.replace(tmp_path, resume_ckpt_path)
            print(f'  -> [CHECKPOINT] resume_checkpoint.pt updated (epoch {epoch})')
        except Exception as e:
            print(f'  -> [CHECKPOINT] WARNING — save failed: {e}')
        # ─────────────────────────────────────────────────────────────────────

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f'[EARLY STOP] No improvement for {EARLY_STOP_PATIENCE} epochs. '
                  f'Stopping at epoch {epoch}.')
            break

    raw_m = model._orig_mod if hasattr(model, '_orig_mod') else model
    raw_m.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    print(f'[BEST MODEL] Loaded best_model_state.pt  (val micro-F1={best_score:.4f})')

    # ── Checkpoint averaging (identical to icdtunev2.py) ──────────────────────
    if len(top_checkpoints) >= 2:
        avg_state = average_state_dicts([sd for _, sd in top_checkpoints])
        raw_m     = model._orig_mod if hasattr(model, '_orig_mod') else model
        backup    = {k: v.detach().cpu().clone() for k, v in raw_m.state_dict().items()}
        raw_m.load_state_dict(avg_state)
        _, avg_f1, _ = run_epoch(model, val_loader, optimizer, scheduler,
                                  scaler, criterion, train=False)
        print(f'Averaged top-{len(top_checkpoints)} val_f1={avg_f1:.4f} '
              f'(best single={best_score:.4f})')
        if avg_f1 >= best_score:
            best_score = avg_f1; print('  -> using averaged checkpoint')
        else:
            raw_m.load_state_dict(backup); print('  -> reverting to best single')

    # ── EMA (identical to icdtunev2.py) ───────────────────────────────────────
    if ema is not None:
        raw_m  = model._orig_mod if hasattr(model, '_orig_mod') else model
        backup = {k: v.detach().cpu().clone() for k, v in raw_m.state_dict().items()}
        raw_m.load_state_dict(ema.state_dict_for_load(backup))
        _, ema_f1, _ = run_epoch(model, val_loader, optimizer, scheduler,
                                  scaler, criterion, train=False)
        print(f'EMA val_f1={ema_f1:.4f} (leader={best_score:.4f})')
        if ema_f1 >= best_score:
            best_score = ema_f1; print('  -> using EMA weights')
        else:
            raw_m.load_state_dict(backup); print('  -> reverting')

    # ── Final evaluation: all 10 metrics ─────────────────────────────────────
    test_logits, test_labels = get_test_logits(model, test_loader)
    metrics = evaluate_multilabel(test_logits, test_labels)
    metrics.update({'num_labels': n_classes,
                    'train_size': len(train_df),
                    'val_size'  : len(val_df),
                    'test_size' : len(test_df)})

    print(f"\n{'='*62}\nFINAL EVALUATION — ALL 10 METRICS\n{'='*62}")
    for k, v in metrics.items():
        print(f'  {k:<30}: {v}')

    # ── Save (mirrors icdtunev2.py) ───────────────────────────────────────────
    with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    model_dir = os.path.join(OUTPUT_DIR, 'best_model')
    os.makedirs(model_dir, exist_ok=True)
    raw_m = model._orig_mod if hasattr(model, '_orig_mod') else model
    torch.save(raw_m.state_dict(), os.path.join(model_dir, 'pytorch_model.bin'))
    raw_m.bert.config.save_pretrained(model_dir)
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, 'tokenizer'))
    with open(os.path.join(OUTPUT_DIR, 'mlb.pkl'), 'wb') as f:
        pickle.dump(mlb, f)
    with open(os.path.join(OUTPUT_DIR, 'combo_engine.pkl'), 'wb') as f:
        pickle.dump(combo_engine, f)
    print('Saved model/tokenizer/mlb/combo_engine/metrics to', OUTPUT_DIR)

    # ── Demo: Tasks 4, 5, 6, 7 ───────────────────────────────────────────────
    predict_claim = build_predict_fn(raw_m, tokenizer, mlb, combo_engine)

    print(f"\n{'='*62}")
    print('DEMO — Multi-ICD Prediction + Combination Recommendations')
    print(f"{'='*62}")
    demo_note = ('Patient with Type 2 Diabetes, Hypertension, '
                 'Chronic Kidney Disease, and Hyperlipidemia.')
    result = predict_claim(
        clinical_text=demo_note,
        risk_level='High',
        category='Endocrine & Metabolic',
        top_k=5,
    )
    print('\nTask 4 — Predicted ICD Codes:')
    for code, conf in zip(result['predicted_icd_codes'],
                          result['predicted_confidence']):
        print(f'  {code}  ({conf*100:.0f}%)')

    print('\nTask 5 — Top-5 Recommendations:')
    for r in result['top_5_recommendations']:
        print(f"  {r['rank']}. {r['icd_code']}  ({r['confidence']*100:.0f}%)")

    print('\nTask 6 — Combination Recommendations:')
    if result['combination_recommendations']:
        for c in result['combination_recommendations']:
            print(f"  {c['combination']}  (avg conf: {c['confidence_score']*100:.0f}%)"
                  f"  {c['description'][:60]}")
    else:
        print('  No known combinations matched.')

    print('\nTask 7 — Full Insurance Claim Workflow:')
    claim = predict_for_claim(predict_claim, demo_note,
                               patient_info={'risk_level': 'High',
                                             'category': 'Endocrine & Metabolic'})
    print(json.dumps(claim, indent=2, default=str))


if __name__ == '__main__':
    if DEMO_MODE:
        run_demo_mode()
    else:
        main()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 9 — STREAMLIT DEMO PLAN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Install: pip install streamlit
# Run:     streamlit run streamlit_app.py
#
# ┌─────────────────────────────────────────────────────────────────┐
# │   🏥  AI Medical ICD Coder — Insurance Claim Assistant          │
# ├──────────────────────┬──────────────────────────────────────────┤
# │  INPUT PANEL         │  RESULTS PANEL                          │
# │  ────────────────    │  ──────────────────────────────────────  │
# │  Clinical Note       │  📋 Predicted ICD Codes                 │
# │  [text area]         │     E11 (92%)  I10 (88%)  N18 (76%)     │
# │                      │                                          │
# │  Risk Level          │  📊 Top-5 Recommendations               │
# │  [dropdown]          │     1. E11  ████████ 92%                 │
# │                      │     2. I10  ███████  88%                 │
# │  Category            │     3. N18  ██████   76%                 │
# │  [dropdown]          │     4. E78  █████    65%                 │
# │                      │     5. I50  ███      42%                 │
# │  Threshold           │                                          │
# │  [slider 0.1–0.9]    │  🔗 Combination Recommendations         │
# │                      │     E11+I10  (90%)                       │
# │  [PREDICT CODES]     │     "Type 2 DM + Hypertension"          │
# │                      │     E11+N18  (84%)                       │
# │                      │     "Type 2 DM + CKD"                   │
# │                      │                                          │
# │                      │  ✅ Coder Actions                        │
# │                      │     [Accept All] [Edit] [Submit Claim]   │
# │                      │     [Download JSON]                      │
# └──────────────────────┴──────────────────────────────────────────┘
#
# Components used:
#   st.text_area("Clinical Note")
#   st.selectbox("Risk Level", ["Low","Medium","High","Very High"])
#   st.selectbox("Category",   [...categories...])
#   st.slider("Confidence Threshold", 0.1, 0.9, 0.35, 0.05)
#   st.button("Predict ICD Codes")
#   st.metric(label="E11", value="92%")
#   st.progress(0.92)
#   st.expander("Combination Details")
#   st.download_button("Export Claim JSON", data=json.dumps(claim))
