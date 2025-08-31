# app/model.py
import os, json
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from scipy import sparse
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import xgboost as xgb

from app.db import get_conn, dict_cur, get_schema
from app.text import encode_text_to_tensor, CHARSET, normalize_text

S = get_schema()

# --------------------- 설정 ---------------------
CAT_COLS = ["mccsno", "block", "event", "sign", "deptcode", "shiptype"]
NUM_COLS = ["duration"]

TEXT_ENCODER = os.getenv("TEXT_ENCODER", "charcnn").lower()  # charcnn | hash
TORCH_DEVICE_PREF = os.getenv("TORCH_DEVICE", "auto").lower()  # auto|cuda|mps|cpu

# Char-CNN 하이퍼파라미터
CHARCNN_MAX_LEN = int(os.getenv("CHARCNN_MAX_LEN", 1024))
CHARCNN_EMB_DIM = int(os.getenv("CHARCNN_EMB_DIM", 48))
CHARCNN_OUT_CH  = int(os.getenv("CHARCNN_OUT_CH", 64))
CHARCNN_KERNELS = tuple(int(k) for k in os.getenv("CHARCNN_KERNELS", "2,3,4,5").split(","))
CHARCNN_DROPOUT = float(os.getenv("CHARCNN_DROPOUT", "0.1"))
CHARCNN_EPOCHS  = int(os.getenv("CHARCNN_EPOCHS", "2"))
CHARCNN_LR      = float(os.getenv("CHARCNN_LR", "0.001"))
CHARCNN_BS      = int(os.getenv("CHARCNN_BATCH", "256"))

# HashingVectorizer (폴백용)
HASH_N_FEATURES = int(os.getenv("XGB_HASH_FEATURES", str(2**18)))
HASH_NGRAM_MIN  = int(os.getenv("XGB_CHAR_NGRAM_MIN", "3"))
HASH_NGRAM_MAX  = int(os.getenv("XGB_CHAR_NGRAM_MAX", "5"))

# --------------------- 디바이스 / XGBoost 선택 ---------------------
def _torch_device() -> torch.device:
    pref = TORCH_DEVICE_PREF
    if pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if pref == "cpu":
        return torch.device("cpu")
    # auto: cuda > mps > cpu
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = _torch_device()

def _choose_xgb_tree_and_predictor() -> Tuple[str, str]:
    """cuda 가능하면 gpu_hist/gpu_predictor, 아니면 hist/auto"""
    pref_method = os.getenv("XGB_TREE_METHOD", "auto").lower()
    pref_pred   = os.getenv("XGB_PREDICTOR", "auto").lower()
    method = pref_method if pref_method != "auto" else ("gpu_hist" if torch.cuda.is_available() else "hist")
    predictor = pref_pred if pref_pred != "auto" else ("gpu_predictor" if method == "gpu_hist" else "auto")
    return method, predictor

def _build_classifier(num_class: int,
                      force_method: Optional[str] = None,
                      force_predictor: Optional[str] = None) -> xgb.XGBClassifier:
    method, predictor = _choose_xgb_tree_and_predictor()
    method = force_method or method
    predictor = force_predictor or predictor
    return xgb.XGBClassifier(
        n_estimators=int(os.getenv("XGB_N_ESTIMATORS", "800")),
        learning_rate=float(os.getenv("XGB_LEARNING_RATE", "0.05")),
        max_depth=int(os.getenv("XGB_MAX_DEPTH", "8")),
        subsample=float(os.getenv("XGB_SUBSAMPLE", "0.9")),
        colsample_bytree=float(os.getenv("XGB_COLSAMPLE_BYTREE", "0.9")),
        reg_lambda=float(os.getenv("XGB_REG_LAMBDA", "1.0")),
        reg_alpha=float(os.getenv("XGB_REG_ALPHA", "0.0")),
        tree_method=method,
        predictor=predictor,
        n_jobs=int(os.getenv("XGB_N_JOBS", "0")),
        eval_metric="mlogloss",
        objective="multi:softprob" if num_class > 2 else "binary:logistic",
        num_class=num_class if num_class > 2 else None,
    )

# --------------------- 전역 상태 ---------------------
MODELS: Dict[str, Dict[str, Any]] = {"item_act": None, "item_mr": None}
DETAILS: Dict[str, Dict[str, Any]] = {"item_act": None, "item_mr": None}
VERSIONS: Dict[str, Optional[int]] = {"item_act": None, "item_mr": None}

# --------------------- Char-CNN ---------------------
VOCAB_SIZE = len(CHARSET) + 1  # 0=pad/unk

class CharCNNEncoder(nn.Module):
    def __init__(self,
                 vocab_size: int = VOCAB_SIZE,
                 emb_dim: int = CHARCNN_EMB_DIM,
                 out_ch: int = CHARCNN_OUT_CH,
                 kernels: Tuple[int, ...] = CHARCNN_KERNELS,
                 dropout: float = CHARCNN_DROPOUT):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, out_ch, k) for k in kernels])
        self.dropout = nn.Dropout(dropout)
        self.out_dim = out_ch * len(kernels)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(ids)           # (B, L, E)
        x = x.transpose(1, 2)       # (B, E, L)
        feats = []
        for conv in self.convs:
            h = torch.relu(conv(x))     # (B, C, L')
            h = torch.max(h, dim=2)[0]  # (B, C)
            feats.append(h)
        z = torch.cat(feats, dim=1)     # (B, C*|kernels|)
        return self.dropout(z)

class CharCNNClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.enc = CharCNNEncoder()
        self.fc  = nn.Linear(self.enc.out_dim, num_classes)
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.fc(self.enc(ids))

class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: Optional[np.ndarray], max_len: int):
        self.ids = [encode_text_to_tensor(t, max_len, device=torch.device("cpu")) for t in texts]
        self.labels = None if labels is None else torch.tensor(labels, dtype=torch.long)
    def __len__(self) -> int: return len(self.ids)
    def __getitem__(self, i: int):
        return (self.ids[i], self.labels[i]) if self.labels is not None else self.ids[i]

def _collate(batch):
    if isinstance(batch[0], tuple):
        ids, ys = zip(*batch)
        ids = pad_sequence(ids, batch_first=True, padding_value=0)
        ys  = torch.stack(ys)
        return ids.to(DEVICE), ys.to(DEVICE)
    ids = pad_sequence(batch, batch_first=True, padding_value=0)
    return ids.to(DEVICE)

@torch.no_grad()
def _encode_charcnn(texts: List[str], enc: CharCNNEncoder, max_len: int, bs: int) -> np.ndarray:
    enc.eval()
    ds = TextDataset([normalize_text(t) for t in texts], None, max_len)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=_collate)
    vecs = []
    for ids in dl:
        z = enc(ids)
        vecs.append(z.detach().cpu().numpy())
    return np.concatenate(vecs, axis=0).astype(np.float32)

def _train_charcnn(texts: List[str], y: np.ndarray, num_classes: int) -> Tuple[CharCNNEncoder, Dict[str, Any]]:
    ds = TextDataset([normalize_text(t) for t in texts], y, CHARCNN_MAX_LEN)
    dl = DataLoader(ds, batch_size=CHARCNN_BS, shuffle=True, collate_fn=_collate)
    model = CharCNNClassifier(num_classes=num_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=CHARCNN_LR)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(CHARCNN_EPOCHS):
        for ids, yy in dl:
            logits = model(ids)
            loss = loss_fn(logits, yy)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    enc = model.enc
    enc.eval()
    info = {
        "type": "charcnn",
        "dim": int(enc.out_dim),
        "max_len": CHARCNN_MAX_LEN,
        "emb_dim": CHARCNN_EMB_DIM,
        "out_ch": CHARCNN_OUT_CH,
        "kernels": list(CHARCNN_KERNELS),
        "dropout": CHARCNN_DROPOUT,
        "device": ("cuda" if torch.cuda.is_available()
                   else "mps" if torch.backends.mps.is_available()
                   else "cpu"),
    }
    return enc, info

# --------------------- 저장/로드 ---------------------
def _save_model(ver: int, task: str, bundle: Dict[str, Any], details: Dict[str, Any]) -> str:
    import json
    os.makedirs("models", exist_ok=True)
    path = f"./models/{task}_v{ver}.joblib"
    joblib.dump({"bundle": bundle, "details": details}, path)

    conn = get_conn(); cur = conn.cursor()
    det = dict(details); det["task"] = task
    cur.execute(
        f"UPDATE {S}.model_versions SET details=:1 WHERE version_id=:2",
        (json.dumps(det, ensure_ascii=False), ver)
    )
    conn.commit(); cur.close(); conn.close()
    return path


def load_latest(task: str) -> None:
    import json
    conn = get_conn(); cur = dict_cur(conn)
    # details에 '"task":"{task}"' 문자열이 들어있다는 가정 (11g JSON 함수 부재)
    cur.execute(f"""
      SELECT version_id, details FROM (
        SELECT version_id, details
        FROM {S}.model_versions
        WHERE details LIKE :task_like
        ORDER BY version_id DESC
      )
      WHERE ROWNUM <= 10
    """, {"task_like": f'%\"task\":\"{task}\"%'})
    rows = cur.fetchall(); cur.close(); conn.close()

    chosen = None
    for r in rows:
        ver = r["version_id"]
        det_raw = r.get("details")
        try:
            det = json.loads(det_raw) if det_raw else {}
        except Exception:
            det = {}
        path = f"./models/{task}_v{ver}.joblib"
        if os.path.exists(path):
            chosen = (ver, det, path); break

    if not chosen:
        MODELS[task] = None; DETAILS[task] = None; VERSIONS[task] = None
        return

    ver, details, path = chosen
    try:
        obj = joblib.load(path)
        bundle = obj["bundle"]
        tinfo = (obj.get("details") or details).get("text_encoder", {})
        if tinfo.get("type") == "charcnn":
            state_path = tinfo.get("state_path")
            enc = CharCNNEncoder()
            if state_path and os.path.exists(state_path):
                enc.load_state_dict(torch.load(state_path, map_location=DEVICE))
            enc.to(DEVICE).eval()
            bundle["char_encoder"] = enc
        MODELS[task] = bundle
        DETAILS[task] = obj.get("details") or details
        VERSIONS[task] = ver
    except Exception as e:
        print(f"[{task}] load warning:", e)
        MODELS[task] = None; DETAILS[task] = None; VERSIONS[task] = None


# --------------------- 데이터 적재 ---------------------
def _fetch_training_rows(task: str) -> List[Dict[str, Any]]:
    assert task in ("item_act", "item_mr")
    view = "vw_training_item_act" if task == "item_act" else "vw_training_item_mr"
    conn = get_conn(); cur = dict_cur(conn)
    if task == "item_act":
        cur.execute(f"""
            SELECT item_name, spec_text,
                   mccsno, block, event, sign, duration, deptcode, shiptype,
                   actocode, actno
            FROM {S}.{view}
        """)
    else:
        cur.execute(f"""
            SELECT item_name, spec_text,
                   mccsno, block, event, sign, duration, deptcode, shiptype,
                   mr_label
            FROM {S}.{view}
        """)
    rows = cur.fetchall(); cur.close(); conn.close()
    return rows

def _build_frame_and_labels(task: str, rows: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, List[str]]:
    texts, mccsno, block, event, sign, duration, deptcode, shiptype, labels = [], [], [], [], [], [], [], [], []
    for r in rows:
        text = " ".join([(r.get("item_name") or ""), (r.get("spec_text") or "")]).strip()
        texts.append(text)
        mccsno.append((r.get("mccsno") or "").strip() or "UNK")
        block.append((r.get("block") or "").strip() or "UNK")
        event.append((r.get("event") or "").strip() or "UNK")
        sign.append((r.get("sign") or "").strip() or "UNK")
        deptcode.append((r.get("deptcode") or "").strip() or "UNK")
        shiptype.append((r.get("shiptype") or "").strip() or "UNK")
        v = r.get("duration", None)
        duration.append(0 if v is None else float(v))
        if task == "item_act":
            lab = f"{r.get('actocode')}:{r.get('actno')}"
        else:
            lab = r.get("mr_label")
        if lab and str(lab).strip():
            labels.append(str(lab))

    df = pd.DataFrame({
        "text": texts, "mccsno": mccsno, "block": block, "event": event, "sign": sign,
        "deptcode": deptcode, "shiptype": shiptype, "duration": duration
    })
    return df, labels

# --------------------- 전처리 구성 ---------------------
def _pre_catnum() -> ColumnTransformer:
    try:
        cat_enc = OneHotEncoder(handle_unknown="ignore", dtype=np.float32, sparse_output=True)
    except TypeError:
        cat_enc = OneHotEncoder(handle_unknown="ignore", dtype=np.float32, sparse=True)
    num_scaler = StandardScaler(with_mean=False)
    return ColumnTransformer(
        transformers=[("cat", cat_enc, CAT_COLS), ("num", num_scaler, NUM_COLS)],
        sparse_threshold=0.1
    )

def _pre_hash() -> Tuple[ColumnTransformer, HashingVectorizer]:
    text_vec = HashingVectorizer(
        n_features=HASH_N_FEATURES,
        analyzer="char",
        ngram_range=(HASH_NGRAM_MIN, HASH_NGRAM_MAX),
        alternate_sign=False,
        norm=None,
        dtype=np.float32,
        preprocessor=normalize_text,
    )
    try:
        cat_enc = OneHotEncoder(handle_unknown="ignore", dtype=np.float32, sparse_output=True)
    except TypeError:
        cat_enc = OneHotEncoder(handle_unknown="ignore", dtype=np.float32, sparse=True)
    num_scaler = StandardScaler(with_mean=False)
    pre = ColumnTransformer(
        transformers=[("text", text_vec, "text"), ("cat", cat_enc, CAT_COLS), ("num", num_scaler, NUM_COLS)],
        sparse_threshold=0.1
    )
    return pre, text_vec

# --------------------- 학습 ---------------------
def _alloc_new_version_id(conn) -> int:
    """
    Oracle 11g: 시퀀스가 있으면 사용, 없으면 MAX+1 폴백
    환경변수 MODEL_VERSION_SEQ (예: MODEL_VERSIONS_SEQ) 지원
    """
    seq = os.getenv("MODEL_VERSION_SEQ", f"{S}.MODEL_VERSIONS_SEQ")
    cur = conn.cursor()
    try:
        cur.execute(f"SELECT {seq}.NEXTVAL FROM dual")
        ver = cur.fetchone()[0]
    except Exception:
        cur.execute(f"SELECT NVL(MAX(version_id),0)+1 FROM {S}.model_versions")
        ver = cur.fetchone()[0]
    cur.close()
    return int(ver)


def train_items(task: str) -> Tuple[int, int]:
    assert task in ("item_act", "item_mr")

    # 라벨 공간 (활성 조건은 환경/스키마에 맞게 조정)
    conn = get_conn(); cur = dict_cur(conn)
    if task == "item_act":
        # Postgres의 active=TRUE → Oracle에서는 숫자/문자 플래그를 쓰는 경우가 많음
        # 필요 시 NVL(active,1)=1 → 'Y'/'N'이면 NVL(active,'Y')='Y'로 바꾸세요.
        cur.execute(f"SELECT actocode, actno FROM {S}.activity_codes WHERE NVL(active,1)=1 ORDER BY actocode, actno")
        labs_master = [f"{r['actocode']}:{r['actno']}" for r in cur.fetchall()]
    else:
        cur.execute(f"""
            SELECT DISTINCT mr_label
            FROM {S}.vw_training_item_mr
            WHERE mr_label IS NOT NULL
            ORDER BY mr_label
        """)
        labs_master = [r["mr_label"] for r in cur.fetchall()]
    cur.close(); conn.close()
    if not labs_master:
        raise RuntimeError("No labels to train (master empty).")

    # 버전 발급 & 빈 row 삽입 (RETURNING 대체)
    conn = get_conn()
    ver = _alloc_new_version_id(conn)
    c0 = conn.cursor()
    c0.execute(f"INSERT INTO {S}.model_versions (version_id, details) VALUES (:1, :2)", (ver, "{}"))
    conn.commit(); c0.close(); conn.close()

    # ----- 이하 원본 로직 동일 (데이터 적재/학습/저장) -----
    rows = _fetch_training_rows(task)
    if not rows:
        raise RuntimeError("No training rows.")
    df, labels = _build_frame_and_labels(task, rows)

    le = LabelEncoder(); le.fit(labs_master)
    y = le.transform([lab if lab in labs_master else labs_master[0] for lab in labels])

    bundle: Dict[str, Any] = {}
    if TEXT_ENCODER == "charcnn":
        enc, tinfo = _train_charcnn(df["text"].tolist(), y, num_classes=len(labs_master))
        text_emb = _encode_charcnn(df["text"].tolist(), enc, CHARCNN_MAX_LEN, CHARCNN_BS)
        pre = _pre_catnum()
        catnum = pre.fit_transform(df)
        X = sparse.hstack([sparse.csr_matrix(text_emb), catnum], format="csr")
        state_path = f"./models/{task}_v{ver}_charcnn.pt"
        torch.save(enc.state_dict(), state_path)
        tinfo["state_path"] = state_path
        bundle["char_encoder"] = enc
    else:
        pre, _ = _pre_hash()
        X = pre.fit_transform(df)
        tinfo = {"type": "hash", "hash_features": HASH_N_FEATURES, "char_ngram": [HASH_NGRAM_MIN, HASH_NGRAM_MAX]}

    val_split = float(os.getenv("XGB_VALID_SPLIT", "0.1"))
    early_rounds = int(os.getenv("XGB_EARLY_STOPPING_ROUNDS", "50"))
    use_valid = (val_split > 0.0) and (len(np.unique(y)) > 1) and (X.shape[0] >= 50)
    if use_valid:
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=val_split, stratify=y, random_state=42)
    else:
        Xtr, ytr = X, y
        Xva, yva = None, None

    clf = _build_classifier(num_class=len(labs_master))
    tree_used = clf.get_xgb_params().get("tree_method")
    pred_used = clf.get_xgb_params().get("predictor")
    try:
        if use_valid:
            clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False, early_stopping_rounds=early_rounds)
        else:
            clf.fit(Xtr, ytr, verbose=False)
    except Exception:
        if str(tree_used).startswith("gpu"):
            clf = _build_classifier(len(labs_master), force_method="hist", force_predictor="auto")
            if use_valid:
                clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False, early_stopping_rounds=early_rounds)
            else:
                clf.fit(Xtr, ytr, verbose=False)
            tree_used, pred_used = "hist", "auto"
        else:
            raise

    details = {
        "task": task,
        "label_keys": labs_master,
        "cat_cols": CAT_COLS,
        "num_cols": NUM_COLS,
        "text_encoder": tinfo,
        "xgb": {
            "n_estimators": int(os.getenv("XGB_N_ESTIMATORS", "800")),
            "learning_rate": float(os.getenv("XGB_LEARNING_RATE", "0.05")),
            "max_depth": int(os.getenv("XGB_MAX_DEPTH", "8")),
            "subsample": float(os.getenv("XGB_SUBSAMPLE", "0.9")),
            "colsample_bytree": float(os.getenv("XGB_COLSAMPLE_BYTREE", "0.9")),
            "tree_method": os.getenv("XGB_TREE_METHOD", "auto"),
            "predictor": os.getenv("XGB_PREDICTOR", "auto"),
            "tree_method_used": tree_used, "predictor_used": pred_used,
        },
    }
    bundle.update({"preprocessor": pre, "clf": clf, "label_encoder": le})
    _save_model(ver, task, bundle, details)

    # 학습 시간 갱신: NOW() → SYSDATE
    conn = get_conn(); cur = conn.cursor()
    cur.execute(f"UPDATE {S}.model_versions SET trained_at=SYSDATE WHERE version_id=:1", (ver,))
    conn.commit(); cur.close(); conn.close()

    load_latest(task)
    return ver, len(labs_master)
# --------------------- 예측 ---------------------
def _rows_to_frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    texts, mccsno, block, event, sign, duration, deptcode, shiptype = [], [], [], [], [], [], [], []
    for r in rows:
        texts.append((" ".join([(r.get("item_name") or ""), (r.get("spec_text") or "")]).strip()))
        mccsno.append((r.get("mccsno") or "").strip() or "UNK")
        block.append((r.get("block") or "").strip() or "UNK")
        event.append((r.get("event") or "").strip() or "UNK")
        sign.append((r.get("sign") or "").strip() or "UNK")
        deptcode.append((r.get("deptcode") or "").strip() or "UNK")
        shiptype.append((r.get("shiptype") or "").strip() or "UNK")
        v = r.get("duration", None)
        duration.append(0 if v is None else float(v))
    return pd.DataFrame({
        "text": texts, "mccsno": mccsno, "block": block, "event": event, "sign": sign,
        "deptcode": deptcode, "shiptype": shiptype, "duration": duration
    })

def predict_items(task: str, header_rows: List[Dict[str, Any]], topk: int = 3):
    if MODELS[task] is None or DETAILS[task] is None:
        raise RuntimeError(f"{task} model not loaded.")
    bundle = MODELS[task]
    pre = bundle["preprocessor"]; clf = bundle["clf"]; le = bundle["label_encoder"]
    label_keys = list(le.classes_)
    details = DETAILS[task]; tinfo = details.get("text_encoder", {"type": "hash"})
    df = _rows_to_frame(header_rows)

    if tinfo.get("type") == "charcnn":
        enc: CharCNNEncoder = bundle.get("char_encoder")
        if enc is None:
            enc = CharCNNEncoder()
            state_path = tinfo.get("state_path")
            if state_path and os.path.exists(state_path):
                enc.load_state_dict(torch.load(state_path, map_location=DEVICE))
            enc.to(DEVICE).eval()
            bundle["char_encoder"] = enc
        emb = _encode_charcnn(df["text"].tolist(), enc, tinfo.get("max_len", CHARCNN_MAX_LEN), CHARCNN_BS)
        catnum = pre.transform(df)
        X = sparse.hstack([sparse.csr_matrix(emb), catnum], format="csr")
    else:
        X = pre.transform(df)

    proba = clf.predict_proba(X)
    out = []
    if proba.ndim == 1 or proba.shape[1] == 1:
        p1 = proba.ravel(); p0 = 1.0 - p1
        probs = np.vstack([p0, p1]).T
    else:
        probs = proba

    for i in range(probs.shape[0]):
        vec = probs[i]; idx = int(np.argmax(vec))
        label = label_keys[idx]
        order = np.argsort(-vec)[:min(topk, len(label_keys))]
        top = [{"label": label_keys[j], "score": float(vec[j])} for j in order]
        out.append((label, float(vec[idx]), top))
    return out, VERSIONS[task]
