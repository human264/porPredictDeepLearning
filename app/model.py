# app/model.py
import os, json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from scipy import sparse
import joblib
import xgboost as xgb

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from app.db import get_conn, dict_cur, get_schema
from app.text import encode_text_to_tensor, CHARSET, normalize_text

S = get_schema()

# --------------------- 설정 ---------------------
CAT_COLS = ["mccsno","block","event","sign","deptcode","shiptype"]
NUM_COLS = ["duration"]

# 텍스트 인코더 선택: charcnn | hash (hash = 완전 CPU 경로 폴백)
TEXT_ENCODER = os.getenv("TEXT_ENCODER", "charcnn").lower()

# Char-CNN 하이퍼파라미터 (.env로 조정)
CHARCNN_MAX_LEN   = int(os.getenv("CHARCNN_MAX_LEN", 1024))
CHARCNN_EMB_DIM   = int(os.getenv("CHARCNN_EMB_DIM", 48))
CHARCNN_OUT_CH    = int(os.getenv("CHARCNN_OUT_CH", 64))
CHARCNN_KERNELS   = tuple(int(k) for k in (os.getenv("CHARCNN_KERNELS","2,3,4,5").split(",")))
CHARCNN_DROPOUT   = float(os.getenv("CHARCNN_DROPOUT", "0.1"))
CHARCNN_EPOCHS    = int(os.getenv("CHARCNN_EPOCHS", "2"))
CHARCNN_LR        = float(os.getenv("CHARCNN_LR", "0.001"))
CHARCNN_BS        = int(os.getenv("CHARCNN_BATCH", "256"))
TORCH_DEVICE_PREF = os.getenv("TORCH_DEVICE", "auto").lower()  # auto|mps|cpu

def _torch_device():
    pref = TORCH_DEVICE_PREF  # auto|mps|cuda|cpu
    if pref == "cuda":
        return torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    if pref == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if pref == "cpu":
        return torch.device("cpu")
    # auto: 우선순위 cuda > mps > cpu
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = _torch_device()

# 해시 벡터라이저(폴백) 설정
HASH_N_FEATURES = int(os.getenv("XGB_HASH_FEATURES", str(2**18)))  # 262,144
HASH_NGRAM_MIN  = int(os.getenv("XGB_CHAR_NGRAM_MIN", "3"))
HASH_NGRAM_MAX  = int(os.getenv("XGB_CHAR_NGRAM_MAX", "5"))
# --- replace _build_classifier ---
def _build_classifier(num_class: int, force_method: str | None = None, force_predictor: str | None = None):
    method, predictor = _choose_xgb_tree_and_predictor()
    if force_method is not None:
        method = force_method
    if force_predictor is not None:
        predictor = force_predictor

    clf = xgb.XGBClassifier(
        n_estimators=int(os.getenv("XGB_N_ESTIMATORS", "800")),
        learning_rate=float(os.getenv("XGB_LEARNING_RATE", "0.05")),
        max_depth=int(os.getenv("XGB_MAX_DEPTH", "8")),
        subsample=float(os.getenv("XGB_SUBSAMPLE", "0.9")),
        colsample_bytree=float(os.getenv("XGB_COLSAMPLE_BYTREE", "0.9")),
        reg_lambda=float(os.getenv("XGB_REG_LAMBDA", "1.0")),
        reg_alpha=float(os.getenv("XGB_REG_ALPHA", "0.0")),
        tree_method=method,                  # auto→gpu_hist/hist
        predictor=predictor,                 # auto→gpu_predictor/auto
        n_jobs=int(os.getenv("XGB_N_JOBS", "0")),
        eval_metric="mlogloss",
        objective="multi:softprob" if num_class > 2 else "binary:logistic",
        num_class=num_class if num_class > 2 else None,
    )
    return clf
# --------------------- 전역 상태 ---------------------
MODELS: Dict[str, Dict] = {"item_act": None, "item_mr": None}
DETAILS: Dict[str, Dict] = {"item_act": None, "item_mr": None}
VERSIONS: Dict[str, int | None] = {"item_act": None, "item_mr": None}

# --------------------- Char-CNN ---------------------
VOCAB_SIZE = len(CHARSET) + 1  # 0=pad/unk

class CharCNNEncoder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, emb_dim=CHARCNN_EMB_DIM,
                 out_ch=CHARCNN_OUT_CH, kernels=CHARCNN_KERNELS, dropout=CHARCNN_DROPOUT):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, out_ch, k) for k in kernels])
        self.dropout = nn.Dropout(dropout)
        self.out_dim = out_ch * len(kernels)

    def forward(self, ids: torch.Tensor):  # ids: (B,L)
        x = self.emb(ids)            # (B,L,E)
        x = x.transpose(1, 2)        # (B,E,L)
        feats = []
        for conv in self.convs:
            h = torch.relu(conv(x))  # (B,C,L')
            h = torch.max(h, dim=2)[0]  # (B,C)
            feats.append(h)
        z = torch.cat(feats, dim=1)  # (B, C*|kernels|)
        return self.dropout(z)

class CharCNNClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.enc = CharCNNEncoder()
        self.fc  = nn.Linear(self.enc.out_dim, num_classes)

    def forward(self, ids: torch.Tensor):
        z = self.enc(ids)           # (B, D)
        return self.fc(z)           # (B, C)

class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: np.ndarray | None, max_len: int):
        self.ids = [encode_text_to_tensor(t, max_len, device=torch.device("cpu")) for t in texts]
        self.labels = None if labels is None else torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        if self.labels is None:
            return self.ids[i]
        return self.ids[i], self.labels[i]

def _collate(batch):
    if isinstance(batch[0], tuple):
        ids, ys = zip(*batch)
        ids = pad_sequence(ids, batch_first=True, padding_value=0)
        ys  = torch.stack(ys)
        return ids.to(DEVICE), ys.to(DEVICE)
    else:
        ids = pad_sequence(batch, batch_first=True, padding_value=0)
        return ids.to(DEVICE)

@torch.no_grad()
def _encode_charcnn(texts: List[str], enc: CharCNNEncoder, max_len: int, bs: int) -> np.ndarray:
    enc.eval()
    ds = TextDataset([normalize_text(t) for t in texts], labels=None, max_len=max_len)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=_collate)
    vecs = []
    for ids in dl:
        z = enc(ids)               # (B, D)
        vecs.append(z.detach().cpu().numpy())
    return np.concatenate(vecs, axis=0).astype(np.float32)

def _train_charcnn(texts: List[str], y: np.ndarray, num_classes: int) -> Tuple[CharCNNEncoder, Dict]:
    ds = TextDataset([normalize_text(t) for t in texts], labels=y, max_len=CHARCNN_MAX_LEN)
    dl = DataLoader(ds, batch_size=CHARCNN_BS, shuffle=True, collate_fn=_collate)

    model = CharCNNClassifier(num_classes=num_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=CHARCNN_LR)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(CHARCNN_EPOCHS):
        for ids, yy in dl:
            logits = model(ids)
            loss = loss_fn(logits, yy)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    enc = model.enc  # 분류 헤드는 폐기하고 인코더만 사용
    enc.eval()
    info = {
        "type": "charcnn",
        "dim": int(enc.out_dim),
        "max_len": CHARCNN_MAX_LEN,
        "emb_dim": CHARCNN_EMB_DIM,
        "out_ch": CHARCNN_OUT_CH,
        "kernels": list(CHARCNN_KERNELS),
        "dropout": CHARCNN_DROPOUT,
        "device": "mps" if torch.backends.mps.is_available() else "cpu",
    }
    return enc, info

# --------------------- 저장/로드 ---------------------
def _save_model(ver: int, task: str, bundle: Dict, details: dict):
    os.makedirs("models", exist_ok=True)
    path = f"./models/{task}_v{ver}.joblib"
    joblib.dump({"bundle": bundle, "details": details}, path)
    conn = get_conn(); cur = dict_cur(conn)
    det = dict(details); det["task"] = task
    cur.execute(f"UPDATE {S}.model_versions SET details=%s WHERE version_id=%s",
                (json.dumps(det), ver))
    conn.commit(); cur.close(); conn.close()
    return path

def _load_latest_meta(task: str) -> Tuple[int | None, dict | None]:
    conn = get_conn(); cur = dict_cur(conn)
    cur.execute(f"""
      SELECT version_id, details
      FROM {S}.model_versions
      WHERE COALESCE(details->>'task','')=%s
      ORDER BY version_id DESC
      LIMIT 1
    """, (task,))
    row = cur.fetchone()
    cur.close(); conn.close()
    if not row: return None, None
    return row["version_id"], (row["details"] or {})

def load_latest(task: str):
    conn = get_conn(); cur = dict_cur(conn)
    cur.execute(f"""
      SELECT version_id, details
      FROM {S}.model_versions
      WHERE COALESCE(details->>'task','')=%s
      ORDER BY version_id DESC
      LIMIT 10
    """, (task,))
    rows = cur.fetchall(); cur.close(); conn.close()

    chosen = None
    for r in rows:
        ver = r["version_id"]; det = r["details"] or {}
        path = f"./models/{task}_v{ver}.joblib"
        if os.path.exists(path):
            chosen = (ver, det, path); break

    if not chosen:
        MODELS[task] = None; DETAILS[task] = None; VERSIONS[task] = None; return

    ver, details, path = chosen
    try:
        obj = joblib.load(path)
        bundle = obj["bundle"]
        # Char-CNN 상태 로딩 (필요 시)
        tinfo = (obj.get("details") or details).get("text_encoder", {})
        if tinfo.get("type") == "charcnn":
            # 인코더 state_dict 파일에서 로드
            state_path = tinfo.get("state_path")
            enc = CharCNNEncoder()
            if state_path and os.path.exists(state_path):
                enc.load_state_dict(torch.load(state_path, map_location=DEVICE))
            enc.to(DEVICE).eval()
            bundle["char_encoder"] = enc
        MODELS[task]  = bundle
        DETAILS[task] = obj.get("details") or details
        VERSIONS[task]= ver
    except Exception as e:
        print(f"[{task}] load warning:", e)
        MODELS[task] = None; DETAILS[task] = None; VERSIONS[task] = None

# --------------------- 데이터 적재 ---------------------
def _fetch_training_rows(task: str) -> List[Dict]:
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

def _build_frame_and_labels(task: str, rows: List[Dict]) -> Tuple[pd.DataFrame, List[str]]:
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
        if not (lab and str(lab).strip()):
            continue
        labels.append(str(lab))

    df = pd.DataFrame({
        "text": texts,
        "mccsno": mccsno,
        "block": block,
        "event": event,
        "sign": sign,
        "deptcode": deptcode,
        "shiptype": shiptype,
        "duration": duration
    })
    return df, labels

# --------------------- 전처리 구성 ---------------------
def _pre_catnum():
    # sklearn 1.4+와 1.3- 모두 호환
    try:
        cat_enc = OneHotEncoder(handle_unknown="ignore", dtype=np.float32, sparse_output=True)
    except TypeError:
        cat_enc = OneHotEncoder(handle_unknown="ignore", dtype=np.float32, sparse=True)
    num_scaler = StandardScaler(with_mean=False)
    pre = ColumnTransformer(
        transformers=[
            ("cat",  cat_enc,  CAT_COLS),
            ("num",  num_scaler, NUM_COLS),
        ],
        sparse_threshold=0.1
    )
    return pre

def _pre_hash():
    text_vec = HashingVectorizer(
        n_features=HASH_N_FEATURES,
        analyzer="char",
        ngram_range=(HASH_NGRAM_MIN, HASH_NGRAM_MAX),
        alternate_sign=False,
        norm=None,
        dtype=np.float32,
        preprocessor=normalize_text
    )
    try:
        cat_enc = OneHotEncoder(handle_unknown="ignore", dtype=np.float32, sparse_output=True)
    except TypeError:
        cat_enc = OneHotEncoder(handle_unknown="ignore", dtype=np.float32, sparse=True)
    num_scaler = StandardScaler(with_mean=False)
    pre = ColumnTransformer(
        transformers=[
            ("text", text_vec, "text"),
            ("cat",  cat_enc,  CAT_COLS),
            ("num",  num_scaler, NUM_COLS),
        ],
        sparse_threshold=0.1
    )
    return pre, text_vec

# --------------------- 학습 ---------------------
def train_items(task: str) -> Tuple[int, int]:
    assert task in ("item_act", "item_mr")

    # (A) 라벨 공간
    conn = get_conn(); cur = dict_cur(conn)
    if task == "item_act":
        cur.execute(f"SELECT actocode, actno FROM {S}.activity_codes WHERE active=TRUE ORDER BY actocode, actno")
        labs_master = [f"{r['actocode']}:{r['actno']}" for r in cur.fetchall()]
    else:
        cur.execute(f"SELECT DISTINCT mr_label FROM {S}.vw_training_item_mr WHERE mr_label IS NOT NULL ORDER BY mr_label")
        labs_master = [r["mr_label"] for r in cur.fetchall()]
    cur.close(); conn.close()
    if not labs_master:
        raise RuntimeError("No labels to train (master empty).")

    # (B) 버전 먼저 발급 (charcnn state 파일명에 사용)
    conn = get_conn(); cur = dict_cur(conn)
    cur.execute(f"INSERT INTO {S}.model_versions(details) VALUES ('{{}}') RETURNING version_id")
    ver = cur.fetchone()["version_id"]; conn.commit(); cur.close(); conn.close()

    # (C) 데이터 적재
    rows = _fetch_training_rows(task)
    if not rows:
        raise RuntimeError("No training rows.")
    df, labels = _build_frame_and_labels(task, rows)

    # (D) 라벨 인코딩(마스터 기준 고정)
    le = LabelEncoder()
    le.fit(labs_master)
    y = le.transform([lab if lab in labs_master else labs_master[0] for lab in labels])

    # (E) 텍스트 인코딩 + cat/num 전처리
    bundle: Dict = {}
    if TEXT_ENCODER == "charcnn":
        # Char-CNN 학습(MPS), 임베딩 추출
        enc, tinfo = _train_charcnn(df["text"].tolist(), y, num_classes=len(labs_master))
        # 임베딩 전체 계산
        text_emb = _encode_charcnn(df["text"].tolist(), enc, CHARCNN_MAX_LEN, CHARCNN_BS)  # (N, D)
        pre = _pre_catnum()
        catnum = pre.fit_transform(df)                                         # (N, S)
        X = sparse.hstack([sparse.csr_matrix(text_emb), catnum], format="csr") # (N, D+S)
        # 인코더 state 저장
        state_path = f"./models/{task}_v{ver}_charcnn.pt"
        torch.save(enc.state_dict(), state_path)
        tinfo["state_path"] = state_path
        bundle["char_encoder"] = enc  # 메모리 내 보관(서버 재시작 시 details로부터 로드)
    else:
        # 폴백: 문자 n-gram 해싱(완전 CPU)
        pre, _ = _pre_hash()
        X = pre.fit_transform(df)
        tinfo = {
            "type": "hash",
            "hash_features": HASH_N_FEATURES,
            "char_ngram": [HASH_NGRAM_MIN, HASH_NGRAM_MAX],
        }

    # (F) 학습/검증 분할
    val_split = float(os.getenv("XGB_VALID_SPLIT", "0.1"))
    early_rounds = int(os.getenv("XGB_EARLY_STOPPING_ROUNDS", "50"))
    use_valid = (val_split > 0.0) and (len(np.unique(y)) > 1) and (X.shape[0] >= 50)

    if use_valid:
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=val_split, stratify=y, random_state=42)
    else:
        Xtr, ytr = X, y
        Xva, yva = None, None

    # (G) XGBoost 학습
    clf = _build_classifier(num_class=len(labs_master))
    tree_used = clf.get_xgb_params().get("tree_method")
    pred_used = clf.get_xgb_params().get("predictor")

    try:
        if use_valid:
            clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False, early_stopping_rounds=early_rounds)
        else:
            clf.fit(Xtr, ytr, verbose=False)
    except Exception as e:
        # GPU 빌드/드라이버 문제로 실패 시 자동 폴백
        # (예: Windows에서 xgboost가 GPU 미지원 빌드인 경우)
        if str(tree_used).startswith("gpu"):
            # CPU로 재시도
            clf = _build_classifier(num_class=len(labs_master), force_method="hist", force_predictor="auto")
            if use_valid:
                clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False, early_stopping_rounds=early_rounds)
            else:
                clf.fit(Xtr, ytr, verbose=False)
            tree_used = "hist";
            pred_used = "auto"
        else:
            raise

    # (H) 저장/등록
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
            "tree_method_used": tree_used,
            "predictor_used": pred_used
        }
    }
    bundle.update({"preprocessor": pre, "clf": clf, "label_encoder": le})
    _save_model(ver, task, bundle, details)

    # 메타 갱신 및 전역 로드
    conn = get_conn(); cur = dict_cur(conn)
    cur.execute(f"UPDATE {S}.model_versions SET trained_at=NOW() WHERE version_id=%s", (ver,))
    conn.commit(); cur.close(); conn.close()

    load_latest(task)
    return ver, len(labs_master)

# --------------------- 예측 ---------------------
def _rows_to_frame(rows: List[Dict]) -> pd.DataFrame:
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
        "text": texts,
        "mccsno": mccsno,
        "block": block,
        "event": event,
        "sign": sign,
        "deptcode": deptcode,
        "shiptype": shiptype,
        "duration": duration
    })

def predict_items(task: str, header_rows: List[Dict], topk: int = 3):
    if MODELS[task] is None or DETAILS[task] is None:
        raise RuntimeError(f"{task} model not loaded.")
    bundle = MODELS[task]
    pre = bundle["preprocessor"]; clf = bundle["clf"]; le = bundle["label_encoder"]
    label_keys = list(le.classes_)
    details = DETAILS[task]; tinfo = details.get("text_encoder", {"type":"hash"})

    df = _rows_to_frame(header_rows)

    if tinfo.get("type") == "charcnn":
        enc: CharCNNEncoder = bundle.get("char_encoder")
        if enc is None:
            # 안전장치: 서버가 재시작되어 메모리 캐시가 없다면 파일에서 로딩
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
        # hash 경로: ColumnTransformer가 text 포함
        X = pre.transform(df)

    proba = clf.predict_proba(X)
    out = []
    if proba.ndim == 1 or proba.shape[1] == 1:
        p1 = proba.ravel(); p0 = 1.0 - p1
        probs = np.vstack([p0, p1]).T
    else:
        probs = proba

    for i in range(probs.shape[0]):
        vec = probs[i]
        idx = int(np.argmax(vec))
        label = label_keys[idx]
        order = np.argsort(-vec)[:min(topk, len(label_keys))]
        top = [{"label": label_keys[j], "score": float(vec[j])} for j in order]
        out.append((label, float(vec[idx]), top))
    return out, VERSIONS[task]

# --- add this helper near _build_classifier ---
def _choose_xgb_tree_and_predictor():
    """XGBoost 트리/프리딕터 자동 결정: cuda 가능하면 gpu_hist, 아니면 hist"""
    pref_method = os.getenv("XGB_TREE_METHOD", "auto").lower()
    pref_pred   = os.getenv("XGB_PREDICTOR", "auto").lower()

    if pref_method != "auto":
        method = pref_method
    else:
        # torch.cuda로 간단히 감지 (없으면 CPU)
        method = "gpu_hist" if (torch.cuda.is_available()) else "hist"

    if pref_pred != "auto":
        predictor = pref_pred
    else:
        predictor = "gpu_predictor" if method == "gpu_hist" else "auto"
    return method, predictor
