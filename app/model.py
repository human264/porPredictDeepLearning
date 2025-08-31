import os, json, math, random
import numpy as np
import torch
import torch.nn as nn
from app.db import get_conn, dict_cur, get_schema
from app.text import encode_text_to_tensor, CHARSET

S = get_schema()

# --------------------- 장치/시드 ---------------------
def set_global_seed(seed: int | None):
    if seed is None: return
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def _select_device():
    pref = os.getenv("DEVICE_PREFERENCE", "auto").lower()
    if pref in ("mps","metal"):
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device("cpu")

DEVICE = _select_device()
print(f"[hybrid-model] DEVICE={DEVICE}; mps_avail={torch.backends.mps.is_available()}")

# --------------------- 문자 CNN ---------------------
def _kernels():
    ks = os.getenv("CNN_KERNEL_SIZES", "2,3,4,5")
    return tuple(int(x) for x in ks.split(",") if x.strip())

EMB_DIM  = int(os.getenv("CNN_EMB_DIM", "48"))
OUT_CH   = int(os.getenv("CNN_OUT_CH",  "64"))
DROPOUT  = float(os.getenv("DROPOUT", "0.1"))
KERNELS  = _kernels()
VOCAB_SIZE = len(CHARSET) + 1
MAX_ITEM_LEN = int(os.getenv("MAX_ITEM_LEN", "2048"))
MLP_HIDDEN = int(os.getenv("MLP_HIDDEN", "256"))

def auto_cat_dim(card: int) -> int:
    # 카드inality에 따른 임베딩 차원 자동 산정
    default = int(os.getenv("CAT_EMB_DIM_DEFAULT", "0") or "0")
    if default > 0: return default
    return min(64, max(4, int(round(card ** 0.25) * 8)))

class ItemCNNEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=EMB_DIM, out_ch=OUT_CH, kernels=KERNELS, dropout=DROPOUT):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, out_ch, k) for k in kernels])
        self.dropout = nn.Dropout(dropout)
        self.out_dim = out_ch * len(kernels)
    def forward(self, char_ids: torch.Tensor):
        # char_ids: (L,)
        x = self.emb(char_ids)           # (L,E)
        x = x.transpose(0,1).unsqueeze(0)  # (1,E,L)
        feats = []
        for conv in self.convs:
            h = torch.relu(conv(x))      # (1,C,L')
            h = torch.max(h, dim=2)[0]   # (1,C)
            feats.append(h)
        z = torch.cat(feats, dim=1).squeeze(0)  # (C*|kernels|)
        return self.dropout(z)

class HybridItemClassifier(nn.Module):
    """
    문자CNN(Text) + 카테고리 임베딩 + 수치피처(정규화) MLP → 분류
    """
    def __init__(self, num_classes: int, cat_cardinalities: dict[str,int],
                 cat_emb_dims: dict[str,int], num_numeric: int):
        super().__init__()
        self.text_enc = ItemCNNEncoder(VOCAB_SIZE)
        self.cat_cols = list(cat_cardinalities.keys())
        self.cat_emb = nn.ModuleDict({
            col: nn.Embedding(cat_cardinalities[col], cat_emb_dims[col])
            for col in self.cat_cols
        })  # 0=UNK 포함하여 cardinality 반영
        cat_total = sum(cat_emb_dims.values())
        in_dim = self.text_enc.out_dim + cat_total + num_numeric
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, MLP_HIDDEN),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(MLP_HIDDEN, num_classes)
        )
    def forward(self, text_ids: torch.Tensor, cat_idxs: dict[str, torch.Tensor], num_feats: torch.Tensor):
        tvec = self.text_enc(text_ids)  # (T)
        embs = []
        for col in self.cat_cols:
            idx = cat_idxs[col].long().clamp(min=0)
            embs.append(self.cat_emb[col](idx.unsqueeze(0)).squeeze(0))
        x = torch.cat([tvec] + embs + [num_feats.float()], dim=0)
        return self.mlp(x)

# --------------------- 전역 상태 ---------------------
# 태스크별: item_act, item_mr
MODELS = {
    "item_act": None,
    "item_mr":  None
}
DETAILS = {          # model_versions.details 캐시
    "item_act": None,
    "item_mr":  None
}
VERSIONS = {
    "item_act": None,
    "item_mr":  None
}

# --------------------- 저장/로드 ---------------------
def _save_model(ver: int, task: str, model: nn.Module, details: dict):
    os.makedirs("models", exist_ok=True)
    path = f"./models/{task}_v{ver}.pt"
    torch.save(model.state_dict(), path)
    conn = get_conn(); cur = dict_cur(conn)
    details_to_store = dict(details)
    details_to_store["task"] = task
    cur.execute(f"UPDATE {S}.model_versions SET details=%s WHERE version_id=%s",
                (json.dumps(details_to_store), ver))
    conn.commit(); cur.close(); conn.close()
    return path

def _build_model_from_details(num_classes: int, details: dict) -> HybridItemClassifier:
    cat_card = {k:int(v) for k,v in details["cat_cardinalities"].items()}
    cat_emb_dims = {k:int(v) for k,v in details["cat_emb_dims"].items()}
    num_numeric = len(details["numeric_stats"])
    return HybridItemClassifier(num_classes, cat_card, cat_emb_dims, num_numeric).to(DEVICE)

def _load_latest(task: str):
    conn = get_conn(); cur = dict_cur(conn)
    cur.execute(f"""
      SELECT version_id, details
      FROM {S}.model_versions
      WHERE COALESCE(details->>'task','')=%s
      ORDER BY version_id DESC LIMIT 1
    """, (task,))
    row = cur.fetchone()
    cur.close(); conn.close()
    if not row: return None, None
    ver, det = row["version_id"], (row["details"] or {})
    return ver, det

def load_latest(task: str):
    ver, details = _load_latest(task)
    if not ver or not details:
        MODELS[task] = None; DETAILS[task] = None; VERSIONS[task] = None; return
    label_keys = details.get("label_keys", [])
    model = _build_model_from_details(len(label_keys), details)
    try:
        state = torch.load(f"./models/{task}_v{ver}.pt", map_location=DEVICE)
        model.load_state_dict(state, strict=False)
    except Exception as e:
        print(f"[{task}] load warning:", e)
    model.eval()
    MODELS[task] = model
    DETAILS[task] = details
    VERSIONS[task] = ver

# --------------------- 학습 유틸 ---------------------
CAT_COLS   = ["mccsno","block","event","sign","deptcode","shiptype"]
NUM_COLS   = ["duration"]   # 필요 시 추가 가능

def _gather_stats_and_vocabs(view_name: str):
    conn = get_conn()
    cats = {c:set() for c in CAT_COLS}
    sums = {n:0.0 for n in NUM_COLS}
    sums2= {n:0.0 for n in NUM_COLS}
    cnts = {n:0   for n in NUM_COLS}
    cur = conn.cursor(name=f"scan_{view_name}", withhold=True)
    cur.itersize = 1000
    cur.execute(f"SELECT item_name, spec_text, {', '.join(CAT_COLS+NUM_COLS)} FROM {S}.{view_name}")
    while True:
        rows = cur.fetchmany(2000)
        if not rows: break
        for (item_name, spec_text, *rest) in rows:
            off = 0
            for c in CAT_COLS:
                v = rest[off]; off += 1
                if v is not None and str(v).strip()!="":
                    cats[c].add(str(v))
            for n in NUM_COLS:
                v = rest[off - (len(CAT_COLS) - CAT_COLS.index(n) - 1)] if False else None
            # 위 한 줄은 사용 안 하므로 제거
            # num은 아래에서 별도로 다시 읽자
        # 숫자 피처만 다시 커서로 읽어 평균/분산
    cur.close()

    # 숫자 컬럼 통계
    cur2 = conn.cursor(name=f"scan_num_{view_name}", withhold=True)
    cur2.itersize = 1000
    cur2.execute(f"SELECT {', '.join(NUM_COLS)} FROM {S}.{view_name}")
    while True:
        rows = cur2.fetchmany(5000)
        if not rows: break
        for tup in rows:
            for i, n in enumerate(NUM_COLS):
                v = tup[i]
                if v is None: continue
                v = float(v)
                sums[n]  += v
                sums2[n] += v*v
                cnts[n]  += 1
    cur2.close(); conn.close()

    cat_vocabs = {c:{v:i+1 for i,v in enumerate(sorted(cats[c]))} for c in CAT_COLS}  # 0=UNK
    numeric_stats = {}
    for n in NUM_COLS:
        if cnts[n] == 0:
            numeric_stats[n] = {"mean":0.0,"std":1.0}
        else:
            mean = sums[n]/cnts[n]
            var  = max(1e-8, sums2[n]/cnts[n] - mean*mean)
            numeric_stats[n] = {"mean":mean,"std":math.sqrt(var)}
    cat_cardinalities = {c:(len(cat_vocabs[c])+1) for c in CAT_COLS}  # +1 for UNK(0)
    cat_emb_dims = {c:auto_cat_dim(cat_cardinalities[c]) for c in CAT_COLS}
    return cat_vocabs, numeric_stats, cat_cardinalities, cat_emb_dims

def _encode_features_row(r, details):
    # 텍스트
    text = " ".join([(r.get("item_name") or ""), (r.get("spec_text") or "")]).strip()
    text_ids = encode_text_to_tensor(text, MAX_ITEM_LEN, DEVICE)

    # 카테고리
    cat_idxs = {}
    vocabs = details["cat_vocabs"]
    for c in CAT_COLS:
        raw = r.get(c)
        idx = vocabs[c].get(str(raw), 0) if (raw is not None and str(raw).strip()!="") else 0
        cat_idxs[c] = torch.tensor(idx, dtype=torch.long, device=DEVICE)

    # 수치
    num_list = []
    stats = details["numeric_stats"]
    for n in NUM_COLS:
        v = r.get(n, None)
        if v is None:
            val = 0.0
        else:
            m, s = stats[n]["mean"], stats[n]["std"]
            val = (float(v)-m)/(s+1e-6)
        num_list.append(val)
    num_feats = torch.tensor(num_list, dtype=torch.float32, device=DEVICE)
    return text_ids, cat_idxs, num_feats

# --------------------- 학습 (item_act / item_mr) ---------------------
def train_items(task: str):
    """
    task='item_act' → vw_training_item_act에서 (actocode:actno)
    task='item_mr'  → vw_training_item_mr  에서 (mr_label)
    """
    assert task in ("item_act","item_mr")
    set_global_seed(int(os.getenv("GLOBAL_SEED","0")) or None)

    EPOCHS      = int(os.getenv("EPOCHS","3"))
    LR          = float(os.getenv("LEARNING_RATE","0.001"))
    ACC_STEPS   = int(os.getenv("ACCUM_STEPS","4"))
    CLIP_NORM   = float(os.getenv("CLIP_NORM","0"))
    view        = "vw_training_item_act" if task=="item_act" else "vw_training_item_mr"

    conn = get_conn(); cur = dict_cur(conn)

    # 라벨 스페이스
    if task == "item_act":
        # 활성 코드를 라벨로
        cur.execute(f"SELECT actocode, actno FROM {S}.activity_codes WHERE active=TRUE ORDER BY actocode, actno")
        labs = [f"{r['actocode']}:{r['actno']}" for r in cur.fetchall()]
    else:
        cur.execute(f"SELECT DISTINCT mr_label FROM {S}.vw_training_item_mr ORDER BY mr_label")
        labs = [r["mr_label"] for r in cur.fetchall() if (r["mr_label"] or "").strip()]
    if not labs:
        raise RuntimeError("No labels to train.")
    key_to_idx = {k:i for i,k in enumerate(labs)}

    # 버전 생성
    cur.execute(f"INSERT INTO {S}.model_versions(details) VALUES ('{{}}') RETURNING version_id")
    ver = cur.fetchone()["version_id"]; conn.commit()

    # 카테고리/수치 통계 & 보카브
    cat_vocabs, numeric_stats, cat_card, cat_emb_dims = _gather_stats_and_vocabs(view)

    # 모델 구성
    model = HybridItemClassifier(num_classes=len(labs), cat_cardinalities=cat_card,
                                 cat_emb_dims=cat_emb_dims, num_numeric=len(NUM_COLS)).to(DEVICE)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    # 스트리밍 학습
    stream = conn.cursor(name=f"{task}_train", withhold=True)
    stream.itersize = 1000
    cols = "item_name,spec_text," + ",".join(CAT_COLS+NUM_COLS)
    label_col = "actocode,actno" if task=="item_act" else "mr_label"
    stream.execute(f"SELECT {cols}, {label_col} FROM {S}.{view} ORDER BY pjtno,porser,porseq,revno,line_no")
    step = 0
    while True:
        rows = stream.fetchmany(1000)
        if not rows: break
        for tup in rows:
            # dict 로 재구성
            d = {}
            i = 0
            d["item_name"] = tup[i]; i+=1
            d["spec_text"] = tup[i]; i+=1
            for c in CAT_COLS+NUM_COLS:
                d[c] = tup[i]; i+=1
            if task=="item_act":
                actocode = tup[i]; actno = tup[i+1]
                ykey = f"{actocode}:{actno}"
            else:
                ykey = tup[i]
            if ykey not in key_to_idx:
                continue
            y = torch.tensor([key_to_idx[ykey]], dtype=torch.long, device=DEVICE)

            details = {
                "cat_vocabs": cat_vocabs,
                "numeric_stats": numeric_stats,
                "cat_cardinalities": cat_card,
                "cat_emb_dims": cat_emb_dims,
                "label_keys": labs
            }
            text_ids, cat_idxs, num_feats = _encode_features_row(d, details)
            logits = model(text_ids, cat_idxs, num_feats)      # (C,)
            loss = loss_fn(logits.unsqueeze(0), y)             # add batch dim
            (loss / ACC_STEPS).backward()
            step += 1
            if step % ACC_STEPS == 0:
                if CLIP_NORM > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                opt.step(); opt.zero_grad()
    stream.close()

    if step % ACC_STEPS != 0:
        if CLIP_NORM > 0:
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        opt.step(); opt.zero_grad()

    # 저장
    details = {
        "task": task,
        "label_keys": labs,
        "cat_vocabs": cat_vocabs,
        "numeric_stats": numeric_stats,
        "cat_cardinalities": cat_card,
        "cat_emb_dims": cat_emb_dims,
        "charset": CHARSET,
        "text_cnn": {"emb_dim": EMB_DIM, "out_ch": OUT_CH, "kernels": list(KERNELS)},
        "mlp_hidden": MLP_HIDDEN,
        "dropout": DROPOUT,
        "num_cols": NUM_COLS,
        "cat_cols": CAT_COLS
    }
    _save_model(ver, task, model, details)
    model.eval()

    # 메타 업데이트
    cur = dict_cur(conn)
    cur.execute(f"UPDATE {S}.model_versions SET trained_at=NOW() WHERE version_id=%s", (ver,))
    conn.commit(); cur.close(); conn.close()

    # 전역 로드
    load_latest(task)
    return ver, len(labs)

# --------------------- 예측 ---------------------
def predict_items(task: str, header_rows: list[dict], topk: int=3):
    if MODELS[task] is None or DETAILS[task] is None:
        raise RuntimeError(f"{task} model not loaded.")
    model = MODELS[task]; details = DETAILS[task]
    labs  = details["label_keys"]
    out = []
    with torch.no_grad():
        for r in header_rows:
            text_ids, cat_idxs, num_feats = _encode_features_row(r, details)
            logits = model(text_ids, cat_idxs, num_feats)
            probs  = torch.softmax(logits, dim=0).cpu().numpy()
            idx = int(np.argmax(probs))
            label = labs[idx]
            order = np.argsort(-probs)[:topk]
            top = [{"label": labs[i], "score": float(probs[i])} for i in order]
            out.append((label, float(probs[idx]), top))
    return out, VERSIONS[task]
