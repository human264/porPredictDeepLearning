# app/model.py
import os, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from app.db import get_conn, dict_cur, get_schema

S = get_schema()

def _select_device():
    pref = os.getenv("DEVICE_PREFERENCE", "auto").lower()
    if pref in ("mps","metal"):
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device("cpu")

DEVICE = _select_device()
print(f"[model] DEVICE={DEVICE}; mps_avail={torch.backends.mps.is_available()}")

# ---------- 문자 집합 ----------
CHARSET = "abcdefghijklmnopqrstuvwxyz0123456789 .-/%xdiam"
char_to_idx = {c: i+1 for i, c in enumerate(CHARSET)}  # 0은 <unk>/<pad>
VOCAB_SIZE = len(char_to_idx) + 1

def encode_text_to_tensor(text: str, max_len: int):
    s = (text or "")[:max_len]
    return torch.tensor([char_to_idx.get(ch, 0) for ch in s], dtype=torch.long, device=DEVICE)

def _kernels():
    ks = os.getenv("CNN_KERNEL_SIZES", "2,3,4,5")
    return tuple(int(x) for x in ks.split(",") if x.strip())

EMB_DIM = int(os.getenv("CNN_EMB_DIM", "48"))
OUT_CH  = int(os.getenv("CNN_OUT_CH",  "64"))
KERNELS = _kernels()
DROPOUT = float(os.getenv("CNN_DROPOUT", "0.1"))

# ---------- 순수 CNN 인코더/분류기 ----------
class ItemCNNEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=EMB_DIM, out_ch=OUT_CH, kernels=KERNELS, dropout=DROPOUT):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, out_ch, kernel_size=k) for k in kernels])
        self.dropout = nn.Dropout(dropout)
        self.out_dim = out_ch * len(kernels)

    def forward(self, char_ids):
        x = self.emb(char_ids)            # (L,E)
        x = x.transpose(0,1).unsqueeze(0) # (1,E,L)
        feats = []
        for conv in self.convs:
            h = torch.relu(conv(x))       # (1,C,L')
            h = torch.max(h, dim=2)[0]    # (1,C)
            feats.append(h)
        z = torch.cat(feats, dim=1)       # (1,C*|kernels|)
        z = self.dropout(z)
        return z.squeeze(0)               # (C*|kernels|,)

class SetCNNClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()
        self.item_enc = ItemCNNEncoder(vocab_size)
        self.fc = nn.Linear(self.item_enc.out_dim, num_classes)

    def forward(self, bundle_tensors):
        if not bundle_tensors:
            M = torch.zeros((1, self.item_enc.out_dim), device=DEVICE)
        else:
            M = torch.stack([self.item_enc(ids) for ids in bundle_tensors])
        Svec = torch.mean(M, dim=0)
        return self.fc(Svec)

# ---------- 전역 상태 ----------
MODEL = None
LABELS = []              # ["B1","B2",...]
CURRENT_MODEL_VERSION = None

def _save_model(ver: int, model: nn.Module, label_list):
    os.makedirs("models", exist_ok=True)
    path = f"./models/model_v{ver}.pt"
    torch.save(model.state_dict(), path)
    conn = get_conn(); cur = dict_cur(conn)
    cur.execute(
        f"UPDATE {S}.model_versions SET details = %s WHERE version_id = %s",
        (json.dumps({"label_list": label_list, "arch":"TextCNN"}), ver)
    )
    conn.commit(); cur.close(); conn.close()
    return path

def load_latest_model():
    """최신 버전 모델 로드(카테고리 B)"""
    global MODEL, LABELS, CURRENT_MODEL_VERSION
    conn = get_conn(); cur = dict_cur(conn)
    cur.execute(f"SELECT version_id, details FROM {S}.model_versions ORDER BY version_id DESC LIMIT 1;")
    row = cur.fetchone(); cur.close(); conn.close()
    if not row:
        MODEL = None; LABELS = []; CURRENT_MODEL_VERSION = None; return
    ver, details = row["version_id"], (row["details"] or {})
    label_list = details.get("label_list", [])
    model = SetCNNClassifier(VOCAB_SIZE, len(label_list)).to(DEVICE)
    try:
        state = torch.load(f"./models/model_v{ver}.pt", map_location=DEVICE)
        model.load_state_dict(state, strict=False)
    except Exception as e:
        print("[model] load warning:", e)
    model.eval()
    MODEL = model; LABELS = label_list; CURRENT_MODEL_VERSION = ver

def train_streaming():
    """bundle_sets/bundle_items 기반 스트리밍 학습 (카테고리 B)"""
    from psycopg2.extras import RealDictCursor
    ACCUM_STEPS = int(os.getenv("ACCUM_STEPS", "4"))
    EPOCHS      = int(os.getenv("EPOCHS", "3"))
    LR          = float(os.getenv("LEARNING_RATE", "0.001"))
    MAX_ITEM_LEN = int(os.getenv("MAX_ITEM_LEN", "512"))
    MAX_ITEMS_PER_BUNDLE = int(os.getenv("MAX_ITEMS_PER_BUNDLE", "64"))

    conn = get_conn(); cur = dict_cur(conn)
    # ✅ 활성 라벨: categories_b.active = TRUE
    cur.execute(f"SELECT code FROM {S}.categories_b WHERE active = TRUE ORDER BY code;")
    label_list = [r["code"] for r in cur.fetchall()]
    label_to_idx = {c:i for i,c in enumerate(label_list)}
    num_classes = len(label_list)

    # 새 버전 생성
    cur.execute(f"INSERT INTO {S}.model_versions(details) VALUES ('{{}}') RETURNING version_id;")
    ver = cur.fetchone()["version_id"]; conn.commit()

    model = SetCNNClassifier(VOCAB_SIZE, num_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(EPOCHS):
        stream = conn.cursor(name=f"ep{ep}_cursor", cursor_factory=RealDictCursor, withhold=True)
        stream.itersize = 512
        stream.execute(f"""
          SELECT s.id AS set_id, s.label,
                 array_agg(i.clean_text ORDER BY i.id) AS items
          FROM {S}.bundle_sets s
          JOIN {S}.bundle_items i ON i.set_id = s.id
          WHERE s.label = ANY(%s)
          GROUP BY s.id, s.label
          ORDER BY s.id
        """, (label_list,))
        step = 0
        while True:
            rows = stream.fetchmany(128)
            if not rows: break
            for r in rows:
                y = label_to_idx[r["label"]]
                items = (r["items"] or [])[:MAX_ITEMS_PER_BUNDLE]
                tensors = [encode_text_to_tensor(t or "", MAX_ITEM_LEN) for t in items]
                logits = model(tensors)
                loss = loss_fn(logits.unsqueeze(0), torch.tensor([y], dtype=torch.long, device=DEVICE))
                loss = loss / ACCUM_STEPS
                loss.backward()
                step += 1
                if step % ACCUM_STEPS == 0:
                    opt.step(); opt.zero_grad()
        stream.close()

    _save_model(ver, model, label_list)
    cur = dict_cur(conn)
    cur.execute(f"UPDATE {S}.model_versions SET trained_at = NOW() WHERE version_id = %s", (ver,))
    conn.commit(); cur.close(); conn.close()
    return ver, label_list
