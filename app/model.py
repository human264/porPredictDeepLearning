import os, json, unicodedata, random
import numpy as np
import torch
import torch.nn as nn
from app.db import get_conn, dict_cur, get_schema

S = get_schema()

# ---------------- Utility: seed ----------------
def set_global_seed(seed: int | None):
    if seed is None: return
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def _select_device():
    pref = os.getenv("DEVICE_PREFERENCE", "auto").lower()
    if pref in ("mps", "metal"):
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device("cpu")

DEVICE = _select_device()
print(f"[model] DEVICE={DEVICE}; mps_avail={torch.backends.mps.is_available()}")

# ---------- Conv 커널 설정 ----------
def _kernels():
    ks = os.getenv("CNN_KERNEL_SIZES", "2,3,4,5")
    return tuple(int(x) for x in ks.split(",") if x.strip())

EMB_DIM = int(os.getenv("CNN_EMB_DIM", "48"))
OUT_CH  = int(os.getenv("CNN_OUT_CH",  "64"))
KERNELS = _kernels()
DROPOUT = float(os.getenv("CNN_DROPOUT", "0.1"))
_MAX_KERNEL = max(KERNELS) if len(KERNELS) > 0 else 1

# ---------- 한글 자모 테이블 ----------
_CHOSEONG = [chr(c) for c in [
    0x1100,0x1101,0x1102,0x1103,0x1104,0x1105,0x1106,0x1107,0x1108,0x1109,
    0x110A,0x110B,0x110C,0x110D,0x110E,0x110F,0x1110,0x1111,0x1112
]]
_JUNGSEONG = [chr(c) for c in [
    0x1161,0x1162,0x1163,0x1164,0x1165,0x1166,0x1167,0x1168,0x1169,0x116A,
    0x116B,0x116C,0x116D,0x116E,0x116F,0x1170,0x1171,0x1172,0x1173,0x1174,
    0x1175
]]
_JONGSEONG = [""] + [chr(c) for c in [
    0x11A8,0x11A9,0x11AA,0x11AB,0x11AC,0x11AD,0x11AE,0x11AF,0x11B0,0x11B1,
    0x11B2,0x11B3,0x11B4,0x11B5,0x11B6,0x11B7,0x11B8,0x11B9,0x11BA,0x11BB,
    0x11BC,0x11BD,0x11BE,0x11BF,0x11C0,0x11C1,0x11C2
]]

def decompose_hangul_char(ch: str):
    code = ord(ch)
    if 0xAC00 <= code <= 0xD7A3:
        sindex = code - 0xAC00
        jong = sindex % 28
        jung = ((sindex - jong) // 28) % 21
        cho  = ((sindex - jong) // 28) // 21
        parts = [_CHOSEONG[cho], _JUNGSEONG[jung]]
        if _JONGSEONG[jong]: parts.append(_JONGSEONG[jong])
        return parts
    if (0x1100 <= code <= 0x1112) or (0x1161 <= code <= 0x1175) or (0x11A8 <= code <= 0x11C2):
        return [ch]
    return [ch]

def _normalize_text(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    return t.lower().strip()

# ---------- 문자 집합 ----------
BASE_ASCII = "abcdefghijklmnopqrstuvwxyz0123456789 .-/%xdiam"
KOREAN_JAMO = "".join(_CHOSEONG) + "".join(_JUNGSEONG) + "".join([j for j in _JONGSEONG if j])
_seen = set(); CHARSET_LIST = []
for c in (BASE_ASCII + KOREAN_JAMO):
    if c not in _seen:
        _seen.add(c); CHARSET_LIST.append(c)
CHARSET = "".join(CHARSET_LIST)
char_to_idx = {c: i + 1 for i, c in enumerate(CHARSET)}  # 0은 <unk>/<pad>
VOCAB_SIZE = len(char_to_idx) + 1

def encode_text_to_tensor(text: str, max_len: int):
    t = _normalize_text(text)
    seq = []
    for ch in t:
        for p in decompose_hangul_char(ch):
            seq.append(char_to_idx.get(p, 0))
            if len(seq) >= max_len: break
        if len(seq) >= max_len: break
    if len(seq) < _MAX_KERNEL:
        seq.extend([0] * (_MAX_KERNEL - len(seq)))
    return torch.tensor(seq[:max_len], dtype=torch.long, device=DEVICE)

# ---------- CNN 인코더/분류기 ----------
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
        return z.squeeze(0)

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
LABEL_KEYS = []          # ["ACT:001", ...]  ← 인덱스→라벨 매핑
CURRENT_MODEL_VERSION = None

def _label_key(actocode: str, actno: str) -> str:
    return f"{(actocode or '').strip()}:{(actno or '').strip()}"

def _save_model(ver: int, model: nn.Module, label_keys):
    os.makedirs("models", exist_ok=True)
    path = f"./models/model_v{ver}.pt"
    torch.save(model.state_dict(), path)
    conn = get_conn(); cur = dict_cur(conn)
    details = {
        "label_keys": label_keys,
        "arch": "TextCNN",
        "charset": CHARSET,
        "normalize": "NFKC_lower",
        "jamo": True,
        "kernels": list(KERNELS),
        "emb_dim": EMB_DIM,
        "out_ch": OUT_CH,
        "dropout": DROPOUT,
    }
    cur.execute(
        f"UPDATE {S}.model_versions SET details = %s WHERE version_id = %s",
        (json.dumps(details), ver)
    )
    conn.commit(); cur.close(); conn.close()
    return path

def load_latest_model():
    """최신 버전 모델 로드(활동 코드)"""
    global MODEL, LABEL_KEYS, CURRENT_MODEL_VERSION
    conn = get_conn(); cur = dict_cur(conn)
    cur.execute(f"SELECT version_id, details FROM {S}.model_versions ORDER BY version_id DESC LIMIT 1;")
    row = cur.fetchone(); cur.close(); conn.close()
    if not row:
        MODEL = None; LABEL_KEYS = []; CURRENT_MODEL_VERSION = None; return
    ver, details = row["version_id"], (row["details"] or {})
    label_keys = details.get("label_keys", [])
    if not label_keys:
        print("[model] load warning: empty label_keys; model disabled.")
        MODEL = None; LABEL_KEYS = []; CURRENT_MODEL_VERSION = ver; return
    model = SetCNNClassifier(VOCAB_SIZE, len(label_keys)).to(DEVICE)
    try:
        state = torch.load(f"./models/model_v{ver}.pt", map_location=DEVICE)
        model.load_state_dict(state, strict=False)
    except Exception as e:
        print("[model] load warning:", e)
    model.eval()
    MODEL = model; LABEL_KEYS = label_keys; CURRENT_MODEL_VERSION = ver

def _dedup_preserve_order(strings: list[str]) -> list[str]:
    seen = set(); out = []
    for s in strings:
        if s not in seen:
            seen.add(s); out.append(s)
    return out

def train_streaming():
    """
    vw_training_activity에서 (pjtno,porser,porseq,revno, actocode,actno, items[])를 스트리밍 학습.
    """
    from psycopg2.extras import RealDictCursor
    set_global_seed(int(os.getenv("GLOBAL_SEED", "0")) or None)

    ACCUM_STEPS = int(os.getenv("ACCUM_STEPS", "4"))
    EPOCHS      = int(os.getenv("EPOCHS", "3"))
    LR          = float(os.getenv("LEARNING_RATE", "0.001"))
    MAX_ITEM_LEN = int(os.getenv("MAX_ITEM_LEN", "4096"))
    MAX_ITEMS_PER_BUNDLE = int(os.getenv("MAX_ITEMS_PER_BUNDLE", "64"))
    CLIP_NORM   = float(os.getenv("CLIP_NORM", "0"))

    conn = get_conn(); cur = dict_cur(conn)

    # 활성 활동 코드 목록
    cur.execute(f"SELECT actocode, actno FROM {S}.activity_codes WHERE active = TRUE ORDER BY actocode, actno;")
    labs = [(_["actocode"], _["actno"]) for _ in cur.fetchall()]
    if not labs:
        raise RuntimeError("No active rows in activity_codes (active=TRUE).")
    label_keys = [_label_key(a,b) for a,b in labs]
    key_to_idx = {k:i for i,k in enumerate(label_keys)}
    num_classes = len(label_keys)

    # 새 버전
    cur.execute(f"INSERT INTO {S}.model_versions(details) VALUES ('{{}}') RETURNING version_id;")
    ver = cur.fetchone()["version_id"]; conn.commit()

    model = SetCNNClassifier(VOCAB_SIZE, num_classes).to(DEVICE)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(EPOCHS):
        stream = conn.cursor(name=f"ep{ep}_cursor", cursor_factory=RealDictCursor, withhold=True)
        stream.itersize = 256
        stream.execute(f"""
          SELECT pjtno, porser, porseq, revno, actocode, actno, items
          FROM {S}.vw_training_activity
          WHERE (actocode, actno) IN (SELECT actocode, actno FROM {S}.activity_codes WHERE active=TRUE)
          ORDER BY pjtno, porser, porseq, revno, actocode, actno
        """)
        step = 0
        while True:
            rows = stream.fetchmany(64)
            if not rows: break
            for r in rows:
                ykey = _label_key(r["actocode"], r["actno"])
                y = key_to_idx.get(ykey, None)
                if y is None:
                    continue
                items_all = (r["items"] or [])
                # 중복/공백 제거
                items = [t for t in _dedup_preserve_order(items_all) if (t or "").strip()]
                if not items:
                    continue
                items = items[:MAX_ITEMS_PER_BUNDLE]
                tensors = [encode_text_to_tensor(t, MAX_ITEM_LEN) for t in items]
                logits = model(tensors)
                loss = loss_fn(logits.unsqueeze(0), torch.tensor([y], dtype=torch.long, device=DEVICE))
                (loss / ACCUM_STEPS).backward()
                step += 1
                if step % ACCUM_STEPS == 0:
                    if CLIP_NORM > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                    opt.step(); opt.zero_grad()
        stream.close()
        if step % ACCUM_STEPS != 0:
            if CLIP_NORM > 0:
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            opt.step(); opt.zero_grad()

    _save_model(ver, model, label_keys)
    model.eval()
    cur = dict_cur(conn)
    cur.execute(f"UPDATE {S}.model_versions SET trained_at = NOW() WHERE version_id = %s", (ver,))
    conn.commit(); cur.close(); conn.close()
    return ver, label_keys
