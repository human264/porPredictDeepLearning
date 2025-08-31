import unicodedata, re
import torch

# 한글 자모
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

BASE_ASCII = "abcdefghijklmnopqrstuvwxyz0123456789 .-/%xdiam"
KOREAN_JAMO = "".join(_CHOSEONG) + "".join(_JUNGSEONG) + "".join([j for j in _JONGSEONG if j])
CHARSET = "".join(dict.fromkeys(BASE_ASCII + KOREAN_JAMO))
char_to_idx = {c: i+1 for i, c in enumerate(CHARSET)}  # 0=pad/unk

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

_ws = re.compile(r"\s+")
def normalize_text(t: str) -> str:
    t = unicodedata.normalize("NFKC", (t or "").strip()).lower()
    t = t.replace("Ø"," diam ").replace("ø"," diam ")
    t = re.sub(r"\bdn(\d+)", r"dn \1", t)
    t = re.sub(r"\bpn(\d+)", r"pn \1", t)
    t = re.sub(r"(?<=\d)(mm|cm|inch)\b", r" \1", t)
    t = re.sub(r"(\d)x(?=\d)", r"\1 x ", t)
    t = _ws.sub(" ", t).strip()
    return t

def encode_text_to_tensor(text: str, max_len: int, device: torch.device):
    t = normalize_text(text)
    seq = []
    for ch in t:
        for p in decompose_hangul_char(ch):
            seq.append(char_to_idx.get(p, 0))
            if len(seq) >= max_len: break
        if len(seq) >= max_len: break
    if not seq:
        seq = [0]
    return torch.tensor(seq[:max_len], dtype=torch.long, device=device)
