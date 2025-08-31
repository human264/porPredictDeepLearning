import os
import json
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from psycopg2.extras import Json
from app.db import get_conn, dict_cur, get_schema
from app.textvec import normalize, text_to_vec
from app.schemas import IngestReq, RuleUpsertReq, PredictReq, PredictResp, TrainResp, RebuildSetsResp
from app.knn import knn_topk
import app.model as mdl

load_dotenv()

PROPOSED_HISTORY = os.getenv("PROPOSED_HISTORY", "0") in ("1", "true", "True")
FORCE_ENSEMBLE = os.getenv("FORCE_ENSEMBLE", "0").lower() in ("1", "true", "on", "yes")
app = FastAPI(title="POR Activity Classifier")
S = get_schema()

SCORE_DECIMALS = os.getenv("SCORE_DECIMALS", "auto").strip().lower()
SCORE_MODE = os.getenv("SCORE_MODE", "number").strip().lower()
SCORE_MAX_DIGITS = int(os.getenv("SCORE_MAX_DIGITS", "15"))


def _fmt_score(v: float):
    x = float(v)
    if SCORE_DECIMALS == "auto":
        if SCORE_MODE == "string":
            return f"{x:.{SCORE_MAX_DIGITS}g}"
        return x
    try:
        n = int(SCORE_DECIMALS)
    except ValueError:
        n = 3
    if SCORE_MODE == "string":
        return f"{x:.{n}f}"
    return round(x, n)


def _to_floats(a: np.ndarray) -> list[float]:
    return np.asarray(a, dtype=float).ravel().tolist()


@app.on_event("startup")
def _startup():
    mdl.load_latest_model()


@app.get("/debug/dbinfo")
def debug_dbinfo():
    conn = get_conn()
    cur = dict_cur(conn)
    try:
        cur.execute("SELECT current_user, session_user")
        u = cur.fetchone()
        cur.execute("SELECT current_database() AS db, inet_server_addr()::text AS host, inet_server_port() AS port")
        d = cur.fetchone()
        cur.execute("SHOW search_path")
        sp = cur.fetchone()
        cur.execute("SELECT version() AS ver")
        ver = cur.fetchone()
        cur.execute("SELECT extname FROM pg_catalog.pg_extension ORDER BY 1")
        exts = [r["extname"] for r in cur.fetchall()]
        return {"user": u, "db": d, "search_path": sp, "version": ver, "extensions": exts, "schema": S}
    finally:
        cur.close()
        conn.close()


# (옵션) 수동 주입
@app.post("/bundle/ingest")
def ingest(req: IngestReq):
    V = int(os.getenv("VEC_DIM", "512"))
    clean = [normalize(x or "") for x in (req.items or [])]
    item_vecs = [text_to_vec(t) for t in clean]
    set_vec = np.mean(item_vecs, axis=0) if item_vecs else np.zeros(V, dtype="float32")
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""INSERT INTO {S}.bundle_sets
                        (pjtno,porser,porseq,revno,label_actocode,label_actno,set_embed)
                        VALUES (%s,%s,%s,%s,%s,%s,%s)
                        RETURNING id;""",
                    (req.pjtno, req.porser, req.porseq, req.revno,
                     req.label_actocode, req.label_actno, _to_floats(set_vec))
                )
                set_id = cur.fetchone()[0]
                for raw, c, v in zip(req.items, clean, item_vecs):
                    cur.execute(
                        f"""INSERT INTO {S}.bundle_items(set_id, raw_text, clean_text, item_embed)
                            VALUES (%s,%s,%s,%s)""",
                        (set_id, raw, c, _to_floats(v)),
                    )
        return {"status": "ok", "set_id": set_id, "items_inserted": len(item_vecs)}
    finally:
        conn.close()


@app.post("/bundle_rules")
def upsert_rule(req: RuleUpsertReq):
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                  INSERT INTO {S}.bundle_rules
                    (pattern, target_actocode, target_actno, priority, enabled,
                     where_mccsno, where_block, where_event, where_sign, where_duration_min, where_duration_max)
                  VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                  ON CONFLICT (rule_id) DO NOTHING
                """,
                    (req.pattern, req.target_actocode, req.target_actno, req.priority, req.enabled,
                     req.where_mccsno, req.where_block, req.where_event, req.where_sign,
                     req.where_duration_min, req.where_duration_max),
                )
        return {"status": "ok"}
    finally:
        conn.close()


@app.post("/bundle/train", response_model=TrainResp)
def train():
    ver, _ = mdl.train_streaming()
    mdl.load_latest_model()
    return {"status": "trained", "model_version": ver}


# 학습셋으로 bundle_sets(임베딩) 재구성 → KNN용
@app.post("/bundle/rebuild_sets", response_model=RebuildSetsResp)
def rebuild_sets():
    from app.textvec import normalize, text_to_vec
    V = int(os.getenv("VEC_DIM", "512"))
    inserted = 0
    updated = 0
    conn = get_conn()
    cur = dict_cur(conn)
    try:
        # 학습 뷰에서 헤더/활동별 items 가져와 set_embed 생성
        cur.execute(f"SELECT pjtno,porser,porseq,revno,actocode,actno,items FROM {S}.vw_training_activity")
        rows = cur.fetchall()
        with conn:
            # ✅ dict cursor 사용 (ex["id"] 접근 가능)
            with dict_cur(conn) as c:
                for r in rows:
                    items = [normalize((t or "")) for t in (r["items"] or []) if (t or "").strip()]
                    if not items:
                        continue
                    vecs = [text_to_vec(t) for t in items]
                    set_vec = np.mean(vecs, axis=0) if vecs else np.zeros(V, dtype="float32")
                    # 존재 여부 확인(동일 헤더+라벨)
                    c.execute(
                        f"""SELECT id FROM {S}.bundle_sets
                            WHERE pjtno=%s AND porser=%s AND porseq=%s AND revno=%s
                              AND label_actocode=%s AND label_actno=%s
                            LIMIT 1;""",
                        (r["pjtno"], r["porser"], r["porseq"], r["revno"], r["actocode"], r["actno"])
                    )
                    ex = c.fetchone()
                    if ex:
                        c.execute(
                            f"UPDATE {S}.bundle_sets SET set_embed=%s WHERE id=%s",
                            (_to_floats(set_vec), ex["id"])
                        )
                        updated += 1
                    else:
                        c.execute(
                            f"""INSERT INTO {S}.bundle_sets
                                (pjtno,porser,porseq,revno,label_actocode,label_actno,set_embed)
                                VALUES (%s,%s,%s,%s,%s,%s,%s)""",
                            (r["pjtno"], r["porser"], r["porseq"], r["revno"], r["actocode"], r["actno"],
                             _to_floats(set_vec))
                        )
                        inserted += 1
        return {"status": "ok", "inserted": inserted, "updated": updated}
    finally:
        cur.close()
        conn.close()


def _apply_rule_if_any(full_text: str, header: dict):
    """
    룰 매칭: pattern ~* full_text
    + where_* 가 NULL이거나, tb_por_detail에 동일 값/범위 존재
    """
    conn = get_conn()
    cur = dict_cur(conn)
    try:
        cur.execute(
            f"""
            WITH hdr AS (
              SELECT %(pjtno)s::text AS pjtno, %(porser)s::text AS porser,
                     %(porseq)s::text AS porseq, %(revno)s::text AS revno
            )
            SELECT br.target_actocode, br.target_actno
            FROM {S}.bundle_rules br, hdr
            WHERE br.enabled = TRUE
              AND %(full_text)s ~* br.pattern
              AND (br.where_mccsno IS NULL OR EXISTS (
                    SELECT 1 FROM {S}.tb_por_detail d
                    WHERE d.pjtno=hdr.pjtno AND d.porser=hdr.porser AND d.porseq=hdr.porseq AND d.revno=hdr.revno
                      AND d.mccsno = br.where_mccsno))
              AND (br.where_block  IS NULL OR EXISTS (
                    SELECT 1 FROM {S}.tb_por_detail d
                    WHERE d.pjtno=hdr.pjtno AND d.porser=hdr.porser AND d.porseq=hdr.porseq AND d.revno=hdr.revno
                      AND d.block = br.where_block))
              AND (br.where_event  IS NULL OR EXISTS (
                    SELECT 1 FROM {S}.tb_por_detail d
                    WHERE d.pjtno=hdr.pjtno AND d.porser=hdr.porser AND d.porseq=hdr.porseq AND d.revno=hdr.revno
                      AND d.event = br.where_event))
              AND (br.where_sign   IS NULL OR EXISTS (
                    SELECT 1 FROM {S}.tb_por_detail d
                    WHERE d.pjtno=hdr.pjtno AND d.porser=hdr.porser AND d.porseq=hdr.porseq AND d.revno=hdr.revno
                      AND d.sign = br.where_sign))
              AND ((br.where_duration_min IS NULL AND br.where_duration_max IS NULL)
                   OR EXISTS (
                    SELECT 1 FROM {S}.tb_por_detail d
                    WHERE d.pjtno=hdr.pjtno AND d.porser=hdr.porser AND d.porseq=hdr.porseq AND d.revno=hdr.revno
                      AND (br.where_duration_min IS NULL OR d.duration >= br.where_duration_min)
                      AND (br.where_duration_max IS NULL OR d.duration <= br.where_duration_max)
                   ))
            ORDER BY br.priority DESC
            LIMIT 1;
            """,
            {"full_text": full_text, **header}
        )
        r = cur.fetchone()
        return (r["target_actocode"], r["target_actno"]) if r else None
    finally:
        cur.close()
        conn.close()


@app.post("/bundle/predict", response_model=PredictResp)
def predict(req: PredictReq):
    if mdl.MODEL is None or not mdl.LABEL_KEYS:
        raise HTTPException(503, "model not loaded")

    alpha = float(os.getenv("ENSEMBLE_ALPHA", "0.5"))
    V = int(os.getenv("VEC_DIM", "512"))
    MAX_ITEM_LEN = int(os.getenv("MAX_ITEM_LEN", "4096"))
    MAX_ITEMS_PER_BUNDLE = int(os.getenv("MAX_ITEMS_PER_BUNDLE", "64"))

    # 1) 입력 확보: 헤더 우선
    header = None
    items = None
    if req.pjtno and req.porser and req.porseq and req.revno:
        header = {"pjtno": req.pjtno, "porser": req.porser, "porseq": req.porseq, "revno": req.revno}
        # POR 라인에서 아이템 구성
        conn = get_conn()
        cur = dict_cur(conn)
        try:
            cur.execute(
                f"""SELECT item_name, spec_text FROM {S}.tb_por_detail
                    WHERE pjtno=%s AND porser=%s AND porseq=%s AND revno=%s
                    ORDER BY line_no""",
                (req.pjtno, req.porser, req.porseq, req.revno)
            )
            rows = cur.fetchall()
            items = [(" ".join([r["item_name"] or "", r["spec_text"] or ""])).strip() for r in rows]
        finally:
            cur.close()
            conn.close()
    elif req.items:
        items = req.items
    else:
        raise HTTPException(400, "Provide POR header(pjtno,porser,porseq,revno) or items[]")

    clean = [normalize(x) for x in items if (x or "").strip()]
    full_text = " ".join(clean)

    force_ensemble = FORCE_ENSEMBLE or bool(getattr(req, "force_ensemble", False))
    model_only = bool(getattr(req, "model_only", False)) or (float(os.getenv("ENSEMBLE_ALPHA", "0.5")) >= 0.9999)

    # 2) 룰
    if header and (not force_ensemble) and (not model_only):
        hit = _apply_rule_if_any(full_text, header)
        if hit:
            actocode, actno = hit
            top3 = [{"label": f"{actocode}:{actno}", "score": _fmt_score(1.0)}]
            conf_out = float(_fmt_score(1.0))
            _log_pred(header, actocode, actno, conf_out, Json(top3), via="rule")
            _upsert_proposed(header, actocode, actno, conf_out, Json(top3))
            return {"actocode": actocode, "actno": actno, "confidence": conf_out,
                    "top3": top3, "model_version": mdl.CURRENT_MODEL_VERSION or 0}

    # 3) KNN
    item_vecs = [text_to_vec(t) for t in clean]
    set_vec = np.mean(item_vecs, axis=0) if item_vecs else np.zeros(V, dtype="float32")
    neigh = knn_topk(set_vec, k=3, active_only=True)
    votes = {}
    for n in neigh:
        sim = 1.0 / (float(n["distance"]) + 1e-6)
        key = f"{n['actocode']}:{n['actno']}"
        votes[key] = votes.get(key, 0.0) + sim
    ssum = sum(votes.values()) or 1.0
    knn_prob = {k: v / ssum for k, v in votes.items()}

    # 4) Model
    tensors = [mdl.encode_text_to_tensor(t, MAX_ITEM_LEN) for t in clean[:MAX_ITEMS_PER_BUNDLE]]
    with torch.no_grad():
        logits = mdl.MODEL(tensors)
        probs = torch.softmax(logits, dim=0).cpu().numpy()
    model_prob = {mdl.LABEL_KEYS[i]: float(probs[i]) for i in range(len(mdl.LABEL_KEYS))}

    if model_only:
        # 모델 점수만으로 Top3 계산
        top = sorted(model_prob.items(), key=lambda x: x[1], reverse=True)
        best_key, conf = top[0]
        actocode, actno = best_key.split(":", 1)
        top3 = [{"label": k, "score": _fmt_score(v)} for k, v in top[:3]]
        conf_out = float(_fmt_score(conf))
        _log_pred(header, actocode, actno, conf_out, Json(top3), via="model_only")
        _upsert_proposed(header, actocode, actno, conf_out, Json(top3))
        return {"actocode": actocode, "actno": actno, "confidence": conf_out,
                "top3": top3, "model_version": mdl.CURRENT_MODEL_VERSION or 0}

    # 5) Ensemble
    labels = set(knn_prob.keys()) | set(model_prob.keys())
    scores = {lab: alpha * model_prob.get(lab, 0.0) + (1 - alpha) * knn_prob.get(lab, 0.0) for lab in labels}
    if not scores:
        raise HTTPException(503, "no candidates")
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_key, conf = top[0]
    actocode, actno = best_key.split(":", 1)
    top3 = [{"label": k, "score": _fmt_score(v)} for k, v in top[:3]]
    conf_out = float(_fmt_score(conf))

    _log_pred(header, actocode, actno, conf_out, Json(top3),
              via=("ensemble_forced" if force_ensemble else "ensemble"), alpha=alpha)
    _upsert_proposed(header, actocode, actno, conf_out, Json(top3))

    return {"actocode": actocode, "actno": actno, "confidence": conf_out,
            "top3": top3, "model_version": mdl.CURRENT_MODEL_VERSION or 0}


def _log_pred(header, actocode, actno, conf: float, top3_json, via="ensemble", alpha=None):
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                if header:
                    explain = {"via": via}
                    if alpha is not None:
                        explain["alpha"] = alpha
                    cur.execute(
                        f"""INSERT INTO {S}.bundle_predictions
                            (pjtno,porser,porseq,revno,predicted_actocode,predicted_actno,confidence,top3,model_version,explain)
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                        (header["pjtno"], header["porser"], header["porseq"], header["revno"],
                         actocode, actno, conf, top3_json, mdl.CURRENT_MODEL_VERSION or 0, Json(explain))
                    )
    finally:
        conn.close()


def _upsert_proposed(header, actocode, actno, confidence, top3_json):
    if not header:
        return
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {S}.tb_mr_proposed
                      (pjtno,porser,porseq,revno,actocode,actno,category_name,confidence,top3,model_version)
                    VALUES
                      (%s,%s,%s,%s,%s,%s,
                       (SELECT name FROM {S}.activity_codes WHERE actocode=%s AND actno=%s),
                       %s,%s,%s)
                    ON CONFLICT (pjtno,porser,porseq,revno,actocode,actno) DO UPDATE
                    SET category_name = EXCLUDED.category_name,
                        confidence    = EXCLUDED.confidence,
                        top3          = EXCLUDED.top3,
                        model_version = EXCLUDED.model_version,
                        predicted_at  = NOW()
                    """,
                    (header["pjtno"], header["porser"], header["porseq"], header["revno"],
                     actocode, actno, actocode, actno, confidence, top3_json, mdl.CURRENT_MODEL_VERSION or 0)
                )
                if PROPOSED_HISTORY:
                    cur.execute(
                        f"""
                        INSERT INTO {S}.tb_mr_proposed_hist
                          (pjtno,porser,porseq,revno,actocode,actno,category_name,confidence,top3,model_version)
                        VALUES
                          (%s,%s,%s,%s,%s,%s,
                           (SELECT name FROM {S}.activity_codes WHERE actocode=%s AND actno=%s),
                           %s,%s,%s)
                        """,
                        (header["pjtno"], header["porser"], header["porseq"], header["revno"],
                         actocode, actno, actocode, actno, confidence, top3_json, mdl.CURRENT_MODEL_VERSION or 0)
                    )
    finally:
        conn.close()
