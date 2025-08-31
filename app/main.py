# app/main.py
import os
import json
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from psycopg2.extras import Json  # ✅ JSONB 안전 삽입용
from app.db import get_conn, dict_cur, get_schema
from app.textvec import normalize, text_to_vec
from app.schemas import IngestReq, RuleUpsertReq, PredictReq, PredictResp, TrainResp
from app.knn import knn_topk
import app.model as mdl

load_dotenv()

PROPOSED_HISTORY = os.getenv("PROPOSED_HISTORY", "0") in ("1", "true", "True")
PERSIST_PREDICT_BUNDLES = os.getenv("PERSIST_PREDICT_BUNDLES", "0") in ("1", "true", "True")
FORCE_ENSEMBLE = os.getenv("FORCE_ENSEMBLE", "0").lower() in ("1","true","on","yes")
app = FastAPI(title="POR Bundle Classifier (Category B)")
S = get_schema()

# ---- 점수 포맷 정책 ----
SCORE_DECIMALS = os.getenv("SCORE_DECIMALS", "auto").strip().lower()  # 'auto' 또는 숫자 문자열
SCORE_MODE     = os.getenv("SCORE_MODE", "number").strip().lower()    # 'number' | 'string'
SCORE_MAX_DIGITS = int(os.getenv("SCORE_MAX_DIGITS", "15"))

def _fmt_score(v: float):
    """
    점수 포맷팅:
      - SCORE_DECIMALS='auto' → 반올림 없이 그대로
      - SCORE_DECIMALS=n(정수) → 소수 n자리 반올림
      - SCORE_MODE='number'  → JSON number로 반환
      - SCORE_MODE='string'  → JSON string로 반환(표시 자릿수 제어)
    """
    x = float(v)
    if SCORE_DECIMALS == "auto":
        if SCORE_MODE == "string":
            return f"{x:.{SCORE_MAX_DIGITS}g}"   # 유효자리 기준 가변 문자열
        return x                                   # 숫자 그대로
    try:
        n = int(SCORE_DECIMALS)
    except ValueError:
        n = 3
    if SCORE_MODE == "string":
        return f"{x:.{n}f}"
    return round(x, n)
# ------------------------

def _to_floats(a: np.ndarray) -> list[float]:
    return np.asarray(a, dtype=float).ravel().tolist()


@app.on_event("startup")
def _startup():
    mdl.load_latest_model()


@app.get("/debug/snapshot")
def debug_snapshot(n: int = 5):
    conn = get_conn()
    cur = dict_cur(conn)
    try:
        cur.execute(f"SELECT id, label, created_at FROM {S}.bundle_sets ORDER BY id DESC LIMIT %s", (n,))
        sets_ = cur.fetchall()
        cur.execute(
            f"SELECT id, set_id, LEFT(raw_text,60) AS raw, LEFT(clean_text,60) AS clean "
            f"FROM {S}.bundle_items ORDER BY id DESC LIMIT %s",
            (n,),
        )
        items_ = cur.fetchall()
        cur.execute(
            f"SELECT id, predicted_label, confidence, top3, model_version, predicted_at "
            f"FROM {S}.bundle_predictions ORDER BY id DESC LIMIT %s",
            (n,),
        )
        preds_ = cur.fetchall()
        cur.execute(
            f"SELECT por_id, category_code, category_name, confidence, top3, model_version, predicted_at "
            f"FROM {S}.tb_mr_proposed ORDER BY predicted_at DESC LIMIT %s",
            (n,),
        )
        props_ = cur.fetchall()
        return {
            "last_sets": sets_,
            "last_items": items_,
            "last_predictions": preds_,
            "last_proposed": props_,
        }
    finally:
        cur.close()
        conn.close()


@app.post("/bundle/ingest")
def ingest(req: IngestReq):
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                clean = [normalize(x or "") for x in req.items]
                item_vecs = [text_to_vec(t) for t in clean]
                V = int(os.getenv("VEC_DIM", "512"))
                set_vec = np.mean(item_vecs, axis=0) if item_vecs else np.zeros(V, dtype="float32")
                cur.execute(
                    f"INSERT INTO {S}.bundle_sets(label, set_embed) VALUES (%s,%s) RETURNING id;",
                    (req.label, _to_floats(set_vec)),
                )
                set_id = cur.fetchone()[0]
                for raw, c, v in zip(req.items, clean, item_vecs):
                    cur.execute(
                        f"""
                        INSERT INTO {S}.bundle_items(set_id, raw_text, clean_text, item_embed)
                        VALUES (%s,%s,%s,%s)
                        """,
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
                  INSERT INTO {S}.bundle_rules(pattern, target_label, priority, enabled)
                  VALUES (%s,%s,%s,%s)
                  ON CONFLICT (pattern)
                  DO UPDATE SET target_label=EXCLUDED.target_label,
                                priority=EXCLUDED.priority,
                                enabled=EXCLUDED.enabled
                """,
                    (req.pattern, req.target_label, req.priority, req.enabled),
                )
        return {"status": "ok"}
    finally:
        conn.close()


@app.get("/stats")
def stats():
    conn = get_conn()
    cur = dict_cur(conn)
    try:
        cur.execute(f"SELECT COUNT(*) c FROM {S}.bundle_sets")
        total = cur.fetchone()["c"]
        cur.execute(f"SELECT COUNT(DISTINCT label) c FROM {S}.bundle_sets")
        labels = cur.fetchone()["c"]
        cur.execute(f"SELECT COUNT(*) c FROM {S}.categories_b WHERE active")
        act = cur.fetchone()["c"]
        cur.execute(
            f"""
          SELECT label, COUNT(*) cnt
          FROM {S}.bundle_sets
          GROUP BY label
          ORDER BY cnt DESC
          LIMIT 5
        """
        )
        top = cur.fetchall()
        return {"total_bundles": total, "unique_labels": labels, "active_labels": act, "top_labels": top}
    finally:
        cur.close()
        conn.close()


@app.post("/bundle/train", response_model=TrainResp)
def train():
    ver, _ = mdl.train_streaming()
    mdl.load_latest_model()
    return {"status": "trained", "model_version": ver}


@app.post("/bundle/predict", response_model=PredictResp)
def predict(req: PredictReq):
    if mdl.MODEL is None or not mdl.LABELS:
        raise HTTPException(503, "model not loaded")

    alpha = float(os.getenv("ENSEMBLE_ALPHA", "0.5"))
    V = int(os.getenv("VEC_DIM", "512"))
    MAX_ITEM_LEN = int(os.getenv("MAX_ITEM_LEN", "512"))
    MAX_ITEMS_PER_BUNDLE = int(os.getenv("MAX_ITEMS_PER_BUNDLE", "64"))

    clean = [normalize(x) for x in req.items]
    full_text = " ".join(clean)

    # ✅ 전역 or 요청 단위 강제 엔SEMBLE
    force_ensemble = FORCE_ENSEMBLE or bool(getattr(req, "force_ensemble", False))

    conn = get_conn()
    try:
        with conn:
            cur = dict_cur(conn)

            # 1) Rule first — ✅ 강제 엔SEMBLE이면 스킵
            if not force_ensemble:
                cur.execute(
                    f"""
                    SELECT target_label
                    FROM {S}.bundle_rules
                    WHERE enabled=TRUE AND %s ~* pattern
                    ORDER BY priority DESC
                    LIMIT 1
                    """,
                    (full_text,),
                )
                r = cur.fetchone()
                if r:
                    chosen = r["target_label"]
                    top3 = [{"label": chosen, "score": _fmt_score(1.0)}]
                    top3_json = Json(top3)
                    conf_out = float(_fmt_score(1.0))

                    # (선택) persist predicted bundle ...
                    # (생략: 기존 그대로 유지)

                    # log prediction
                    explain = Json({"via":"rule"})
                    cur.execute(
                        f"""INSERT INTO {S}.bundle_predictions(set_id, predicted_label, confidence, top3, model_version)
                            VALUES (NULL, %s, %s, %s, %s)""",
                        (chosen, conf_out, top3_json, mdl.CURRENT_MODEL_VERSION or 0),
                    )
                    _upsert_tb_mr_proposed(cur, req.por_id, chosen, conf_out, top3_json)
                    return {
                        "chosen_label": chosen,
                        "confidence": conf_out,
                        "top3": top3,
                        "model_version": mdl.CURRENT_MODEL_VERSION or 0,
                    }

            # 2) KNN (강제일 때도 여기로 진행)
            item_vecs = [text_to_vec(t) for t in clean]
            set_vec = np.mean(item_vecs, axis=0) if item_vecs else np.zeros(V, dtype="float32")
            neigh = knn_topk(set_vec, k=3, active_only=True)
            votes = {}
            for n in neigh:
                sim = 1.0 / (float(n["distance"]) + 1e-6)
                votes[n["label"]] = votes.get(n["label"], 0.0) + sim
            ssum = sum(votes.values()) or 1.0
            knn_prob = {k: v / ssum for k, v in votes.items()}

            # 3) Model
            tensors = [mdl.encode_text_to_tensor(t, MAX_ITEM_LEN) for t in clean[:MAX_ITEMS_PER_BUNDLE]]
            with torch.no_grad():
                logits = mdl.MODEL(tensors)
                probs = torch.softmax(logits, dim=0).cpu().numpy()
            model_prob = {mdl.LABELS[i]: float(probs[i]) for i in range(len(mdl.LABELS))}

            # 4) Ensemble
            labels = set(knn_prob.keys()) | set(model_prob.keys())
            scores = {lab: alpha*model_prob.get(lab,0.0) + (1-alpha)*knn_prob.get(lab,0.0) for lab in labels}
            if not scores:
                raise HTTPException(503, "no candidates")
            top = sorted(scores.items(), key=lambda x:x[1], reverse=True)
            chosen, conf = top[0]
            top3 = [{"label": k, "score": _fmt_score(v)} for k, v in top[:3]]
            top3_json = Json(top3)
            conf_out = float(_fmt_score(conf))

            # log prediction — ✅ explain에 강제 여부 표시
            explain = Json({"via": "ensemble_forced" if force_ensemble else "ensemble", "alpha": alpha})
            cur.execute(
                f"""INSERT INTO {S}.bundle_predictions(set_id, predicted_label, confidence, top3, model_version)
                    VALUES (NULL, %s, %s, %s, %s)""",
                (chosen, conf_out, top3_json, mdl.CURRENT_MODEL_VERSION or 0),
            )
            _upsert_tb_mr_proposed(cur, req.por_id, chosen, conf_out, top3_json)

            return {
                "chosen_label": chosen,
                "confidence": conf_out,
                "top3": top3,
                "model_version": mdl.CURRENT_MODEL_VERSION or 0,
            }
    finally:
        conn.close()



@app.get("/debug/dbinfo")
def debug_dbinfo():
    conn = get_conn()
    cur = dict_cur(conn)
    try:
        cur.execute("SELECT current_user AS current_user, session_user AS session_user")
        u = cur.fetchone()
        cur.execute("SELECT current_database() AS db, inet_server_addr()::text AS host, inet_server_port() AS port")
        d = cur.fetchone()
        cur.execute("SHOW search_path")
        sp = cur.fetchone()
        cur.execute("SELECT version() AS ver")
        ver = cur.fetchone()
        cur.execute("SELECT extname FROM pg_catalog.pg_extension ORDER BY 1")
        exts = [r["extname"] for r in cur.fetchall()]
        cur.execute(
            f"""
          SELECT
            (SELECT COUNT(*) FROM {S}.bundle_sets)        AS bundle_sets,
            (SELECT COUNT(*) FROM {S}.bundle_items)       AS bundle_items,
            (SELECT COUNT(*) FROM {S}.bundle_predictions) AS bundle_predictions,
            (SELECT COUNT(*) FROM {S}.model_versions)     AS model_versions,
            (SELECT COUNT(*) FROM {S}.tb_mr_proposed)     AS tb_mr_proposed,
            (SELECT COUNT(*) FROM {S}.categories_b)       AS categories_b;
        """
        )
        cnt = cur.fetchone()
        return {"user": u, "db": d, "search_path": sp, "version": ver, "extensions": exts, "counts": cnt, "schema": S}
    finally:
        cur.close()
        conn.close()


def _upsert_tb_mr_proposed(cur, por_id: int, label: str, confidence: float, top3_json):
    """tb_mr_proposed UPSERT + (옵션) tb_mr_proposed_hist append"""
    if por_id is None:
        return
    cur.execute(
        f"""
        INSERT INTO {S}.tb_mr_proposed
          (por_id, category_code, category_name, confidence, top3, model_version)
        VALUES
          (%s,%s,(SELECT name FROM {S}.categories_b WHERE code=%s),%s,%s,%s)
        ON CONFLICT (por_id) DO UPDATE
        SET category_code=EXCLUDED.category_code,
            category_name=EXCLUDED.category_name,
            confidence=EXCLUDED.confidence,
            top3=EXCLUDED.top3,
            model_version=EXCLUDED.model_version,
            predicted_at=NOW()
        """,
        (por_id, label, label, confidence, top3_json, mdl.CURRENT_MODEL_VERSION or 0),
    )
    if PROPOSED_HISTORY:
        cur.execute(
            f"""
            INSERT INTO {S}.tb_mr_proposed_hist
              (por_id, category_code, category_name, confidence, top3, model_version)
            VALUES
              (%s,%s,(SELECT name FROM {S}.categories_b WHERE code=%s),%s,%s,%s)
            """,
            (por_id, label, label, confidence, top3_json, mdl.CURRENT_MODEL_VERSION or 0),
        )
