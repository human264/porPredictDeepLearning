import logging

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

import app.model as mdl
# ✅ XGBoost 라우터 임포트 (제가 드린 파일 구조 기준)
# app/api/routes/xgb_routes.py 에서 router 객체를 가져옵니다.
from app.api.routes.xgb_routes import router as xgb_router
from app.db import get_conn, dict_cur, get_schema
from app.schemas import PredictItemReq, TrainResp, ItemPredictResp
load_dotenv()

logger = logging.getLogger("por-ml")
app = FastAPI(title="POR Item Classifier (Hybrid)")
S = get_schema()

# ✅ /ml/* 엔드포인트 추가 (XGBoost 파이프라인)
app.include_router(xgb_router)


@app.on_event("startup")
def _startup():
    # 기존 사내 모델 2종을 기동 시 로드
    for bucket in ("item_act", "item_mr"):
        try:
            mdl.load_latest(bucket)
            logger.info("Loaded latest model: %s", bucket)
        except Exception as e:
            # 모델이 아직 없거나, 최초 기동 시 실패해도 서버는 뜨도록 경고만 남김
            logger.warning("Model load skipped for %s: %s", bucket, e)


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

        return {"user": u, "db": d, "search_path": sp, "version": ver, "schema": S}
    finally:
        cur.close()
        conn.close()


# -------- 학습 --------
@app.post("/item/train_act", response_model=TrainResp)
def item_train_act():
    try:
        ver, n = mdl.train_items("item_act")
        return {"status": "trained_item_act", "model_version": ver, "num_labels": n}
    except Exception as e:
        # detail에 문자열을 명시적으로 전달
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/item/train_mr", response_model=TrainResp)
def item_train_mr():
    try:
        ver, n = mdl.train_items("item_mr")
        return {"status": "trained_item_mr", "model_version": ver, "num_labels": n}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------- 예측 --------
@app.post("/item/predict", response_model=ItemPredictResp)
def item_predict(req: PredictItemReq):
    # 헤더로 라인들 가져오기
    conn = get_conn()
    cur = dict_cur(conn)
    try:
        # 주의: 스키마(S)는 서버 설정에서 결정되므로 f-string 사용.
        # 파라미터는 바인딩으로 안전하게 전달.
        cur.execute(
            f"""
            SELECT line_no, item_name, spec_text, mccsno, block, event, sign, duration, deptcode, shiptype
            FROM {S}.tb_por_detail
            WHERE pjtno=%s AND porser=%s AND porseq=%s AND revno=%s
            ORDER BY line_no
            """,
            (req.pjtno, req.porser, req.porseq, req.revno),
        )
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    if not rows:
        raise HTTPException(status_code=404, detail="No lines found for the header.")

    results = []

    if req.want_act:
        act_out, act_ver = mdl.predict_items("item_act", rows, topk=req.topk)
    else:
        act_out, act_ver = ([(None, None, None)] * len(rows), None)

    if req.want_mr:
        mr_out, mr_ver = mdl.predict_items("item_mr", rows, topk=req.topk)
    else:
        mr_out, mr_ver = ([(None, None, None)] * len(rows), None)

    for i, r in enumerate(rows):
        rec = {
            "line_no": r["line_no"],
            "text": " ".join([(r.get("item_name") or ""), (r.get("spec_text") or "")]).strip(),
        }

        if req.want_act:
            lab, conf, top = act_out[i]
            if lab is not None:
                actocode, actno = lab.split(":", 1)
                rec.update(
                    {
                        "actocode": actocode,
                        "actno": actno,
                        "act_confidence": conf,
                        "act_topk": top,
                    }
                )

        if req.want_mr:
            lab, conf, top = mr_out[i]
            if lab is not None:
                rec.update({"mr_label": lab, "mr_confidence": conf, "mr_topk": top})

        results.append(rec)

    return {
        "pjtno": req.pjtno,
        "porser": req.porser,
        "porseq": req.porseq,
        "revno": req.revno,
        "count": len(results),
        "results": results,
        "item_act_model_version": act_ver,
        "item_mr_model_version": mr_ver,
    }
