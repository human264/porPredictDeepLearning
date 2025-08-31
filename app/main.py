# app/main.py
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
app = FastAPI(title="POR Item Classifier (XGBoost-only)")
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
    conn = get_conn(); cur = dict_cur(conn)
    try:
        cur.execute("SELECT USER AS current_user FROM dual")
        u = cur.fetchone()

        cur.execute("""
            SELECT
              SYS_CONTEXT('USERENV','DB_NAME')      AS db,
              SYS_CONTEXT('USERENV','INSTANCE_NAME') AS instance,
              SYS_CONTEXT('USERENV','SERVER_HOST')   AS host
            FROM dual
        """)
        d = cur.fetchone()

        ver = {}
        try:
            c2 = conn.cursor()
            c2.execute("SELECT banner FROM v$version")
            banners = [r[0] for r in c2.fetchall()]
            c2.close()
            ver = {"banners": banners}
        except Exception:
            # 권한 없을 수 있으니 무시
            ver = {}

        return {"user": u, "db": d, "schema": S, "version": ver}
    finally:
        cur.close(); conn.close()


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
    conn = get_conn()
    cur = dict_cur(conn)
    try:
        cur.execute(
            f"""
            SELECT line_no, item_name, spec_text, mccsno, block, event, sign, duration, deptcode, shiptype
            FROM {S}.tb_por_detail
            WHERE pjtno=:pjtno AND porser=:porser AND porseq=:porseq AND revno=:revno
            ORDER BY line_no
            """,
            {"pjtno": req.pjtno, "porser": req.porser, "porseq": req.porseq, "revno": req.revno},
        )
        rows = cur.fetchall()
    finally:
        cur.close(); conn.close()

    if not rows:
        raise HTTPException(status_code=404, detail="No lines found for the header.")
    # 이하 로직 동일 ...
    # (생략: 원본 함수 그대로 유지)
