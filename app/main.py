import os
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from app.db import get_conn, dict_cur, get_schema
import app.model as mdl
from app.schemas import PredictItemReq, TrainResp, ItemPredictResp

load_dotenv()
app = FastAPI(title="POR Item Classifier (Hybrid)")
S = get_schema()

@app.on_event("startup")
def _startup():
    mdl.load_latest("item_act")
    mdl.load_latest("item_mr")

@app.get("/debug/dbinfo")
def debug_dbinfo():
    conn = get_conn(); cur = dict_cur(conn)
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
        cur.close(); conn.close()

# -------- 학습 --------
@app.post("/item/train_act", response_model=TrainResp)
def item_train_act():
    try:
        ver, n = mdl.train_items("item_act")
        return {"status": "trained_item_act", "model_version": ver, "num_labels": n}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/item/train_mr", response_model=TrainResp)
def item_train_mr():
    try:
        ver, n = mdl.train_items("item_mr")
        return {"status": "trained_item_mr", "model_version": ver, "num_labels": n}
    except Exception as e:
        raise HTTPException(500, str(e))

# -------- 예측 --------
@app.post("/item/predict", response_model=ItemPredictResp)
def item_predict(req: PredictItemReq):
    # 헤더로 라인들 가져오기
    conn = get_conn(); cur = dict_cur(conn)
    try:
        cur.execute(
            f"""SELECT line_no, item_name, spec_text, mccsno, block, event, sign, duration, deptcode, shiptype
                FROM {S}.tb_por_detail
                WHERE pjtno=%s AND porser=%s AND porseq=%s AND revno=%s
                ORDER BY line_no""",
            (req.pjtno, req.porser, req.porseq, req.revno)
        )
        rows = cur.fetchall()
    finally:
        cur.close(); conn.close()
    if not rows:
        raise HTTPException(404, "No lines found for the header.")

    results = []
    if req.want_act:
        act_out, act_ver = mdl.predict_items("item_act", rows, topk=req.topk)
    else:
        act_out, act_ver = ([(None,None,None)]*len(rows), None)

    if req.want_mr:
        mr_out, mr_ver = mdl.predict_items("item_mr", rows, topk=req.topk)
    else:
        mr_out, mr_ver = ([(None,None,None)]*len(rows), None)

    for i, r in enumerate(rows):
        rec = {
            "line_no": r["line_no"],
            "text": " ".join([(r["item_name"] or ""), (r["spec_text"] or "")]).strip()
        }
        if req.want_act:
            lab, conf, top = act_out[i]
            if lab is not None:
                actocode, actno = lab.split(":",1)
                rec.update({
                    "actocode": actocode, "actno": actno,
                    "act_confidence": conf, "act_topk": top
                })
        if req.want_mr:
            lab, conf, top = mr_out[i]
            if lab is not None:
                rec.update({
                    "mr_label": lab,
                    "mr_confidence": conf, "mr_topk": top
                })
        results.append(rec)

    return {
        "pjtno": req.pjtno, "porser": req.porser, "porseq": req.porseq, "revno": req.revno,
        "count": len(results),
        "results": results,
        "item_act_model_version": act_ver,
        "item_mr_model_version": mr_ver
    }
