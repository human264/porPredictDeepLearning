# app/api/routes/xgb_routes.py
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.ml.xgb_model import train_and_save_from_db, predict_header

router = APIRouter(prefix="/ml", tags=["ml"])

# ---------- 요청 스키마 ----------
class HeaderReq(BaseModel):
    pjtno: str
    porser: str
    porseq: str
    revno: str
    topk: Optional[int] = Field(default=5, ge=1, le=50)

# ---------- 학습 (DB에서 자동 수집) ----------
@router.post("/mr/train_db")
def train_mr_from_db():
    try:
        info = train_and_save_from_db("mr")
        return {"message": "MR 모델 학습/저장 완료", **info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/act/train_db")
def train_act_from_db():
    try:
        info = train_and_save_from_db("act")
        return {"message": "ACT 모델 학습/저장 완료", **info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- 예측 (헤더로 tb_por_detail 라인 예측) ----------
@router.post("/mr/predict_header")
def predict_mr(req: HeaderReq):
    try:
        header = {"pjtno": req.pjtno, "porser": req.porser, "porseq": req.porseq, "revno": req.revno}
        res = predict_header("mr", header, topk=req.topk or 5)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/act/predict_header")
def predict_act(req: HeaderReq):
    try:
        header = {"pjtno": req.pjtno, "porser": req.porser, "porseq": req.porseq, "revno": req.revno}
        res = predict_header("act", header, topk=req.topk or 5)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
