# app/schemas.py
from pydantic import BaseModel
from typing import List, Optional, Any

class IngestReq(BaseModel):
    label: str
    items: List[str]

class RuleUpsertReq(BaseModel):
    pattern: str
    target_label: str
    priority: int = 0
    enabled: bool = True

class PredictReq(BaseModel):
    items: List[str]
    por_id: Optional[int] = None  # tb_mr_proposed 저장시 사용

class PredictResp(BaseModel):
    chosen_label: str
    confidence: float
    top3: Any
    model_version: int

class TrainResp(BaseModel):
    status: str
    model_version: int
