# app/schemas.py
from pydantic import BaseModel, field_validator
from typing import List, Optional, Any
from pydantic import BaseModel, ConfigDict


class ApiModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

class IngestReq(BaseModel):
    label: str
    items: List[Any]
    @field_validator("items", mode="before")
    @classmethod
    def coerce_items(cls, v):
        if isinstance(v, str):
            return [v]
        if isinstance(v, (list, tuple)):
            out = []
            for x in v:
                if x is None:
                    out.append("")
                elif isinstance(x, bytes):
                    out.append(x.decode())
                else:
                    out.append(str(x))
            return out
        raise TypeError("items must be a string or a list/tuple")

class RuleUpsertReq(BaseModel):
    pattern: str
    target_label: str
    priority: int = 0
    enabled: bool = True


class PredictReq(BaseModel):
    items: List[Any]
    por_id: Optional[int] = None
    force_ensemble: Optional[bool] = False  # ✅ 요청 단위로 룰을 건너뛰기

    @field_validator("items", mode="before")
    @classmethod
    def coerce_items(cls, v):
        if isinstance(v, str):
            return [v]
        if isinstance(v, (list, tuple)):
            out = []
            for x in v:
                if x is None:
                    out.append("")
                elif isinstance(x, bytes):
                    out.append(x.decode())
                else:
                    out.append(str(x))
            return out
        raise TypeError("items must be a string or a list/tuple")

# 그리고 응답 모델들에 적용
class PredictResp(ApiModel):
    chosen_label: str
    confidence: float
    top3: Any
    model_version: int

class TrainResp(ApiModel):
    status: str
    model_version: int
