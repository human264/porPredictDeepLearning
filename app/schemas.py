from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict, field_validator

__all__ = [
    "ApiModel",
    "IngestReq",        # (옵션) 수동 주입용
    "RuleUpsertReq",
    "PredictReq",
    "PredictResp",
    "TrainResp",
    "RebuildSetsResp",
]

class ApiModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

class IngestReq(BaseModel):
    # (옵션) 직접 bundle_sets/items 넣고 싶을 때 사용
    pjtno: str
    porser: str
    porseq: str
    revno: str
    label_actocode: Optional[str] = None
    label_actno: Optional[str] = None
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
    target_actocode: str
    target_actno: str
    priority: int = 0
    enabled: bool = True
    where_mccsno: Optional[str] = None
    where_block: Optional[str] = None
    where_event: Optional[str] = None
    where_sign: Optional[str] = None
    where_duration_min: Optional[int] = None
    where_duration_max: Optional[int] = None

class PredictReq(BaseModel):
    # ① POR 헤더로 예측하는 경우 (권장)
    pjtno: Optional[str] = None
    porser: Optional[str] = None
    porseq: Optional[str] = None
    revno: Optional[str] = None
    # ② 직접 아이템을 넣는 경우 (백업)
    items: Optional[List[Any]] = None
    # 옵션
    force_ensemble: Optional[bool] = False

    @field_validator("items", mode="before")
    @classmethod
    def coerce_items(cls, v):
        if v is None:
            return None
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

class PredictResp(ApiModel):
    actocode: str
    actno: str
    confidence: float
    top3: Any
    model_version: int

class TrainResp(ApiModel):
    status: str
    model_version: int

class RebuildSetsResp(ApiModel):
    status: str
    inserted: int
    updated: int
