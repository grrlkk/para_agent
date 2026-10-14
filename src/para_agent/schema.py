from pydantic import BaseModel, ConfigDict


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, serialize_by_alias=True)

