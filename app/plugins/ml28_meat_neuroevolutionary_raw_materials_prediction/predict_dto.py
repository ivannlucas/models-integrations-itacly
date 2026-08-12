"""Transport contract for ml28.

Unlike every other DNSL-family plugin in this repo, the prediction unit here is genuinely a
single row (date + raw_material_id + destination_profile) — verified in manifest-extraction that
the rules engine has no cross-row dependency (lags/windows), so predict_inline is a real
single-row mode, not a workaround.
"""
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class PredictBatchRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["batch"] = "batch"
    data_path: str = Field(..., description="Path to CSV following docs/input_contract.md: one row per date+raw_material_id+destination_profile")
    mlflow_run_id: str = Field(default="", description="Ignored — ml28 has no trainable artifact (training.supported=false)")


class PredictBatchResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    predictions: list[dict[str, Any]]
    summary: dict[str, Any] | None = Field(default=None, description="Aggregate policy-simulation metrics across all rows (row_count, triggered_orders, aggregate_excess_reduction_pct, stockout_guardrail_pass, ...)")
    output_path: str | None = None


class PredictInlineRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["inline"] = "inline"
    date: str = Field(..., description="ISO or parseable date of the operational need")
    raw_material_id: str = Field(..., description="Meat raw material identifier")
    destination_profile: str = Field(..., description="Intended production destination of the raw material (not the purchased product)")
    current_inventory_tons: float = Field(..., ge=0, description="Available inventory at the start of the horizon")
    expected_requirement_tons: float = Field(..., ge=0, description="Expected raw material requirement")
    lead_time_days: float = Field(..., ge=0, description="Expected days until receipt")
    safety_coverage_days: float = Field(..., ge=0, description="Target safety coverage in days")
    expected_yield_rate: float = Field(..., ge=0, le=1, description="Expected raw material yield rate")
    expected_waste_rate: float = Field(..., ge=0, le=1, description="Expected waste rate")
    unit_purchase_cost: float = Field(..., ge=0, description="Expected unit cost per ton")
    shelf_life_days: float = Field(..., ge=0, description="Expected shelf life of the raw material/lot")
    model_key: str | None = None
    threshold: float | None = None
    mlflow_run_id: str = Field(default="", description="Ignored — ml28 has no trainable artifact (training.supported=false)")


class PredictInlineResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    date: str
    raw_material_id: str
    destination_profile: str
    current_inventory_tons: float
    expected_requirement_tons: float
    lead_time_days: float
    safety_coverage_days: float
    expected_yield_rate: float
    expected_waste_rate: float
    unit_purchase_cost: float
    shelf_life_days: float
    purchase_trigger_proba: float = Field(..., description="Sigmoid score over the inventory coverage gap")
    purchase_trigger_flag: int = Field(..., description="1 if projected stock after lead time is below safety stock")
    recommended_action: str = Field(..., description="BUY or DO_NOT_BUY")
    quantity_optimizer_recommendation_tons: float = Field(..., description="Raw recommendation BEFORE gating — never present as the final decision")
    order_quantity_tons: float = Field(..., description="Final recommended quantity — 0.0 whenever purchase_trigger_flag=0")
    decision_reason: str
    projected_stock_after_lead_time_tons: float
    safety_stock_tons: float
    coverage_gap_tons: float
    risk_level: str = Field(..., description="LOW | MEDIUM | HIGH")
    baseline_order_quantity_tons: float
    delta_order_vs_baseline_tons: float
    excess_tons: float
    stockout_tons: float


PredictRequest = Annotated[
    Union[PredictBatchRequest, PredictInlineRequest],
    Field(discriminator="mode"),
]

PredictResponse = Union[PredictBatchResponse, PredictInlineResponse]
