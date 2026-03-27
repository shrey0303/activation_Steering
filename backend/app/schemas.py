"""Pydantic request/response models for all API endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class SteeringConfig(BaseModel):
    layer: int = Field(..., ge=0, description="Target layer index")
    strength: float = Field(
        0.0, ge=-10.0, le=10.0, description="Intervention strength"
    )
    direction_vector: Optional[List[float]] = Field(
        None, description="Custom direction vector (auto-detected if null)"
    )
    source: str = Field(
        "mathematical_analysis",
        description="Origin: 'mathematical_analysis' or 'manual'"
    )


class PerformanceMetrics(BaseModel):
    latency_ms: float = 0.0
    tokens_per_sec: float = 0.0
    memory_used_mb: float = 0.0
    steering_overhead_ms: float = 0.0


# --- Health ---

class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool = False
    device: str = "cpu"
    memory: Dict[str, float] = {}
    uptime_seconds: float = 0.0
    version: str = "1.0.0"


# --- Model ---

class ModelInfo(BaseModel):
    id: str
    name: str
    loaded: bool = False
    parameters: str = ""
    quantized: bool = False
    quantization_bits: int = 0
    memory_mb: float = 0.0
    device: str = "cpu"
    num_layers: int = 0
    hidden_dim: int = 0
    scanned: bool = False

class ModelsResponse(BaseModel):
    models: List[ModelInfo] = []
    active_model: Optional[str] = None


class LoadModelRequest(BaseModel):
    model_name: str = Field(
        ..., min_length=1,
        description="HuggingFace model ID, e.g. 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'",
    )
    quantize: bool = Field(True, description="Enable quantization (CUDA only)")
    quantization_bits: int = Field(4, ge=4, le=8, description="Quantization bits (4 or 8)")
    device: str = Field("auto", description="'auto' | 'cuda' | 'mps' | 'cpu'")


class LoadModelResponse(BaseModel):
    success: bool = True
    model: Optional[ModelInfo] = None
    message: str = ""
    load_time_seconds: float = 0.0


# --- Scan ---

class ScanRequest(BaseModel):
    model_name: Optional[str] = Field(
        None, description="Model to scan (uses loaded model if null)"
    )
    force_rescan: bool = Field(
        False, description="Re-scan even if cached results exist"
    )

class LayerProfile(BaseModel):
    layer_index: int
    category: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    behavioral_role: str = ""
    weight_stats: Dict[str, float] = {}
    description: str = ""

class ScanResponse(BaseModel):
    model_name: str
    num_layers: int
    hidden_dim: int
    architecture: str = ""
    layer_profiles: List[LayerProfile] = []
    scan_time_ms: float = 0.0
    from_cache: bool = False


# --- Analysis ---

class AnalyzeRequest(BaseModel):
    prompt: str = Field(
        "", max_length=5000,
        description="The prompt sent to the model"
    )
    expected_response: str = Field(
        "", max_length=5000,
        description="What the user expects the model to produce"
    )
    behavior_description: Optional[str] = Field(
        None, max_length=2000,
        description=(
            "Describe desired behaviour instead of exact response. "
            "E.g. 'be rude and dismissive', 'act like a professor'."
        ),
    )
    analysis_type: str = Field(
        "full", description="'full' or 'quick'"
    )

class LayerAnalysis(BaseModel):
    layer_index: int
    anomaly_score: float = 0.0
    confidence: float = 0.0
    category: str = ""
    behavioral_role: str = ""
    explanation: str = ""
    recommended_intervention: Dict[str, Any] = {}
    statistics: Dict[str, float] = {}

class AnalyzeResponse(BaseModel):
    analysis_id: str = Field(default_factory=lambda: str(uuid4()))
    prompt: str
    expected_response: str
    detected_layers: List[int] = []
    detailed_analysis: Dict[int, LayerAnalysis] = {}
    overall_confidence: float = 0.0
    interpretation: Dict[str, Any] = {}
    processing_time_ms: float = 0.0


# --- Generation ---

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000)
    max_tokens: int = Field(200, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    steering: Optional[SteeringConfig] = None

class GenerateResponse(BaseModel):
    text: str
    prompt: str
    tokens_generated: int = 0
    latency_ms: float = 0.0
    tokens_per_sec: float = 0.0
    steering_applied: bool = False
    steering_overhead_ms: float = 0.0
    metrics: Dict[str, Any] = {}


# --- Activations ---

class ActivationsRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000)
    layers: Optional[List[int]] = Field(
        None, description="Specific layers to capture (null = all)"
    )
    aggregation: str = Field("mean", description="'mean' | 'max' | 'l2norm'")

class ActivationsResponse(BaseModel):
    activations: Dict[int, float] = {}
    prompt: str
    num_layers: int = 0
    capture_time_ms: float = 0.0


# --- Patches ---

class InterventionSpec(BaseModel):
    layer: int
    strength: float
    direction_vector: Optional[List[float]] = None
    notes: str = ""

class PatchExportRequest(BaseModel):
    patch_name: str = Field(..., min_length=1)
    description: str = ""
    interventions: List[InterventionSpec] = []
    validation_data: Dict[str, Any] = {}

class PatchMetadata(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    model: str = ""
    version: str = "1.0"
    created: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )
    description: str = ""
    interventions_count: int = 0
    file_size_kb: float = 0.0

class PatchExportResponse(BaseModel):
    patch_id: str
    download_url: str = ""
    file_size_kb: float = 0.0
    patch: Dict[str, Any] = {}

class PatchListResponse(BaseModel):
    patches: List[PatchMetadata] = []
    total: int = 0


# --- WebSocket ---

class WSGenerateRequest(BaseModel):
    type: str = "generate"
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    prompt: str = ""
    max_tokens: int = 200
    temperature: float = 0.7
    steering: Optional[SteeringConfig] = None

class WSTokenMessage(BaseModel):
    type: str = "token"
    request_id: str
    text: str
    token_id: int = 0

class WSDoneMessage(BaseModel):
    type: str = "done"
    request_id: str
    metadata: Dict[str, Any] = {}

class WSErrorMessage(BaseModel):
    type: str = "error"
    request_id: str = ""
    error: Dict[str, Any] = {}


# --- Error ---

class ErrorResponse(BaseModel):
    error: str
    detail: str = ""
    code: str = "INTERNAL_ERROR"
