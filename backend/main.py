"""
DISA — FastAPI Backend
=======================
REST API endpoints for the Driver Input Smoothness Analyzer.

Endpoints:
  GET  /health              — Health check
  POST /analyze             — Upload CSV → full score breakdown
  GET  /simulate            — Generate synthetic dataset
  POST /optimize-weights    — Upload labeled CSV → optimized weights
  POST /validate            — Run validation suite
  POST /report              — Generate PDF report
  GET  /sessions            — List stored sessions
  GET  /sessions/{id}       — Get session detail
"""

from __future__ import annotations

import io
import json
import os
import sys
import uuid
import tempfile
from pathlib import Path
from typing import List, Optional

# Ensure the project root is on sys.path
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response

from backend.preprocessing import preprocess, preprocess_for_features, SIGNAL_COLUMNS
from backend.features import extract_features, get_event_highlight_mask
from backend.scoring import compute_smoothness_score, compute_timeseries_score, rank_sessions, DEFAULT_WEIGHTS
from backend.optimizer import prepare_training_data, optimize_weights
from backend.anomaly import detect_all_anomalies
from backend.database import init_db, save_session, get_session, list_sessions, save_optimizer_run

# ---------------------------------------------------------------------------
# App initialization
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DISA — Driver Input Smoothness Analyzer",
    description=(
        "A research-grade API for evaluating driving smoothness from time-series "
        "vehicle sensor data. Outputs explainable scores with sub-components."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DB on startup
init_db()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "service": "DISA", "version": "1.0.0"}


# ---------------------------------------------------------------------------
# Analyze endpoint
# ---------------------------------------------------------------------------

@app.post("/analyze", tags=["Analysis"])
async def analyze(
    file: UploadFile = File(..., description="CSV file with driving time-series data"),
    fs: float = Form(25.0, description="Sample rate in Hz"),
    weights: Optional[str] = Form(None, description="JSON array of 4 weights [w_S, w_J, w_V, w_P]"),
    save_result: bool = Form(True),
):
    """
    Upload a CSV driving session and receive a complete smoothness analysis.

    Required CSV columns:
      timestamp, steering_angle, throttle_position, brake_pressure,
      acceleration_x, acceleration_y, acceleration_z, gyroscope_yaw_rate

    Optional column:
      expected_score (ground truth for validation)
    """
    # --- Parse CSV ---
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    # --- Validate required columns ---
    missing = [c for c in SIGNAL_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required columns: {missing}. "
                   f"Available: {list(df.columns)}"
        )

    # --- Parse weights ---
    w = None
    if weights:
        try:
            w = np.array(json.loads(weights), dtype=float)
            if len(w) != 4:
                raise ValueError("weights must have exactly 4 elements")
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid weights: {e}")

    # --- Pipeline ---
    # Feature extraction uses raw (filtered, un-normalized) signals
    # to preserve magnitude differences between driver styles
    feat_df = preprocess_for_features(df, fs=fs)
    fm = extract_features(feat_df, fs=fs)
    result = compute_smoothness_score(fm, weights=w)

    # Normalized data for visualization / rolling score
    processed_df, norm_stats = preprocess(df, fs=fs)

    # Time-series score (on normalized data for display)
    ts_scores = compute_timeseries_score(feat_df, fs=fs, weights=result.weights)

    # Anomaly detection (on raw filtered data)
    anomalies = detect_all_anomalies(feat_df, fs=fs)

    # Highlight mask
    highlight_mask = get_event_highlight_mask(fm)

    # Duration
    n_samples = len(df)
    duration_s = n_samples / fs

    # --- Persist ---
    session_id = str(uuid.uuid4())[:8]
    if save_result:
        save_session(
            session_id=session_id,
            filename=file.filename or "upload.csv",
            fs=fs,
            n_samples=n_samples,
            duration_s=duration_s,
            score_result=result.to_dict(),
            anomalies=[ev.to_dict() for ev in anomalies],
        )

    return {
        "session_id": session_id,
        "filename": file.filename,
        "n_samples": n_samples,
        "duration_s": round(duration_s, 2),
        "fs": fs,
        "analysis": result.to_dict(),
        "anomaly_events": [ev.to_dict() for ev in anomalies],
        "timeseries_score": [round(float(s), 2) for s in ts_scores.tolist()],
        "event_highlight_mask": highlight_mask.tolist() if len(highlight_mask) > 0 else [],
    }


# ---------------------------------------------------------------------------
# Simulate endpoint
# ---------------------------------------------------------------------------

@app.get("/simulate", tags=["Simulation"])
async def simulate(
    profile: str = Query("smooth", description="Driver profile: smooth | aggressive | mixed"),
    duration_s: float = Query(120.0, description="Duration of simulation in seconds"),
    fs: float = Query(25.0, description="Sample rate in Hz"),
    seed: int = Query(42, description="Random seed for reproducibility"),
    download: bool = Query(False, description="Return CSV file download instead of JSON"),
):
    """
    Generate a synthetic driving dataset for the given driver profile.
    Returns JSON (default) or a downloadable CSV.
    """
    from simulator.simulator import generate_smooth_driver, generate_aggressive_driver, generate_mixed_driver

    generators = {
        "smooth": lambda: generate_smooth_driver(duration_s=duration_s, fs=fs, seed=seed),
        "aggressive": lambda: generate_aggressive_driver(duration_s=duration_s, fs=fs, seed=seed),
        "mixed": lambda: generate_mixed_driver(duration_s=duration_s, fs=fs, seed=seed),
    }

    if profile not in generators:
        raise HTTPException(status_code=400, detail=f"Unknown profile '{profile}'. Choose: smooth | aggressive | mixed")

    df, events = generators[profile]()

    if download:
        csv_bytes = df.to_csv(index=False).encode()
        return Response(
            content=csv_bytes,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={profile}.csv"},
        )

    return {
        "profile": profile,
        "n_samples": len(df),
        "duration_s": round(duration_s, 2),
        "fs": fs,
        "seed": seed,
        "n_events": len(events),
        "events": [{"name": e.name, "start_idx": e.start_idx, "end_idx": e.end_idx, "severity": e.severity} for e in events],
        "preview": df.head(50).to_dict(orient="records"),
        "stats": {
            col: {
                "mean": round(float(df[col].mean()), 4),
                "std": round(float(df[col].std()), 4),
                "min": round(float(df[col].min()), 4),
                "max": round(float(df[col].max()), 4),
            }
            for col in df.columns if col not in ("timestamp", "expected_score")
        },
    }


# ---------------------------------------------------------------------------
# Optimize weights endpoint
# ---------------------------------------------------------------------------

@app.post("/optimize-weights", tags=["Optimization"])
async def optimize_weights_endpoint(
    files: List[UploadFile] = File(..., description="Labeled CSV files (must have expected_score column)"),
    method: str = Form("slsqp", description="Optimization method: slsqp | grid | both"),
    fs: float = Form(25.0),
    segment_length_s: float = Form(10.0),
):
    """
    Upload one or more labeled CSV files and optimize scoring weights.
    """
    # Write uploads to temp files
    tmp_paths = []
    try:
        for f in files:
            contents = await f.read()
            suffix = Path(f.filename or "data.csv").suffix or ".csv"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(contents)
                tmp_paths.append(tmp.name)

        feature_list, expected_scores = prepare_training_data(tmp_paths, fs=fs, segment_length_s=segment_length_s)

        if len(feature_list) < 3:
            raise HTTPException(
                status_code=422,
                detail=f"Not enough training data. Got {len(feature_list)} segments, need ≥ 3."
            )

        opt_result = optimize_weights(
            feature_list=feature_list,
            expected_scores=expected_scores,
            method=method,
        )

        result_dict = opt_result.to_dict()
        save_optimizer_run({**result_dict, "csv_files": [f.filename for f in files]})

        return result_dict

    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Validate endpoint
# ---------------------------------------------------------------------------

@app.post("/validate", tags=["Validation"])
async def validate_endpoint(
    files: List[UploadFile] = File(..., description="Labeled CSV files for validation"),
    fs: float = Form(25.0),
    weights: Optional[str] = Form(None, description="JSON array of 4 weights to use"),
):
    """
    Run the full validation suite on labeled datasets.
    Returns correlation, MAE, MSE per profile and robustness test results.
    """
    from validation.validate import run_validation

    tmp_paths = []
    try:
        for f in files:
            contents = await f.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(contents)
                tmp_paths.append((f.filename or "data.csv", tmp.name))

        w = None
        if weights:
            w = np.array(json.loads(weights), dtype=float)

        results = run_validation(
            csv_paths=[(name, path) for name, path in tmp_paths],
            fs=fs,
            weights=w,
        )
        return results

    finally:
        for _, p in tmp_paths:
            try:
                os.unlink(p)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Report endpoint
# ---------------------------------------------------------------------------

@app.post("/report", tags=["Reports"])
async def generate_report_endpoint(
    file: UploadFile = File(...),
    fs: float = Form(25.0),
    weights: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
):
    """Generate a PDF report for an uploaded driving session."""
    try:
        from backend.reports import generate_report
    except ImportError:
        raise HTTPException(status_code=501, detail="fpdf2 not installed. Run: pip install fpdf2")

    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    w = None
    if weights:
        w = np.array(json.loads(weights), dtype=float)

    processed_df, _ = preprocess(df, fs=fs)
    fm = extract_features(processed_df, fs=fs)
    result = compute_smoothness_score(fm, weights=w)

    raw_df, _ = preprocess(df, fs=fs, normalize=False)
    anomalies = detect_all_anomalies(raw_df, fs=fs)

    sid = session_id or str(uuid.uuid4())[:8]
    pdf_bytes = generate_report(result, anomalies, session_id=sid)

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=disa_report_{sid}.pdf"},
    )


# ---------------------------------------------------------------------------
# Session history endpoints
# ---------------------------------------------------------------------------

@app.get("/sessions", tags=["Sessions"])
async def get_sessions():
    """List all stored analysis sessions."""
    return list_sessions()


@app.get("/sessions/{session_id}", tags=["Sessions"])
async def get_session_detail(session_id: str):
    """Retrieve stored details for a specific session."""
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return session


# ---------------------------------------------------------------------------
# Driver ranking endpoint
# ---------------------------------------------------------------------------

@app.post("/rank", tags=["Analysis"])
async def rank_drivers(
    files: List[UploadFile] = File(..., description="Multiple CSV sessions to rank"),
    fs: float = Form(25.0),
    weights: Optional[str] = Form(None),
):
    """Upload multiple sessions and return a ranked comparison table."""
    w = None
    if weights:
        w = np.array(json.loads(weights), dtype=float)

    session_results = []
    for f in files:
        contents = await f.read()
        try:
            df = pd.read_csv(io.BytesIO(contents))
            processed_df, _ = preprocess(df, fs=fs)
            fm = extract_features(processed_df, fs=fs)
            result = compute_smoothness_score(fm, weights=w)
            session_id = Path(f.filename or "session").stem
            session_results.append((session_id, result))
        except Exception as e:
            session_results.append((f.filename or "unknown", None))

    valid = [(sid, res) for sid, res in session_results if res is not None]
    if not valid:
        raise HTTPException(status_code=422, detail="No valid sessions could be analyzed")

    ranking = rank_sessions(valid)
    return {"ranking": ranking}


# ---------------------------------------------------------------------------
# Run server if executed directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
