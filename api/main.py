"""
api/main.py
-----------
FastAPI backend for MedBridge.

3 endpoints:
    POST /analyze         → accepts PDF, returns full JSON result
    GET  /analyze/stream  → accepts PDF, streams live progress via SSE
    GET  /health          → returns server status

Why FastAPI:
- Async by default — handles multiple requests
- Auto-generates docs at /docs
- Built-in file upload handling
- Easy SSE (Server-Sent Events) for live progress
"""

import os
import sys
import json
import asyncio
from typing import AsyncGenerator

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.medbridge.pipeline import run_pipeline_from_bytes
from src.medbridge.llm.schemas import MedBridgeOutput

# ── App setup ─────────────────────────────────────────────────
app = FastAPI(
    title="MedBridge API",
    description="Translates medical documents into plain English",
    version="1.0.0"
)

# CORS — allows React frontend to call this API
# In production, replace * with your actual frontend URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """
    Health check endpoint.
    Used by Railway to verify the server is running.
    """
    return {
        "status": "ok",
        "version": "1.0.0",
        "service": "MedBridge API"
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Main endpoint. Accepts a PDF file, returns full analysis.

    Request:  multipart/form-data with PDF file
    Response: JSON matching MedBridgeOutput schema
    """
    # validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted"
        )

    # read file bytes
    pdf_bytes = await file.read()

    # validate file size (max 20MB)
    max_size = 20 * 1024 * 1024
    if len(pdf_bytes) > max_size:
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 20MB"
        )

    try:
        # run the full pipeline
        result = run_pipeline_from_bytes(
            pdf_bytes=pdf_bytes,
            filename=file.filename
        )
        return result.model_dump()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@app.post("/analyze/stream")
async def analyze_stream(file: UploadFile = File(...)):
    """
    Streaming endpoint. Accepts PDF, streams progress via SSE.

    The frontend listens to this endpoint and updates the
    processing screen as each pipeline step completes.

    Each SSE event is a JSON object:
    {
        "step": "parsing_done",
        "detail": "1 pages, 136 words extracted",
        "done": false
    }

    Final event has "done": true and includes the full result.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted"
        )

    pdf_bytes = await file.read()

    async def event_generator() -> AsyncGenerator[str, None]:
        """Generates SSE events as pipeline progresses."""
        progress_events = []

        def progress_callback(step: str, detail: str):
            """Called by pipeline after each step."""
            progress_events.append({
                "step": step,
                "detail": detail,
                "done": False
            })

        # run pipeline in a thread so it doesn't block async
        loop = asyncio.get_event_loop()

        result_container = {"result": None, "error": None}

        def run_sync():
            try:
                result_container["result"] = run_pipeline_from_bytes(
                    pdf_bytes=pdf_bytes,
                    filename=file.filename,
                    progress_callback=progress_callback
                )
            except Exception as e:
                result_container["error"] = str(e)

        # start pipeline in background thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = loop.run_in_executor(pool, run_sync)

            # stream progress events while pipeline runs
            sent_count = 0
            while not future.done():
                # send any new progress events
                while sent_count < len(progress_events):
                    event = progress_events[sent_count]
                    yield f"data: {json.dumps(event)}\n\n"
                    sent_count += 1
                await asyncio.sleep(0.1)

            # send any remaining events
            while sent_count < len(progress_events):
                event = progress_events[sent_count]
                yield f"data: {json.dumps(event)}\n\n"
                sent_count += 1

        # send final event
        if result_container["error"]:
            yield f"data: {json.dumps({'step': 'error', 'detail': result_container['error'], 'done': True})}\n\n"
        else:
            result = result_container["result"]
            final_event = {
                "step": "complete",
                "detail": "Analysis complete",
                "done": True,
                "result": result.model_dump()
            }
            yield f"data: {json.dumps(final_event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# ── Run directly ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )