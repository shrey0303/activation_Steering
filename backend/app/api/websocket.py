"""
WebSocket endpoint for streaming text generation.

Handles token-by-token streaming with optional steering.
Supports stop signal to interrupt generation mid-stream.
"""

from __future__ import annotations

import asyncio
import json
import time
import traceback
from uuid import uuid4

import torch
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from app.core.config import get_settings
from app.core.engine import SteeringEngine
from app.core.loader import ModelManager

router = APIRouter()

# Track active generation cancellation flags per connection
_stop_flags: dict[int, bool] = {}
_active_connections: int = 0
_conn_lock = asyncio.Lock()


@router.websocket("/api/v1/ws/generate")
async def websocket_generate(ws: WebSocket):
    """
    WebSocket endpoint for streaming generation.
    """
    global _active_connections
    settings = get_settings()

    async with _conn_lock:
        if _active_connections >= settings.ws_max_connections:
            await ws.close(code=1013, reason="Server busy — max connections reached")
            return
        await ws.accept()
        _active_connections += 1

    conn_id = id(ws)
    _stop_flags[conn_id] = False
    logger.info(f"WebSocket client connected (active={_active_connections})")

    try:
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)
            msg_type = data.get("type", "")

            if msg_type == "generate":
                _stop_flags[conn_id] = False
                await _handle_generate(ws, data, conn_id)
            elif msg_type == "stop":
                _stop_flags[conn_id] = True
                logger.info("Stop signal received from client")
                await ws.send_json({"type": "stopped"})
            elif msg_type == "ping":
                await ws.send_json({"type": "pong"})
            else:
                await ws.send_json({
                    "type": "error",
                    "error": {
                        "code": "UNKNOWN_MESSAGE",
                        "message": f"Unknown message type: {msg_type}",
                    },
                })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await ws.send_json({
                "type": "error",
                "error": {
                    "code": "SERVER_ERROR",
                    "message": str(e),
                    "recoverable": False,
                },
            })
        except Exception:
            pass
    finally:
        async with _conn_lock:
            _active_connections = max(0, _active_connections - 1)
        _stop_flags.pop(conn_id, None)
        logger.info(f"WebSocket client cleaned up (active={_active_connections})")


async def _handle_generate(ws: WebSocket, data: dict, conn_id: int):
    """Handle a single generation request over WebSocket."""
    request_id = data.get("request_id", str(uuid4()))
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 200)
    temperature = data.get("temperature", 0.7)
    top_p = data.get("top_p", 0.9)
    steering_config = data.get("steering", None)

    if not prompt:
        await ws.send_json({
            "type": "error",
            "request_id": request_id,
            "error": {"code": "EMPTY_PROMPT", "message": "Prompt cannot be empty"},
        })
        return

    # Ensure model is loaded — do NOT auto-load (security: prevents DoS via WS)
    mm = ModelManager.get_instance()
    if not mm.loaded:
        await ws.send_json({
            "type": "error",
            "request_id": request_id,
            "error": {
                "code": "NO_MODEL",
                "message": "No model loaded. Load a model first via the Control Panel.",
                "recoverable": True,
            },
        })
        return

    engine = SteeringEngine.get_instance(mm)

    # Apply steering if configured
    if steering_config:
        direction_vec = None
        if steering_config.get("direction_vector"):
            direction_vec = torch.tensor(
                steering_config["direction_vector"], dtype=torch.float32
            )
        engine.add_intervention(
            layer_idx=steering_config.get("layer", 0),
            strength=steering_config.get("strength", 0.0),
            direction_vector=direction_vec,
            gate_threshold=steering_config.get("gate_threshold"),
            norm_tolerance=steering_config.get("norm_tolerance", 0.05),
            decay_rate=steering_config.get("decay_rate", 0.006),
        )

    _GENERATE_WS_TIMEOUT = 180  # hard safety cap for streaming

    t0 = time.perf_counter()
    tokens_generated = 0
    was_stopped = False

    # Async queue: sync generator pushes tokens from thread, async handler reads.
    token_queue: asyncio.Queue = asyncio.Queue()

    def _run_sync_generator():
        """Run the blocking sync generator in a thread, pushing tokens to queue."""
        try:
            for token_data in engine.generate_stream(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            ):
                if _stop_flags.get(conn_id, False):
                    break
                token_queue.put_nowait(token_data)
            token_queue.put_nowait(None)
        except Exception as e:
            token_queue.put_nowait(e)

    try:
        # Start sync generator in thread pool
        gen_task = asyncio.get_event_loop().run_in_executor(None, _run_sync_generator)

        while True:
            try:
                token_data = await asyncio.wait_for(
                    token_queue.get(), timeout=_GENERATE_WS_TIMEOUT,
                )
            except asyncio.TimeoutError:
                was_stopped = True
                logger.warning("WS generation timed out")
                break

            if token_data is None:
                break

            if isinstance(token_data, Exception):
                raise token_data

            if _stop_flags.get(conn_id, False):
                was_stopped = True
                logger.info(f"Generation stopped after {tokens_generated} tokens")
                break

            tokens_generated += 1
            msg = {
                "type": "token",
                "request_id": request_id,
                "text": token_data["text"],
                "token_id": token_data["token_id"],
            }
            if "diagnostics" in token_data:
                diag = token_data["diagnostics"]
                diag["active_interventions"] = engine.active_interventions
                msg["diagnostics"] = diag
            await ws.send_json(msg)

        await gen_task

        elapsed = time.perf_counter() - t0
        await ws.send_json({
            "type": "done",
            "request_id": request_id,
            "metadata": {
                "total_tokens": tokens_generated,
                "latency_ms": round(elapsed * 1000, 1),
                "tokens_per_sec": round(
                    tokens_generated / max(elapsed, 0.001), 1
                ),
                "steering_overhead_ms": round(engine.total_overhead_ms, 2),
                "steering_applied": len(engine._hooks) > 0,
                "was_stopped": was_stopped,
                "active_interventions": engine.active_interventions,
            },
        })

    except Exception as e:
        logger.error(f"Generation error: {e}\n{traceback.format_exc()}")
        await ws.send_json({
            "type": "error",
            "request_id": request_id,
            "error": {
                "code": "GENERATION_ERROR",
                "message": str(e),
                "recoverable": True,
            },
        })
    finally:
        engine.clear_interventions()
        _stop_flags[conn_id] = False
