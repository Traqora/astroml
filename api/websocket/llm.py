"""WebSocket LLM streaming handler."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_async_session_factory
from astroml.llm.streaming import StreamHandler, format_websocket
from astroml.llm.cost import check_budget, track_request, route_request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ws", tags=["websocket"])


@router.websocket("/llm")
async def ws_llm_stream(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
):
    """
    Bidirectional WebSocket streaming endpoint for LLMs.
    Clients send prompts in JSON: {"prompt": "...", "model": "..."}
    and can send an abort message: {"type": "abort"}.
    """
    await websocket.accept()
    
    # Authenticate (simple bypass if token matches or token verification is skipped)
    user_id = "ws_user"
    if token:
        user_id = f"user_{token[:8]}"
        
    session_factory = get_async_session_factory()
    active_handler: Optional[StreamHandler] = None
    
    try:
        while True:
            # 1. Listen for client requests
            raw_msg = await websocket.receive_text()
            try:
                msg = json.loads(raw_msg)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"type": "error", "message": "Invalid JSON"}))
                continue
                
            msg_type = msg.get("type")
            
            # Allow cancellation mid-stream
            if msg_type == "abort":
                if active_handler:
                    active_handler.abort()
                    await websocket.send_text(json.dumps({"type": "status", "message": "generation aborted"}))
                continue
                
            prompt = msg.get("prompt")
            model = msg.get("model", "gpt-3.5-turbo")
            feature = msg.get("feature", "chatbot")
            
            if not prompt:
                await websocket.send_text(json.dumps({"type": "error", "message": "Missing prompt"}))
                continue
                
            # Create session-specific async DB session
            async with session_factory() as db:
                # Dynamic model routing
                routed_model = await route_request(db, user_id, model, prompt)
                
                # Check budget
                try:
                    await check_budget(db, user_id, routed_model)
                except Exception as e:
                    await websocket.send_text(json.dumps({"type": "error", "message": f"Budget block: {str(e)}"}))
                    continue
                    
                # Set up active handler
                active_handler = StreamHandler(session_id=f"ws_{user_id}_{int(time.time())}")
                
                # Setup mock token generator
                mock_text = f"This is a progressive response for your query '{prompt}' using {routed_model}."
                words = mock_text.split(" ")
                
                async def mock_word_generator():
                    for i, word in enumerate(words):
                        # Allow check for early abort
                        if active_handler.buffer.is_aborted:
                            break
                        await asyncio.sleep(0.05)
                        yield word + " " if i < len(words) - 1 else word
                        
                start_time = time.perf_counter()
                total_tokens = 0
                
                try:
                    async for token_chunk in active_handler.process_stream(mock_word_generator()):
                        total_tokens += 1
                        ws_payload = format_websocket(token=token_chunk, finished=False)
                        await websocket.send_text(ws_payload)
                        
                    if not active_handler.buffer.is_aborted:
                        duration = (time.perf_counter() - start_time) * 1000
                        usage = {"prompt_tokens": len(prompt) // 4 + 1, "completion_tokens": total_tokens}
                        
                        # Save cost/usage record
                        await track_request(
                            db=db,
                            user_id=user_id,
                            feature=feature,
                            model_name=routed_model,
                            input_tokens=usage["prompt_tokens"],
                            output_tokens=usage["completion_tokens"],
                            latency_ms=duration,
                        )
                        
                        ws_done = format_websocket(token=None, finished=True, usage={"total_tokens": usage["prompt_tokens"] + usage["completion_tokens"]})
                        await websocket.send_text(ws_done)
                except Exception as e:
                    logger.error("Error during WS LLM stream processing: %s", e)
                    await websocket.send_text(json.dumps({"type": "error", "message": "Stream interrupted"}))
                finally:
                    active_handler = None
                    
    except WebSocketDisconnect:
        logger.info("WS LLM client disconnected")
        if active_handler:
            active_handler.abort()
    except Exception as e:
        logger.error("WS general error: %s", e)
        if active_handler:
            active_handler.abort()
