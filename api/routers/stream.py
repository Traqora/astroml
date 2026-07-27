import asyncio
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional
from astroml.llm.stream.server import get_streaming_server
from astroml.llm.stream.reconnect import get_reconnection_manager
from astroml.llm.stream.protocol import StreamProtocol

router = APIRouter(prefix="/api/v1/llm-stream", tags=["llm-stream"])

@router.get("/sse")
async def sse_stream(
    prompt: str = Query(..., description="Prompt to stream response for"),
    session_id: Optional[str] = Query(None, description="Optional session ID to resume"),
    last_token_idx: int = Query(0, ge=0, description="Optional last token index to resume from")
):
    reconnect_mgr = get_reconnection_manager()
    server = get_streaming_server()
    
    if session_id:
        resumed_tokens = reconnect_mgr.resume_stream(session_id, last_token_idx)
        if resumed_tokens is None:
            # Session expired/not found, create new session
            session_id = reconnect_mgr.create_session(prompt)
        else:
            async def resume_generator():
                for t in resumed_tokens:
                    yield StreamProtocol.format_sse("token", {"token": t, "session_id": session_id})
                    await asyncio.sleep(0.01)
            return StreamingResponse(resume_generator(), media_type="text/event-stream")
    else:
        session_id = reconnect_mgr.create_session(prompt)

    # Simulated generation content
    response_text = f"This is a simulated streaming response for your prompt: '{prompt}'. It demonstrates adaptive buffering, backpressure compensation, and reconnection capabilities."

    async def event_generator():
        yield StreamProtocol.format_sse("start", {"session_id": session_id})
        async for token in server.stream_tokens(response_text, session_id):
            yield StreamProtocol.format_sse("token", {"token": token, "session_id": session_id})
        yield StreamProtocol.format_sse("done", {"session_id": session_id})

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.websocket("/ws")
async def ws_stream(websocket: WebSocket):
    await websocket.accept()
    reconnect_mgr = get_reconnection_manager()
    server = get_streaming_server()
    
    try:
        # Initial message
        data = await websocket.receive_json()
        prompt = data.get("prompt", "")
        session_id = reconnect_mgr.create_session(prompt)
        
        await websocket.send_json(StreamProtocol.format_ws("start", {"session_id": session_id}))
        
        response_text = f"This is a simulated streaming response for your prompt: '{prompt}' delivered over WebSockets."
        
        async for token in server.stream_tokens(response_text, session_id):
            await websocket.send_json(StreamProtocol.format_ws("token", {"token": token, "session_id": session_id}))
            
        await websocket.send_json(StreamProtocol.format_ws("done", {"session_id": session_id}))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"event": "error", "message": "An error occurred"})
        except:
            pass
    finally:
        await websocket.close()
