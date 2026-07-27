"""Reports API endpoints (issue XXX)."""
from __future__ import annotations

import io
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth.dependencies import get_current_auth, AuthContext
from api.database import get_async_session_factory
from api.services.report_generator import report_generator

router = APIRouter(prefix="/api/v1/reports", tags=["reports"])


class GenerateReportRequest(BaseModel):
    days: Optional[int] = 90
    format: str = "pdf"  # pdf or markdown
    include_summary: Optional[bool] = True


@router.post("/generate")
async def generate_report(
    request: GenerateReportRequest,
    auth: AuthContext = Depends(get_current_auth),
    db: AsyncSession = Depends(get_async_session_factory),
):
    """Generate a report with LLM insights and charts."""
    try:
        report_data = await report_generator.fetch_report_data(db, days=request.days)
        
        # Generate LLM summary (placeholder for now)
        llm_summary = None
        
        markdown_content = report_generator.generate_markdown(report_data, llm_summary=llm_summary)
        
        if request.format == "markdown":
            return JSONResponse(content={"markdown": markdown_content})
        else:
            pdf_bytes = await report_generator.generate_pdf(markdown_content)
            
            if not pdf_bytes:
                # Fallback to markdown if PDF generation fails
                return JSONResponse(content={"markdown": markdown_content, "warning": "PDF generation failed"})
            
            return StreamingResponse(
                io.BytesIO(pdf_bytes),
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f"attachment; filename=astroml-report-{datetime.utcnow().strftime('%Y%m%d')}.pdf"
                }
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to generate report")
