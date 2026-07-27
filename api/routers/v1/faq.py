"""FAQ router for issue #307.

Provides endpoints for:
- Listing and searching FAQs
- Managing FAQs (admin)
- Submitting feedback
- Suggesting new FAQs
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_db
from api.models.orm import FAQ, FAQFeedback, FAQSuggestion
from api.schemas import (
    FAQOut,
    FAQIn,
    FAQUpdateIn,
    FAQListResponse,
    FAQFeedbackIn,
    FAQFeedbackOut,
    FAQSuggestionIn,
    FAQSuggestionOut,
    FAQSuggestionListResponse,
)

router = APIRouter(prefix="/faq", tags=["faq"])


@router.get("", response_model=FAQListResponse)
async def list_faqs(
    category: Optional[str] = None,
    search: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """List all published FAQs with optional category filter and full-text search."""
    query = select(FAQ).where(FAQ.is_published == True)

    if category:
        query = query.where(FAQ.category == category)

    if search:
        search_pattern = f"%{search}%"
        query = query.where(
            or_(
                FAQ.question.ilike(search_pattern),
                FAQ.answer.ilike(search_pattern),
                FAQ.category.ilike(search_pattern),
            )
        )

    query = query.order_by(FAQ.order, FAQ.id)

    result = await db.execute(query)
    faqs = result.scalars().all()

    # Get distinct categories
    categories_query = (
        select(FAQ.category)
        .where(FAQ.is_published == True)
        .distinct()
        .order_by(FAQ.category)
    )
    categories_result = await db.execute(categories_query)
    categories = [row[0] for row in categories_result.fetchall()]

    return FAQListResponse(
        data=[FAQOut.model_validate(faq) for faq in faqs],
        categories=categories,
        total=len(faqs),
    )


@router.get("/{faq_id}", response_model=FAQOut)
async def get_faq(faq_id: int, db: AsyncSession = Depends(get_db)):
    """Get a specific FAQ by ID."""
    query = select(FAQ).where(FAQ.id == faq_id, FAQ.is_published == True)
    result = await db.execute(query)
    faq = result.scalar_one_or_none()

    if not faq:
        raise HTTPException(status_code=404, detail="FAQ not found")

    return FAQOut.model_validate(faq)


@router.post("", response_model=FAQOut, status_code=201)
async def create_faq(faq_in: FAQIn, db: AsyncSession = Depends(get_db)):
    """Create a new FAQ (admin)."""
    faq = FAQ(**faq_in.model_dump())
    db.add(faq)
    await db.commit()
    await db.refresh(faq)
    return FAQOut.model_validate(faq)


@router.put("/{faq_id}", response_model=FAQOut)
async def update_faq(
    faq_id: int, faq_in: FAQUpdateIn, db: AsyncSession = Depends(get_db)
):
    """Update an existing FAQ (admin)."""
    query = select(FAQ).where(FAQ.id == faq_id)
    result = await db.execute(query)
    faq = result.scalar_one_or_none()

    if not faq:
        raise HTTPException(status_code=404, detail="FAQ not found")

    update_data = faq_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(faq, field, value)

    await db.commit()
    await db.refresh(faq)
    return FAQOut.model_validate(faq)


@router.delete("/{faq_id}", status_code=204)
async def delete_faq(faq_id: int, db: AsyncSession = Depends(get_db)):
    """Delete an FAQ (admin)."""
    query = select(FAQ).where(FAQ.id == faq_id)
    result = await db.execute(query)
    faq = result.scalar_one_or_none()

    if not faq:
        raise HTTPException(status_code=404, detail="FAQ not found")

    await db.delete(faq)
    await db.commit()


@router.post("/{faq_id}/feedback", response_model=FAQFeedbackOut, status_code=201)
async def submit_feedback(
    faq_id: int, feedback_in: FAQFeedbackIn, db: AsyncSession = Depends(get_db)
):
    """Submit feedback on FAQ helpfulness."""
    # Verify FAQ exists
    faq_query = select(FAQ).where(FAQ.id == faq_id)
    faq_result = await db.execute(faq_query)
    if not faq_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="FAQ not found")

    feedback = FAQFeedback(faq_id=faq_id, **feedback_in.model_dump())
    db.add(feedback)
    await db.commit()
    await db.refresh(feedback)
    return FAQFeedbackOut.model_validate(feedback)


@router.get("/{faq_id}/feedback/stats")
async def get_feedback_stats(faq_id: int, db: AsyncSession = Depends(get_db)):
    """Get feedback statistics for an FAQ."""
    # Verify FAQ exists
    faq_query = select(FAQ).where(FAQ.id == faq_id)
    faq_result = await db.execute(faq_query)
    if not faq_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="FAQ not found")

    helpful_query = select(func.count(FAQFeedback.id)).where(
        FAQFeedback.faq_id == faq_id, FAQFeedback.is_helpful == True
    )
    not_helpful_query = select(func.count(FAQFeedback.id)).where(
        FAQFeedback.faq_id == faq_id, FAQFeedback.is_helpful == False
    )

    helpful_result = await db.execute(helpful_query)
    not_helpful_result = await db.execute(not_helpful_query)

    helpful_count = helpful_result.scalar() or 0
    not_helpful_count = not_helpful_result.scalar() or 0
    total = helpful_count + not_helpful_count

    return {
        "faq_id": faq_id,
        "helpful": helpful_count,
        "not_helpful": not_helpful_count,
        "total": total,
        "helpful_percentage": round(helpful_count / total * 100, 1) if total > 0 else 0,
    }


@router.post("/suggestions", response_model=FAQSuggestionOut, status_code=201)
async def submit_suggestion(
    suggestion_in: FAQSuggestionIn, db: AsyncSession = Depends(get_db)
):
    """Submit a new FAQ suggestion."""
    suggestion = FAQSuggestion(**suggestion_in.model_dump())
    db.add(suggestion)
    await db.commit()
    await db.refresh(suggestion)
    return FAQSuggestionOut.model_validate(suggestion)


@router.get("/suggestions/list", response_model=FAQSuggestionListResponse)
async def list_suggestions(
    status: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """List FAQ suggestions (admin)."""
    query = select(FAQSuggestion)

    if status:
        query = query.where(FAQSuggestion.status == status)

    total_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(total_query)
    total = total_result.scalar()

    query = query.order_by(FAQSuggestion.created_at.desc())
    query = query.offset((page - 1) * page_size).limit(page_size)

    result = await db.execute(query)
    suggestions = result.scalars().all()

    return FAQSuggestionListResponse(
        data=[FAQSuggestionOut.model_validate(s) for s in suggestions],
        page=page,
        page_size=page_size,
        total=total,
    )


@router.put("/suggestions/{suggestion_id}/status")
async def update_suggestion_status(
    suggestion_id: int, status: str, db: AsyncSession = Depends(get_db)
):
    """Update suggestion status (admin)."""
    if status not in ["pending", "approved", "rejected"]:
        raise HTTPException(status_code=400, detail="Invalid status")

    query = select(FAQSuggestion).where(FAQSuggestion.id == suggestion_id)
    result = await db.execute(query)
    suggestion = result.scalar_one_or_none()

    if not suggestion:
        raise HTTPException(status_code=404, detail="Suggestion not found")

    suggestion.status = status
    await db.commit()
    await db.refresh(suggestion)
    return FAQSuggestionOut.model_validate(suggestion)
