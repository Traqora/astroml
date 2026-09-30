"""GraphQL schema definition."""

from __future__ import annotations

from typing import List, Optional

import strawberry
from strawberry import ID
from strawberry.tools import merge_types
from strawberry.types import Info

from api.database import _sync_session_factory
from api.graphql.context import GraphQLContext, get_graphql_context
from api.graphql.types import FAQ as GFAQ
from api.graphql.types import (
    Account,
    AccountConnection,
    CreateAccountInput,
    CreateFraudAlertInput,
    FraudAlertConnection,
    MutationResult,
    PageInfo,
    Transaction,
    TransactionConnection,
    UpdateLoyaltyPointsInput,
)
from api.graphql.types import ApiKey as GApiKey
from api.graphql.types import AuditLog as GAuditLog
from api.graphql.types import FraudAlert as GFraudAlert
from api.graphql.types import LoyaltyPoints as GLoyaltyPoints
from api.graphql.types import Mentee as GMentee
from api.graphql.types import Mentor as GMentor
from api.graphql.types import Mentorship as GMentorship
from api.graphql.types import ModelRegistry as GModelRegistry
from api.graphql.types import Notification as GNotification
from api.graphql.types import PointsTransaction as GPointsTransaction
from api.graphql.types import User as GUser
from api.models.orm import (
    FAQ,
    ApiAccount,
    ApiKey,
    ApiTransaction,
    AuditLog,
    FraudAlert as FraudAlertModel,
    LoyaltyPoints,
    Mentee,
    Mentor,
    Mentorship,
    ModelRegistry,
    Notification,
    PointsTransaction,
    User,
)


# Query resolvers
@strawberry.type
class Query:
    """GraphQL queries."""

    @strawberry.field
    def account(self, id: ID, info: Info) -> Optional[Account]:
        """Get an account by ID."""
        account = info.context.session.query(ApiAccount).filter_by(id=int(id)).first()
        if not account:
            return None
        return Account(
            id=strawberry.ID(str(account.id)),
            public_key=account.public_key,
            first_seen=account.first_seen,
            last_active=account.last_active,
            balance=account.balance,
            home_domain=account.home_domain,
            created_at=account.created_at,
        )

    @strawberry.field
    def account_by_public_key(self, public_key: str, info: Info) -> Optional[Account]:
        """Get an account by public key."""
        account = info.context.session.query(ApiAccount).filter_by(public_key=public_key).first()
        if not account:
            return None
        return Account(
            id=strawberry.ID(str(account.id)),
            public_key=account.public_key,
            first_seen=account.first_seen,
            last_active=account.last_active,
            balance=account.balance,
            home_domain=account.home_domain,
            created_at=account.created_at,
        )

    @strawberry.field
    def accounts(
        self,
        info: Info,
        limit: int = 20,
        offset: int = 0,
    ) -> AccountConnection:
        """Get paginated accounts."""
        query = info.context.session.query(ApiAccount)
        total = query.count()
        accounts = query.offset(offset).limit(limit).all()

        return AccountConnection(
            edges=[
                Account(
                    id=strawberry.ID(str(a.id)),
                    public_key=a.public_key,
                    first_seen=a.first_seen,
                    last_active=a.last_active,
                    balance=a.balance,
                    home_domain=a.home_domain,
                    created_at=a.created_at,
                )
                for a in accounts
            ],
            page_info=PageInfo(
                has_next_page=offset + limit < total,
                has_previous_page=offset > 0,
            ),
            total_count=total,
        )

    @strawberry.field
    def transaction(self, hash: str, info: Info) -> Optional[Transaction]:
        """Get a transaction by hash."""
        tx = info.context.session.query(ApiTransaction).filter_by(hash=hash).first()
        if not tx:
            return None
        return Transaction(
            hash=tx.hash,
            ledger_sequence=tx.ledger_sequence,
            source_account=tx.source_account,
            destination_account=tx.destination_account,
            amount=tx.amount,
            asset_code=tx.asset_code,
            asset_issuer=tx.asset_issuer,
            fee=tx.fee,
            operation_type=tx.operation_type,
            successful=tx.successful,
            memo_type=tx.memo_type,
            created_at=tx.created_at,
        )

    @strawberry.field
    def transactions(
        self,
        info: Info,
        source_account: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> TransactionConnection:
        """Get paginated transactions."""
        query = info.context.session.query(ApiTransaction)
        if source_account:
            query = query.filter_by(source_account=source_account)
        total = query.count()
        txs = query.order_by(ApiTransaction.created_at.desc()).offset(offset).limit(limit).all()

        return TransactionConnection(
            edges=[
                Transaction(
                    hash=tx.hash,
                    ledger_sequence=tx.ledger_sequence,
                    source_account=tx.source_account,
                    destination_account=tx.destination_account,
                    amount=tx.amount,
                    asset_code=tx.asset_code,
                    asset_issuer=tx.asset_issuer,
                    fee=tx.fee,
                    operation_type=tx.operation_type,
                    successful=tx.successful,
                    memo_type=tx.memo_type,
                    created_at=tx.created_at,
                )
                for tx in txs
            ],
            page_info=PageInfo(
                has_next_page=offset + limit < total,
                has_previous_page=offset > 0,
            ),
            total_count=total,
        )

    @strawberry.field
    def fraud_alerts(
        self,
        info: Info,
        account_id: Optional[str] = None,
        resolved: Optional[bool] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> FraudAlertConnection:
        """Get paginated fraud alerts."""
        query = info.context.session.query(FraudAlertModel)
        if account_id:
            query = query.filter_by(account_id=account_id)
        if resolved is not None:
            query = query.filter_by(resolved=resolved)
        total = query.count()
        alerts = query.order_by(FraudAlertModel.detected_at.desc()).offset(offset).limit(limit).all()

        return FraudAlertConnection(
            edges=[
                GFraudAlert(
                    id=strawberry.ID(str(a.id)),
                    account_id=a.account_id,
                    pattern=a.pattern,
                    risk_score=a.risk_score,
                    risk_level=a.risk_level,
                    description=a.description,
                    detected_at=a.detected_at,
                    resolved=a.resolved,
                )
                for a in alerts
            ],
            page_info=PageInfo(
                has_next_page=offset + limit < total,
                has_previous_page=offset > 0,
            ),
            total_count=total,
        )

    @strawberry.field
    def loyalty_points(self, account_id: str, info: Info) -> Optional[GLoyaltyPoints]:
        """Get loyalty points for an account."""
        lp = info.context.session.query(LoyaltyPoints).filter_by(account_id=account_id).first()
        if not lp:
            return None
        return GLoyaltyPoints(
            id=strawberry.ID(str(lp.id)),
            account_id=lp.account_id,
            balance=lp.balance,
            tier=lp.tier,
            multiplier=lp.multiplier,
            updated_at=lp.updated_at,
        )

    @strawberry.field
    def loyalty_points_transactions(
        self,
        info: Info,
        account_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> List[GPointsTransaction]:
        """Get loyalty points transactions for an account."""
        txs = (
            info.context.session.query(PointsTransaction)
            .filter_by(account_id=account_id)
            .order_by(PointsTransaction.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        return [
            GPointsTransaction(
                id=strawberry.ID(str(tx.id)),
                account_id=tx.account_id,
                type=tx.type,
                points=tx.points,
                source=tx.source,
                note=tx.note,
                created_at=tx.created_at,
            )
            for tx in txs
        ]

    @strawberry.field
    def model_versions(
        self,
        info: Info,
        name: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[GModelRegistry]:
        """Get model versions."""
        query = info.context.session.query(ModelRegistry)
        if name:
            query = query.filter_by(name=name)
        if status:
            query = query.filter_by(status=status)
        models = query.order_by(ModelRegistry.created_at.desc()).offset(offset).limit(limit).all()

        return [
            GModelRegistry(
                id=strawberry.ID(str(m.id)),
                name=m.name,
                version=m.version,
                path=m.path,
                owner=m.owner,
                tags=m.tags,
                mlflow_run_id=m.mlflow_run_id,
                metrics=strawberry.scalars.JSON(m.metrics) if m.metrics else None,
                status=m.status,
                parent_id=m.parent_id,
                created_at=m.created_at,
            )
            for m in models
        ]

    @strawberry.field
    def user(self, id: ID, info: Info) -> Optional[GUser]:
        """Get a user by ID."""
        user = info.context.session.query(User).filter_by(id=int(id)).first()
        if not user:
            return None
        return GUser(
            id=strawberry.ID(str(user.id)),
            username=user.username,
            scopes=user.scopes or [],
            is_active=user.is_active,
            created_at=user.created_at,
        )

    @strawberry.field
    def me(self, info: Info) -> Optional[GUser]:
        """Get the current authenticated user."""
        if not info.context.user:
            return None
        user = (
            info.context.session.query(User).filter_by(id=info.context.user.get("user_id")).first()
        )
        if not user:
            return None
        return GUser(
            id=strawberry.ID(str(user.id)),
            username=user.username,
            scopes=user.scopes or [],
            is_active=user.is_active,
            created_at=user.created_at,
        )

    @strawberry.field
    def notifications(
        self,
        info: Info,
        is_read: Optional[bool] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[GNotification]:
        """Get notifications for the current user."""
        if not info.context.user:
            return []
        query = info.context.session.query(Notification).filter_by(
            user_id=info.context.user.get("user_id")
        )
        if is_read is not None:
            query = query.filter_by(is_read=is_read)
        notifications = (
            query.order_by(Notification.created_at.desc()).offset(offset).limit(limit).all()
        )

        return [
            GNotification(
                id=strawberry.ID(str(n.id)),
                user_id=n.user_id,
                event_type=n.event_type,
                title=n.title,
                content=n.content,
                link=n.link,
                actor=n.actor,
                is_read=n.is_read,
                created_at=n.created_at,
            )
            for n in notifications
        ]

    @strawberry.field
    def faqs(
        self,
        info: Info,
        category: Optional[str] = None,
        is_published: bool = True,
    ) -> List[GFAQ]:
        """Get FAQs."""
        query = info.context.session.query(FAQ).filter_by(is_published=is_published)
        if category:
            query = query.filter_by(category=category)
        faqs = query.order_by(FAQ.order.asc()).all()

        return [
            GFAQ(
                id=strawberry.ID(str(f.id)),
                category=f.category,
                question=f.question,
                answer=f.answer,
                order=f.order,
                is_published=f.is_published,
                created_at=f.created_at,
                updated_at=f.updated_at,
            )
            for f in faqs
        ]


# Mutation resolvers
@strawberry.type
class Mutation:
    """GraphQL mutations."""

    @strawberry.mutation
    def create_account(self, input: CreateAccountInput, info: Info) -> MutationResult:
        """Create a new account."""
        existing = (
            info.context.session.query(ApiAccount).filter_by(public_key=input.public_key).first()
        )
        if existing:
            return MutationResult(success=False, message="Account already exists")

        account = ApiAccount(
            public_key=input.public_key,
            home_domain=input.home_domain,
            first_seen=datetime.now(),
            last_active=datetime.now(),
        )
        info.context.session.add(account)
        info.context.session.commit()

        return MutationResult(success=True, message="Account created", id=str(account.id))

    @strawberry.mutation
    def create_fraud_alert(
        self,
        input: CreateFraudAlertInput,
        info: Info,
    ) -> MutationResult:
        """Create a new fraud alert."""
        alert = FraudAlertModel(
            account_id=input.account_id,
            pattern=input.pattern,
            risk_score=input.risk_score,
            risk_level=FraudAlertModel.risk_level_for_score(input.risk_score),
            description=input.description,
        )
        info.context.session.add(alert)
        info.context.session.commit()

        return MutationResult(success=True, message="Fraud alert created", id=str(alert.id))

    @strawberry.mutation
    def resolve_fraud_alert(self, id: ID, info: Info) -> MutationResult:
        """Resolve a fraud alert."""
        alert = info.context.session.query(FraudAlertModel).filter_by(id=int(id)).first()
        if not alert:
            return MutationResult(success=False, message="Fraud alert not found")

        alert.resolved = True
        info.context.session.commit()

        return MutationResult(success=True, message="Fraud alert resolved", id=str(alert.id))

    @strawberry.mutation
    def update_loyalty_points(
        self,
        input: UpdateLoyaltyPointsInput,
        info: Info,
    ) -> MutationResult:
        """Update loyalty points for an account."""
        lp = (
            info.context.session.query(LoyaltyPoints).filter_by(account_id=input.account_id).first()
        )
        if not lp:
            lp = LoyaltyPoints(account_id=input.account_id, balance=0)
            info.context.session.add(lp)

        lp.balance += input.points
        if lp.balance < 0:
            lp.balance = 0

        # Create transaction record
        tx = PointsTransaction(
            account_id=input.account_id,
            type="adjust" if input.points < 0 else "earn",
            points=abs(input.points),
            source=input.source,
            note=input.note,
        )
        info.context.session.add(tx)
        info.context.session.commit()

        return MutationResult(success=True, message="Loyalty points updated", id=str(lp.id))

    @strawberry.mutation
    def mark_notification_read(self, id: ID, info: Info) -> MutationResult:
        """Mark a notification as read."""
        notification = info.context.session.query(Notification).filter_by(id=int(id)).first()
        if not notification:
            return MutationResult(success=False, message="Notification not found")

        notification.is_read = True
        info.context.session.commit()

        return MutationResult(
            success=True, message="Notification marked as read", id=str(notification.id)
        )

    @strawberry.mutation
    def mark_all_notifications_read(self, info: Info) -> MutationResult:
        """Mark all notifications as read for the current user."""
        if not info.context.user:
            return MutationResult(success=False, message="Not authenticated")

        info.context.session.query(Notification).filter_by(
            user_id=info.context.user.get("user_id"),
            is_read=False,
        ).update({"is_read": True})
        info.context.session.commit()

        return MutationResult(success=True, message="All notifications marked as read")


# Subscriptions resolvers
@strawberry.type
class Subscription:
    """GraphQL subscriptions."""

    @strawberry.subscription
    async def transaction_created(self) -> Transaction:
        """Subscribe to new transactions."""
        import asyncio

        from api.graphql.subscriptions import transaction_queue

        while True:
            try:
                tx_data = await transaction_queue.get()
                yield Transaction(
                    hash=tx_data.get("hash", ""),
                    ledger_sequence=tx_data.get("ledger_sequence", 0),
                    source_account=tx_data.get("source_account", ""),
                    destination_account=tx_data.get("destination_account"),
                    amount=tx_data.get("amount"),
                    asset_code=tx_data.get("asset_code"),
                    asset_issuer=tx_data.get("asset_issuer"),
                    fee=tx_data.get("fee", 0),
                    operation_type=tx_data.get("operation_type"),
                    successful=tx_data.get("successful", True),
                    memo_type=tx_data.get("memo_type"),
                    created_at=tx_data.get("created_at", datetime.now()),
                )
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(0.1)

    @strawberry.subscription
    async def fraud_alert_created(self) -> GFraudAlert:
        """Subscribe to new fraud alerts."""
        import asyncio

        from api.graphql.subscriptions import fraud_alert_queue

        while True:
            try:
                alert_data = await fraud_alert_queue.get()
                yield GFraudAlert(
                    id=strawberry.ID(str(alert_data.get("id", 0))),
                    account_id=alert_data.get("account_id", ""),
                    pattern=alert_data.get("pattern"),
                    risk_score=alert_data.get("risk_score", 0.0),
                    risk_level=alert_data.get("risk_level", "low"),
                    description=alert_data.get("description"),
                    detected_at=alert_data.get("detected_at", datetime.now()),
                    resolved=False,
                )
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(0.1)

    @strawberry.subscription
    async def loyalty_points_updated(self) -> GLoyaltyPoints:
        """Subscribe to loyalty points updates."""
        import asyncio

        from api.graphql.subscriptions import loyalty_points_queue

        while True:
            try:
                lp_data = await loyalty_points_queue.get()
                yield GLoyaltyPoints(
                    id=strawberry.ID(str(lp_data.get("id", 0))),
                    account_id=lp_data.get("account_id", ""),
                    balance=lp_data.get("balance", 0),
                    tier=lp_data.get("tier", "bronze"),
                    multiplier=lp_data.get("multiplier", 1.0),
                    updated_at=lp_data.get("updated_at", datetime.now()),
                )
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(0.1)


# Create schema
schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    subscription=Subscription,
)
