from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from jumpbot.config import get_settings
from jumpbot.db.models import Plan, Subscription, SubscriptionStatus, UsageCounter


def week_start(now: datetime | None = None) -> datetime:
    current = now or datetime.now(UTC)
    midnight = current.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight - timedelta(days=midnight.weekday())


async def consume_analysis(session: AsyncSession, user_id: int) -> bool:
    pro = await session.scalar(
        select(Subscription.id).where(
            Subscription.user_id == user_id,
            Subscription.plan == Plan.PRO,
            Subscription.status.in_([SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIALING]),
            Subscription.current_period_end > datetime.now(UTC),
        )
    )
    if pro:
        return True

    period = week_start()
    counter = await session.scalar(
        select(UsageCounter)
        .where(UsageCounter.user_id == user_id, UsageCounter.period_start == period)
        .with_for_update()
    )
    if counter is None:
        counter = UsageCounter(user_id=user_id, period_start=period, analyses_count=0)
        session.add(counter)
    if counter.analyses_count >= get_settings().free_analyses_per_week:
        return False
    counter.analyses_count += 1
    await session.flush()
    return True
