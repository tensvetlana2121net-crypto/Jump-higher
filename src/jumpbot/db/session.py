from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from jumpbot.config import get_settings

settings = get_settings()
# Celery invokes each synchronous task through asyncio.run(), which creates a
# fresh event loop. asyncpg connections are bound to the loop that created them,
# so they must not be reused by a later task running on another loop.
engine = create_async_engine(settings.database_url, poolclass=NullPool)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session
