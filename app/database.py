from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings

DATABASE_URL = f"sqlite+aiosqlite:///{settings.STORAGE_DIR / 'flux_service.db'}"

engine = create_async_engine(DATABASE_URL, echo=False)
async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db() -> None:
    from app.models.job import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncSession:  # type: ignore[misc]
    async with async_session_factory() as session:
        yield session  # type: ignore[misc]
