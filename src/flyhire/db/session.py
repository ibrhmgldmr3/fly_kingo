from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from flyhire.config import get_settings


def make_engine(url: str | None = None):
    # connect_timeout olmadan kapalı porta bağlanma Windows'ta dakikalarca bekleyebiliyor.
    return create_engine(
        url or get_settings().database_url, pool_pre_ping=True, connect_args={"connect_timeout": 5}
    )


def make_session_factory(url: str | None = None) -> sessionmaker[Session]:
    return sessionmaker(make_engine(url), expire_on_commit=False)
