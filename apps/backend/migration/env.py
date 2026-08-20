import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from api.db.models import Base  # noqa: E402, I001

target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def _db_url() -> str:
    """Resolve the migration database URL from the environment.

    Alembic is synchronous and connects via psycopg2, so the ``+asyncpg``
    dialect suffix the application runtime uses is stripped here.

    Raises when ``DB_URL`` is unset. There is deliberately no fallback: the
    previous version fell back to a DSN committed in ``alembic.ini``, so a
    missing environment variable silently migrated whatever database that file
    happened to name rather than failing.
    """
    url = os.getenv("DB_URL")
    if not url:
        raise RuntimeError(
            "DB_URL is not set, and there is no default. Alembic needs it to "
            "know which database to migrate. Example: "
            "DB_URL=postgresql+asyncpg://USER:PASSWORD@localhost:5432/DBNAME"
        )
    return url.replace("postgresql+asyncpg://", "postgresql://")


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = _db_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = _db_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
