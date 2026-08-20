"""Shared Azure factories for all Airflow DAGs.

Credentials and config are resolved from the Airflow connection
``azure_ml_conn`` (its Extra JSON), falling back to environment variables when
a value is not present there. This lets cloud/on-prem deployments use the
managed connection while local Docker runs keep using ``.env``.

The connection's Extra is expected to hold the service-principal/ML-workspace
keys plus the feedback storage values and the retrain threshold:

    {
      "tenant_id": "...",
      "client_id": "...",
      "client_secret": "...",
      "subscription_id": "...",
      "resource_group": "...",
      "workspace_name": "...",
      "FEEDBACK_DB_URL": "...",
      "FEEDBACK_BLOB_CONNECTION_STRING": "...",
      "FEEDBACK_BLOB_CONTAINER": "...",
      "RETRAIN_FEEDBACK_THRESHOLD": "5"
    }
"""

import logging
import os
from typing import TYPE_CHECKING

from airflow.exceptions import AirflowNotFoundException
from airflow.hooks.base import BaseHook
from azure.ai.ml import MLClient
from azure.identity import ClientSecretCredential

if TYPE_CHECKING:
    from azure.storage.blob import ContainerClient
    from psycopg2.extensions import connection as PgConnection

logger = logging.getLogger(__name__)

AZURE_ML_CONN_ID = "azure_ml_conn"

# --- ML client credentials -------------------------------------------------

# Internal canonical keys -> Airflow connection Extra keys.
_REQUIRED_KEYS = (
    "tenant_id",
    "client_id",
    "client_secret",
    "subscription_id",
    "resource_group",
    "workspace_name",
)

# Internal canonical keys -> environment variable names (fallback path).
_ENV_VAR_BY_KEY = {
    "tenant_id": "AZURE_TENANT_ID",
    "client_id": "AZURE_CLIENT_ID",
    "client_secret": "AZURE_CLIENT_SECRET",
    "subscription_id": "AZURE_SUBSCRIPTION_ID",
    "resource_group": "AZURE_RESOURCE_GROUP",
    "workspace_name": "AZURE_WORKSPACE_NAME",
}

# --- Feedback storage + retrain config -------------------------------------
# Stored in the azure_ml_conn Extra under the same names as the fallback env
# vars, so a single name resolves both.
_FEEDBACK_DB_URL = "FEEDBACK_DB_URL"
_FEEDBACK_BLOB_CONN_STR = "FEEDBACK_BLOB_CONNECTION_STRING"
_FEEDBACK_BLOB_CONTAINER = "FEEDBACK_BLOB_CONTAINER"
_RETRAIN_FEEDBACK_THRESHOLD = "RETRAIN_FEEDBACK_THRESHOLD"


def _credentials_from_connection() -> dict[str, str]:
    """Read ML credentials from the ``azure_ml_conn`` connection Extra.

    Returns:
        dict[str, str]: Mapping keyed by the internal canonical names in
        ``_REQUIRED_KEYS``.

    Raises:
        AirflowNotFoundException: If the connection is not defined. Callers
            use this to trigger the environment-variable fallback.
        KeyError: If the connection exists but its Extra is missing one or
            more required keys.
    """
    conn = BaseHook.get_connection(AZURE_ML_CONN_ID)
    extra = conn.extra_dejson

    missing = [key for key in _REQUIRED_KEYS if not extra.get(key)]
    if missing:
        raise KeyError(
            f"Airflow connection '{AZURE_ML_CONN_ID}' is missing required Extra "
            f"keys: {', '.join(missing)}"
        )

    return {key: extra[key] for key in _REQUIRED_KEYS}


def _credentials_from_env() -> dict[str, str]:
    """Read ML credentials from ``AZURE_*`` environment variables (fallback).

    Returns:
        dict[str, str]: Mapping keyed by the internal canonical names in
        ``_REQUIRED_KEYS``.

    Raises:
        KeyError: If one or more required environment variables are unset.
    """
    missing = [env for env in _ENV_VAR_BY_KEY.values() if not os.getenv(env)]
    if missing:
        raise KeyError(
            f"No '{AZURE_ML_CONN_ID}' connection found and required environment "
            f"variables are unset: {', '.join(missing)}"
        )

    return {key: os.environ[env] for key, env in _ENV_VAR_BY_KEY.items()}


def _load_credentials() -> dict[str, str]:
    """Resolve ML credentials, preferring the connection over env vars."""
    try:
        creds = _credentials_from_connection()
        logger.info(
            "Loaded Azure ML credentials from connection '%s'.", AZURE_ML_CONN_ID
        )
        return creds
    except AirflowNotFoundException:
        logger.info(
            "Connection '%s' not found; falling back to AZURE_* environment variables.",
            AZURE_ML_CONN_ID,
        )
        return _credentials_from_env()


def _conn_or_env(name: str) -> str | None:
    """Look up one value by ``name``: connection Extra first, then env var.

    Args:
        name: Key used both in the ``azure_ml_conn`` Extra and as the env var.

    Returns:
        The resolved value, or None if set in neither source.
    """
    try:
        value = BaseHook.get_connection(AZURE_ML_CONN_ID).extra_dejson.get(name)
        if value:
            return value
    except AirflowNotFoundException:
        pass
    return os.getenv(name)


def _required(name: str) -> str:
    """Resolve a required value via :func:`_conn_or_env`.

    Raises:
        KeyError: If the value is in neither the connection Extra nor the env.
    """
    value = _conn_or_env(name)
    if not value:
        raise KeyError(
            f"'{name}' is not set in connection '{AZURE_ML_CONN_ID}' Extra "
            f"nor as an environment variable."
        )
    return value


def get_ml_client() -> MLClient:
    """Create an authenticated MLClient for the team Azure ML workspace.

    Credentials are resolved by :func:`_load_credentials`: the ``azure_ml_conn``
    Airflow connection is used when available, otherwise the ``AZURE_*``
    environment variables are used.

    Returns:
        MLClient: Authenticated client for the team Azure ML workspace.

    Raises:
        KeyError: If credentials cannot be fully resolved from either source.
    """
    creds = _load_credentials()

    credential = ClientSecretCredential(
        tenant_id=creds["tenant_id"],
        client_id=creds["client_id"],
        client_secret=creds["client_secret"],
    )
    return MLClient(
        credential=credential,
        subscription_id=creds["subscription_id"],
        resource_group_name=creds["resource_group"],
        workspace_name=creds["workspace_name"],
    )


def get_feedback_db_conn() -> "PgConnection":
    """Open a psycopg2 connection to the cloud feedback Postgres.

    The DSN is resolved from the ``azure_ml_conn`` Extra key ``FEEDBACK_DB_URL``,
    falling back to the ``FEEDBACK_DB_URL`` env var.

    Returns:
        psycopg2 connection: An open connection. The caller must close it.

    Raises:
        KeyError: If the DSN is in neither the connection Extra nor the env.
    """
    import psycopg2

    return psycopg2.connect(_required(_FEEDBACK_DB_URL))


def get_feedback_container_client() -> "ContainerClient":
    """Build the blob ContainerClient for the feedback datastore container.

    The connection string and container name are resolved from the
    ``azure_ml_conn`` Extra keys ``FEEDBACK_BLOB_CONNECTION_STRING`` and
    ``FEEDBACK_BLOB_CONTAINER``, falling back to the env vars of the same names.

    Returns:
        ContainerClient: Client for the configured blob container.

    Raises:
        KeyError: If a value is in neither the connection Extra nor the env.
    """
    from azure.storage.blob import BlobServiceClient

    service = BlobServiceClient.from_connection_string(
        _required(_FEEDBACK_BLOB_CONN_STR)
    )
    return service.get_container_client(_required(_FEEDBACK_BLOB_CONTAINER))


def get_retrain_threshold(default: int = 50) -> int:
    """Resolve the retrain feedback threshold.

    Read from the ``azure_ml_conn`` Extra key ``RETRAIN_FEEDBACK_THRESHOLD``,
    falling back to the env var of the same name, then to ``default``.

    Args:
        default: Value used when the threshold is set in neither source.

    Returns:
        int: The threshold (minimum ready rows required to fire retraining).
    """
    raw = _conn_or_env(_RETRAIN_FEEDBACK_THRESHOLD)
    return int(raw) if raw else default
