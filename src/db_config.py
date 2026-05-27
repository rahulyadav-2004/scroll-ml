import os
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def _to_bool(value, fallback=False):
    if value is None or value == "":
        return fallback
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on", "enabled"}:
        return True
    if normalized in {"0", "false", "no", "off", "disabled"}:
        return False
    return fallback


def _with_query_param(url, key, value):
    parts = urlsplit(url)
    params = dict(parse_qsl(parts.query, keep_blank_values=True))
    params.setdefault(key, value)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(params), parts.fragment))


def resolve_database_url():
    """
    Resolve the database URL for the ML service.

    Hyperdrive bindings are only available inside Cloudflare Workers/Pages
    Functions, not inside this Railway-hosted Python service. If the service is
    ever moved into a Cloudflare runtime and a generated Hyperdrive connection
    string is injected, prefer it. Otherwise use the normal Postgres URLs.
    """
    hyperdrive_enabled = _to_bool(os.getenv("HYPERDRIVE_ENABLED"), False)
    cloudflare_runtime = _to_bool(os.getenv("CLOUDFLARE_WORKER"), False)
    hyperdrive_url = (os.getenv("HYPERDRIVE_CONNECTION_STRING") or "").strip().strip("\"'")

    db_url = (
        (hyperdrive_url if hyperdrive_enabled and cloudflare_runtime else "")
        or os.getenv("DATABASE_URL")
        or os.getenv("DATABASE_PUBLIC_URL")
        or ""
    ).strip().strip("\"'")

    if not db_url:
        return ""

    ssl_mode = (os.getenv("PGSSLMODE") or os.getenv("DB_SSL") or "").strip().lower()
    if ssl_mode in {"require", "prefer", "verify-ca", "verify-full"} and "sslmode=" not in db_url.lower():
        db_url = _with_query_param(db_url, "sslmode", ssl_mode)

    return db_url


def resolve_sqlalchemy_database_url():
    db_url = resolve_database_url()
    if db_url.startswith("postgresql://"):
        return db_url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return db_url
