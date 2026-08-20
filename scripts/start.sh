#!/usr/bin/env sh
# One-command local startup for Linux and macOS.
# Usage: ./scripts/start.sh [extra docker compose flags]
set -eu

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$REPO_ROOT/configs/env/.env"
COMPOSE_FILE="$REPO_ROOT/infra/local/docker-compose.yml"

choose_env_template() {
    if [ -f "$REPO_ROOT/configs/env/.env.example" ]; then
        printf '%s\n' "$REPO_ROOT/configs/env/.env.example"
        return 0
    fi

    if [ -f "$REPO_ROOT/configs/env/env.example" ]; then
        printf '%s\n' "$REPO_ROOT/configs/env/env.example"
        return 0
    fi

    return 1
}

generate_secret() {
    if command -v openssl > /dev/null 2>&1; then
        openssl rand -hex 32
        return 0
    fi

    echo "ERROR: openssl not found. Install openssl or set the required values in $ENV_FILE manually."
    exit 1
}

set_env_value() {
    key="$1"
    value="$2"

    if grep -q "^${key}=" "$ENV_FILE"; then
        sed "s|^${key}=.*|${key}=${value}|" "$ENV_FILE" > "$ENV_FILE.tmp"
        mv "$ENV_FILE.tmp" "$ENV_FILE"
    else
        printf '%s=%s\n' "$key" "$value" >> "$ENV_FILE"
    fi
}

ensure_secret() {
    key="$1"
    placeholder="$2"
    current_value="$(grep "^${key}=" "$ENV_FILE" | head -n 1 | cut -d= -f2- || true)"

    if [ -z "$current_value" ] || [ "$current_value" = "$placeholder" ]; then
        set_env_value "$key" "$(generate_secret)"
    fi
}

# ---- prerequisites ----

if ! command -v docker > /dev/null 2>&1; then
    echo "ERROR: Docker not found. Install Docker Desktop: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! docker compose version > /dev/null 2>&1; then
    echo "ERROR: Docker Compose plugin not found."
    exit 1
fi

# ---- first-run .env setup ----

if [ ! -f "$ENV_FILE" ]; then
    ENV_TEMPLATE="$(choose_env_template)"

    if [ -z "${ENV_TEMPLATE:-}" ]; then
        echo "ERROR: Could not find configs/env/.env.example or configs/env/env.example."
        exit 1
    fi

    cp "$ENV_TEMPLATE" "$ENV_FILE"
    echo "Created $ENV_FILE from $(basename "$ENV_TEMPLATE")."
fi

# ---- ensure required secrets ----

ensure_secret "API_KEY" "your-api-key-here"
ensure_secret "POSTGRES_PASSWORD" ""
ensure_secret "ADMIN_API_KEY" "your-admin-api-key-here"
ensure_secret "JWT_SIGNING_KEY" ""
ensure_secret "SESSION_SECRET" ""

API_KEY_VALUE="$(grep '^API_KEY=' "$ENV_FILE" | head -n 1 | cut -d= -f2-)"
if [ -z "$API_KEY_VALUE" ]; then
    echo "ERROR: API_KEY is missing or empty in $ENV_FILE."
    exit 1
fi

export API_KEY="$API_KEY_VALUE"
export POSTGRES_PASSWORD="$(grep '^POSTGRES_PASSWORD=' "$ENV_FILE" | head -n 1 | cut -d= -f2-)"
export ADMIN_API_KEY="$(grep '^ADMIN_API_KEY=' "$ENV_FILE" | head -n 1 | cut -d= -f2-)"
export JWT_SIGNING_KEY="$(grep '^JWT_SIGNING_KEY=' "$ENV_FILE" | head -n 1 | cut -d= -f2-)"
export SESSION_SECRET="$(grep '^SESSION_SECRET=' "$ENV_FILE" | head -n 1 | cut -d= -f2-)"

echo "Using environment from $ENV_FILE"

# ---- start ----

echo "Starting CV Platform..."
echo "  Frontend : http://localhost:3000"
echo "  Backend  : http://localhost:8000/docs"
echo ""
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up --build "$@"
