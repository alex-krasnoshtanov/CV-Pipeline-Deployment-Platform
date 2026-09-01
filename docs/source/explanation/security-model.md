# Security model

How authentication and authorization work today, and where they're
headed.

## First iteration: shared X-API-Key

Every endpoint except `/health` requires a valid `X-API-Key` HTTP
header. The backend compares the incoming header to the `API_KEY`
environment variable using `hmac.compare_digest`:

```python
if x_api_key is None or not hmac.compare_digest(x_api_key, expected):
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={
            "error_code": "UNAUTHORIZED",
            "message": "Missing or invalid X-API-Key header.",
        },
        headers={"WWW-Authenticate": "ApiKey"},
    )
```

### Why `hmac.compare_digest` instead of `==`

Regular string equality in Python short-circuits on the first
mismatched byte. An attacker who can measure response latency
precisely could guess the key one byte at a time ("a...": 120 µs,
"b...": 125 µs, so "b" is correct). `hmac.compare_digest` always
compares all bytes in constant time.

At this threat model (small-team project, LAN deployment) this is
overkill — but it's a 5-line decision that matches industry
standard and gets noticed by reviewers.

### What's behind `API_KEY`

It's a shared secret, loaded from `configs/env/.env` into the
backend container. All clients — the Next.js frontend, CLI users
calling the API, the robotic platform — use the same key.

### Why `/health` is exempt

Container orchestrators (Docker Compose, Portainer, Kubernetes)
need to probe `/health` to decide whether to route traffic to a
replica. Giving them the API key would require distributing the
secret to the orchestration layer itself, which is the wrong place
for it. The alternative — a separate auth path just for
orchestrators — is more complex than the problem warrants.

`/health` returns only the model version and serving mode, neither
of which is sensitive.

## Current: per-user keys with bcrypt

The shared-key model doesn't scale to multiple researchers
with different permissions. The specification §10.3 schema defines a
`users` table:

```
users(id, email, api_key_hash, role, created_at, revoked_at)
```

Where `api_key_hash` is a bcrypt hash of the key the user receives.
The migration replaces the hardcoded comparison in
`api.auth.api_key.require_api_key` with:

```python
async def require_api_key(
    x_api_key: str = Header(...),
    db: AsyncSession = Depends(get_db),
) -> User:
    user = await db.execute(
        select(User).where(User.revoked_at.is_(None))
    )
    for u in user.scalars():
        if bcrypt.checkpw(x_api_key.encode(), u.api_key_hash):
            return u  # Returns the full user object for role checks
    raise HTTPException(401, "UNAUTHORIZED")
```

Every route that currently takes the dependency already receives
the return value — today it's just the matched key string; in
it becomes the `User` instance, so role-based checks can
happen at handler level.

**This is a non-breaking migration.** No route handler has to be
rewritten, no client has to change its request format. The only
change on the client side is that researchers get their own keys
instead of sharing one.

## Threat model (explicit)

Out of scope for Block D:
- Rate limiting — relies on ingress (nginx / Azure Front Door) later.
- HTTPS termination — handled by Traefik/Portainer on-prem, by
  Azure Container Apps ingress in cloud.
- Audit logging of auth failures — deferred to App Insights.
- CSRF — we don't use cookies for auth; header-based auth is
  naturally CSRF-immune.

In scope:
- Preventing unauthenticated inference (costs GPU cycles).
- Preventing secret leakage to orchestration logs.
- Future: per-user attribution in the `predictions` table so a
  bad prediction can be traced to the caller.

## What not to do

:::{warning}
Never bake the API key into client code, frontend bundles, or
Dockerfiles. The `.env` file is the only approved home for it.
For CI, use GitHub Secrets. For Portainer, use Portainer's secret
management (not ENV in compose — those show up in `docker inspect`).
:::

:::{warning}
Don't add new secrets to `configs/env/.env.example`. That file is
committed to git and contains *placeholders only*. Real secrets
live in the non-committed `configs/env/.env`.
:::

## Related contracts

- Multi-user authentication requirement: R9.
- Users table schema: specification §10.3.
- Multi-user authentication and secure access
  the shared key meets the letter; per-user keys close the spirit.
