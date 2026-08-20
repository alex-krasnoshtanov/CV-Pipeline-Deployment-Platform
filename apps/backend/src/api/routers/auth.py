"""Auth router: register, login, logout, OAuth, and identity probe.

Endpoint layout:
- POST /auth/register  -> create account, set session cookie
- POST /auth/login     -> validate creds, set session cookie
- GET  /auth/github/login    -> redirect browser to GitHub OAuth
- GET  /auth/github/callback -> exchange code, mint JWT, redirect home
- POST /auth/logout    -> delete session, clear cookie
- GET  /auth/me        -> identity of the current caller

/auth/me accepts the session cookie, the OAuth JWT, or the legacy
X-API-Key header via the shared ``require_user`` dep. That is what
lets us keep the Streamlit frontend on the old path while we build
the new web frontend on cookies and OAuth.

Why a cookie and not a JSON token in the body? httpOnly means
JavaScript cannot read it, which blocks the XSS-then-exfil-token
attack pattern that plagues JWT-in-localStorage setups. SameSite=
Lax then mitigates the most common CSRF vector (cross-site POSTs).
The cost is that a SPA needs ``credentials: 'include'`` on fetch
calls and the backend needs allow_credentials=True in CORS - see
main.py.
"""

from __future__ import annotations

import logging
import os

from fastapi import (
    APIRouter,
    Cookie,
    Depends,
    Header,
    HTTPException,
    Request,
    Response,
    status,
)
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth.dependencies import SESSION_COOKIE_NAME, require_user
from api.auth.jwt import mint_backend_jwt
from api.auth.oauth import oauth
from api.auth.password import hash_password, verify_password
from api.auth.session import create_session, delete_session
from api.db import get_db
from api.db.models import User
from api.schemas.auth import LoginRequest, RegisterRequest, UserResponse
from api.services.user_oauth_service import upsert_oauth_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


# Cookie TTL in seconds. Must match (or be shorter than) the
# session row TTL or the browser will keep sending an expired
# cookie that the server then rejects.
_COOKIE_MAX_AGE = 7 * 24 * 60 * 60  # 7 days

# Well-formed but unsatisfiable bcrypt hash. Used as a dummy
# target in /login so verify_password runs even when the email
# does not exist - prevents a timing oracle on email existence.
_DUMMY_HASH = "$2b$12$" + "A" * 53


def _set_session_cookie(response: Response, session_id: str) -> None:
    """Attach the session cookie to *response*.

    Cookie attributes:
    - httponly=True: JavaScript cannot read it. Blocks XSS-exfil.
    - samesite='lax': allows the cookie on top-level navigation
      but not on cross-site POSTs. Sane CSRF default.
    - secure: True in production (https), False on localhost so
      dev does not break over http://. Driven by COOKIE_SECURE
      env var; default off so the local docker-compose stack
      works out of the box.
    - path='/': cookie applies to the whole API.
    """
    secure = os.getenv("COOKIE_SECURE", "false").lower() == "true"
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        max_age=_COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=secure,
        path="/",
    )


def _clear_session_cookie(response: Response) -> None:
    """Tell the browser to drop the session cookie."""
    response.delete_cookie(key=SESSION_COOKIE_NAME, path="/")


def _user_to_response(user: User) -> UserResponse:
    """Project the ORM row onto the public schema, no secrets."""
    return UserResponse(
        id=str(user.id),
        name=user.name,
        role=user.role,
        email=user.email,
    )


def _auth_method_for(
    user: User,
    authorization: str | None,
    session_id: str | None,
    x_api_key: str | None,
) -> str:
    """Determine which credential the request used.

    Called only after the dependency hierarchy has authenticated the
    user, so at least one of the three credentials must be present.
    This function is unreachable for the fallback case: require_user
    raises 401 before this runs if all credentials are absent.
    """
    _ = user
    if authorization and authorization.lower().startswith("bearer "):
        return "oauth"
    if session_id is not None:
        return "password"
    if x_api_key is not None:
        return "api_key"
    # Unreachable: require_user gate ensures at least one credential exists.
    return "unknown"


@router.get("/github/login", summary="Initiate GitHub OAuth login.")
async def github_login(request: Request) -> RedirectResponse:
    """Redirect the browser to GitHub's authorisation page.

    Authlib stores a state nonce in the session cookie (managed by
    SessionMiddleware, registered in main.py P3-09). The callback
    verifies the same nonce, which is what stops CSRF on the redirect.
    """
    redirect_uri = os.getenv("OAUTH_REDIRECT_URI")
    if not redirect_uri:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "MISCONFIGURED",
                "message": "OAUTH_REDIRECT_URI is not set.",
            },
        )
    return await oauth.github.authorize_redirect(request, redirect_uri)


@router.get("/github/callback", summary="Handle GitHub OAuth redirect.")
async def github_callback(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> RedirectResponse:
    """Exchange the OAuth code for a token, upsert the user, mint a JWT.

    Steps:
      1. Exchange ?code= for an access token (Authlib does this).
      2. Fetch the user profile from /user. If email is private, fall
         back to /user/emails for the primary verified email.
      3. Upsert a User row keyed on (provider, subject).
      4. Mint a backend JWT.
      5. Redirect the browser to the frontend with ?token=<jwt>.

    Raises:
        HTTPException: 400 if GitHub rejects the code exchange (most
            commonly because the state nonce expired).
    """
    try:
        token = await oauth.github.authorize_access_token(request)
    except Exception as exc:
        logger.warning("OAuth code exchange failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": "OAUTH_FAILED",
                "message": "GitHub sign-in failed. Try again.",
            },
        ) from exc

    resp = await oauth.github.get("user", token=token)
    profile = resp.json()

    email: str | None = profile.get("email")
    if not email:
        emails_resp = await oauth.github.get("user/emails", token=token)
        for entry in emails_resp.json():
            if entry.get("primary") and entry.get("verified"):
                email = entry.get("email")
                break

    user = await upsert_oauth_user(
        db,
        provider="github",
        subject=str(profile["id"]),
        email=email,
        name=profile.get("name") or profile.get("login") or "GitHub User",
    )

    jwt_token = mint_backend_jwt(user)
    logger.info("User id=%s signed in via GitHub OAuth.", user.id)

    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:8501").rstrip("/")
    return RedirectResponse(
        url=f"{frontend_url}/?token={jwt_token}",
        status_code=status.HTTP_302_FOUND,
    )


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new email/password account and log in.",
)
async def register(
    body: RegisterRequest,
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    """Create a new researcher account and start a session.

    New accounts always start with role='researcher'. Admin
    promotion is a separate, deliberate action via POST /users
    (admin only).

    Returns 409 if the email is already taken. For a school
    project, the convenience of a clear error outweighs the
    theoretical email-enumeration risk; swap for a generic
    "could not create account" if you ever want to harden this.
    """
    existing = await db.execute(select(User).where(User.email == body.email))
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error_code": "EMAIL_TAKEN",
                "message": "An account with this email already exists.",
            },
        )

    user = User(
        name=body.name,
        email=body.email,
        password_hash=hash_password(body.password),
        role="researcher",
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    session_row = await create_session(db, user.id)
    _set_session_cookie(response, str(session_row.id))

    logger.info("Registered new user '%s' (id=%s).", user.email, user.id)
    return _user_to_response(user)


@router.post(
    "/login",
    response_model=UserResponse,
    summary="Validate credentials and start a session.",
)
async def login(
    body: LoginRequest,
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    """Validate email/password and set a session cookie.

    Returns 401 for both "no such email" and "wrong password" with
    the same message. Distinguishing them would let an attacker
    enumerate valid emails. We also always run verify_password
    (against a dummy hash if the email does not exist) so the
    response time does not leak whether the email was found.
    """
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()

    candidate_hash = (
        user.password_hash if user is not None and user.password_hash else _DUMMY_HASH
    )
    ok = verify_password(body.password, candidate_hash)

    if user is None or not ok or user.password_hash is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error_code": "INVALID_CREDENTIALS",
                "message": "Email or password is incorrect.",
            },
        )

    session_row = await create_session(db, user.id)
    _set_session_cookie(response, str(session_row.id))

    logger.info("Login succeeded for user '%s' (id=%s).", user.email, user.id)
    return _user_to_response(user)


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="End the current session.",
)
async def logout(
    response: Response,
    session_id: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_user),
) -> Response:
    """Delete the active session row and clear the cookie.

    require_user gates this so an unauthenticated call returns
    401 rather than silently succeeding. Logout is intentionally
    idempotent for any session_id (delete_session is a no-op for
    missing rows), so a double-click on the logout button does
    not 404.

    For JWT callers (Authorization: Bearer ...), the auth dep
    accepts the token and this route becomes a no-op on the DB
    side: there is no session row to delete, and the cookie clear
    only affects browser-based callers.

    For X-API-Key callers (no cookie present), only the cookie
    clear runs - they have no session row to delete. That is
    fine: the legacy flow does not need stateful logout.
    """
    _ = current_user  # auth gate only
    if session_id is not None:
        await delete_session(db, session_id)
    _clear_session_cookie(response)
    response.status_code = status.HTTP_204_NO_CONTENT
    return response


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Return the authenticated user's identity.",
)
async def get_me(
    authorization: str | None = Header(default=None),
    session_id: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    current_user: User = Depends(require_user),
) -> UserResponse:
    """Return name, role, id, email, and the credential used.

    Accepts either the session cookie, the OAuth JWT, or the
    X-API-Key header via the shared ``require_user`` dep. The
    frontend uses this on app load to detect "am I logged in"
    and to populate the header ("Logged in as ...").
    """
    return UserResponse(
        id=str(current_user.id),
        name=current_user.name,
        role=current_user.role,
        email=current_user.email,
        auth_method=_auth_method_for(
            current_user,
            authorization,
            session_id,
            x_api_key,
        ),
    )
