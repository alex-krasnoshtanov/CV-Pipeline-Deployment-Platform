# Backend API Reference

This is an OpenAPI-derived reference for the backend API.

:::{note}
This page is generated from the live FastAPI app by
`scripts/generate_openapi_docs.py`, which the `docs.yml` workflow runs before
Sphinx on every build — so the published site always reflects the running app.
The committed copy is a snapshot and can lag the code; it was last manually
synced with the routers on 2026-06-10 (adding `POST /monitoring/check`). If it
disagrees with the code, the code wins.
:::

## GET `/metrics`

**Summary:** Metrics  
**Description:** Endpoint that serves Prometheus metrics.  

### Responses
- **200**: Successful Response

## GET `/health`

**Summary:** Healthcheck  
**Description:** Return backend readiness state.

Args:
    request: FastAPI request, used to reach the shared service
        state published by the lifespan in ``api.main``.
    response: FastAPI response, used to override the status code
        to 503 when the model has not yet loaded.

Returns:
    A ``HealthResponse`` with ``status="ok"`` when ready,
    ``status="loading"`` with HTTP 503 otherwise.  

### Responses
- **200**: Successful Response

## POST `/infer`

**Summary:** Run segmentation + landmark detection on a single image.  
**Description:** Accept a plant image and return the inference result.

Args:
    request: FastAPI request (used to reach the shared model on
        ``app.state`` and to expose ``request.state`` for slowapi).
    image: Uploaded image via multipart/form-data.
    plate_id: Optional Petri dish identifier (pass-through).
    experiment_id: Optional experiment identifier (pass-through).
    timestamp: Optional ISO 8601 capture timestamp (pass-through).
    current_user: Authenticated user or None for anonymous calls.
    db: Async database session injected by ``get_db``.

Returns:
    A ``dict`` that FastAPI serialises through ``InferenceResponse``.

Raises:
    HTTPException: 413/422 on cv_pipeline validation failure,
        503 if the model is not loaded, 429 on rate-limit
        breach (raised by slowapi, handled in main.py).  

### Parameters
- **`authorization`** *(Optional)* (header): 
- **`X-API-Key`** *(Optional)* (header): 
- **`session_id`** *(Optional)* (cookie): 

### Responses
- **200**: Successful Response
- **422**: Validation Error

## POST `/explain`

**Summary:** Compute a Seg-Grad-CAM explanation heatmap for a single image.  
**Description:** Accept a plant image and return a Grad-CAM heatmap.

Args:
    request: FastAPI request (used to reach ``app`` for model resolution
        and to expose ``request.state`` for slowapi).
    image: Uploaded image via multipart/form-data.
    plate_id: Optional Petri dish identifier (pass-through).
    experiment_id: Optional experiment identifier (pass-through).
    timestamp: Optional ISO 8601 capture timestamp (pass-through).
    current_user: Authenticated user or None for anonymous calls.

Returns:
    A ``dict`` serialised through ``ExplainResponse``.

Raises:
    HTTPException: 413/422 on cv_pipeline validation failure, 503 if no
        model is available, 504 if the explanation times out, 500 on any
        other explanation failure, 429 on rate-limit breach (slowapi).  

### Parameters
- **`authorization`** *(Optional)* (header): 
- **`X-API-Key`** *(Optional)* (header): 
- **`session_id`** *(Optional)* (cookie): 

### Responses
- **200**: Successful Response
- **422**: Validation Error

## POST `/feedback`

**Summary:** Flag a prediction as good, bad, or uncertain.  
**Description:** Record feedback on a prediction.

Args:
    body: Feedback payload with prediction_id, flag, and optional
        notes or corrected mask.
    current_user: Authenticated user (cookie OR X-API-Key).
    db: Async database session.

Returns:
    A dict serialised through ``FeedbackResponse``.

Raises:
    HTTPException: 401 if not authenticated, 404 if prediction
        not found, 422 if the prediction_id format is invalid.  

### Parameters
- **`authorization`** *(Optional)* (header): 
- **`X-API-Key`** *(Optional)* (header): 
- **`session_id`** *(Optional)* (cookie): 

### Responses
- **200**: Successful Response
- **422**: Validation Error

## GET `/stats`

**Summary:** Monitoring dashboard statistics.  
**Description:** Return aggregated business and operational statistics.

Aggregates prediction volume, confidence distribution, feedback rates,
and per-model-version metrics from the database.  All heavy computation
is pushed to SQL; the handler is a thin delegation layer.

Args:
    request: FastAPI request, used to read app state (serving version)
        and detect whether the /metrics route is registered.
    current_user: Authenticated user (cookie or X-API-Key).
    db: Async database session.

Returns:
    :class:`~api.schemas.stats.StatsResponse` with ``business`` and
    ``operational`` sub-objects.

Raises:
    HTTPException: 401 if not authenticated.  

### Parameters
- **`authorization`** *(Optional)* (header): 
- **`X-API-Key`** *(Optional)* (header): 
- **`session_id`** *(Optional)* (cookie): 

### Responses
- **200**: Successful Response
- **422**: Validation Error

## POST `/monitoring/check`

**Summary:** Run rolling-confidence drift detection.  
**Description:** Run rolling-confidence drift detection.  

### Parameters
- **`authorization`** *(Optional)* (header): 
- **`X-API-Key`** *(Optional)* (header): 
- **`session_id`** *(Optional)* (cookie): 

### Responses
- **200**: Successful Response
- **422**: Validation Error

## POST `/users`

**Summary:** Create a new API-key user (admin only).  
**Description:** Create a new user and return their generated API key.

Admin-only. The plaintext API key is included in the response
and never stored - the admin must copy it immediately and
deliver it to the new user out-of-band.

Args:
    body: User creation payload with name and role.
    current_admin: Authenticated admin (enforced by dep).
    db: Async database session.

Returns:
    A dict serialised through ``CreateUserResponse``.

Raises:
    HTTPException: 401 if not authenticated, 403 if not admin.  

### Parameters
- **`authorization`** *(Optional)* (header): 
- **`X-API-Key`** *(Optional)* (header): 
- **`session_id`** *(Optional)* (cookie): 

### Responses
- **200**: Successful Response
- **422**: Validation Error

## GET `/auth/github/login`

**Summary:** Initiate GitHub OAuth login.  
**Description:** Redirect the browser to GitHub's authorisation page.

Authlib stores a state nonce in the session cookie (managed by
SessionMiddleware, registered in main.py P3-09). The callback
verifies the same nonce, which is what stops CSRF on the redirect.  

### Responses
- **200**: Successful Response

## GET `/auth/github/callback`

**Summary:** Handle GitHub OAuth redirect.  
**Description:** Exchange the OAuth code for a token, upsert the user, mint a JWT.

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

### Responses
- **200**: Successful Response

## POST `/auth/register`

**Summary:** Create a new email/password account and log in.  
**Description:** Create a new researcher account and start a session.

New accounts always start with role='researcher'. Admin
promotion is a separate, deliberate action via POST /users
(admin only).

Returns 409 if the email is already taken. For a school
project, the convenience of a clear error outweighs the
theoretical email-enumeration risk; swap for a generic
"could not create account" if you ever want to harden this.  

### Responses
- **201**: Successful Response
- **422**: Validation Error

## POST `/auth/login`

**Summary:** Validate credentials and start a session.  
**Description:** Validate email/password and set a session cookie.

Returns 401 for both "no such email" and "wrong password" with
the same message. Distinguishing them would let an attacker
enumerate valid emails. We also always run verify_password
(against a dummy hash if the email does not exist) so the
response time does not leak whether the email was found.  

### Responses
- **200**: Successful Response
- **422**: Validation Error

## POST `/auth/logout`

**Summary:** End the current session.  
**Description:** Delete the active session row and clear the cookie.

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

### Parameters
- **`authorization`** *(Optional)* (header): 
- **`X-API-Key`** *(Optional)* (header): 
- **`session_id`** *(Optional)* (cookie): 

### Responses
- **204**: Successful Response
- **422**: Validation Error

## GET `/auth/me`

**Summary:** Return the authenticated user's identity.  
**Description:** Return name, role, id, email, and the credential used.

Accepts either the session cookie, the OAuth JWT, or the
X-API-Key header via the shared ``require_user`` dep. The
frontend uses this on app load to detect "am I logged in"
and to populate the header ("Logged in as ...").  

### Parameters
- **`authorization`** *(Optional)* (header): 
- **`X-API-Key`** *(Optional)* (header): 
- **`session_id`** *(Optional)* (cookie): 

### Responses
- **200**: Successful Response
- **422**: Validation Error
