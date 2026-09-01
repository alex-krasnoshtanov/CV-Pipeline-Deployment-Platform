# Backend API Reference

This is an OpenAPI-derived reference for the backend API.

:::{note}
This page is generated from the live FastAPI app by
`scripts/generate_openapi_docs.py`, which the `docs.yml` workflow runs
before Sphinx on every build, so the published site always reflects the
running app. Do not edit it by hand — regeneration discards the edit.

Descriptions are the first paragraph of each route's docstring. The full
docstrings, including `Args`, `Returns` and `Raises`, are in the
autoapi reference.
:::

## GET `/metrics`

**Summary:** Metrics  
**Description:** Endpoint that serves Prometheus metrics.  

### Responses
- **200**: Successful Response

## GET `/health`

**Summary:** Healthcheck  
**Description:** Return backend readiness state.  

### Responses
- **200**: Successful Response

## POST `/infer`

**Summary:** Run segmentation + landmark detection on a single image.  
**Description:** Accept a plant image and return the inference result.  

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

### Parameters
- **`authorization`** *(Optional)* (header): 
- **`X-API-Key`** *(Optional)* (header): 
- **`session_id`** *(Optional)* (cookie): 

### Responses
- **200**: Successful Response
- **422**: Validation Error

## GET `/feedback/review-queue`

**Summary:** List predictions awaiting reviewer correction (admin).  
**Description:** Return predictions flagged bad/uncertain and not yet resolved.  

### Parameters
- **`limit`** *(Optional)* (query): 
- **`offset`** *(Optional)* (query): 
- **`authorization`** *(Optional)* (header): 
- **`X-API-Key`** *(Optional)* (header): 
- **`session_id`** *(Optional)* (cookie): 

### Responses
- **200**: Successful Response
- **422**: Validation Error

## POST `/feedback/relabel`

**Summary:** Submit a corrected mask or resolved verdict (admin).  
**Description:** Record an admin correction as a new feedback row.  

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

### Responses
- **200**: Successful Response

## GET `/auth/github/callback`

**Summary:** Handle GitHub OAuth redirect.  
**Description:** Exchange the OAuth code for a token, upsert the user, mint a JWT.  

### Responses
- **200**: Successful Response

## POST `/auth/register`

**Summary:** Create a new email/password account and log in.  
**Description:** Create a new researcher account and start a session.  

### Responses
- **201**: Successful Response
- **422**: Validation Error

## POST `/auth/login`

**Summary:** Validate credentials and start a session.  
**Description:** Validate email/password and set a session cookie.  

### Responses
- **200**: Successful Response
- **422**: Validation Error

## POST `/auth/logout`

**Summary:** End the current session.  
**Description:** Delete the active session row and clear the cookie.  

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

### Parameters
- **`authorization`** *(Optional)* (header): 
- **`X-API-Key`** *(Optional)* (header): 
- **`session_id`** *(Optional)* (cookie): 

### Responses
- **200**: Successful Response
- **422**: Validation Error
