"""Authlib OAuth client configuration.

One ``oauth.register(...)`` block per provider. Adding Google (or any
other OIDC provider) later is literally another ``oauth.register(...)``
call — no frontend changes required.

The ``oauth`` instance is imported by the auth router and by tests that
need to mock it.
"""

from __future__ import annotations

import os

from authlib.integrations.starlette_client import OAuth

oauth = OAuth()

oauth.register(
    name="github",
    client_id=os.getenv("GITHUB_OAUTH_CLIENT_ID"),
    client_secret=os.getenv("GITHUB_OAUTH_CLIENT_SECRET"),
    access_token_url="https://github.com/login/oauth/access_token",
    authorize_url="https://github.com/login/oauth/authorize",
    api_base_url="https://api.github.com/",
    client_kwargs={"scope": "read:user user:email"},
    # GitHub's profile endpoint often withholds email if set to private.
    # The callback handler falls back to GET /user/emails if that happens.
)
