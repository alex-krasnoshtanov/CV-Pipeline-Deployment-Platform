# Explanation

:::{note}
**Explanation** pages help you understand how the system works and
why it's built the way it is. They don't tell you how to do
anything — for that, see [How-to](../how-to/index).
:::

```{toctree}
:maxdepth: 1
:hidden:

architecture
frontend
deploy
security-model
error-codes
```

## What you'll find here

- {doc}`architecture` — the big picture: CLI, API, and Azure ML
  scoring all share one pipeline; why FastAPI, why U-Net, why
  patch-based inference.
- {doc}`error-codes` — how errors are reported, and why we
  separated `error_code` from `message`.
- {doc}`security-model` — X-API-Key today, bcrypt + users table
  tomorrow, and the threat model we're defending against.
