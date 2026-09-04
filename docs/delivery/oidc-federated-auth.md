# Cloud authentication with no stored credentials

The pipeline deploys to Azure without a client secret anywhere in the
repository. Authentication goes through OpenID Connect federated credentials,
so what the workflow presents to Azure is a short-lived token minted for that
one run.

## The exchange

Every run on `main` does this before touching Azure:

1. The job asks GitHub's OIDC provider for a signed JWT describing itself:
   which repository, which ref, which run.
2. The [`azure/login`](../../.github/workflows/cd.yml) action presents that JWT to
   Microsoft Entra ID.
3. Entra checks the JWT's claims against a federated credential registered
   against the deploy identity.
4. If the repository and ref match, Entra issues an Azure access token, valid
   about an hour.
5. The `az` commands in the rest of the job use that token.

No password exists at any point in the chain, and the token cannot be replayed
outside the run that requested it.

## Configuration

Five repository *variables*, not secrets:

| Variable | Holds |
|---|---|
| `AZURE_CLIENT_ID` | client id of the deploy managed identity |
| `AZURE_TENANT_ID` | Entra tenant |
| `AZURE_SUBSCRIPTION_ID` | target subscription |
| `AZURE_RESOURCE_GROUP` | resource group the apps live in |
| `AZURE_BACKEND_APP` / `AZURE_FRONTEND_APP` | Container App names |

They are `vars.*` deliberately. They are identifiers, and knowing them buys an
attacker nothing: using them requires minting a valid OIDC token from this
repository on the trusted ref, which requires the ability to run a workflow in
it. Treating identifiers as secrets makes debugging harder and protects
nothing.

The federated credential itself is registered once, on the Azure side:

```
subject  = repo:<org>/<repo>:ref:refs/heads/main
audience = api://AzureADTokenExchange
```

## The constraint that costs an afternoon if you miss it

That subject is pinned to `main`, and **three separate things have to agree
with it**:

- the workflow trigger (`on: push: branches: [main]`)
- the deployment branch rule on the `production` environment
- the credential subject above

Binding a job to a GitHub Environment rewrites the OIDC subject to
`repo:<org>/<repo>:environment:<name>`. A credential registered against the
`ref:refs/heads/main` form then stops matching, and Entra rejects the exchange
with no useful diagnostic. This is why `azure-auth-test` and `deploy-azure` in
`cd.yml` carry an explicit comment saying not to add an `environment:` binding,
and why the approval gate is enforced by branch protection instead. If you
re-register the credential against the environment form, the binding can go
back.

The other requirement is `permissions: id-token: write` on the job. Without it
the JWT fetch never happens and `azure/login` fails with `AADSTS70016`, which
reads like a tenant problem rather than a permissions one.

## What the auth job actually does

`azure-auth-test` verifies the exchange end to end without deploying anything:
it prints the subscription context, confirms it can read the resource group,
and lists Container Apps in it. It runs after the images are pushed and before
any deploy job, so a broken credential fails fast and cheaply rather than
half-way through a rollout.
