# On-Prem Deployment (Portainer)

This folder stores deployment assets for the on-premise server managed through Portainer.

Expected contents:

- `stack/compose.yml`: Portainer stack definition for backend, frontend, and database.
- `stack/.env.example`: Environment variables consumed by the stack.
- `runbooks/`: Operational notes for update, rollback, and incident handling.

Deployment baseline:

- Pull images from GHCR that are produced by `.github/workflows/cd.yml`.
- Trigger updates through the Portainer webhook configured in GitHub Actions.
- Keep runtime secrets in Portainer environment variables or secret storage, never in git.
