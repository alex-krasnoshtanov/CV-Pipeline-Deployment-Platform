# Branching and Commit Strategy

## How We Work With Git

We use trunk-based development with short-lived feature branches. There is one long-lived branch — `main` — which is protected. All changes go through pull requests. Every merged PR results in a single squashed commit on `main`.

## Branch Naming

Format: `type/ticket-id-short-description`

- `feature/` — new functionality (e.g. `feature/42-plant-segmentation`)
- `bugfix/` — fixing broken behaviour (e.g. `bugfix/57-fix-health-endpoint`)
- `hotfix/` — urgent fix that cannot wait for a normal review cycle (e.g. `hotfix/61-api-crash-on-startup`)
- `chore/` — docs, config, CI, refactoring, dependencies (e.g. `chore/63-update-architecture-diagrams`)

The ticket ID is the Azure DevOps work item number. This links branches and commits back to the board.

## Commit Messages

We follow conventional commits with ticket references:

```
type(#ticket-id): short description
```

Types:

- `feat` — new feature
- `fix` — bug fix
- `docs` — documentation only
- `chore` — config, CI, dependencies, refactoring
- `test` — adding or updating tests
- `style` — formatting (no logic changes)

Examples:

```
feat(#42): add segmentation inference endpoint
fix(#57): handle missing image path in pipeline
docs(#63): add architecture diagrams
test(#42): add unit tests for segmentation output
```

## Branch Protection on `main`

- 1 approval required before merging
- CI status checks (ruff lint + format) must pass
- Merge strategy: squash and merge
- Branches are deleted after merge

## Pull Request Process

1. Pick a work item from the Azure DevOps board.
2. Branch off `main` using the naming convention above.
3. Commit your work using the conventional commit format.
4. Open a PR to `main`. In the description, include:
   - Link to the Azure DevOps work item
   - What was changed and why
   - How it was tested
5. CI runs automatically. The merge button is blocked until it passes.
6. A team member reviews the PR.
7. After approval, anyone can squash and merge.
8. The branch gets deleted.

## What Reviewers Check

- Logic is correct and edge cases are handled
- New functionality has unit tests
- Existing tests still pass
- Functions and classes have docstrings
- No hardcoded values — use config or environment variables
- Code passes ruff formatting and linting
- No data files, cache, or other junk committed
