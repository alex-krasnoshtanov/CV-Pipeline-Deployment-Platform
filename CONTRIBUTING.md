# Contributing

This document covers the conventions this repository is built on: commit and
PR format, branching, and what "done" means before review.

## Getting set up

```bash
uv sync                 # installs the workspace, including dev tooling
uv run pre-commit install
cd apps/frontend && npm ci
```

The workspace resolves CPU PyTorch wheels. If you need a CUDA build for local
training, override the index at sync time rather than editing `pyproject.toml` —
see the comment in that file for why the default is CPU.

## Definition of done

A change is done when all of the following hold. Apply this bar before
requesting review.

- CI is green: lint, format, tests, and the image build.
- Tests cover the changed behaviour. The suite gates at 85% line coverage.
- Documentation is updated if behaviour or configuration changed.
- The PR description explains *why*, not just what.
- One concern per PR. Unrelated changes go in their own branch.
- No secrets, credentials, personal identifiers, or internal hostnames.

That last item is not boilerplate. This repository was extracted from a
university group project, and leaked identifiers — a student number in a
download URL, an internal hostname in a docstring — were the single most
common thing that had to be cleaned up.

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): short imperative description
```

Allowed types: `feat`, `fix`, `docs`, `chore`, `test`, `style`, `refactor`,
`ci`, `perf`, `build`.

`commitizen` is configured in `pyproject.toml`; `uv run cz commit` gives a
guided prompt if you want one.

## PR titles

PR titles follow the same pattern and are enforced by the `pr-title` workflow,
which fails the check rather than the build.

```
feat(backend): add drift detection endpoint
fix(cv-pipeline): handle RGBA input without silent conversion
docs: document the model version registry
```

## Branch naming

```
type/short-slug
```

For example `feat/drift-endpoint`, `fix/rgba-input`, `docs/weights-registry`.
Include an issue number when one exists: `fix/473-rgba-input`.

## Workflow

Trunk-based development against `main`:

1. `git checkout main && git pull`
2. `git checkout -b type/short-slug`
3. Make the change. Run the local gate:
   ```bash
   uv run ruff format .
   uv run ruff check .
   uv run pytest -m unit
   ```
4. `git push -u origin <branch>` and open a PR.
5. Address feedback in additional commits on the same branch.
6. Merge once CI is green, keeping `main` linear.

Squash when the branch accumulated work-in-progress commits — the kind
that say "fix typo" or "address review". Rebase when each commit is
already a self-contained change with a message worth keeping. The test
is whether someone reading `git log` a year from now benefits from the
separation.

`main` is protected. Branches are deleted after merge.

## Testing

```bash
uv run pytest -m unit                     # fast, no external dependencies
uv run pytest -m integration              # requires a running stack
cd apps/frontend && npm run test:ci       # vitest
```

Unit tests must not require a database, a network call, or model weights. If a
test needs one of those, mark it `integration`.

## Documentation

Sphinx sources live in `docs/source/` and follow the
[Diátaxis](https://diataxis.fr/) split — tutorials, how-to, reference,
explanation. Put a new page in the quadrant that matches its purpose rather
than the component it describes.

The API reference is generated; do not hand-edit
`docs/source/reference/backend-api.md`. Regenerate it with:

```bash
uv run scripts/generate_openapi_docs.py
```

The [CV pipeline specification](docs/source/reference/specification.md) is the contract between
the package and its consumers. Its section numbers are referenced from code
comments and other documents, so sections are amended in place and never
renumbered.
