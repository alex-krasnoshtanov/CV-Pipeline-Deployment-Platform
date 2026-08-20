# Contributing

This document captures the team's working agreements for the
NPEC CV Pipeline project. It is the single source of truth for
PR conventions, review expectations, and branching rules.

## PR review SLA

The team has agreed on the following review service-level agreement
to keep delivery flowing and avoid the "stale PR" pattern observed
in Sprint 2 retrospective (action item #2314):

- **Target turnaround:** 24 business hours (Mon-Fri, 09:00-18:00 CET)
  between PR opening and first reviewer response.
- **Reviewer assignment:** the PR author requests a review at PR
  creation time. Default rotation by area:
  - Backend / API -> Filipp or Danil
  - Frontend -> Maksym
  - Infrastructure / Docker / CI / Deploy -> Alex
  - Tests / docs -> Marin or Danil
- **What counts as a response:** an approval, a "request changes"
  review, or an explicit comment that the reviewer has started the
  review and needs more time. Silence does not count.
- **Escalation:** if 24 business hours pass with no response, the
  author pings the team channel. No blame -- this is just a signal
  to pull in a backup reviewer.
- **Tracking:** breach patterns are noted in the next sprint retro.
  Misses are signals about workload and dependencies, not faults of
  individual reviewers.

## PR title format

PR titles must follow the conventional-commits pattern:

```
type(#issue): short imperative description
```

Allowed `type` values: `feat`, `fix`, `hotfix`, `docs`, `chore`,
`test`, `style`, `refactor`, `ci`.

Examples:

- `feat(#1232): add MLflow-logged training script`
- `fix(#473): correct em-dash mojibake in docs`
- `docs(#2314): document PR review SLA`

The `pr-title` GitHub Action enforces this pattern at PR creation.

## Branch naming

Use the pattern `type/issue-short-slug`:

- `feat/1232-mlflow-training`
- `fix/473-em-dash-mojibake`
- `docs/2314-pr-review-sla`

## Commit messages

Conventional Commits, same `type(#issue): description` shape as PR
titles. `commitizen` is configured in `pyproject.toml`; run
`uv run cz commit` if you want guided commit creation.

## Workflow

We use trunk-based development:

1. `git checkout main && git pull`
2. `git checkout -b type/issue-short-slug`
3. Make changes, run `uv run ruff format` + `uv run ruff check` +
   `uv run pytest` locally.
4. `git push -u origin <branch>` and open a PR.
5. Request review from the area owner above.
6. Address feedback in additional commits on the same branch.
7. Squash-merge to `main` once approved + CI green.

Branches are deleted automatically after merge.

## Self-review checklist

Before requesting review, the author confirms:

- [ ] PR title matches `type(#issue): description`
- [ ] All CI checks green (lint, test, docs build)
- [ ] No unrelated changes mixed in (one concern per PR)
- [ ] Description explains the why, links the issue, lists any
  follow-up work
- [ ] Tests added or updated for new behaviour
- [ ] Docs updated if the change affects user-facing behaviour

## Issue references

Every PR closes or relates to an Azure DevOps work item. Use the
`#NNNN` form in the title and again in the body so the link is
unambiguous.

## Board hygiene

Sprint 2 retrospective identified that the Azure DevOps board was
not actively maintained during the sprint, making blockers and
progress invisible until standup. Action item #2312 sets the
following expectation for Sprint 3 and beyond:

- **Daily state update.** Each member moves their assigned tasks
  through `To Do` -> `Doing` -> `Done` on the same day the work
  changes phase. Aim for at least one board interaction per
  working day, even if it is just to confirm a task is still in
  progress.
- **Blocker comments.** If a task is stuck for more than half a
  day, add a comment on the work item describing the blocker and
  who or what is needed to unblock it. Do not wait for standup.
- **Standup board review.** The first five minutes of every daily
  standup is spent walking the Sprint board together. We start
  from the rightmost column (Done) and move left, so blockers in
  Doing are surfaced before new work is picked up from To Do.
- **End-of-sprint sweep.** On the last day of the sprint, the
  Scrum Master verifies every parent Issue whose child Tasks are
  all Done has itself been moved to Done. Open Issues with no
  open Tasks are a board-hygiene defect.
- **Tracking.** Adherence is reviewed in each sprint retro. The
  metric we care about is "did blockers surface within a day of
  appearing", not raw click counts.

This protocol is also referenced from the Sprint 2 retro action
item #2312 and applies to all five team members regardless of
role.
