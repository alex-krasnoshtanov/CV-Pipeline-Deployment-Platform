# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

This repository was reassembled from the original four-person group repository
for public release. History was rewritten rather than carried over, so it does
not preserve the group project's commit log — see
[Attribution](README.md#attribution) for authorship.

### Added

- Dependabot configuration covering GitHub Actions, three uv lockfiles, the
  frontend npm project and the container base images, grouped monthly.
- Issue and pull-request templates.
- This changelog.

### Changed

- `torch` and `torchvision` now resolve from the CPU wheel index. The serving
  image previously pulled thirteen `nvidia-*` wheels plus triton (~2.5 GB) that
  no code path in that image could use, which exhausted the CI runner's disk.
  Resolution moved torch 2.6.0 → 2.13.0 and torchvision 0.21.0 → 0.28.0.
- Model weights now resolve from GitHub Release assets instead of a personal
  cloud-storage share link.
- `README.md` rewritten: architecture and lifecycle diagrams, per-component
  authorship breakdown, and an explicit statement that neither the dataset nor
  a trained checkpoint is redistributed.

### Removed

- Personal and institutional identifiers: a student number embedded in a
  weights download URL, an internal campus hostname in a service docstring, a
  collaborator's name and student number in a documentation caption, and
  references to internal file-sharing infrastructure.
- Coursework deliverables (learning-outcome mappings, sprint evidence,
  ceremony records, product-owner sign-off tracking) that documented the
  assessment rather than the system.
- The group repository's deploy pipeline, which targeted a campus IP and a
  provisioned cloud environment that no longer exist.

### Known gaps

- No public weights release exists yet, so a fresh clone cannot run inference
  without supplying `MODEL_PATH` or `MODEL_ENDPOINT_URL`.
- The Sphinx site builds in CI but is not published; GitHub Pages requires the
  repository to be public.
