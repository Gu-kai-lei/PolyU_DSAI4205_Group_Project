# Member 5 First Release Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline; fresh whole-release read-only reviewer at end.

**Goal:** Publish the completed M5 first delivery to main with an ordinary Jupyter entry point.
**Architecture:** Curated Member5 files from saved development source; generic notebook and local-only snapshots; committed-evidence numeric verification separated from historical full fit verification.
**Tech Stack:** Python3.12, PyTorch CPU/MPS/CUDA, nbformat/nbclient/ipykernel, existing sklearn and pandas.
**Spec:** docs/superpowers/specs/2026-11-13-m5-first-release-design.md

## Global Constraints
- No cloud imports/authentication, fixed provider directories, credentials or personal account IDs in published M5 code.
- Preserve model/data/split protocol; no new tuning or performance claims.
- No shared-file edits beyond own results; no heavy arrays/weights/env in Git.
- User authorizes main push, no force push or protection bypass.

## Review Focus
- Default notebook works in a fresh kernel from root and Member5 directory without caches or cloud SDK.
- Missing optional caches produce actionable errors before fitting; training is opt-in.
- Published historical scores independently match stored OOF; new code provenance is distinguished.
- Main race is caught by final fetch/non-force push; other members' paths preserved.
- No cloud SDK, private provider ID or home path remains in published executable code.

### Task1: Curate and make runtime portable
- [x] Copy only M5 core code/tests/MVP and completed local evidence.
- [x] Add failing provider-free/notebook/evidence tests, then remove provider code and create generic Jupyter entrypoint.
- [x] Keep raw historical evidence immutable where possible; strip personal path metadata via explicit publication manifest.

### Task2: Verify and publish
- [x] Recompute committed metrics, execute both entry notebooks using nbclient and run all scoped M5 tests including local training checks.
- [x] Fresh reviewer, one fixes pass, green checks and concrete release commit.
- [ ] Fetch main again; push non-force if unchanged, reconcile/revalidate if changed; verify remote SHA/file after push.
