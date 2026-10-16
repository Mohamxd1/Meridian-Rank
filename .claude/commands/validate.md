---
description: Validate Meridian Rank against mathematical, API, and storage contracts
---

Read AGENTS.md and docs/implementation.md. Use the project virtual environment.
Run Ruff, pytest, and the package build. If local Compose services are available,
set the documented RANK_TEST_DATABASE_URL and RANK_TEST_REDIS_URL to include the
real storage checks. Report passed, skipped, and failed checks separately.
Diagnose failures before changing expectations. Review tenant ownership, sparse
matrix orientation, and probability conservation. Never invent benchmark values.
