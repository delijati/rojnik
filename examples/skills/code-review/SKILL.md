---
name: code-review
description: Review Python changes for correctness, regressions, security, and missing tests.
---

# Code Review

Inspect the implementation and its surrounding call paths before drawing conclusions.

## Workflow

1. Read the changed code and relevant tests.
2. Prioritize correctness, security, data loss, and behavioral regressions.
3. Cite concrete files and lines for each finding.
4. Check failure paths and concurrency behavior.
5. Report missing tests when they leave important behavior unverified.

Lead with findings ordered by severity. Keep summaries secondary.
