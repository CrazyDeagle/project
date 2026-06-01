<!--
Thanks for opening a PR. Please fill in the sections below — they help
reviewers move quickly. Delete anything that does not apply.
-->

## Summary

<!-- What does this PR change, in one or two sentences? -->

## Motivation

<!-- Why is this change needed? Link to an issue or ADR if one exists. -->

Closes #

## Changes

- 
- 
- 

## Validation

<!-- How did you verify the change? Paste the relevant command and a one-line
result. CI runs the full lint + CPU test suite automatically. -->

```
pytest -q
ruff check .
```

## Checklist

- [ ] Tests added or updated.
- [ ] `ruff check` and `ruff format --check .` pass locally.
- [ ] `CHANGELOG.md` updated under *Unreleased*.
- [ ] No unrelated reformatting churn.
- [ ] If this changes CUDA kernels or checkpoint format, the change is called
      out explicitly in the PR description.
