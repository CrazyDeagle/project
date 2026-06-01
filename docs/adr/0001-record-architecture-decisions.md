# ADR-0001: Record Architecture Decisions

- **Status:** Accepted
- **Date:** 2025-01-01
- **Deciders:** maintainers

## Context

The SilexCode codebase touches a deliberately narrow but unusual set of
ideas: packed ternary weights, FWHT-based deterministic initialisation,
block K-FAC updates restricted to plastic adapters, and a curriculum that
escalates from byte-level structure up to formal supervised targets.

Decisions in any one of these areas are easy to forget the *reason* for once
the code lands. We have already accumulated several non-obvious choices
(plastic-only bootstrap, output-adapter SGD path, deterministic vs packed
backbone runtime modes) that would be expensive to re-derive from `git log`.

## Decision

We will record architectural decisions as Markdown files in `docs/adr/`,
numbered sequentially (`NNNN-title.md`). Each ADR follows this template:

```markdown
# ADR-NNNN: <Title>

- **Status:** Proposed | Accepted | Superseded by ADR-MMMM | Deprecated
- **Date:** YYYY-MM-DD
- **Deciders:** <names or roles>

## Context
<What is the situation that forces a decision?>

## Decision
<What did we decide, in active voice?>

## Consequences
<Positive, negative, and follow-on consequences.>
```

ADRs are append-only. If a decision is reversed, write a new ADR with status
*Accepted* that explicitly supersedes the old one, and update the old ADR's
status to *Superseded by ADR-MMMM*. Do not edit the body of a superseded ADR.

## Consequences

- New contributors have a single place to learn *why* the code looks the way
  it does.
- PRs that change a previously-recorded decision must update or supersede the
  corresponding ADR, which surfaces design drift early.
- The discipline cost is one extra file per substantive design choice. We
  accept that cost.
