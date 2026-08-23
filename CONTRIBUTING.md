# Contributing to Silica

Thanks for looking. Silica is a solo, pre-1.0 project under active development, so the bar is simple: keep the diff small, keep the vault safe, and match the code that's already there.

## Dev setup

Silica uses [uv](https://github.com/astral-sh/uv). Everything runs through it:

```bash
git clone https://github.com/kiycoh/silica-harness.git
cd silica-harness
uv pip install -e '.[mcp]'   # drop [mcp] if you don't touch the MCP server
uv run silica doctor         # sanity-check the environment
```

## Before you open a PR

```bash
uv run pytest                # tests must pass
```

- Every change to non-trivial logic (a branch, a parser, a write/gate path) leaves at least one runnable test behind. Follow the existing `tests/test_*.py` style; no new frameworks or fixtures unless the change genuinely needs them.
- If you touch the write path, the invariant you must not break is the one the whole project exists for: **no mutation reaches the vault except through the Injector FSM**, and every write is verify-or-revert. A PR that adds a side channel to the vault will be rejected on principle, not on style.

## Conventions

- **English only.** All code, comments, identifiers, UI copy, and error messages are in English, even though a vault's *content* may be in any language. This keeps the codebase navigable.
- **Conventional commits.** The changelog is generated from history with [git-cliff](https://github.com/orhun/git-cliff), so commit messages matter: `feat(scope): …`, `fix(scope): …`, `docs: …`, `refactor: …`, `test: …`. One logical change per commit.
- **Smallest diff that works.** Follow existing patterns before introducing new ones. No speculative abstractions, no dependency added for what a few lines of stdlib can do.
- **Flag, don't work around.** If you find a leak or an anti-pattern, name it and fix the root cause. Don't route around it with a side channel.

## Reporting

- **Bugs / features:** open a GitHub issue with a minimal reproduction.
- **Security:** do **not** open a public issue. Follow [SECURITY.md](SECURITY.md).

## License

Silica is licensed under **AGPL-3.0-or-later**. By submitting a contribution you agree it is licensed
under the same terms, and that every source file keeps its `SPDX-License-Identifier: AGPL-3.0-or-later`
header. See [LICENSE](LICENSE).

You keep the copyright on what you write. On top of the AGPL grant above, you also grant Alessandro
Carosia a perpetual, worldwide, irrevocable, non-exclusive right to use your contribution and to
relicense it under other terms, permissive or commercial. Silica is single-owner today, and that is
what keeps a future relicensing possible at all: one merged PR without this grant would pin the
project to AGPL forever. Opening a PR is your agreement, there is no separate form to sign. If you
are contributing on behalf of an employer, make sure you have the authority to grant this.
