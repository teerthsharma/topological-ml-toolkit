# Contribution Standard

Small repo does not mean small standards.

## Code

- Rust APIs must expose typed errors, not panics.
- Production Rust must avoid `unwrap()` unless the invariant is local and obvious.
- Unsafe code needs a `Safety` section and a test that exercises the boundary.
- Optional dependencies must stay optional.
- Public Python APIs must accept NumPy arrays and return graph-ready structures.

## Docs

Every concept page needs:

- formula;
- plain-English explanation;
- generated graph;
- ML use case;
- failure mode.

## Benchmarks

Every speed claim needs:

- raw artifact;
- baseline list;
- hardware/software environment;
- correctness metric;
- seed or determinism note.

## Commits

Do not spam commits. Group related work into coherent commits that can be
reviewed alone.

