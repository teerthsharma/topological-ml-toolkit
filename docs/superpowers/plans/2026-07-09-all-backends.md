# All Backends Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote every planned backend from a README promise into a tested contract, adapter, or guarded implementation surface.

**Architecture:** Keep `topoml-core` as the correctness source. Add a backend metadata contract in Rust and Python, then route C++/ASM/Triton/PyTorch/TensorFlow through opt-in adapters with dense/safe fallbacks. Heavy dependencies must remain optional.

**Tech Stack:** Rust 2021, Python 3.10+, NumPy, optional PyTorch/TensorFlow/Triton, C++ header-only adapter surface, MkDocs.

## Global Constraints

- Safe Rust remains the truth source for persistent homology correctness.
- Python remains importable without PyTorch, TensorFlow, Triton, ripser, GUDHI, C++, or ASM tooling installed.
- C++ must be a portable native extension path with documented ABI and barcode equivalence gate.
- ASM AVX-512 must have CPUID gating and safe fallback; no speed claim without correctness equivalence.
- Triton must expose schedule metadata and require dense SDPA/FlashAttention baseline plus same-budget ablations before speed claims.
- PyTorch must provide dense fallback and document torch.compile-safe behavior before any module rewrite claim.
- TensorFlow must distinguish eager and graph-mode parity before any adapter claim.
- Do not add benchmark speed numbers to README unless generated raw artifacts exist.

---

### Task 1: Shared Backend Contract

**Files:**
- Modify: `crates/topoml-core/src/lib.rs`
- Test: `crates/topoml-core/tests/backend_contract.rs`
- Modify: `python/topoml/core.py`
- Test: `python/tests/test_backend_contract.py`

**Interfaces:**
- Produces Rust `BackendId`, `BackendCapability`, `BackendMetadata`, `BackendWarning`, `select_backend`.
- Produces Python `BackendMetadata`, `available_backends()`, and `select_backend(name)`.

- [ ] **Step 1: Add failing Rust backend contract tests**
- [ ] **Step 2: Verify Rust tests fail for missing backend API**
- [ ] **Step 3: Implement typed backend metadata and safe fallback**
- [ ] **Step 4: Add failing Python backend contract tests**
- [ ] **Step 5: Implement Python backend metadata mirror**
- [ ] **Step 6: Run Rust and Python tests**

### Task 2: Framework Adapters

**Files:**
- Create: `python/topoml/adapters/pytorch.py`
- Create: `python/topoml/adapters/tensorflow.py`
- Create: `python/topoml/adapters/triton.py`
- Create: `python/topoml/adapters/__init__.py`
- Test: `python/tests/test_framework_adapters.py`
- Docs: `docs/backends/framework-adapters.md`

**Interfaces:**
- Produces lazy optional imports.
- Produces NumPy conversion helpers without importing heavy frameworks at `import topoml` time.
- Produces adapter capability records for docs and tests.

- [ ] **Step 1: Add failing optional-adapter tests**
- [ ] **Step 2: Implement lazy import guards**
- [ ] **Step 3: Implement dense/safe fallback metadata**
- [ ] **Step 4: Document PyTorch, TensorFlow, and Triton gates**
- [ ] **Step 5: Run Python tests and docs build**

### Task 3: C++ and ASM Contracts

**Files:**
- Create: `cpp/include/topoml/topoml.hpp`
- Create: `cpp/README.md`
- Create: `asm/README.md`
- Modify: `backends/README.md`
- Docs: `docs/backends/native-and-asm.md`

**Interfaces:**
- Produces a portable C++ wrapper contract around the future C ABI.
- Produces ASM backend CPUID/fallback contract and correctness gate.

- [ ] **Step 1: Add native backend documentation**
- [ ] **Step 2: Add C++ header-only API sketch with explicit ABI boundaries**
- [ ] **Step 3: Add ASM CPUID/fallback gate documentation**
- [ ] **Step 4: Run docs build**

### Task 4: Broad Topology Gap Map

**Files:**
- Create: `docs/concepts/topology-landscape.md`
- Modify: `mkdocs.yml`

**Interfaces:**
- Produces taxonomy of topology families not adequately covered by Aether/Epsilon.
- Produces implementability classification: docs-only, prototype, core later.

- [ ] **Step 1: Add topology landscape doc**
- [ ] **Step 2: Link it from MkDocs nav**
- [ ] **Step 3: Run docs build**

