# Bug report: recursive SMARTS `$(...)` return no matches for small target molecules

**Component:** `nvmolkit.substructure` (GPU substructure search)
**Severity (reporter's assessment):** High — incorrect results are returned without any error, for a
feature the docstrings describe as supported.
**Status:** Reproduces deterministically in the environment below.
**Reproducer:** [`benchmarks/recursive_smarts_repro.py`](./recursive_smarts_repro.py)

---

## Summary

In our tests, `hasSubstructMatch` / `countSubstructMatches` / `getSubstructMatches` returned **no match**
for recursive SMARTS environments `$(...)` when the **target** molecule had **128 or fewer atoms**.
Recursive SMARTS matched RDKit once the target had **≥ 129 atoms**, and non-recursive SMARTS matched
RDKit at all sizes we tried. Every molecule we tested below 129 atoms showed the discrepancy, but we
have not tested exhaustively, so we describe this as the behavior we observed rather than a proven
universal rule.

The flip point in our runs coincided with `kMaxTargetAtoms = 128` in
`src/substruct/substruct_constants.h`. The docstrings for all three entry points state
*"Supports recursive SMARTS queries"*, and no exception is raised, so the incorrect result is silent.

## Environment

| Field | Value |
|---|---|
| nvMolKit | `0.5.0` (wheel `nvmolkit-0.5.0.post1`) |
| Repo commit | `6f967ed` (VERSION 0.5.0, `main`) |
| RDKit | `2026.03.1` |
| PyTorch | `2.12.0+cu126` (CUDA 12.6) |
| GPU | NVIDIA A100 80GB PCIe (sm_80) |
| Driver | 565.57.01 |
| Python / OS | 3.12.13 / Linux x86_64 (glibc 2.35) |

We have only reproduced this on the single configuration above; behavior on other GPUs / RDKit
versions / builds is unknown.

## Affected API

`hasSubstructMatch`, `countSubstructMatches`, `getSubstructMatches` (shared preprocessing path;
demonstrated below with `hasSubstructMatch`).

## Test molecules

Query: `[$(NC=O)]` (a recursive SMARTS matching a nitrogen bonded to a carbonyl, i.e. an amide-type N).

Small molecules where we observed RDKit and nvMolKit **disagree** (all contain an amide N):

| Molecule | SMILES | Atoms | RDKit | nvMolKit (observed) |
|---|---|---:|:---:|:---:|
| acetamide | `CC(=O)N` | 4 | match | no match |
| formamide | `NC=O` | 3 | match | no match |
| N-methylacetamide | `CC(=O)NC` | 5 | match | no match |
| benzamide | `NC(=O)c1ccccc1` | 9 | match | no match |
| urea | `NC(=O)N` | 4 | match | no match |

Controls where RDKit and nvMolKit **agreed** (true negatives — no amide N):

| Molecule | SMILES | Atoms | RDKit | nvMolKit (observed) |
|---|---|---:|:---:|:---:|
| methylamine | `CN` | 2 | no match | no match |
| acetic acid | `CC(=O)O` | 4 | no match | no match |

Copy-paste list:

```python
disagree = ["CC(=O)N", "NC=O", "CC(=O)NC", "NC(=O)c1ccccc1", "NC(=O)N"]  # RDKit match, nvMolKit no match
agree    = ["CN", "CC(=O)O"]                                              # both no match
```

To probe the size threshold, glycine chains `H-(Gly)ₙ-OH` (`"NCC(=O)" + "NCC(=O)"*(n-1) + "O"`) give a
clean atom-count sweep where every chain with n ≥ 2 contains a peptide (amide) bond:

```python
# n_res -> atoms : 8->33, 24->97, 30->121, 31->125, 32->129, 33->133
```

## Steps to reproduce

```bash
python benchmarks/recursive_smarts_repro.py
# exit code 1 = bug reproduced, 0 = not reproduced
```

Minimal standalone snippet:

```python
from rdkit.Chem import MolFromSmiles, MolFromSmarts
from nvmolkit.substructure import hasSubstructMatch

q = MolFromSmarts("[$(NC=O)]")          # recursive: an N bonded to a carbonyl
m = MolFromSmiles("CC(=O)N")            # acetamide (4 atoms)

print(m.HasSubstructMatch(q))           # RDKit    -> True
print(int(hasSubstructMatch([m], [q])[0, 0]))  # nvMolKit -> 0 in our runs
```

## Observed behavior

**1. Small molecules — recursive queries returned 0 matches; non-recursive queries agreed with RDKit:**

```
pattern      recursive  rd_hits  nv_hits   agree  verdict
[$(NC=O)]         True        5        0     50%   disagree
[$([OH])]         True        2        0     80%   disagree
[$([#6])]         True       10        0      0%   disagree   <- "any carbon" wrapped recursively, also 0
C=O              False        6        6    100%   agree
[OH]             False        2        2    100%   agree
[#6]             False       10       10    100%   agree
```

`[$([#6])]` wraps "any carbon" in a recursive environment; logically it should match every
carbon-containing molecule, but it returned zero matches.

**2. Size threshold — recursive `[$(NC=O)]` on glycine chains:**

```
n_res atoms  RDKit  nvMolKit
    8    33   True     False
   24    97   True     False
   30   121   True     False
   31   125   True     False
   32   129   True      True   <-- nvMolKit started matching here
   33   133   True      True
```

In our runs the boundary was sharp: 0% recall at ≤ 128 atoms, correct at ≥ 129 atoms. We also saw this
on a 2,000-molecule ChEMBL sample (≈0% recall for ≤128-atom molecules, ≈91% for ≥129-atom molecules).
For the large molecules where nvMolKit did return matches, the atom indices were identical to RDKit, so
the core matching appears correct in those cases — the smaller molecules were the ones skipped.

**3. Deterministic:** identical across 5 repeated runs and across `batchSize` ∈ {16 … 100000} in our
testing — we did not observe run-to-run or batch-ordering variation.

> Note on why this can be easy to miss: a query panel restricted to small molecules can fail this way
> while only showing up against an RDKit comparison; and a molecule set whose first entries are large
> (≥ 129 atoms) can show partial agreement. Comparing a recursive query on a small molecule against
> RDKit is the most direct check.

## Expected behavior

We would expect recursive SMARTS results to match RDKit `Mol.HasSubstructMatch` regardless of molecule
size, as non-recursive SMARTS already did in our tests. For example,
`hasSubstructMatch([acetamide], [MolFromSmarts("[$(NC=O)]")])` would be expected to return `1`.

## Root-cause hypothesis (speculative)

The threshold we observed equals `kMaxTargetAtoms = 128` (`src/substruct/substruct_constants.h`; kernel
template `Config_T128_Q64_B8`). One possibility is that the recursive match-bit buffer
(`recursiveMatchBits`, stride `maxTargetAtoms` — see `src/substruct/substruct_kernels.h` and the
"paint mode" kernel for recursive SMARTS preprocessing) is only populated for targets spanning more than
one 128-atom tile, so molecules fitting in a single tile (≤ 128 atoms) skip the recursive-bit step. We
have not confirmed this in the source — it is only a hypothesis consistent with the observed threshold.

Places that might be worth checking:
- The condition gating the recursive-bit "paint" launch on target atom count / tile count.
- Indexing of `recursiveMatchBits` by `maxTargetAtoms` stride for single-tile molecules.

A regression test comparing a recursive query on a small molecule against RDKit would likely catch this;
the existing suite may pass because it does not appear to cover that specific case.
