# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deterministic reproducer for a recursive-SMARTS discrepancy in nvMolKit.

In our tests, ``nvmolkit.substructure.hasSubstructMatch`` returns NO matches for
recursive SMARTS environments ``$(...)`` when the *target* molecule has 128 or
fewer atoms, while non-recursive SMARTS match RDKit at all sizes.  Recursive
SMARTS matched RDKit again once the target had >= 129 atoms.  The flip point in
our runs coincided with ``kMaxTargetAtoms = 128`` in
``src/substruct/substruct_constants.h``.  We have not tested exhaustively, so the
output below describes observed behavior rather than a proven universal rule.

The script is fully self-contained (all molecules are hard-coded SMILES, no data
files needed) and deterministic in our testing.

Usage:
    python benchmarks/recursive_smarts_repro.py

Exit code:
    1  bug reproduced (recursive SMARTS disagree with RDKit on small molecules)
    0  bug NOT reproduced (likely fixed)
"""

from __future__ import annotations

import sys

import numpy as np
from rdkit import RDLogger
from rdkit.Chem import MolFromSmarts, MolFromSmiles

from nvmolkit.substructure import hasSubstructMatch

RDLogger.DisableLog("rdApp.*")


def nv_has(targets, queries) -> np.ndarray:
    """nvMolKit boolean match matrix, shape (n_targets, n_queries)."""
    return np.asarray(hasSubstructMatch(targets, queries)).astype(bool)


def rd_has(targets, queries) -> np.ndarray:
    """RDKit boolean match matrix (ground truth), shape (n_targets, n_queries)."""
    return np.array([[t.HasSubstructMatch(q) for q in queries] for t in targets], dtype=bool)


def print_env() -> None:
    import platform

    import nvmolkit
    import rdkit
    import torch

    print("=" * 72)
    print("ENVIRONMENT")
    print("=" * 72)
    print(f"  python   : {sys.version.split()[0]}  ({platform.platform()})")
    print(f"  nvmolkit : {nvmolkit.__version__}")
    print(f"  rdkit    : {rdkit.__version__}")
    print(f"  torch    : {torch.__version__}  (cuda {torch.version.cuda})")
    if torch.cuda.is_available():
        print(f"  gpu      : {torch.cuda.get_device_name(0)}")
    else:
        print("  gpu      : CUDA NOT AVAILABLE")
    print()


def section(title: str) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------------
# Part A -- minimal, unambiguous cases (small, common molecules)
# ---------------------------------------------------------------------------
def part_a() -> bool:
    section("A. Minimal cases: recursive [$(NC=O)] on small amide-containing molecules")
    q = MolFromSmarts("[$(NC=O)]")  # matches an N bonded to a carbonyl (amide-type N)
    cases = [
        ("acetamide", "CC(=O)N"),
        ("formamide", "NC=O"),
        ("N-methylacetamide", "CC(=O)NC"),
        ("benzamide", "NC(=O)c1ccccc1"),
        ("urea", "NC(=O)N"),
        ("methylamine (true negative)", "CN"),
        ("acetic acid (true negative)", "CC(=O)O"),
    ]
    mols = [MolFromSmiles(s) for _, s in cases]
    rd = rd_has(mols, [q])[:, 0]
    nv = nv_has(mols, [q])[:, 0]
    print(f"  {'molecule':32s} {'atoms':>5} {'RDKit':>6} {'nvMolKit':>9} {'':>4}")
    bug = False
    for i, (name, smi) in enumerate(cases):
        flag = "FAIL" if rd[i] != nv[i] else "ok"
        if rd[i] != nv[i]:
            bug = True
        print(f"  {name:32s} {mols[i].GetNumAtoms():>5} {str(bool(rd[i])):>6} {str(bool(nv[i])):>9} {flag:>4}")
    print()
    return bug


# ---------------------------------------------------------------------------
# Part B -- recursion is the trigger (non-recursive controls all agree)
# ---------------------------------------------------------------------------
def part_b() -> bool:
    section("B. Recursion is the trigger: recursive queries fail, non-recursive agree")
    smis = ["CC(=O)N", "NC=O", "CC(=O)NC", "NC(=O)c1ccccc1", "NC(=O)N", "CN", "CCN", "CC(=O)O", "CCO", "c1ccccc1"]
    mols = [MolFromSmiles(s) for s in smis]
    probes = [
        ("[$(NC=O)]", True),   # recursive
        ("[$([OH])]", True),   # recursive
        ("[$([#6])]", True),   # recursive, trivial: ANY carbon
        ("C=O", False),        # non-recursive control
        ("[OH]", False),       # non-recursive control
        ("[#6]", False),       # non-recursive control: ANY carbon
    ]
    queries = [MolFromSmarts(p) for p, _ in probes]
    rd = rd_has(mols, queries)
    nv = nv_has(mols, queries)
    print(f"  {'pattern':12s} {'recursive':>9} {'rd_hits':>8} {'nv_hits':>8} {'agree':>7} {'verdict':>8}")
    bug = False
    for j, (p, rec) in enumerate(probes):
        agree = 100.0 * (rd[:, j] == nv[:, j]).mean()
        verdict = "ok" if agree == 100 else "BROKEN"
        if rec and agree != 100:
            bug = True
        print(f"  {p:12s} {str(rec):>9} {int(rd[:, j].sum()):>8} {int(nv[:, j].sum()):>8} {agree:>6.0f}% {verdict:>8}")
    print()
    return bug


# ---------------------------------------------------------------------------
# Part C -- exact 128-atom threshold via glycine chains
# ---------------------------------------------------------------------------
def glycine_chain(n_res: int) -> str:
    """SMILES for H-(Gly)_n-OH; every residue after the first adds a peptide (amide) bond."""
    return "NCC(=O)" + "NCC(=O)" * (n_res - 1) + "O"


def part_c() -> bool:
    section("C. Exact size threshold: recursive [$(NC=O)] on glycine chains (Gly)_n")
    q = MolFromSmarts("[$(NC=O)]")
    print(f"  {'n_res':>5} {'atoms':>5} {'RDKit':>6} {'nvMolKit':>9} {'note'}")
    prev = None
    flip_atoms = None
    chains = [1, 8, 16, 20, 24, 28, 30, 31, 32, 33, 36]
    bug = False
    for n in chains:
        m = MolFromSmiles(glycine_chain(n))
        na = m.GetNumAtoms()
        rd = m.HasSubstructMatch(q)
        nv = bool(nv_has([m], [q])[0, 0])
        note = ""
        if prev is not None and prev != nv:
            note = "<-- nvMolKit flips correct here"
            flip_atoms = na
        prev = nv
        if rd and not nv:
            bug = True
        print(f"  {n:>5} {na:>5} {str(rd):>6} {str(nv):>9}  {note}")
    if flip_atoms is not None:
        print(f"\n  nvMolKit recursive matching turns on at {flip_atoms} atoms "
              f"(kMaxTargetAtoms = 128 -> first multi-tile size).")
    print()
    return bug


# ---------------------------------------------------------------------------
# Part D -- determinism (identical across repeated runs / batch sizes)
# ---------------------------------------------------------------------------
def part_d() -> None:
    section("D. Determinism check (recursive [$(NC=O)], small molecule, 5 runs)")
    q = [MolFromSmarts("[$(NC=O)]")]
    m = [MolFromSmiles("CC(=O)N")]  # acetamide
    results = [int(nv_has(m, q)[0, 0]) for _ in range(5)]
    print(f"  acetamide nvMolKit result across 5 runs: {results}  "
          f"({'deterministic' if len(set(results)) == 1 else 'NONDETERMINISTIC'})")
    print()


def main() -> int:
    print_env()
    a = part_a()
    b = part_b()
    c = part_c()
    part_d()

    section("VERDICT")
    reproduced = a or b or c
    if reproduced:
        print("  BUG REPRODUCED: recursive SMARTS $(...) disagree with RDKit on molecules")
        print("  with <= 128 atoms. See parts A/B/C above.")
        return 1
    print("  Bug NOT reproduced -- recursive SMARTS agree with RDKit (likely fixed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
