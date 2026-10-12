#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recompute the paper's natural-fault numbers (Sec. 5.4) from labels_and_scores.csv. No command-line arguments:
edit the block below and run the file. Needs only numpy.

Labels: each rater chose 1 = correct, 2 = listener also moves the mouth, 3 = speaker does not move the mouth,
23 = wrong person speaks, 4 = mouth on the wrong face / extra face, 5 = cannot see. Binary label: fault = 2/3/23/4,
correct = 1, 5 and unanswered = missing. Reference label = majority vote; ties or fewer than two valid votes are dropped.
Intervals: 2000 bootstrap resamples of whole inputs (seed 20261012)."""
import csv, os
import numpy as np

# ===== Edit here =====
CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "labels_and_scores.csv")
METRICS = ["SLT_gated", "SLT_ungated", "per_person_D", "S", "L_Bind", "T_offset", "mixed_negC"]   # larger = more anomalous
NBOOT, SEED = 2000, 20261012
# =====================

rows = list(csv.DictReader(open(CSV)))
FAULT, OK = {"2", "3", "23", "4"}, {"1"}
def binary(a): return 1 if a in FAULT else 0 if a in OK else None
def majority(r):
    v = [binary(r[f"rater{i}"]) for i in (1, 2, 3)]; v = [x for x in v if x is not None]
    return None if len(v) < 2 or sum(v) * 2 == len(v) else int(sum(v) * 2 > len(v))
def fleiss(rs):
    rs = [r for r in rs if all(binary(r[f"rater{i}"]) is not None for i in (1, 2, 3))]
    x = np.array([[sum(binary(r[f"rater{i}"]) == j for i in (1, 2, 3)) for j in (0, 1)] for r in rs], float)
    P = ((x * (x - 1)).sum(1) / 6).mean(); p = x.sum(0) / x.sum(); Pe = (p ** 2).sum(); return (P - Pe) / (1 - Pe)
def auc(rs, m):
    p = np.array([float(r[m]) for r in rs if r["y"] == 1 and r[m] != ""]); q = np.array([float(r[m]) for r in rs if r["y"] == 0 and r[m] != ""])
    return float(((p[:, None] > q[None]) + 0.5 * (p[:, None] == q[None])).mean()) if len(p) and len(q) else np.nan
for gen in sorted({r["generator"] for r in rows}):
    R = [r for r in rows if r["generator"] == gen]; K = fleiss(R)
    for r in R: r["y"] = majority(r)
    L = [r for r in R if r["y"] is not None]; ids = sorted({r["input_id"] for r in L}); by = {i: [r for r in L if r["input_id"] == i] for i in ids}
    rng = np.random.default_rng(SEED); B = [[r for i in rng.choice(ids, len(ids)) for r in by[i]] for _ in range(NBOOT)]
    print(f"== {gen}: {len(R)} clips, majority fault {sum(r['y'] for r in L)} / correct {sum(1 - r['y'] for r in L)}, Fleiss kappa {K:.2f}")
    for m in METRICS:
        bs = [auc(b, m) for b in B]; print(f"   {m:14s} AUC {auc(L, m):.2f} [{np.nanpercentile(bs, 2.5):.2f}, {np.nanpercentile(bs, 97.5):.2f}]")
