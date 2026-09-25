"""Main study: which harm-preserving rewrites evade which detectors, and does
canonicalizing (transpiling before detection) close the gap?

Views: raw = circuit as submitted (BadQubits setup); L1/L3 = transpiled at that
optimization level before detection. The provider runs the circuit at level 2
for the raw view (Qiskit default) and at the canonicalization level otherwise.
Evasion only counts if the attack's harm survives at the run level.
"""
import csv
import json
import random
import sys
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

from detectors import DETECTORS
from features import LEVELS, REWRITE_NAMES

VIEWS = {"raw": 2, "L1": 1, "L3": 3}  # view -> level the circuit runs at
KEEP = 0.9
AUG_DETECTORS = ("ngram_lr", "feat_gb")  # cheap enough for leave-one-rewrite-out


def load(data="../data/dataset.jsonl", views="../data/views.json"):
    rows = [json.loads(l) for l in open(data)]
    cache = json.load(open(views))
    idx = {r["id"]: i for i, r in enumerate(rows)}
    return rows, cache, idx


def harm_kept(rows, cache, idx, i, rw, lvl):
    r = rows[i]
    if r["label"] == 0:
        return False
    x, x0 = cache[f"{i}|{rw}"], cache[f"{i}|none"]
    if r["attack"] == "swap_storm":
        b = cache[f"{idx[r['base']]}|none"]
        ref = x0["cx0"] - b["cx0"]
        return ref > 0 and x[f"cx{lvl}"] - b[f"cx{lvl}"] >= KEEP * ref
    ref = x0["mm0"]
    return ref > 0 and x[f"mm{lvl}"] >= KEEP * ref


def split(rows, seed, test=0.3):
    bases = sorted({r["base"] for r in rows})
    random.Random(seed).shuffle(bases)
    tb = set(bases[: int(len(bases) * test)])
    tr = [i for i, r in enumerate(rows) if r["base"] not in tb]
    te = [i for i, r in enumerate(rows) if r["base"] in tb]
    return tr, te


def ece(p, y, bins=10):
    p, y = np.asarray(p), np.asarray(y)
    e = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (p >= lo) & (p < lo + 1 / bins)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def train_set(rows, cache, idx, ids, view, rws):
    X, y = [], []
    for i in ids:
        for rw in rws:
            lab = rows[i]["label"]
            if lab and view != "raw" and not harm_kept(rows, cache, idx, i, rw, VIEWS[view]):
                continue  # harm gone at run level: not a real attack, drop it
            X.append(cache[f"{i}|{rw}"][view])
            y.append(lab)
    return X, y


def evaluate(model, rows, cache, idx, te, view, rw, det, seed, variant):
    lvl = VIEWS[view]
    harm = [i for i in te if harm_kept(rows, cache, idx, i, rw, lvl)]
    ben = [i for i in te if rows[i]["label"] == 0]
    ph = model.proba([cache[f"{i}|{rw}"][view] for i in harm]) if harm else np.array([])
    pb = model.proba([cache[f"{i}|{rw}"][view] for i in ben])
    miss = ph[ph < 0.5]
    out = dict(seed=seed, detector=det, variant=variant, view=view, rewrite=rw,
               n_harm=len(harm), recall=float((ph >= 0.5).mean()) if len(ph) else np.nan,
               miss_mean_p=float(miss.mean()) if len(miss) else np.nan,
               conf_miss=float((ph < 0.1).mean()) if len(ph) else np.nan,
               fpr=float((pb >= 0.5).mean()))
    for a in ("swap_storm", "early_measure"):
        m = np.array([rows[i]["attack"] == a for i in harm], dtype=bool)
        out[f"recall_{a}"] = float((ph[m] >= 0.5).mean()) if m.any() else np.nan
    if rw == "none":
        y = [1] * len(ph) + [0] * len(pb)
        p = np.concatenate([ph, pb])
        out["auc"] = roc_auc_score(y, p)
        out["ece"] = ece(p, y)
        dj = [k for k, i in enumerate(ben) if rows[i]["family"] == "dj"]
        out["dj_fpr"] = float((pb[dj] >= 0.5).mean()) if dj else np.nan
    return out


def survival(rows, cache, idx, out="../results/harm_survival.csv"):
    res = []
    for a in ("swap_storm", "early_measure"):
        ids = [i for i, r in enumerate(rows) if r["attack"] == a]
        for rw in REWRITE_NAMES:
            for lvl in LEVELS:
                k = [harm_kept(rows, cache, idx, i, rw, lvl) for i in ids]
                res.append(dict(attack=a, rewrite=rw, opt_level=lvl, n=len(k), harm_kept=np.mean(k)))
    write(res, out)
    return res


def write(res, path):
    keys = list(dict.fromkeys(k for r in res for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(res)


def main(seeds=3):
    rows, cache, idx = load()
    survival(rows, cache, idx)
    res = []
    for seed in range(int(seeds)):
        tr, te = split(rows, seed)
        for view in VIEWS:
            for det, D in DETECTORS.items():
                X, y = train_set(rows, cache, idx, tr, view, ["none"])
                m = D().fit(X, y)
                for rw in REWRITE_NAMES:
                    res.append(evaluate(m, rows, cache, idx, te, view, rw, det, seed, "clean"))
                print(seed, view, det, "clean done", flush=True)
                if det not in AUG_DETECTORS:
                    continue
                for rw in REWRITE_NAMES[1:]:  # leave-one-rewrite-out augmentation
                    seen = [r for r in REWRITE_NAMES if r != rw]
                    X, y = train_set(rows, cache, idx, tr, view, seen)
                    m = D().fit(X, y)
                    res.append(evaluate(m, rows, cache, idx, te, view, rw, det, seed, "aug_loro"))
                print(seed, view, det, "aug done", flush=True)
        write(res, "../results/detection.csv")


if __name__ == "__main__":
    main(*sys.argv[1:])
