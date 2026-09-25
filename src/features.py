"""Per-circuit caches: transpiled views, token sequences, harm metrics."""
import json
import random
import warnings
from multiprocessing import Pool

from qiskit import qasm2, transpile
from qiskit.providers.fake_provider import GenericBackendV2

import rewrites as R

warnings.filterwarnings("ignore")
BACKEND = GenericBackendV2(num_qubits=16, seed=1)  # stand-in for provider backend
LEVELS = (0, 1, 2, 3)
REWRITE_NAMES = ["none", "swap_to_3cx", "identity_pad", "cx_pad", "decompose", "relabel", "combined"]
SKIP = {"barrier", "delay"}


def apply_rewrite(name, qc, seed):
    if name == "none":
        return qc
    if name == "swap_to_3cx":
        return R.swap_to_3cx(qc)
    if name == "identity_pad":
        return R.identity_pad(qc, seed=seed)
    if name == "cx_pad":
        return R.cx_pad(qc, seed=seed)
    if name == "decompose":
        return R.decompose(qc)
    if name == "relabel":
        return R.relabel(qc, seed=seed)
    if name == "combined":
        return R.relabel(R.identity_pad(R.cx_pad(R.swap_to_3cx(qc), seed=seed), seed=seed), seed=seed)
    raise KeyError(name)


def tokens(qc):
    return " ".join(i.operation.name for i in qc.data if i.operation.name not in SKIP)


def cx2q(qc):
    return sum(1 for i in qc.data if i.operation.num_qubits == 2 and i.operation.name not in SKIP)


def mid_meas(qc):
    """Measurements followed by a later gate on the same qubit."""
    pending, n = set(), 0
    for i in qc.data:
        name = i.operation.name
        if name in SKIP:
            continue
        qs = {qc.find_bit(q).index for q in i.qubits}
        if name == "measure":
            pending |= qs
        else:
            n += len(pending & qs)
            pending -= qs
    return n


def load(qasm):
    return qasm2.loads(qasm, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)


def job(args):
    idx, qasm, rw = args
    qc = apply_rewrite(rw, load(qasm), seed=idx)
    out = {"raw": tokens(qc)}
    for lvl in LEVELS:
        t = transpile(qc, BACKEND, optimization_level=lvl, seed_transpiler=1)
        out[f"cx{lvl}"] = cx2q(t)
        out[f"mm{lvl}"] = mid_meas(t)
        if lvl in (1, 3):
            out[f"L{lvl}"] = tokens(t)
    return idx, rw, out


def build(data="../data/dataset.jsonl", out="../data/views.json", procs=4):
    rows = [json.loads(l) for l in open(data)]
    jobs = [(i, r["qasm"], rw) for i, r in enumerate(rows) for rw in REWRITE_NAMES]
    random.Random(0).shuffle(jobs)  # spread big circuits across workers
    cache = {}
    with Pool(procs) as p:
        for k, (i, rw, o) in enumerate(p.imap_unordered(job, jobs, chunksize=8)):
            cache[f"{i}|{rw}"] = o
            if k % 1000 == 0:
                print(k, "/", len(jobs), flush=True)
    json.dump(cache, open(out, "w"))


if __name__ == "__main__":
    build()
