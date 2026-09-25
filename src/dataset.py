"""Rebuild a BadQubits-style dataset: benign MQT Bench circuits + harmful
variants made by injecting attack blocks into them.

This approximates BadQubits' generator from its description (SWAP-heavy
blocks, early/mid-circuit measurement). Swap in their code when released.
Circuits are flattened to a broad standard gate set with no optimization,
matching BadQubits' "flatten, don't transpile" setup.
"""
import json
import random
import sys
import warnings

from qiskit import QuantumCircuit, qasm2, transpile
from mqt.bench import get_benchmark
from mqt.bench.benchmark_generation import BenchmarkLevel
from mqt.bench.benchmarks import get_available_benchmark_names

warnings.filterwarnings("ignore")

STD = ["id", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "sxdg", "rx", "ry", "rz",
       "p", "u", "cx", "cy", "cz", "ch", "cp", "crx", "cry", "crz", "cu", "swap", "ccx",
       "cswap", "rzz", "rxx", "ryy", "measure", "reset"]
CONTROL_FLOW = {"if_else", "while_loop", "for_loop", "switch_case"}
SIZES = range(3, 17)
MAX_OPS = 1500  # skip huge circuits to keep text/transpile cost sane


def flatten(qc):
    return transpile(qc, basis_gates=STD, optimization_level=0)


def inject_swap_storm(qc, rng):
    """Insert a block of SWAP rounds (heating/crosstalk) at a random point."""
    n = qc.num_qubits
    reps = rng.randint(3, 15)
    block = QuantumCircuit(n)
    for _ in range(reps):
        for q in range(rng.randint(0, 1), n - 1, rng.choice([1, 2])):
            block.swap(q, q + 1)
        block.barrier()  # stop trivial SWAP*SWAP cancellation
    return _splice(qc, block, rng)


def inject_early_measure(qc, rng):
    """Measure a subset of qubits partway through, then keep going."""
    n = qc.num_qubits
    k = rng.randint(1, n)
    qs = rng.sample(range(n), k)
    block = QuantumCircuit(n, n)
    for q in qs:
        block.measure(q, q)
    return _splice(qc, block, rng, lo=0.1, hi=0.5)


def _splice(qc, block, rng, lo=0.2, hi=0.8):
    body = [i for i in qc.data if i.operation.name != "measure"]
    tail = [i for i in qc.data if i.operation.name == "measure"]
    cut = int(len(body) * rng.uniform(lo, hi))
    out = qc.copy_empty_like()
    if block.num_clbits and not out.num_clbits:
        out.add_bits(block.clbits[: 0])
    for i in body[:cut]:
        out.append(i)
    out.compose(block, qubits=range(qc.num_qubits),
                clbits=range(min(block.num_clbits, out.num_clbits)), inplace=True)
    for i in body[cut:] + tail:
        out.append(i)
    return out


def ensure_clbits(qc):
    if qc.num_clbits >= qc.num_qubits:
        return qc
    out = QuantumCircuit(qc.num_qubits, qc.num_qubits)
    out.compose(qc, qubits=range(qc.num_qubits), clbits=range(qc.num_clbits), inplace=True)
    return out


ATTACKS = {"swap_storm": inject_swap_storm, "early_measure": inject_early_measure}


def main(out="../data/dataset.jsonl", seed=0):
    rng = random.Random(seed)
    rows = []
    for name in get_available_benchmark_names():
        for n in SIZES:
            try:
                qc = flatten(get_benchmark(name, BenchmarkLevel.ALG, n))
            except Exception:
                continue
            if qc.size() > MAX_OPS:
                continue
            if any(i.operation.name in CONTROL_FLOW for i in qc.data):
                continue  # dynamic circuits don't export to QASM2; skip them
            qc = ensure_clbits(qc)
            base_id = f"{name}_{n}"
            rows.append(dict(id=base_id, base=base_id, family=name, label=0, attack="none",
                             qasm=qasm2.dumps(qc)))
            for aname, inject in ATTACKS.items():
                for v in range(1):
                    h = flatten(inject(qc, rng))
                    rows.append(dict(id=f"{base_id}_{aname}{v}", base=base_id, family=name,
                                     label=1, attack=aname, qasm=qasm2.dumps(h)))
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} circuits: {sum(r['label'] == 0 for r in rows)} benign, "
          f"{sum(r['label'] for r in rows)} harmful, "
          f"{len({r['family'] for r in rows})} families")


if __name__ == "__main__":
    main(*sys.argv[1:])
