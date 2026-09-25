"""Harm-preserving rewrites. Each takes a circuit and returns a new one that
should do the same physical damage but look different to a detector."""
import random

from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate, HGate, SGate, SdgGate, XGate


def swap_to_3cx(qc):
    """SWAP = 3 CX. Unlike BadQubits' perturbation (SWAP -> 1 CX), this keeps
    the unitary and the two-qubit load."""
    out = qc.copy_empty_like()
    for inst in qc.data:
        if inst.operation.name == "swap":
            a, b = inst.qubits
            out.cx(a, b); out.cx(b, a); out.cx(a, b)
        else:
            out.append(inst)
    return out


def identity_pad(qc, rate=0.5, seed=0):
    """Insert G G^dagger pairs after gates. Measurement order is untouched."""
    rng = random.Random(seed)
    pairs = [(HGate(), HGate()), (XGate(), XGate()), (SGate(), SdgGate())]
    out = qc.copy_empty_like()
    for inst in qc.data:
        out.append(inst)
        if inst.operation.name in ("measure", "barrier") or rng.random() > rate:
            continue
        q = inst.qubits[0]
        g, gi = rng.choice(pairs)
        out.append(g, [q]); out.append(gi, [q])
    return out


def cx_pad(qc, rate=0.5, seed=0):
    """Insert CX CX pairs (identity, but two-qubit)."""
    rng = random.Random(seed)
    out = qc.copy_empty_like()
    n = qc.num_qubits
    for inst in qc.data:
        out.append(inst)
        if inst.operation.name in ("measure", "barrier") or rng.random() > rate:
            continue
        a, b = rng.sample(range(n), 2)
        out.append(CXGate(), [a, b]); out.append(CXGate(), [a, b])
    return out


def decompose(qc):
    """Expand every gate one level into its definition."""
    return qc.decompose()


def relabel(qc, seed=0):
    """Permute logical qubit indices."""
    perm = list(range(qc.num_qubits))
    random.Random(seed).shuffle(perm)
    out = qc.copy_empty_like()
    for inst in qc.data:
        qs = [out.qubits[perm[qc.find_bit(q).index]] for q in inst.qubits]
        out.append(inst.operation, qs, inst.clbits)
    return out


def all_combined(qc):
    return relabel(identity_pad(cx_pad(swap_to_3cx(qc))))


REWRITES = {
    "none": lambda qc: qc,
    "swap_to_3cx": swap_to_3cx,
    "identity_pad": identity_pad,
    "cx_pad": cx_pad,
    "decompose": decompose,
    "relabel": relabel,
    "combined": all_combined,
}
