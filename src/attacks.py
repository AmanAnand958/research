"""Toy stand-ins for BadQubits-style harmful circuits.

BadQubits' dataset is not public yet, so these mimic the two harm signals the
paper's detectors key on: heavy two-qubit (SWAP/CX) load and early measurement.
Swap in the regenerated MQT Bench + BadQubits generator circuits when ready.
"""
from qiskit import QuantumCircuit


def swap_storm(n=5, reps=20):
    """Heating/crosstalk style: many entangling ops. Barriers stop the
    transpiler from cancelling back-to-back SWAPs (SWAP*SWAP = I)."""
    qc = QuantumCircuit(n, n)
    for r in range(reps):
        for q in range(n - 1):
            qc.swap(q, q + 1)
        qc.barrier()
    qc.measure(range(n), range(n))
    return qc


def early_measure(n=5, reps=10):
    """Measure right away, then keep driving qubits (state-collapse style)."""
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    qc.measure(range(n), range(n))
    for r in range(reps):
        for q in range(n - 1):
            qc.cx(q, q + 1)
        qc.barrier()
    return qc


def deutsch_jozsa(n=4):
    """Benign control: balanced-oracle DJ. BadQubits flags 15/21 of these."""
    qc = QuantumCircuit(n + 1, n)
    qc.x(n)
    qc.h(range(n + 1))
    for q in range(n):
        qc.cx(q, n)
    qc.h(range(n))
    qc.measure(range(n), range(n))
    return qc


ATTACKS = {"swap_storm": swap_storm, "early_measure": early_measure}
BENIGN = {"deutsch_jozsa": deutsch_jozsa}
