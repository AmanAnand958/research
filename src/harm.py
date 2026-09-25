"""Harm metrics measured on the circuit that actually hits hardware
(i.e. after the provider's transpiler)."""


def two_qubit_count(qc):
    return sum(1 for i in qc.data if i.operation.num_qubits == 2
               and i.operation.name != "barrier")


def ops_after_first_measure(qc):
    """Gate count after the earliest measurement (0 = measure-at-end only)."""
    seen = False; n = 0
    for i in qc.data:
        name = i.operation.name
        if name == "measure":
            seen = True
        elif seen and name not in ("barrier",):
            n += 1
    return n


def harm(qc):
    return {"cx2q": two_qubit_count(qc), "post_meas": ops_after_first_measure(qc)}
