"""Week-1 kill-switch test: does each rewrite survive the transpiler with its
harm intact? Reference = original attack transpiled at opt level 0 (the harm
the attacker intended). Harm is 'kept' if the transpiled rewritten circuit
keeps >= 90% of the reference's 2q gates and post-measurement ops."""
import csv
import sys

from qiskit import transpile
from qiskit.providers.fake_provider import GenericBackendV2

from attacks import ATTACKS, BENIGN
from harm import harm
from rewrites import REWRITES

KEEP = 0.9


def main(out="results/survival.csv"):
    backend = GenericBackendV2(num_qubits=7, seed=1)
    rows = []
    for cname, make in {**ATTACKS, **BENIGN}.items():
        base = make()
        ref = harm(transpile(base, backend, optimization_level=0, seed_transpiler=1))
        for lvl in range(4):
            for rname, rw in REWRITES.items():
                rc = rw(base)
                pre = harm(rc)
                post = harm(transpile(rc, backend, optimization_level=lvl, seed_transpiler=1))
                kept = all(post[k] >= KEEP * ref[k] for k in ref)
                rows.append(dict(circuit=cname, opt_level=lvl, rewrite=rname,
                                 ref_2q=ref["cx2q"], pre_2q=pre["cx2q"], post_2q=post["cx2q"],
                                 ref_post_meas=ref["post_meas"], post_post_meas=post["post_meas"],
                                 harm_kept=kept))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    for r in rows:
        print("{circuit:14s} L{opt_level} {rewrite:13s} 2q ref/pre/post={ref_2q:4d}/{pre_2q:4d}/{post_2q:4d}"
              "  postmeas ref/post={ref_post_meas:4d}/{post_post_meas:4d}  kept={harm_kept}".format(**r))


if __name__ == "__main__":
    main(*sys.argv[1:])
