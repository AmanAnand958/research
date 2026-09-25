# Harm-preserving rewrites vs quantum harmful-circuit detectors

Question: which rewrites (1) still harm hardware after the provider's
transpiler and (2) evade BadQubits-style detectors? Does canonicalizing
(transpiling) before detection close the gap?

Idea 1 (Qiskit version prediction) is the fallback: a "run under old version,
check deprecation warnings" baseline likely beats ML there.

## Layout

- `src/attacks.py` – toy harmful circuits (SWAP storm, early measurement) + benign Deutsch–Jozsa control. Stand-ins until BadQubits data is regenerated.
- `src/rewrites.py` – SWAP→3CX, identity padding, CX-CX padding, gate decomposition, qubit relabeling, all combined.
- `src/harm.py` – harm metrics on the transpiled circuit: 2-qubit gate count, gates after first measurement.
- `src/survival.py` – transpile every (circuit, rewrite) at opt levels 0–3 on a 7-qubit `GenericBackendV2`; writes `results/survival.csv`.

Run: `pip install -r requirements.txt && cd src && python3 survival.py ../results/survival.csv`

## Pilot result (step 3, toy circuits, Qiskit 2.5.2)

Reference harm = original attack transpiled at level 0. "Kept" = ≥90% of reference.

| rewrite | harm kept (2 attacks × 4 levels) |
|---|---|
| none | 6/8 |
| swap_to_3cx | 8/8 |
| decompose | 8/8 |
| combined | 8/8 |
| identity_pad / cx_pad / relabel | 6/8 |

Findings:

1. **Transpiler already kills raw SWAP attacks at level 2+.** Qiskit's `ElidePermutations` pass deletes SWAPs (240 → 0 two-qubit gates), even with barriers. Level 2 is Qiskit's default. So some "harmful" SWAP circuits in BadQubits may be harmless on the real submission path → their labels depend on optimization level. Check this on their data.
2. **SWAP→3CX survives every level (240 2q gates).** It is both a disguise and the thing that makes the attack work at level 2+. Main evasion candidate.
3. **Padding (G·G†, CX·CX) is stripped at level 1+.** Harm survives (it was the original's), but the padding is gone, so "canonicalize before detecting" should undo padding-only evasion. Real test is whether it also normalizes SWAP→3CX (it won't: 3CX stays 3CX).
4. Early-measurement harm survives all rewrites/levels.
5. Kill-switch **not triggered**: transpiler does not undo SWAP→3CX or decomposition.

Caveats: toy circuits, one fake backend, crude harm proxies (gate counts, not
measured heating/crosstalk). Detector side (does it evade?) not tested yet.

## Next

1. Regenerate BadQubits data (MQT Bench + their generator) or email authors; rerun `survival.py` on it; count how many "harmful" labels vanish at level 2.
2. Fine-tune Laya + one small code LLM detector; measure recall on each rewrite, raw vs canonicalized (transpile L1/L3 → then detect).
3. Log detector confidence on missed rewrites (calibration angle).
4. Recheck arXiv in week 4; add compiler-resistant obfuscation (2512.19314) as the adaptive attacker.
