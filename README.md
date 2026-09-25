# Harm-preserving rewrites vs quantum harmful-circuit detectors

Question: which rewrites (1) still harm hardware after the provider's
transpiler and (2) evade BadQubits-style detectors? Does canonicalizing
(transpiling) before detection close the gap?

Idea 1 (Qiskit version prediction) is the fallback: a "run under old version,
check deprecation warnings" baseline likely beats ML there.

## Layout

- `src/dataset.py` – BadQubits-style dataset: 317 benign MQT Bench circuits (30 families, 3–16 qubits) + 634 harmful (each benign circuit with an injected SWAP storm or an early-measurement block). Approximates their generator from its description; flattened, not transpiled. → `data/dataset.jsonl`
- `src/rewrites.py` – SWAP→3CX, identity padding, CX·CX padding, decompose, qubit relabel, all combined.
- `src/features.py` – for every (circuit, rewrite): transpile at levels 0–3 on a 16-qubit `GenericBackendV2`, store gate tokens (raw / L1 / L3) and harm metrics (extra 2q gates vs benign base; mid-circuit measurements). → `data/views.json` (not committed, ~10 min to rebuild)
- `src/detectors.py` – `ngram_lr` (gate 1–4-gram TF-IDF + logistic regression, cheap text-model proxy), `cnn` (1D CNN like BadQubits' baseline), `feat_gb` (gradient boosting on structural counts).
- `src/experiment.py` – 3 group splits (70/30 by base circuit) × 3 views × 3 detectors × 7 rewrites; plus leave-one-rewrite-out augmented training. → `results/detection.csv`, `results/harm_survival.csv`
- `src/summarize.py` → `results/summary.md` (all tables)
- `src/survival.py`, `src/attacks.py` – earlier toy pilot.

Run: `pip install -r requirements.txt && cd src && python3 dataset.py && python3 features.py && python3 experiment.py 3 && python3 summarize.py > ../results/summary.md`

Views: `raw` = detector sees circuit as submitted, provider runs it at level 2 (Qiskit default). `L1`/`L3` = transpile first, detect on that, run at that level. Evasion only counts for attacks whose harm survives at the run level (≥90% of the level-0 harm).

## Results (mean of 3 seeds; seed std ≤ 0.03)

### 1. Half the "harmful" labels are harmless at Qiskit's default level

| attack | rewrite | harm kept L0 | L1 | L2 | L3 |
|---|---|---|---|---|---|
| swap_storm | none / padding / relabel | 1.00 | 1.00 | **0.00** | **0.00** |
| swap_storm | swap_to_3cx / decompose / combined | 1.00 | 1.00 | 0.98 | 0.98 |
| early_measure | any | 1.00 | 0.96 | 0.96 | 0.96 |

`ElidePermutations` (level 2+) deletes SWAPs even with barriers. A BadQubits-style
dataset that labels SWAP-heavy circuits "harmful" is mislabeled for any provider
running level 2+. Only SWAP→3CX or decomposition makes these attacks work there.

### 2. The rewrite that makes the attack work is also the one that evades

Recall on attacks whose harm survives (all rewrites):

| detector | view | none | swap→3cx | id_pad | cx_pad | decompose | relabel | combined |
|---|---|---|---|---|---|---|---|---|
| cnn | raw | 1.00 | **0.49** | 0.97 | 0.97 | **0.49** | 1.00 | **0.44** |
| feat_gb | raw | 1.00 | **0.49** | 1.00 | 1.00 | **0.49** | 1.00 | **0.49** |
| ngram_lr | raw | 0.73 | 0.32 | 0.58 | 0.65 | 0.32 | 0.73 | 0.16 |
| cnn | L3 | 1.00 | 0.49 | 0.99 | 1.00 | 0.49 | 1.00 | 0.49 |
| feat_gb | L3 | 1.00 | 0.49 | 0.99 | 1.00 | 0.49 | 1.00 | 0.49 |
| cnn | **L1** | 0.94 | **0.94** | 0.94 | 0.94 | **0.94** | 0.94 | **0.94** |
| feat_gb | **L1** | 1.00 | **1.00** | 1.00 | 1.00 | **1.00** | 1.00 | **1.00** |
| ngram_lr | **L1** | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 | 0.90 | 0.90 |

The ~0.49 is entirely SWAP storms: surviving SWAP-storm recall is **0.00** for every
detector in raw and L3 views; early-measure recall stays ~1.0.

- **Confident but wrong:** CNN and feat_gb give p(harmful) < 0.1 to 51% of surviving attacks under swap→3cx (raw view). Not borderline misses.
- **Padding and relabeling don't matter**: detectors catch them, and the transpiler strips padding anyway.
- **L3 canonicalization does not help**: at L3 the native SWAP storms are harmless, so training has no example of a working SWAP storm → detector is blind to the rewritten ones.
- **L1 canonicalization closes the gap**: L1 translates SWAP→CX but doesn't elide permutations, so native and rewritten attacks look the same. Recall 0.90–1.00 on every rewrite, FPR ≤ 0.05.

### 3. Augmentation is weaker than canonicalization

Leave-one-rewrite-out augmented training (train on the other rewrites):
feat_gb raw recovers single rewrites (0.98) but not `combined` (0.49); ngram_lr raw
stays 0.41–0.53 on swap→3cx / padding / combined. L1 canonicalization beats it
with no augmentation.

### Other
- Clean (no rewrite) AUC 0.99–1.00 for all; calibration ECE 0.00–0.16 (ngram_lr worst).
- Deutsch–Jozsa FPR 0.00: MQT's DJ measures at the end, so BadQubits' 15/21 DJ false-positive is not reproduced by this dataset.

## Limits
- Attack generator is my reconstruction, not BadQubits' code. Only 2 attack types.
- Harm = gate-count / mid-measure proxies on a fake backend, not measured heating or crosstalk.
- **LLM detectors (Laya, small code LLM) not run**: `huggingface.co` is blocked by this environment's network policy, and there is no GPU. `ngram_lr` is a weak stand-in.
- Split is by base circuit, not by family; family-held-out would be harder.
- Adaptive attacker vs L1 canonicalization (e.g. compiler-resistant obfuscation, 2512.19314) not tested.

## Next
1. Get BadQubits' code/data (email authors) → re-run; check how many of *their* harmful labels die at level 2.
2. Allow huggingface.co (+ GPU env) → fine-tune Laya + a small code LLM, same pipeline.
3. Adaptive attacker against L1 canonicalization: rewrites that survive L1 but differ from native form (e.g. SWAP as CX-H-CZ-H mixes, routing-dependent layouts, obfuscation from 2512.19314).
4. Family-held-out split; more attack types.
5. Recheck arXiv in week 4.
