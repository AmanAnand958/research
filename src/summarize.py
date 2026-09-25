"""Print markdown tables from results/*.csv (mean over split seeds)."""
import pandas as pd

d = pd.read_csv("../results/detection.csv")
s = pd.read_csv("../results/harm_survival.csv")
ORDER = ["none", "swap_to_3cx", "identity_pad", "cx_pad", "decompose", "relabel", "combined"]


def md(df):
    return df.to_markdown(floatfmt=".2f")


print("## Harm kept after transpiling (fraction of attacks)\n")
print(md(s.pivot_table(index=["attack", "rewrite"], columns="opt_level", values="harm_kept")))

c = d[d.variant == "clean"]
print("\n## Clean test (no rewrite)\n")
print(md(c[c.rewrite == "none"].groupby(["detector", "view"])[["auc", "recall", "fpr", "ece", "dj_fpr"]].mean()))

for metric in ["recall", "recall_swap_storm", "recall_early_measure", "conf_miss", "fpr"]:
    print(f"\n## {metric} by rewrite (clean-trained; only attacks whose harm survives)\n")
    t = c.pivot_table(index=["detector", "view"], columns="rewrite", values=metric)[ORDER]
    print(md(t))

a = d[d.variant == "aug_loro"]
print("\n## recall, leave-one-rewrite-out augmented training\n")
print(md(a.pivot_table(index=["detector", "view"], columns="rewrite", values="recall")[ORDER[1:]]))
print("\n## recall std across seeds (clean)\n")
print(md(c.pivot_table(index=["detector", "view"], columns="rewrite", values="recall", aggfunc="std")[ORDER]))
