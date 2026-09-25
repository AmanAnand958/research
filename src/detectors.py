"""Three detector types. All map a gate-token string -> P(harmful).

- ngram_lr: TF-IDF over gate-name 1-4 grams + logistic regression. Cheap
  text-model proxy (the LLM detectors need HuggingFace weights, unavailable here).
- cnn: 1D CNN over gate tokens, like BadQubits' CNN baseline.
- feat_gb: gradient boosting on structural counts (2q gates, SWAPs,
  mid-circuit measures, ...). Sees 'what the circuit does', not its text.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

TWO_Q = {"cx", "cz", "cy", "ch", "cp", "crx", "cry", "crz", "cu", "swap", "rzz", "rxx", "ryy", "ecr"}


class NgramLR:
    def fit(self, X, y):
        self.m = make_pipeline(TfidfVectorizer(ngram_range=(1, 4), token_pattern=r"\S+", min_df=2,
                                               sublinear_tf=True, max_features=50000),
                               LogisticRegression(C=10, max_iter=3000, class_weight="balanced"))
        self.m.fit(X, y)
        return self

    def proba(self, X):
        return self.m.predict_proba(X)[:, 1]


def struct_feats(s):
    t = s.split()
    n = max(len(t), 1)
    two = sum(x in TWO_Q for x in t)
    swaps = t.count("swap")
    meas = [k for k, x in enumerate(t) if x == "measure"]
    first_m = meas[0] / n if meas else 1.0
    after_m = (n - meas[0] - len(meas)) / n if meas else 0.0
    # longest run of 2q gates (a SWAP/CX storm shows up as a long run)
    run = best = 0
    for x in t:
        run = run + 1 if x in TWO_Q else 0
        best = max(best, run)
    return [n, two, two / n, swaps, swaps / n, len(meas), first_m, after_m, best, best / n,
            t.count("cx") / n, t.count("h") / n]


class FeatGB:
    def fit(self, X, y):
        self.m = GradientBoostingClassifier(random_state=0).fit([struct_feats(s) for s in X], y)
        return self

    def proba(self, X):
        return self.m.predict_proba([struct_feats(s) for s in X])[:, 1]


class _Net(nn.Module):
    def __init__(self, vocab, emb=32, ch=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb, ch, k, padding=k // 2) for k in (3, 5, 9)])
        self.out = nn.Linear(3 * ch, 1)

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)
        h = torch.cat([torch.relu(c(e)).max(dim=2).values for c in self.convs], 1)
        return self.out(h).squeeze(1)


class CNN:
    MAXLEN = 4096

    def __init__(self, epochs=12, seed=0):
        self.epochs, self.seed = epochs, seed

    def _enc(self, X):
        arr = np.zeros((len(X), self.MAXLEN), dtype=np.int64)
        for i, s in enumerate(X):
            ids = [self.vocab.get(t, 1) for t in s.split()[: self.MAXLEN]]
            arr[i, : len(ids)] = ids
        return torch.from_numpy(arr)

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        toks = sorted({t for s in X for t in s.split()})
        self.vocab = {t: i + 2 for i, t in enumerate(toks)}  # 0 pad, 1 unk
        self.net = _Net(len(self.vocab) + 2)
        Xe, ye = self._enc(X), torch.tensor(y, dtype=torch.float32)
        pos = ye.mean().item()
        lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos))
        opt = torch.optim.Adam(self.net.parameters(), lr=2e-3)
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            self.net.train()
            for b in torch.randperm(len(Xe), generator=g).split(32):
                opt.zero_grad()
                lossf(self.net(Xe[b]), ye[b]).backward()
                opt.step()
        return self

    @torch.no_grad()
    def proba(self, X):
        self.net.eval()
        Xe = self._enc(X)
        return torch.cat([torch.sigmoid(self.net(Xe[b])) for b in torch.arange(len(Xe)).split(64)]).numpy()


DETECTORS = {"ngram_lr": NgramLR, "cnn": CNN, "feat_gb": FeatGB}
