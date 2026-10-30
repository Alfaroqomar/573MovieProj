"""Neural Collaborative Filtering (NeuMF, He et al. 2017) — deep-learning baseline.

Role in the project (answers the proposal feedback): NeuMF is an
INDEPENDENT baseline. It produces its own top-10 lists, scored with the same
metrics on the same users as every other model. It does not add candidates
to the Stage 1 shortlist or to the LLM stage.

Model:
* GMF branch: user and movie vectors of size NEUMF_GMF_DIM, multiplied
  element-wise (generalised matrix factorisation).
* MLP branch: separate user and movie vectors of size layers[0] / 2 each,
  concatenated and passed through ReLU layers NEUMF_MLP_LAYERS
  (default 64 -> 32 -> 16).
* Output: one linear layer over [GMF output, MLP output] and a sigmoid, the
  estimated chance that the user has watched (rated) the movie.

Training (as in the paper): every training rating is a positive example;
for each positive, NEUMF_NEGATIVES (default 4) random movies are drawn as
negatives, fresh every round. Loss: binary cross-entropy. Optimiser: Adam
(learning rate NEUMF_LEARNING_RATE). After each round the model is scored on
500 validation users (NDCG@10); training stops when that has not improved
for NEUMF_PATIENCE rounds and the best round is kept.

Requires PyTorch (optional install). Uses an Apple Silicon GPU (MPS) or an
NVIDIA GPU when available, otherwise the CPU.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from moviematch.models.common import top_k

Log = Callable[[str], None]


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def pick_device(preference: str):
    import torch

    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_network(n_users: int, n_items: int, gmf_dim: int, layers: tuple[int, ...]):
    import torch
    from torch import nn

    class NeuMF(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            half = layers[0] // 2
            self.user_gmf = nn.Embedding(n_users, gmf_dim)
            self.item_gmf = nn.Embedding(n_items, gmf_dim)
            self.user_mlp = nn.Embedding(n_users, half)
            self.item_mlp = nn.Embedding(n_items, half)
            mlp: list[nn.Module] = []
            width = half * 2
            for size in layers[1:]:
                mlp += [nn.Linear(width, size), nn.ReLU()]
                width = size
            self.mlp = nn.Sequential(*mlp)
            self.out = nn.Linear(gmf_dim + width, 1)
            for emb in (self.user_gmf, self.item_gmf, self.user_mlp, self.item_mlp):
                nn.init.normal_(emb.weight, std=0.01)

        def forward(self, users, items):
            gmf = self.user_gmf(users) * self.item_gmf(items)
            mlp = self.mlp(torch.cat([self.user_mlp(users), self.item_mlp(items)], dim=-1))
            return self.out(torch.cat([gmf, mlp], dim=-1)).squeeze(-1)  # logits

        def score_all_items(self, user: int, device):
            """Logit for every movie for one user."""
            n = self.item_gmf.num_embeddings
            items = torch.arange(n, device=device)
            users = torch.full((n,), user, dtype=torch.long, device=device)
            return self.forward(users, items)

    return NeuMF()


def _validation_ndcg(net, device, users, train: sp.csr_matrix, relevant: dict[int, set[int]], k: int = 10) -> float:
    import torch

    net.eval()
    total = 0.0
    with torch.no_grad():
        for u in users:
            scores = net.score_all_items(int(u), device).float().cpu().numpy()
            seen = train.indices[train.indptr[u] : train.indptr[u + 1]]
            ranked = top_k(scores, k, exclude=seen)
            rel = relevant[int(u)]
            hits = np.array([int(i) in rel for i in ranked])
            dcg = float((hits / np.log2(np.arange(2, len(ranked) + 2))).sum())
            ideal = float((1 / np.log2(np.arange(2, min(len(rel), k) + 2))).sum())
            total += dcg / ideal if ideal else 0.0
    net.train()
    return total / max(len(users), 1)


def fit_neumf(
    train: sp.csr_matrix,
    validation_relevant: dict[int, set[int]],
    *,
    gmf_dim: int,
    layers: tuple[int, ...],
    negatives: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    max_train_ratings: int,
    device_preference: str,
    seed: int,
    log: Log = print,
):
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = pick_device(device_preference)
    n_users, n_items = train.shape
    coo = train.tocoo()
    pos_u = coo.row.astype(np.int32)
    pos_i = coo.col.astype(np.int32)
    del coo
    if max_train_ratings and len(pos_u) > max_train_ratings:
        keep = rng.choice(len(pos_u), size=max_train_ratings, replace=False)
        pos_u, pos_i = pos_u[keep], pos_i[keep]
    # Negatives are drawn from movies that have at least one training rating.
    candidates = np.flatnonzero(np.diff(train.tocsc().indptr) > 0).astype(np.int32)

    net = build_network(n_users, n_items, gmf_dim, layers).to(device)
    optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    val_users = np.array(sorted(validation_relevant))
    if len(val_users) > 500:
        val_users = np.sort(rng.choice(val_users, size=500, replace=False))

    log(
        f"  NeuMF on {device.type}: {len(pos_u):,} positives x (1 + {negatives} negatives) per round, "
        f"{n_users:,} users, {n_items:,} movies"
    )
    best, best_state, best_epoch, waited, history = -1.0, None, 0, 0, []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        neg_u = np.repeat(pos_u, negatives)
        neg_i = candidates[rng.integers(0, len(candidates), size=len(neg_u))]
        users = np.concatenate([pos_u, neg_u])
        items = np.concatenate([pos_i, neg_i])
        labels = np.concatenate([np.ones(len(pos_u), np.float32), np.zeros(len(neg_u), np.float32)])
        del neg_u, neg_i
        order = rng.permutation(len(users)).astype(np.int64)
        users = users[order]
        items = items[order]
        labels = labels[order]
        del order
        users_t = torch.from_numpy(users)
        items_t = torch.from_numpy(items)
        labels_t = torch.from_numpy(labels)
        running = 0.0
        steps = 0
        for s in range(0, len(users_t), batch_size):
            u = users_t[s : s + batch_size].to(device, non_blocking=True).long()
            i = items_t[s : s + batch_size].to(device, non_blocking=True).long()
            y = labels_t[s : s + batch_size].to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            loss = loss_fn(net(u, i), y)
            loss.backward()
            optimiser.step()
            if steps % 200 == 0:
                running += float(loss.detach()) * 200
            steps += 1
        del users_t, items_t, labels_t, users, items, labels
        ndcg = _validation_ndcg(net, device, val_users, train, validation_relevant) if len(val_users) else 0.0
        seconds = time.perf_counter() - started
        history.append({"epoch": epoch, "loss": running / max(steps, 1), "validationNdcg@10": ndcg, "seconds": round(seconds, 1)})
        log(f"    round {epoch}: loss {running / max(steps, 1):.4f}, validation NDCG@10 {ndcg:.4f} ({seconds:.0f}s)")
        if not len(val_users):
            # No held-out users to stop on (e.g. trained on train + validation):
            # keep the latest round and run every round.
            best, best_epoch = ndcg, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        elif ndcg > best:
            best, best_epoch, waited = ndcg, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                log(f"    stopping early: no improvement for {patience} rounds")
                break
    return best_state, {
        "bestEpoch": best_epoch,
        "bestValidationNdcg@10": best,
        "history": history,
        "device": device.type,
        "positivesPerRound": int(len(pos_u)),
    }


def save_neumf(state, path: Path, meta_path: Path, meta: dict) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    meta_path.write_text(json.dumps(meta, indent=2))


class NeuMFModel:
    """Inference wrapper (CPU). Users with no training ratings are 'unknown'."""

    def __init__(self, net, known_users: np.ndarray, trained_items: np.ndarray):
        self.net = net
        self.known_users = known_users
        self.trained_items = trained_items

    @classmethod
    def load(cls, path: Path, meta_path: Path) -> NeuMFModel:
        import torch

        meta = json.loads(meta_path.read_text())
        net = build_network(meta["nUsers"], meta["nItems"], meta["gmfDim"], tuple(meta["layers"]))
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        net.eval()
        known = np.zeros(meta["nUsers"], dtype=bool)
        trained = np.zeros(meta["nItems"], dtype=bool)
        with np.load(path.with_suffix(".support.npz")) as data:
            known[data["users"]] = True
            trained[data["items"]] = True
        return cls(net, known, trained)

    def knows(self, user_idx: int) -> bool:
        return bool(self.known_users[user_idx])

    def scores(self, user_idx: int) -> np.ndarray:
        import torch

        with torch.no_grad():
            out = self.net.score_all_items(user_idx, torch.device("cpu")).numpy().astype(np.float64)
        out[~self.trained_items] = -np.inf
        return out
