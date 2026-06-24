"""
Phase 1: train the keyframe agent.
  - predictor: rolls forward (autoregressively) from the last stored keyframe
  - policy:    decides STORE vs SKIP at each frame (discrete, via straight-through)
  - loss:      #stored + lambda * recon_error
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

D = 32            # stim-vector dim
H = 64            # hidden dim
LAMBDA = 4.0      # weight on reconstruction (must dominate or it stores nothing)
LR = 1e-3


class Predictor(nn.Module):
    """Given a running hidden state, predict the next frame."""
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRUCell(D, H)
        self.out = nn.Linear(H, D)

    def step(self, x, h):          # one autoregressive step
        h = self.rnn(x, h)
        return self.out(h), h


class Policy(nn.Module):
    """Decide STORE vs SKIP from (actual, predicted, drift)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * D + 1, H), nn.ReLU(),
            nn.Linear(H, 2)          # logits: [skip, store]
        )

    def forward(self, actual, pred, drift):
        feat = torch.cat([actual, pred, drift.unsqueeze(0)], dim=-1)
        return self.net(feat)


def run_video(video, predictor, policy, tau=1.0, train=True):
    """
    Walk one video frame by frame.
    Returns total loss, #stored (soft), reconstruction list.
    `video`: tensor [T, D]
    """
    T = video.shape[0]
    h = torch.zeros(H)
    recon = []
    store_mass = 0.0               # soft count of stored frames
    gates = []                     # per-frame store decision (for plots)

    # always store the first frame to seed the rollout
    cur = video[0]
    _, h = predictor.step(cur, h)
    recon.append(video[0])
    anchor_h = h                   # hidden state at last stored frame

    for t in range(1, T):
        pred, h = predictor.step(cur, h)        # autoregressive: feeds last frame
        actual = video[t]
        drift = F.mse_loss(pred, actual).detach()

        logits = policy(actual, pred, drift)
        # straight-through Gumbel: hard one-hot forward, soft gradient backward
        gate = F.gumbel_softmax(logits, tau=tau, hard=True)   # [skip, store]
        store = gate[1]

        # if STORE -> next input is the real frame, reset hidden to anchor lineage
        # if SKIP  -> next input is the prediction (true autoregressive rollout)
        cur = store * actual + (1 - store) * pred
        recon.append(cur)
        store_mass = store_mass + store
        gates.append(store.detach().item())

    recon = torch.stack(recon)
    recon_error = F.mse_loss(recon, video)
    frac_stored = store_mass / T                  # normalized count (0-1)
    loss = frac_stored + LAMBDA * recon_error
    return loss, store_mass, recon, gates


def train(train_videos, epochs=20, eval_fn=None, eval_every=5):
    predictor = Predictor()
    policy = Policy()
    opt = torch.optim.Adam(list(predictor.parameters()) + list(policy.parameters()), lr=LR)

    hist = {"epoch": [], "loss": [], "stored": [], "recon": []}

    for epoch in range(epochs):
        # anneal Gumbel temperature: soft -> sharp over training
        tau = max(0.5, 1.0 * (0.95 ** epoch))

        ep_loss = ep_stored = ep_recon = 0.0
        for video in train_videos:
            loss, n_stored, recon, _ = run_video(video, predictor, policy, tau=tau, train=True)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss   += loss.item()
            ep_stored += n_stored.item()
            ep_recon  += F.mse_loss(recon, video).item()

        n = len(train_videos)
        hist["epoch"].append(epoch)
        hist["loss"].append(ep_loss / n)
        hist["stored"].append(ep_stored / n)
        hist["recon"].append(ep_recon / n)

        # periodic frozen eval on held-out set (Phase 2)
        if eval_fn is not None and epoch % eval_every == 0:
            eval_fn(predictor, policy, epoch)

    return predictor, policy, hist


def plot_results(hist, predictor, policy, sample_video, true_boundaries):
    """Two panels: training curves, and store-decisions vs true boundaries."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    # 1: keyframe count tapering over training
    ax[0].plot(hist["epoch"], hist["stored"], color="tab:blue")
    ax[0].set_title("Keyframes stored over training")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("# stored (per video)")

    # 2: loss + recon error
    ax[1].plot(hist["epoch"], hist["loss"], label="total loss", color="tab:red")
    ax[1].plot(hist["epoch"], hist["recon"], label="recon error", color="tab:orange")
    ax[1].set_title("Loss / reconstruction"); ax[1].set_xlabel("epoch"); ax[1].legend()

    # 3: where keyframes land vs true event onsets
    with torch.no_grad():
        _, _, _, gates = run_video(sample_video, predictor, policy, tau=0.5, train=False)
    ax[2].bar(range(len(gates)), gates, color="tab:blue", label="store decision")
    for b in true_boundaries:
        ax[2].axvline(b, color="k", linestyle="--", alpha=0.6)
    ax[2].axvline(true_boundaries[0], color="k", linestyle="--", alpha=0.6, label="true onset")
    ax[2].set_title("Keyframes vs true boundaries")
    ax[2].set_xlabel("frame"); ax[2].set_ylabel("stored"); ax[2].legend()

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/phase1_results.png", dpi=120)
    print("saved plot -> phase1_results.png")


if __name__ == "__main__":
    N_EVENTS, EV_LEN = 10, 10
    # boundaries = first frame of each event
    BOUNDARIES = [i * EV_LEN for i in range(N_EVENTS)]

    # toy synthetic data: 10 events x 10 frames, autoreg within event
    def make_video(n_events=N_EVENTS, ev_len=EV_LEN):
        frames = []
        for _ in range(n_events):
            x = torch.randn(D)                 # event seed (boundary = fresh draw)
            for _ in range(ev_len):
                x = 0.9 * x + 0.1 * torch.randn(D)   # autoreg within event
                frames.append(x.clone())
        return torch.stack(frames)

    train_videos = [make_video() for _ in range(5)]

    def quick_eval(predictor, policy, epoch):
        with torch.no_grad():
            v = make_video()
            _, n_stored, _, _ = run_video(v, predictor, policy, tau=0.5, train=False)
            print(f"epoch {epoch:3d}  held-out stored ~ {n_stored.item():.1f} / {v.shape[0]}")

    predictor, policy, hist = train(train_videos, epochs=200, eval_fn=quick_eval, eval_every=20)

    # print actual stored indices vs boundaries to check lag direction
    with torch.no_grad():
        v = make_video()
        _, _, _, gates = run_video(v, predictor, policy, tau=0.5, train=False)
    # gates[i] corresponds to frame index i+1 (frame 0 is force-stored, not in gates)
    stored_idx = [0] + [i + 1 for i, g in enumerate(gates) if g > 0.5]
    print("\nboundaries :", BOUNDARIES)
    print("stored idx :", stored_idx)
    print("\nnearest boundary -> stored frame (lag = stored - boundary):")
    for b in BOUNDARIES:
        nearest = min(stored_idx, key=lambda s: abs(s - b))
        print(f"  boundary {b:3d} -> stored {nearest:3d}  (lag {nearest - b:+d})")

    plot_results(hist, predictor, policy, v, BOUNDARIES)
