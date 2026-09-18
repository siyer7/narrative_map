# see markdowns/3_compositions.md
import numpy as np, matplotlib.pyplot as plt, os, itertools
from utils import plot_style; plot_style('talk')
figs_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'figs'); os.makedirs(figs_dir, exist_ok=True)

colors = {'narr1': 'green', 'narr2': 'purple', 'narr3': 'darkorange'}
def get_narrs(n_narrs):                                                                               # narr1 holds the floor, the rest are even-handed, and only narr1 starts on
    narrs = [f'narr{i + 1}' for i in range(n_narrs)]
    stay_probs = {narr: {'on': .95, 'off': .8} if i == 0 else {'on': .93, 'off': .93} for i, narr in enumerate(narrs)}
    return narrs, stay_probs, {narr: i == 0 for i, narr in enumerate(narrs)}
narrs, stay_probs, starts_on = get_narrs(3)


### which narrs are on: a markov chain over the sets of narrs that can be on together, one step per timestep. a set always holds at least one narr, so the story never goes silent
def get_on(narrs, stay_probs, starts_on, n_time=100, min_run=4, seed=3):
    rng = np.random.default_rng(seed)
    def get_weight(narr, is_on, was_on, run):                                                         # each narr carries its own tendency to persist, and its run has to last min_run before it may switch
        if run < min_run: return float(is_on == was_on)
        stays = stay_probs[narr]['on' if was_on else 'off']; return stays if is_on == was_on else 1 - stays
    on = {narr: [starts_on[narr]] for narr in narrs}; run = {narr: 1 for narr in narrs}
    for _ in range(n_time - 1):
        sets = [s for s in itertools.product([True, False], repeat=len(narrs)) if any(s)]             # every set but the silent one
        weights = np.array([np.prod([get_weight(narr, is_on, on[narr][-1], run[narr]) for narr, is_on in zip(narrs, s)]) for s in sets])
        for narr, is_on in zip(narrs, sets[rng.choice(len(sets), p=weights / weights.sum())]):
            run[narr] = run[narr] + 1 if is_on == on[narr][-1] else 1; on[narr].append(is_on)
    return {narr: np.array(is_on) for narr, is_on in on.items()}


### compositions: a narr's match is 0 while it is off and 1/sqrt(n_narrs_on) while it is on, so the match moves when that narr switches, and also when it holds while the count changes. the composition at a switch is the set of narrs whose match moved
on = get_on(narrs, stay_probs, starts_on); n_time = len(on['narr1'])
n_narrs_on = sum(on[narr].astype(int) for narr in narrs)
match = {narr: np.where(on[narr], 1 / np.sqrt(n_narrs_on), 0) for narr in narrs}
moved = {narr: np.abs(np.diff(match[narr])) > 1e-9 for narr in narrs}
compositions = [frozenset(narr for narr in narrs if moved[narr][t]) for t in range(n_time - 1)]
compositions = [c for c in compositions if c]                                                         # the empty one is not a composition: it is a timestep where nothing switched

# the combos a composition can take: every subset of the narrs, less the empty one (nothing switched) and less the singletons (a narr never moves alone, since a switch changes the count for everyone still on)
num_compositional_narrs = 2 ** len(narrs) - 1 - len(narrs)
possible = [frozenset(c) for k in range(2, len(narrs) + 1) for c in itertools.combinations(narrs, k)]
assert len(possible) == num_compositional_narrs
assert all(len(c) > 1 for c in compositions) and set(compositions) <= set(possible)
print(f'{len(compositions)} compositions over {n_time} timesteps, taking {len(set(compositions))} of the {num_compositional_narrs} possible combos')
for c in possible: print(f'  {sorted(c)}: {compositions.count(c)}')


### plot: how the number of possible compositions grows with n_narrs, and the map the story traces through them
fig, ax = plt.subplots(1, 2, figsize=(14.5, 6.5)); fig.suptitle('Compositional Advantage', fontsize=19, fontweight='bold')
ax[0].set(xlabel='number of narratives', ylabel='count')
n_range = np.arange(2, 6)
ax[0].plot(n_range, 2 ** n_range - 1 - n_range, 'o-', color='crimson', label='# compositional events:  $2^n\\!-\\!1\\!-\\!n$')
n_seeds = 20                                                                                          # a bayesian surprisal event is a timestep where any narr switches on, so narrs coming on together make one event, not two. it is a count of what happened, so it is averaged over seeds and carries the standard error of that mean
peaks_by_seed = np.array([[np.any([np.diff(o.astype(int)) == 1 for o in get_on(*get_narrs(n), seed=seed).values()], axis=0).sum() for seed in range(n_seeds)] for n in n_range])
bayesian_peaks, bayesian_err = peaks_by_seed.mean(axis=1), peaks_by_seed.std(axis=1) / np.sqrt(n_seeds)
ax[0].errorbar(n_range, bayesian_peaks, yerr=bayesian_err, fmt='o-', color='C0', capsize=4, label='# bayesian surprisal events')   # the same blue bayesian surprisal is drawn in 2_surprisal_vs_discontinuity
legend = ax[0].legend(handlelength=0, handletextpad=0)                                                # no marker in the box: each entry is just its line's colour
for entry, color in zip(legend.get_texts(), ['crimson', 'C0']): entry.set_color(color)
for artist in legend.findobj(lambda a: a.__class__.__name__ in ('Line2D', 'LineCollection')): artist.set_visible(False)   # the errorbar's marker is its own artist, so hide every line in the box rather than just the handles
ax[0].set_xticks(n_range)

# the cube: one axis per narr, each 0 or 1 for whether that narr is in the composition, so the 2**n_narrs corners are every subset. a composition cannot sit on the origin (nothing switched) nor on its 3 neighbours (one narr alone), leaving the corners at least 2 edges out
ax[1].remove(); ax[1] = fig.add_subplot(1, 2, 2, projection='3d')
ax[1].set(xticks=[], yticks=[], zticks=[], xlim=(0, 1.3), ylim=(0, 1.3), zlim=(0, 1.3))
ax[1].view_init(elev=18, azim=-60); ax[1].set_box_aspect((1, 1, 1), zoom=1.08); ax[1].set_axis_off()
for i, narr in enumerate(narrs):                                                                      # an arrow out along each narr's axis, from 0 to 1, named at its tip so the name sits on the arrow rather than on a back spine
    ax[1].quiver(0, 0, 0, *(1.26 * np.eye(3)[i]), color=colors[narr], arrow_length_ratio=.09, lw=2)
    ax[1].text(*(1.36 * np.eye(3)[i]), narr, color=colors[narr], fontsize=13, ha='left' if i == 0 else 'right', va='center')   # narr1's arrow points right so its name sits right of the tip; narr2 and narr3 take theirs on the left
corners = list(itertools.product([0, 1], repeat=len(narrs)))
for a, b in itertools.combinations(corners, 2):                                                       # the cube's edges: corners one narr apart
    if sum(abs(i - j) for i, j in zip(a, b)) == 1: ax[1].plot(*zip(a, b), color='lightgrey', lw=1.5)
jitter = np.random.default_rng(0)
for corner in corners:
    composition = frozenset(narr for narr, is_in in zip(narrs, corner) if is_in)
    if len(composition) < 2: continue                                                                 # the origin and the singletons are not drawn at all: a composition cannot sit there
    cloud = np.array(corner)[:, None] + .11 * jitter.normal(size=(3, compositions.count(composition))) # one jittered circle per occurrence of that composition
    for point in cloud.T: ax[1].plot(*zip(corner, point), color='lightgrey', lw=.8, alpha=.7, zorder=3)   # a faint stem back to the corner, so each circle reads at its own depth
    ax[1].scatter(*cloud, s=70, color='crimson', edgecolor='white', lw=1, depthshade=False, zorder=4)
plt.tight_layout(); plt.savefig(os.path.join(figs_dir, '3_compositions.png'), dpi=200, bbox_inches='tight'); plt.close()
