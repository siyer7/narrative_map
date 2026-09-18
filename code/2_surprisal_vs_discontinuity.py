# see markdowns/2_surprisal_vs_discontinuity.md
import numpy as np, matplotlib.pyplot as plt, os, itertools
from matplotlib.colors import LinearSegmentedColormap
from utils import plot_style; plot_style('notebook')
figs_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'figs'); os.makedirs(figs_dir, exist_ok=True)

# each timestep is a 15-dim embedding: narr1 features 1-5, narr2 features 6-10, narr3 features 11-15
n_features, n_time = 15, 100
narrs = {'narr1': slice(0, 5), 'narr2': slice(5, 10), 'narr3': slice(10, 15)}
stay_probs = {'narr1': {'on': .95, 'off': .8}, 'narr2': {'on': .93, 'off': .93}, 'narr3': {'on': .93, 'off': .93}}   # p(stay on | on), p(stay off | off): narr1 holds the floor most of the time, and all are high so narrs switch rarely
starts_on = {'narr1': True, 'narr2': False, 'narr3': False}
colors = {'narr1': 'green', 'narr2': 'purple', 'narr3': 'darkorange'}                                      # narr1 green, narr2 purple, narr3 orange, everywhere they appear
mean_feature_value = 1                                                                                    # every narr shares one value, so none reads as brighter than the others; the matrix shades against 1.5 so the wobble around 1 does not clip flat


### inputs
def make_input(n_time=n_time, stay_probs=stay_probs, starts_on=starts_on, mean_feature_value=mean_feature_value, min_run=4, narr_on_feat_noise=0, narr_off_feat_noise=0, seed=0):
    rng = np.random.default_rng(seed)
    # which narrs are on: a markov chain over the sets of narrs that can be on together, one step per timestep. a set always holds at least one narr, so the story never goes silent
    def get_weight(narr, is_on, was_on, run):                                                         # each narr carries its own tendency to persist, and its run has to last min_run before it may switch
        if run < min_run: return float(is_on == was_on)
        stays = stay_probs[narr]['on' if was_on else 'off']; return stays if is_on == was_on else 1 - stays
    on = {narr: [starts_on[narr]] for narr in narrs}; run = {narr: 1 for narr in narrs}
    for _ in range(n_time - 1):
        sets = [s for s in itertools.product([True, False], repeat=len(narrs)) if any(s)]             # every set but the silent one, so 2**n_narrs - 1 of them
        assert len(sets) == 2 ** len(narrs) - 1
        weights = np.array([np.prod([get_weight(narr, is_on, on[narr][-1], run[narr]) for narr, is_on in zip(narrs, s)]) for s in sets])
        for narr, is_on in zip(narrs, sets[rng.choice(len(sets), p=weights / weights.sum())]):
            run[narr] = run[narr] + 1 if is_on == on[narr][-1] else 1; on[narr].append(is_on)
    on = {narr: np.array(is_on) for narr, is_on in on.items()}
    # the input: each narr owns 5 features, which carry the mean feature value while it is on and 0 while it is off
    inputs = rng.normal(0, narr_off_feat_noise, (n_time, n_features))
    for narr, features in narrs.items():
        for t in np.flatnonzero(on[narr]): inputs[t, features] = rng.normal(mean_feature_value, narr_on_feat_noise, 5)
    return inputs, on


### metrics, read off the input, not off which narrs are on: how much a narr's features increased since the previous timestep, summed over its 5 features, read as the prob that the narr resumed; defined from t = 1
def prob_narr_resumed(inputs, features): return np.maximum(np.diff(inputs[:, features], axis=0), 0).sum(axis=1)

inputs, on = make_input(seed=3)
bayesian_surprisal = sum(prob_narr_resumed(inputs, features) for features in narrs.values())          # big at every event start, i.e. any narr coming on

# a narr's template: the shared value on its own features, 0 everywhere else. cosine to the template says how much the input looks like that narr right now: a level, not a change, so it is high all through the narr's events and low in the gaps
narr_template = {narr: np.zeros(n_features) for narr in narrs}
for narr, features in narrs.items(): narr_template[narr][features] = mean_feature_value
def get_cos(inputs, template): return inputs @ template / (np.linalg.norm(inputs, axis=1) * np.linalg.norm(template))

# projection: how far the input reaches along the template's direction. it is the cosine's numerator, before the input's norm divides it, so it keeps the magnitude (~0 when the narr's features are ~0) and the other narrs never enter it
narr_direction = {narr: narr_template[narr] / np.linalg.norm(narr_template[narr]) for narr in narrs}
narrative_continuity = {narr: inputs @ narr_direction[narr] for narr in narrs}


cos_vs_input = {narr: get_cos(inputs, narr_template[narr]) for narr in narrs}                    # normalised by the whole 15-dim input, so another narr coming on pulls this one down

commit_threshold = .5                                                                                 # at .5 the projection and the cosine pick out the same timesteps, but only while some narr is on
cos_diff = {narr: np.diff(cos_vs_input[narr]) for narr in narrs}                                 # the change in the cosine, which moves for a narr whenever any narr switches, since they all sit in its denominator
narrative_discontinuity = {narr: np.abs(cos_diff[narr]) > 1e-9 for narr in narrs}  # where the narr's cosine moves at all: one boolean per timestep per narr, so a switch spikes every narr it touches, not only the one that switched

continuity_above_threshold = {narr: narrative_continuity[narr] > commit_threshold for narr in narrs}
assert np.any([on[narr] for narr in narrs], axis=0).all()                                             # the joint chain guarantees it: at least one narr is on at every timestep, so the input is never all zeros and the cosine never divides by 0
for narr in narrs: assert (continuity_above_threshold[narr] == (cos_vs_input[narr] > commit_threshold)).all()
for narr in narrs: assert (continuity_above_threshold[narr] == on[narr]).all()                    # the commits raster recovers exactly the timesteps the narr is on, read off the input alone
n_narrs_spiking = sum(narrative_discontinuity[narr].astype(int) for narr in narrs)               # a switch changes the input's norm, so it moves the cosine of every narr that is on either side of it, and the no-silence rule means there is always at least one besides the one that switched
num_compositional_narrs = 2 ** len(narrs) - 1 - len(narrs)                                            # the combos a spike can take: every subset of the narrs, less the empty one (silence cannot happen) and less the n singletons (a narr never spikes alone), so the subsets of size 2 or more
assert ((n_narrs_spiking == 0) | (n_narrs_spiking > 1)).all()
assert len({tuple(narrative_discontinuity[narr][t] for narr in narrs) for t in np.flatnonzero(n_narrs_spiking)}) <= num_compositional_narrs

# check: each narr's on and off times, to read the spikes against
switched = {narr: np.diff(on[narr].astype(int)) for narr in narrs}                                    # +1 where the narr came on, -1 where it went off
for narr in narrs: print(f'{narr:9s} on at t = {(np.flatnonzero(switched[narr] == 1) + 1).tolist()}, off at t = {(np.flatnonzero(switched[narr] == -1) + 1).tolist()}')

# check, in 2d so it can be read by hand: features [narr1, narr2], narr1's direction is [1, 0]. with both on the input is [1, 1], whose length is pythagoras, sqrt(2). projection stays 1 whether or not narr2 is on, since the direction is 0 there; the cosine falls to 1/sqrt(2) because narr2 sits in its denominator
narr1_direction_2d, narr1_alone_2d, both_on_2d = np.array([1, 0]), np.array([1, 0]), np.array([1, 1])
assert np.isclose(np.linalg.norm(both_on_2d), np.sqrt(2))
assert np.isclose(narr1_alone_2d @ narr1_direction_2d, 1) and np.isclose(both_on_2d @ narr1_direction_2d, 1)
assert np.isclose(get_cos(narr1_alone_2d[None], narr1_direction_2d), 1) and np.isclose(get_cos(both_on_2d[None], narr1_direction_2d), 1 / np.sqrt(2))
for narr in narrs: assert np.allclose(narrative_continuity[narr], cos_vs_input[narr] * np.linalg.norm(inputs, axis=1))   # the projection does not normalize by feature length, which is the only difference between the two

# check: do projection and cos vs input tell on from off, and does another narr coming on drag them down? cos vs input is normalised by every feature, so the other narrs sit in its denominator; projection never divides, so they cannot reach it
for narr in narrs:
    for name, cos in [('continuity', narrative_continuity[narr]), ('cos vs input', cos_vs_input[narr])]:
        print(f'{narr} {name:19s} on: min {cos[on[narr]].min():5.2f} | off: max {cos[~on[narr]].max():5.2f}, mean {cos[~on[narr]].mean():5.2f}')
for n_others_on in [0, 1, 2]:
    narr1_with_n_others = on['narr1'] & (on['narr2'].astype(int) + on['narr3'].astype(int) == n_others_on)
    print(f'narr1 on with {n_others_on} other narrs on ({narr1_with_n_others.sum():2d} timesteps): continuity {narrative_continuity["narr1"][narr1_with_n_others].mean():.2f}, cos vs input {cos_vs_input["narr1"][narr1_with_n_others].mean():.2f}')


### plot: input matrix (rows = features, cols = time, each narr's rows in its own color, shade = value), then bayesian surprisal, the narrative alignment read as cosine, the memory commits and the sparse memory nodes
plt.rcParams.update({'axes.titlesize': 17, 'axes.titleweight': 'bold', 'axes.titlepad': 10})   # each title heads its row
fig, ax = plt.subplots(5, 1, figsize=(8, 11), sharex=True)
ax[0].set(title='Input', yticks=[2, 7, 12], yticklabels=['narr1', 'narr2', 'narr3']); ax[1].set(title='Bayesian surprisal', yticks=[]); ax[2].set(title='Narrative Alignment', ylabel='cosine similarity', yticks=[]); ax[3].set(title='Memory Commits', yticks=[]); ax[4].set(title='Sparse Memory Nodes', ylabel=r'$\Delta$ cosine similarity', xlabel='time', xticks=[], yticks=[])
img = np.zeros((n_features, n_time, 4))
for narr, features in narrs.items(): img[features] = LinearSegmentedColormap.from_list(narr, ['white', colors[narr]])(np.clip(inputs[:, features].T / 1.5, 0, 1))
ax[0].imshow(img, aspect='auto', interpolation='none')
ax[1].plot(np.arange(1, n_time), bayesian_surprisal)
for narr in narrs: ax[2].plot(cos_vs_input[narr], color=colors[narr])   # from t = 0, since it is a level
ax[3].eventplot([np.flatnonzero(continuity_above_threshold[narr]) for narr in narrs], colors=[colors[narr] for narr in narrs], lineoffsets=[2, 1, 0], linelengths=.8)
ax[4].eventplot([np.flatnonzero(narrative_discontinuity[narr]) + 1 for narr in narrs], colors=[colors[narr] for narr in narrs], lineoffsets=[2, 1, 0], linelengths=.8)   # from t = 1, since it is a change
plt.tight_layout(); plt.savefig(os.path.join(figs_dir, '2_surprisal_vs_discontinuity.png'), dpi=200, bbox_inches='tight'); plt.close()
