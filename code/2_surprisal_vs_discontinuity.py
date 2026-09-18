# see markdowns/2_surprisal_vs_discontinuity.md
import numpy as np, matplotlib.pyplot as plt, os
from matplotlib.colors import LinearSegmentedColormap
from utils import plot_style; plot_style('notebook')
figs_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'figs'); os.makedirs(figs_dir, exist_ok=True)

# each timestep is a 15-dim embedding: narr1 features 1-5, narr2 features 6-10, side narr features 11-15
n_features, n_time = 15, 100
narrs = {'narr1': slice(0, 5), 'narr2': slice(5, 10), 'side_narr': slice(10, 15)}; main_narrs = ['narr1', 'narr2']
stay_probs = {'narr1': {'on': .95, 'off': .8}, 'narr2': {'on': .93, 'off': .93}, 'side_narr': {'on': .9, 'off': .95}}   # p(stay on | on), p(stay off | off): narr1 is mostly on, side narr mostly off; all high so narrs switch rarely
starts_on = {'narr1': True, 'narr2': False, 'side_narr': False}
colors = {'narr1': 'green', 'narr2': 'purple', 'side_narr': 'grey'}                                      # narr1 green, narr2 purple, everywhere they appear
mean_feature_value = 1                                                                                    # every narr shares one value, so none reads as brighter than the others; the matrix shades against 1.5 so the wobble around 1 does not clip flat


### inputs
def make_input(n_time=n_time, stay_probs=stay_probs, starts_on=starts_on, mean_feature_value=mean_feature_value, min_run=4, feature_sd=.15, inactive_noise=.05, seed=0):
    rng = np.random.default_rng(seed)
    # which narrs are on: each narr is its own on/off markov chain, one step per timestep; a run has to last min_run before it can switch
    on = {}
    for narr in narrs:
        is_on = [starts_on[narr]]; run = 1
        for _ in range(n_time - 1):
            if run < min_run or rng.random() < stay_probs[narr]['on' if is_on[-1] else 'off']: is_on.append(is_on[-1]); run += 1
            else: is_on.append(not is_on[-1]); run = 1
        on[narr] = np.array(is_on)
    # while a narr is on, each of its 5 features is drawn from a gaussian over the shared value; while off, they are roughly 0
    inputs = rng.normal(0, inactive_noise, (n_time, n_features))
    for narr, features in narrs.items():
        for t in np.flatnonzero(on[narr]): inputs[t, features] = rng.normal(mean_feature_value, feature_sd, 5)
    return inputs, on


### metrics, read off the input, not off which narrs are on: how much a narr's features increased since the previous timestep, summed over its 5 features, read as the prob that the narr resumed; defined from t = 1
def prob_narr_resumed(inputs, features): return np.maximum(np.diff(inputs[:, features], axis=0), 0).sum(axis=1)

inputs, on = make_input(seed=0)
local_surprisal = sum(prob_narr_resumed(inputs, features) for features in narrs.values())             # big at every event start, i.e. any narr coming on
bayesian_surprisal = sum(prob_narr_resumed(inputs, narrs[narr]) for narr in main_narrs)               # big at every event start of a narrative; side narr coming on does not count

# a narr's template: the shared value on its own features, 0 everywhere else. cosine to the template says how much the input looks like that narr right now: a level, not a change, so it is high all through the narr's events and low in the gaps
narr_template = {narr: np.zeros(n_features) for narr in narrs}
for narr, features in narrs.items(): narr_template[narr][features] = mean_feature_value
def get_cos(inputs, template): return inputs @ template / (np.linalg.norm(inputs, axis=1) * np.linalg.norm(template))

# projection: how far the input reaches along the template's direction. it is the numerator both cosines share, before either divides, so it keeps the magnitude (~0 when the narr's features are ~0) and the other narrs never enter it
narr_direction = {narr: narr_template[narr] / np.linalg.norm(narr_template[narr]) for narr in narrs}
narrative_continuity = {narr: inputs @ narr_direction[narr] for narr in main_narrs}

continuity_diff = {narr: np.diff(narrative_continuity[narr]) for narr in main_narrs}                   # the change in continuity: down at every event end, up at every event start. it diffs the summed continuity, so the five features' wobble cancels instead of piling up
departure_threshold = -1                                                                              # a plain diff < 0 fires at half of all timesteps, since the diff of wobble is negative half the time; real ends sit near -2 and the background tops out at -0.55, so -1 sits in the gap
narrative_departure = {narr: continuity_diff[narr] < departure_threshold for narr in main_narrs}      # where the narr departs: one boolean per timestep per narr

cos_vs_input = {narr: get_cos(inputs, narr_template[narr]) for narr in main_narrs}                    # normalised by the whole 15-dim input, so another narr coming on pulls this one down

# check: each narr's on and off times, to read the spikes against
switched = {narr: np.diff(on[narr].astype(int)) for narr in narrs}                                    # +1 where the narr came on, -1 where it went off
for narr in narrs: print(f'{narr:9s} on at t = {(np.flatnonzero(switched[narr] == 1) + 1).tolist()}, off at t = {(np.flatnonzero(switched[narr] == -1) + 1).tolist()}')

# check, in 2d so it can be read by hand: features [narr1, narr2], narr1's direction is [1, 0]. with both on the input is [1, 1], whose length is pythagoras, sqrt(2). projection stays 1 whether or not narr2 is on, since the direction is 0 there; the cosine falls to 1/sqrt(2) because narr2 sits in its denominator
narr1_direction_2d, narr1_alone_2d, both_on_2d = np.array([1, 0]), np.array([1, 0]), np.array([1, 1])
assert np.isclose(np.linalg.norm(both_on_2d), np.sqrt(2))
assert np.isclose(narr1_alone_2d @ narr1_direction_2d, 1) and np.isclose(both_on_2d @ narr1_direction_2d, 1)
assert np.isclose(get_cos(narr1_alone_2d[None], narr1_direction_2d), 1) and np.isclose(get_cos(both_on_2d[None], narr1_direction_2d), 1 / np.sqrt(2))
for narr in main_narrs: assert np.allclose(narrative_continuity[narr], cos_vs_input[narr] * np.linalg.norm(inputs, axis=1))   # the projection does not normalize by feature length, which is the only difference between the two

# check: do projection and cos vs input tell on from off, and does another narr coming on drag them down? cos vs input is normalised by every feature, so the other narrs sit in its denominator; projection never divides, so they cannot reach it
for narr in main_narrs:
    for name, cos in [('continuity', narrative_continuity[narr]), ('cos vs input', cos_vs_input[narr])]:
        print(f'{narr} {name:19s} on: min {cos[on[narr]].min():5.2f} | off: max {cos[~on[narr]].max():5.2f}, mean {cos[~on[narr]].mean():5.2f}')
for n_others_on in [0, 1, 2]:
    narr1_with_n_others = on['narr1'] & (on['narr2'].astype(int) + on['side_narr'].astype(int) == n_others_on)
    print(f'narr1 on with {n_others_on} other narrs on ({narr1_with_n_others.sum():2d} timesteps): continuity {narrative_continuity["narr1"][narr1_with_n_others].mean():.2f}, cos vs input {cos_vs_input["narr1"][narr1_with_n_others].mean():.2f}')


### plot: input matrix (rows = features, cols = time, each narr's rows in its own color, shade = value), then narrative continuity, narrative departure, the cosine, bayesian surprisal, local surprisal
fig, ax = plt.subplots(6, 1, figsize=(8, 13), sharex=True)
ax[0].set(title='Input', yticks=[2, 7, 12], yticklabels=['narr1', 'narr2', 'side narr']); ax[1].set(title='Narrative Continuity', ylabel='projection', yticks=[]); ax[2].set(title='Narrative Departure', ylabel='diff < threshold', yticks=[]); ax[3].set(title='Narrative Continuity', ylabel='cosine', yticks=[]); ax[4].set(title='Prediction Error account', ylabel='bayesian surprisal', yticks=[]); ax[5].set(ylabel='local surprisal', xlabel='time', xticks=[], yticks=[])
img = np.zeros((n_features, n_time, 4))
for narr, features in narrs.items(): img[features] = LinearSegmentedColormap.from_list(narr, ['white', colors[narr]])(np.clip(inputs[:, features].T / 1.5, 0, 1))
ax[0].imshow(img, aspect='auto', interpolation='none')
for narr in main_narrs: ax[1].plot(narrative_continuity[narr], label=narr, color=colors[narr]); ax[3].plot(cos_vs_input[narr], color=colors[narr])   # from t = 0, since they are levels
for narr in main_narrs: ax[2].plot(np.arange(1, n_time), narrative_departure[narr].astype(int), color=colors[narr])   # from t = 1, since it is a change
ax[4].plot(np.arange(1, n_time), bayesian_surprisal); ax[5].plot(np.arange(1, n_time), local_surprisal)
ax[1].legend(loc='center right', framealpha=.9)   # mid-height on the right is empty, green sits high there and purple low
plt.tight_layout(); plt.savefig(os.path.join(figs_dir, '2_surprisal_vs_discontinuity.png'), dpi=200, bbox_inches='tight'); plt.close()
