# see markdowns/2_surprisal_vs_departure.md
import numpy as np, matplotlib.pyplot as plt, os
from matplotlib.colors import LinearSegmentedColormap
from utils import plot_style; plot_style('notebook')
figs_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'figs'); os.makedirs(figs_dir, exist_ok=True)

# each timestep is a 15-dim embedding: narr1 features 1-5, narr2 features 6-10, non-narr features 11-15
n_features, n_time = 15, 100
streams = {'narr1': slice(0, 5), 'narr2': slice(5, 10), 'non_narr': slice(10, 15)}; narrs = ['narr1', 'narr2']
stay_probs = {'narr1': {'on': .95, 'off': .8}, 'narr2': {'on': .93, 'off': .93}, 'non_narr': {'on': .9, 'off': .95}}   # p(stay on | on), p(stay off | off): narr1 is mostly on, non-narr mostly off; all high so streams flip rarely
starts_on = {'narr1': True, 'narr2': False, 'non_narr': False}
colors = {'narr1': 'purple', 'narr2': 'green', 'non_narr': 'grey'}                                      # narr1 purple, narr2 green, everywhere they appear
shared_value = .7                                                                                         # every stream shares one value, so none reads as brighter than the others


### inputs
def make_input(n_time=n_time, stay_probs=stay_probs, starts_on=starts_on, shared_value=shared_value, min_run=4, feature_sd=.15, inactive_noise=.05, seed=0):
    rng = np.random.default_rng(seed)
    # which streams are on: each stream is its own on/off markov chain, one step per timestep; a run has to last min_run before it can flip
    on = {}
    for stream in streams:
        is_on = [starts_on[stream]]; run = 1
        for _ in range(n_time - 1):
            if run < min_run or rng.random() < stay_probs[stream]['on' if is_on[-1] else 'off']: is_on.append(is_on[-1]); run += 1
            else: is_on.append(not is_on[-1]); run = 1
        on[stream] = np.array(is_on)
    # while a stream is on, each of its 5 features is drawn from a gaussian over the shared value; while off, they are roughly 0
    inputs = rng.normal(0, inactive_noise, (n_time, n_features))
    for stream, features in streams.items():
        for t in np.flatnonzero(on[stream]): inputs[t, features] = rng.normal(shared_value, feature_sd, 5)
    return inputs, on


### metrics, read off the input, not off which streams are on: how much a stream's features went up (or down) since the previous timestep, summed over its 5 features; all defined from t = 1
def get_went_up(inputs, features): return np.maximum(np.diff(inputs[:, features], axis=0), 0).sum(axis=1)
def get_went_down(inputs, features): return np.maximum(-np.diff(inputs[:, features], axis=0), 0).sum(axis=1)

inputs, on = make_input(seed=0)
local_surprisal = sum(get_went_up(inputs, features) for features in streams.values())                  # big at every event start, i.e. any stream coming on
bayesian_surprisal = sum(get_went_up(inputs, streams[narr]) for narr in narrs)                            # big at every event start of a narrative; non-narr coming on does not count
narrative_departure = {narr: get_went_down(inputs, streams[narr]) for narr in narrs}                      # big at every event end of its own narrative; one metric per narrative

# check: each stream's on and off times, to read the spikes against
flipped = {stream: np.diff(on[stream].astype(int)) for stream in streams}                                # +1 where the stream came on, -1 where it went off
for stream in streams: print(f'{stream:8s} on at t = {(np.flatnonzero(flipped[stream] == 1) + 1).tolist()}, off at t = {(np.flatnonzero(flipped[stream] == -1) + 1).tolist()}')


### plot: input matrix (rows = features, cols = time, each stream's rows in its own color, shade = value), local surprisal, bayesian surprisal, narrative departure
fig, ax = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
ax[0].set(title='Input', yticks=[2, 7, 12], yticklabels=['narr1', 'narr2', 'non-narr']); ax[1].set(ylabel='local\nsurprisal', yticks=[]); ax[2].set(ylabel='bayesian\nsurprisal', yticks=[]); ax[3].set(ylabel='narrative\ndeparture', xlabel='time', xticks=[], yticks=[])
img = np.zeros((n_features, n_time, 4))
for stream, features in streams.items(): img[features] = LinearSegmentedColormap.from_list(stream, ['white', colors[stream]])(np.clip(inputs[:, features].T, 0, 1))
ax[0].imshow(img, aspect='auto', interpolation='none')
ax[1].plot(np.arange(1, n_time), local_surprisal); ax[2].plot(np.arange(1, n_time), bayesian_surprisal)
for narr in narrs: ax[3].plot(np.arange(1, n_time), narrative_departure[narr], label=narr, color=colors[narr])
ax[3].legend()
plt.tight_layout(); plt.savefig(os.path.join(figs_dir, '2_surprisal_vs_departure.png'), dpi=200, bbox_inches='tight'); plt.close()
