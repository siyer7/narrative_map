# see markdowns/2_surprise_vs_coherence.md
import numpy as np, matplotlib.pyplot as plt, os
from utils import plot_style; plot_style('notebook')
figs_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'figs'); os.makedirs(figs_dir, exist_ok=True)

# each timestep is a 10-dim embedding: narrative1 features 1-5 (the main narrative), narrative2 features 6-10
n_features, n_time = 10, 100
narrative1_features, narrative2_features = slice(0, 5), slice(5, 10)
narratives = {'narrative1': narrative1_features, 'narrative2': narrative2_features}
stay_probs = {'narrative1': .95, 'narrative2': .85}                                                       # staying within a narrative is likelier than switching, even more so for narrative1
other = {'narrative1': 'narrative2', 'narrative2': 'narrative1'}


### inputs
def make_input(n_time=n_time, stay_probs=stay_probs, within_event_noise=.05, inactive_noise=.05, seed=0):
    rng = np.random.default_rng(seed)
    # which narrative is on: markov chain, one step per timestep, starting in narrative1
    narrative = ['narrative1']
    for _ in range(n_time - 1): narrative.append(narrative[-1] if rng.random() < stay_probs[narrative[-1]] else other[narrative[-1]])
    narrative = np.array(narrative)
    # each narrative has a random pattern on its own features, held across the story; while on, its features = pattern + jitter; the other narrative's features are roughly 0
    patterns = {name: rng.uniform(0, 1, 5) for name in narratives}
    inputs = rng.normal(0, inactive_noise, (n_time, n_features))
    for t, name in enumerate(narrative): inputs[t, narratives[name]] = patterns[name] + rng.normal(0, within_event_noise, 5)
    return inputs, narrative, patterns


### surprisal: how far the input moved since the previous timestep; spikes above the threshold when the narrative has switched
def get_surprisal(inputs): return np.linalg.norm(np.diff(inputs, axis=0), axis=1)                        # defined from t = 1
surprisal_threshold = 1                                                                                   # reasonably big: within-event jitter stays well under it


### coherence: how close the input is to narrative1's template (its pattern on its features, 0s on narrative2's); high while narrative1 is on, low while narrative2 is on
def get_coherence(inputs, template, eps=1e-3): return 1 / (np.linalg.norm(inputs - template, axis=1) + eps)   # eps avoids 1/0


inputs, narrative, patterns = make_input(seed=0)
template = np.zeros(n_features); template[narrative1_features] = patterns['narrative1']
surprisal, coherence = get_surprisal(inputs), get_coherence(inputs, template)

# check: surprisal spikes should line up with narrative switches
switches = np.flatnonzero(narrative[1:] != narrative[:-1]) + 1; spikes = np.flatnonzero(surprisal > surprisal_threshold) + 1
print(f'switches at t = {switches.tolist()}\nspikes   at t = {spikes.tolist()}')


### plot: input matrix (rows = features, cols = time), surprisal, coherence
fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
ax[0].set(title='Input', yticks=[2, 7], yticklabels=['narrative1', 'narrative2']); ax[1].set(ylabel='surprisal'); ax[2].set(ylabel='coherence', xlabel='time')
ax[0].imshow(inputs.T, aspect='auto', interpolation='none', cmap='Greys', vmin=0, vmax=1)
ax[1].plot(np.arange(1, n_time), surprisal); ax[1].axhline(surprisal_threshold, ls='--', c='gray')
ax[2].plot(coherence)
plt.tight_layout(); plt.savefig(os.path.join(figs_dir, '2_surprise_vs_coherence.png'), dpi=200, bbox_inches='tight'); plt.close()
