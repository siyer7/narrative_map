# see markdown/formalization.md
import numpy as np, matplotlib.pyplot as plt, os
from matplotlib.transforms import blended_transform_factory
from utils import plot_style, norm01; plot_style('notebook')
figs_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'figs'); os.makedirs(figs_dir, exist_ok=True)

# each scene is a 12-dim embedding: social features 1-5, locational features 6-10, noise features 11/12
n_features = 12
social_features, locational_features, noise_features = slice(0, 5), slice(5, 10), slice(10, 12)
narratives = {'social': social_features, 'locational': locational_features}
event_type_names = ['social', 'locational', 'simultaneous']


### story
def make_story(n_events=6, avg_event_len=10, sd_event_len=2, build_up_increment=1, within_event_noise=.2, noise_sd=.5, seed=0):
    rng = np.random.default_rng(seed)
    event_types = rng.choice(event_type_names, n_events)                                                # bouts sampled at random
    event_lens = rng.integers(avg_event_len - sd_event_len, avg_event_len + sd_event_len + 1, n_events)
    event_starts = np.concatenate(([0], np.cumsum(event_lens)[:-1]))

    # build up: each narrative's level increments per scene during its bouts, resumes from last recorded level after a gap
    story = np.zeros((event_lens.sum(), n_features)); social_level, locational_level = 0, 0; scene = 0
    for event_type, event_len in zip(event_types, event_lens):
        for _ in range(event_len):
            if event_type in ('social', 'simultaneous'):
                social_level += build_up_increment; story[scene, social_features] = social_level + rng.normal(0, within_event_noise, 5)
            if event_type in ('locational', 'simultaneous'):
                locational_level += build_up_increment; story[scene, locational_features] = locational_level + rng.normal(0, within_event_noise, 5)
            story[scene, noise_features] = rng.normal(0, noise_sd, 2)                                    # inactive features stay exactly 0
            scene += 1
    return story, event_types, event_lens, event_starts


### sort
def chunk_story(story, sort_unit_len):
    # sort unit = median of subsumed scenes; keep which scenes each unit stands for
    unit_starts = range(0, len(story), sort_unit_len)
    units = [np.median(story[start:start + sort_unit_len], axis=0) for start in unit_starts]
    unit_scenes = [np.arange(start, min(start + sort_unit_len, len(story))) for start in unit_starts]
    return units, unit_scenes

def narrative_level(unit, narrative_features): return unit[narrative_features].mean()                   # build up = avg value of the narrative's features

def insertion_sort(units, unit_scenes, narrative_features):
    # units inactive in this narrative (level 0) are excluded; active units inserted in order of build up
    sorted_units, sorted_scenes, num_comparisons = [], [], 0
    for unit, scenes in zip(units, unit_scenes):
        level = narrative_level(unit, narrative_features)
        if level <= 0: continue
        for insert_at in range(len(sorted_units)):
            num_comparisons += 1
            if level < narrative_level(sorted_units[insert_at], narrative_features): break
        else: insert_at = len(sorted_units)
        sorted_units.insert(insert_at, unit); sorted_scenes.insert(insert_at, scenes)
    return sorted_units, sorted_scenes, num_comparisons

def sort_story(story, sort_unit_len):
    # social scenes clustered first, then locational; simultaneous scenes appear in both
    units, unit_scenes = chunk_story(story, sort_unit_len)
    sorted_output, num_comparisons, squared_errors, narrative_bounds = [], 0, [], []
    for narrative_features in narratives.values():
        sorted_units, sorted_scenes, narrative_comparisons = insertion_sort(units, unit_scenes, narrative_features)
        num_comparisons += narrative_comparisons
        for unit, scenes in zip(sorted_units, sorted_scenes):
            sorted_output.extend([unit] * len(scenes))                                                    # each scene stands in as its unit's median
            squared_errors.extend(((story[scenes] - unit) ** 2).mean(axis=1))                             # precision error = blur between scene and its unit
        narrative_bounds.append(len(sorted_output))
    precision_error = np.mean(squared_errors)
    return np.array(sorted_output), num_comparisons, precision_error, narrative_bounds[:-1]

def get_optimal_sort_unit_len(story, max_sort_unit_len=None):
    # sort unit ranges from each scene up to half of the input length
    sort_unit_lens = list(range(1, (max_sort_unit_len or len(story) // 2) + 1)); num_comparisons_all, precision_errors = [], []
    for sort_unit_len in sort_unit_lens:
        _, num_comparisons, precision_error, _ = sort_story(story, sort_unit_len)
        num_comparisons_all.append(num_comparisons); precision_errors.append(precision_error)
    optimal_sort_unit_len = sort_unit_lens[np.argmin(norm01(np.array(num_comparisons_all)) + norm01(np.array(precision_errors)))]
    return sort_unit_lens, num_comparisons_all, precision_errors, optimal_sort_unit_len


### schematic: rows = sort_unit_len 1 vs avg_event_len; cols = processed_input vs sorted_output; each panel n_features x n_scenes
seed, n_events, avg_event_len, sd_event_len = 0, 6, 10, 2
story, event_types, event_lens, event_starts = make_story(seed=seed, n_events=n_events, avg_event_len=avg_event_len, sd_event_len=sd_event_len)

cmaps = {'social': plt.cm.Reds, 'locational': plt.cm.YlOrBr, 'noise': plt.cm.Greys}
def colorize(stream):
    # social rows in reds, locational in yellows, noise in greys; shade = level relative to the story's max level
    max_level = story[:, :noise_features.start].max(); img = np.zeros((n_features, len(stream), 4))
    for name, features in [('social', social_features), ('locational', locational_features), ('noise', noise_features)]:
        img[features] = cmaps[name](.2 + .6 * np.clip(stream[:, features].T / max_level, 0, 1))
    return img

fig, ax = plt.subplots(2, 2, figsize=(8, 4)); fig.suptitle('Sorting scenes by build up of each narrative')
for row, sort_unit_len in enumerate([1, avg_event_len]):
    units, unit_scenes = chunk_story(story, sort_unit_len)
    processed_input = np.vstack([[unit] * len(scenes) for unit, scenes in zip(units, unit_scenes)])
    sorted_output, num_comparisons, precision_error, narrative_bounds = sort_story(story, sort_unit_len)
    for col, (title, stream, bounds) in enumerate([('processed input', processed_input, event_starts[1:]), ('sorted output', sorted_output, narrative_bounds)]):
        ax[row, col].imshow(colorize(stream), aspect='auto', interpolation='none')
        ax[row, col].set(title=title if row == 0 else '', xticks=[], yticks=[2, 7, 10.5], yticklabels=['social', 'locational', 'noise'] if col == 0 else [])
        for bound in bounds: ax[row, col].axvline(bound - .5, color='black', lw=1.5)
    ax[row, 0].set(ylabel=f'sort_unit_len = {sort_unit_len}')
    ax[row, 1].text(1.02, .5, f'comparisons: {num_comparisons}\nprecision error: {precision_error:.2f}', transform=ax[row, 1].transAxes, va='center', fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(figs_dir, 'schematic.png'), dpi=200, bbox_inches='tight'); plt.close()


### optimization: sweep sort_unit_len across seeds, and sweep avg_event_len
fig, (ax_sweep, ax_event_len) = plt.subplots(1, 2, figsize=(9, 3.2), gridspec_kw={'width_ratios': [1.5, 1]}); ax_err = ax_sweep.twinx()

# across seeds: distribution over the comparisons / precision error curves
n_reps = 20
stories = [make_story(seed=rep_seed, n_events=n_events, avg_event_len=avg_event_len, sd_event_len=sd_event_len) for rep_seed in range(n_reps)]
max_sort_unit_len = min(len(story) for story, *_ in stories) // 2                                        # common sweep range across seeds
num_comparisons_reps, precision_errors_reps, optimal_sort_unit_lens_reps, event_lens_reps = [], [], [], []
for story, event_types, event_lens, event_starts in stories:
    sort_unit_lens, num_comparisons_all, precision_errors, optimal_sort_unit_len = get_optimal_sort_unit_len(story, max_sort_unit_len)
    num_comparisons_reps.append(num_comparisons_all); precision_errors_reps.append(precision_errors); optimal_sort_unit_lens_reps.append(optimal_sort_unit_len); event_lens_reps.extend(event_lens)
num_comparisons_reps, precision_errors_reps = np.array(num_comparisons_reps), np.array(precision_errors_reps)
optimal_mean, optimal_sd = np.mean(optimal_sort_unit_lens_reps), np.std(optimal_sort_unit_lens_reps)
event_len_mean, event_len_sd = np.mean(event_lens_reps), np.std(event_lens_reps)
print(f'optimal_sort_unit_lens_reps: {optimal_sort_unit_lens_reps}\nmean: {optimal_mean:.1f}, avg_event_len: {avg_event_len}')

ax_sweep.set(xlabel='sort unit length', ylabel='comparisons'); ax_err.set(ylabel='precision error')
ax_sweep.text(0, 1.05, 'Optimal sort unit', transform=ax_sweep.transAxes, color='purple'); ax_sweep.text(.5, 1.05, '~', transform=ax_sweep.transAxes); ax_sweep.text(.57, 1.05, 'Avg. event length', transform=ax_sweep.transAxes, color='green')
ax_sweep.plot(sort_unit_lens, num_comparisons_reps.mean(0), c='orangered')
ax_sweep.fill_between(sort_unit_lens, num_comparisons_reps.mean(0) - num_comparisons_reps.std(0), num_comparisons_reps.mean(0) + num_comparisons_reps.std(0), color='orangered', alpha=.2)
ax_err.plot(sort_unit_lens, precision_errors_reps.mean(0), c='deepskyblue')
ax_err.fill_between(sort_unit_lens, precision_errors_reps.mean(0) - precision_errors_reps.std(0), precision_errors_reps.mean(0) + precision_errors_reps.std(0), color='deepskyblue', alpha=.2)
ax_sweep.axvline(optimal_mean, ls='--', c='purple'); ax_sweep.axvline(event_len_mean, ls='--', c='green')
ax_sweep.spines['left'].set_color('orangered'); ax_sweep.tick_params(axis='y', colors='orangered'); ax_sweep.yaxis.label.set_color('orangered')
ax_err.spines['right'].set_visible(True); ax_err.spines['right'].set_color('deepskyblue'); ax_err.tick_params(axis='y', colors='deepskyblue'); ax_err.yaxis.label.set_color('deepskyblue')
blend = blended_transform_factory(ax_sweep.transData, ax_sweep.transAxes)                                 # x = data, y = axes fraction
ax_sweep.errorbar(optimal_mean, .93, xerr=optimal_sd, fmt='o', c='purple', capsize=3, transform=blend, clip_on=False)
ax_sweep.errorbar(event_len_mean, .86, xerr=event_len_sd, fmt='o', c='green', capsize=3, transform=blend, clip_on=False)

# across avg_event_len: does the optimal sort unit track event length?
avg_event_lens = np.linspace(5, 15, 10).astype(int); optimal_sort_unit_lens = []
for avg_event_len_sweep in avg_event_lens:
    story, *_ = make_story(seed=seed, n_events=n_events, avg_event_len=avg_event_len_sweep, sd_event_len=sd_event_len)
    *_, optimal_sort_unit_len = get_optimal_sort_unit_len(story); optimal_sort_unit_lens.append(optimal_sort_unit_len)
ax_event_len.set(xlabel='avg. event length', ylabel='optimal sort unit length', xlim=(4, 16), ylim=(4, 16), xticks=[5, 10, 15], yticks=[5, 10, 15])
ax_event_len.plot([4, 16], [4, 16], ls='--', c='gray', alpha=.5); ax_event_len.scatter(avg_event_lens, optimal_sort_unit_lens)
plt.tight_layout(); plt.savefig(os.path.join(figs_dir, 'optimization.png'), dpi=200, bbox_inches='tight'); plt.close()
