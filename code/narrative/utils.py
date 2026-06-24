def plot_style():
    import seaborn as sns, matplotlib.pyplot as plt, matplotlib as mpl

    # sets fontsize etc. appropriate for presentation/paper, etc.
    sns.set(context='talk', style='white', palette='deep')

    # keep text editable in svg
    plt.rcParams['svg.fonttype'] = 'none'

    # push ticks inward
    mpl.rcParams['xtick.direction'], mpl.rcParams['ytick.direction'] = 'in', 'in'
    # remove top and right splines
    mpl.rcParams['axes.spines.top'], mpl.rcParams['axes.spines.right'] = False, False

def norm01(x):
    import numpy as np
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))

def generate_stim(stim_dim=10, total_time=100, n_events=10, noise_std=0.2, seed=None):
    import numpy as np
    rng = np.random.default_rng(seed)

    # generate event lengths via normal dist
    event_avg_len = total_time / n_events
    event_lengths = rng.normal(event_avg_len, event_avg_len * 0.2, n_events).astype(int)
    event_lengths = np.clip(event_lengths, 5, None)  # minimum event length of 5
    event_lengths[-1] = total_time - event_lengths[:-1].sum()  # ensure final event sums to total_time

    # Per-event autoregressive metadata
    autoreg_feat_sizes = []  # number of autoregressive features per event
    autoreg_features  = []  # which feature indices are autoregressive per event
    autoreg_directions = []  # 0=decay (1→0), 1=build (0→1) per event

    # Generate stimulus matrix with autoregressive event structure
    stim = np.zeros((total_time, stim_dim))
    event_templates = []   # (n_events, stim_dim) — ground-truth event fingerprints
    event_start_t   = 0   # global timepoint where current event starts

    for event_idx in range(n_events):
        event_len = event_lengths[event_idx]

        # Sample per-event autoregressive metadata and append to lists
        size = rng.integers(stim_dim // 4, stim_dim // 2 + 1)
        autoreg_feat_sizes.append(size)

        # mutually exhaustive autoreg & stable feats
        ev_autoreg_features = rng.choice(stim_dim, size, replace=False)
        autoreg_features.append(ev_autoreg_features)  # append to the list
        ev_stable_features = np.setdiff1d(np.arange(stim_dim), ev_autoreg_features)

        # direction of autoregression
        direction = int(rng.integers(0, 2))
        autoreg_directions.append(direction)
        target        = 0.0 if direction == 0 else 1.0  # decay→0, build→1
        pull_strength = 1.0 / event_len  # norm by event_len

        # Sample event template (continuous uniform [0,1]) — unique fingerprint for this event
        event_template = rng.random(stim_dim)
        event_templates.append(event_template.copy())

        # Initialize autoreg feature values at event template
        x_prev = event_template[ev_autoreg_features].copy()

        for event_t in range(event_len):
            t = event_start_t + event_t

            # OU-style AR(1): pull toward target proportional to remaining distance, plus noise
            dist_to_target = target - x_prev
            x_t            = x_prev + pull_strength * dist_to_target + rng.normal(0, noise_std, len(ev_autoreg_features))
            x_t            = np.clip(x_t, 0, 1)  # keep in valid range
            stim[t, ev_autoreg_features] = x_t
            x_prev                       = x_t

            # Stable features: jitter around template value to avoid perfect within-event lock
            stim[t, ev_stable_features] = np.clip(event_template[ev_stable_features] + rng.normal(0, noise_std, len(ev_stable_features)), 0, 1)

        event_start_t += event_len

    autoreg_directions = np.array(autoreg_directions)

    return dict(stim=stim, event_lengths=event_lengths,
                event_templates=np.array(event_templates),
                autoreg_directions=autoreg_directions,
                autoreg_features=autoreg_features)