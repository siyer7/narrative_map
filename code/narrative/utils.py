def plot_style(context):
    import seaborn as sns, matplotlib.pyplot as plt, matplotlib as mpl

    # sets fontsize etc. appropriate for presentation/paper, etc.
    sns.set(context=context, style='white', palette='deep')

    # keep text editable in svg
    plt.rcParams['svg.fonttype'] = 'none'

    # push ticks inward
    mpl.rcParams['xtick.direction'], mpl.rcParams['ytick.direction'] = 'in', 'in'
    # remove top and right splines
    mpl.rcParams['axes.spines.top'], mpl.rcParams['axes.spines.right'] = False, False

def norm01(x):
    import numpy as np
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))

def generate_stim(stim_dim=10, event_dim=5, n_stims=100, min_events=5, mean_events=5, max_events=5, event_len_var = .2, stim_noise=0.05,
                  prop_related_events=0.2, related_event_template_noise=0.05, seed=None):
    import numpy as np
    rng = np.random.default_rng(seed)

    n_events = int(np.clip(rng.poisson(lam=mean_events), min_events, max_events))

    # event_feats: dims shared by all events that carry identity (rest are autoregressive)
    event_feats     = rng.choice(stim_dim, event_dim, replace=False)
    non_event_feats = np.setdiff1d(np.arange(stim_dim), event_feats)

    # Narrative-level template — sparse: only event_feats are active, rest are 0
    narrative_template = np.zeros(stim_dim)
    narrative_template[event_feats] = rng.random(event_dim)

    # generate event templates — sparse: only event_feats are active, rest are 0
    event_relatedness = rng.random(n_events) < prop_related_events  # True = related to narrative
    event_templates   = np.zeros((n_events, stim_dim))
    for event_idx in range(n_events):
        if event_relatedness[event_idx]:
            event_templates[event_idx, event_feats] = np.clip(narrative_template[event_feats] + rng.normal(0, related_event_template_noise, event_dim), 0, 1)
        else:
            event_templates[event_idx, event_feats] = rng.random(event_dim)  # completely random activity

    # generate event lengths via normal dist; normalize before int cast so no event overshoots n_stims
    event_avg_len = n_stims / n_events
    raw_lengths   = np.maximum(rng.normal(event_avg_len, event_avg_len * event_len_var, n_events), 5)
    event_lengths = (raw_lengths / raw_lengths.sum() * n_stims).astype(int)
    event_lengths = np.clip(event_lengths, 5, None)  # minimum event length of 5
    event_lengths[-1] = n_stims - event_lengths[:-1].sum()  # ensure exact sum = n_stims

    # Generate stimulus matrix with autoregressive event structure
    autoreg_directions = []  # 0=decay (1→0), 1=build (0→1) per event
    stim          = np.zeros((stim_dim, n_stims))
    event_start_t = 0   # global timepoint where current event starts

    for event_idx in range(n_events):
        event_len      = event_lengths[event_idx]
        event_template = event_templates[event_idx]

        direction = int(rng.integers(0, 2))
        autoreg_directions.append(direction)
        target        = 0.0 if direction == 0 else 1.0  # decay→0, build→1
        pull_strength = 1.0 / event_len  # norm by event_len

        # non_event_feats (autoreg feats): randomly initialized, then build/decay autoregressively
        x_prev = rng.random(len(non_event_feats))

        for event_t in range(event_len):
            t = event_start_t + event_t

            # OU-style AR(1): pull toward target proportional to remaining distance, plus noise
            dist_to_target = target - x_prev
            x_t            = x_prev + pull_strength * dist_to_target + rng.normal(0, stim_noise, len(non_event_feats))
            x_t            = np.clip(x_t, 0, 1)  # keep in valid range
            stim[non_event_feats, t] = x_t
            x_prev                   = x_t

            # event_feats: low variation between stims — jitter around template value
            stim[event_feats, t] = np.clip(event_template[event_feats] + rng.normal(0, stim_noise, len(event_feats)), 0, 1)

        event_start_t += event_len

    return dict(stim=stim, event_lengths=event_lengths,
                event_boundaries=np.concatenate([[0], np.cumsum(event_lengths)[:-1]]),
                event_feats=event_feats, non_event_feats=non_event_feats,
                narrative_template=narrative_template,
                event_relatedness=event_relatedness,
                event_templates=event_templates,
                autoreg_directions=np.array(autoreg_directions))