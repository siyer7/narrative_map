* Consider the inputs stage of the recipe (meta/pipeline1.md): contrast surprisal vs coherence on one input. Consider an input where each timestep is a 10-dim embedding dominated by 2 narratives: narrative1 (the main narrative) captured by features 1-5, narrative2 by features 6-10. Say the input is 100 timesteps long.
* Say that which narrative is on follows a markov chain, one step per timestep, starting in narrative1. Staying within a narrative is likelier than switching (p(stay) = .85 for narrative2), and even more so for narrative1, the main narrative (p(stay) = .95).
* Say that each narrative has a random pattern (U(0, 1)) on its own features, held across the story. While a narrative is on, its features are that pattern plus jitter (sd .05); the other narrative's features are roughly 0 (noise, sd .05).
* Say that surprisal(t) = ||input(t) - input(t-1)|| (L2). It should mostly spike above a reasonably big threshold (1) when the narrative has switched, since within-event jitter stays well under it. Plotted raw with the threshold as a dashed line, so near-misses are visible.
* Say that coherence(t) = 1 / ||input(t) - template||, where template = narrative1's pattern on its features and 0s on narrative2's (+ 1e-3 in the denominator to avoid 1/0). Coherence is high while narrative1 is on and low while narrative2 is on.

## Notes

* surprisal spikes at every switch; coherence is a level that tracks whether narrative1 is on. So the two are separable: they move together at narrative2 → narrative1 and apart at narrative1 → narrative2.
* pattern is held within a narrative rather than fresh random values per timestep: fresh values would make within-event surprisal nearly as big as a switch, leaving the threshold nowhere clean to sit. The coherence template is the same pattern narrative1 jitters around; an independent draw would make coherence during narrative1 only mildly higher than during narrative2.
* the recipe names three input quantities (bayesian surprisal, bayesian surprise, connectedness); this stage covers surprisal and coherence only.
