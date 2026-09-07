# Formalization

* Consider a story as an input, where each scene is a high dimensional embedding, and is primarily dominated by 2 narratives: one about the social dynamics between the 2 characters, and the other about the places they are visiting through the story. Say that social dynamics are captured by features 1-5, and places 6-10, and 11/12 are noise.
* Say that the social dynamics and locational (place) info gets built over the course of the story, often in alternating bouts, but sometimes simultaneously. Say that bouts are of 3 types: social, locational, simultaneous. Say that this build up is captured by an increase in the avg value of the corresponding features.
    * During social bouts, say that locational features = 0, and vice versa. Thereafter, when the locational or simultaneous bout resumes, locational features are an increment of the last recorded features (increment reflecting the build up).

* Say that the goal is to cluster/sort inputs based on the building of each narrative. Assume insertion sort. Assume unit of sorting can be each scene (sort_unit_len=1), or up to half of the input length. In the latter case, the unit is the median of subsumed scenes. Assume costs of num_comparisons and precision_error, where the latter is a comparison of matched scenes between input and sorted_output (as in optimal_chunking).

## Decisions from discussion

* bouts are called `event` in code. event type is sampled at random per event (same type may repeat); n_events = 10; event length ~ avg_event_len ± sd_event_len (sd = 0 for the schematic only, so chunks land on events). simultaneous events dropped for now.
* level increments once per scene within an event. each feature gets its own random increment per scene (~ U(0, 2 × build_up_increment), so the mean rate is build_up_increment), so rows within a narrative rise at different, jagged paces. small iid noise on active features; inactive features stay exactly 0. features 11/12 are iid noise.
* the story stays in temporal order (events are not shuffled). the sort pulls same-narrative scenes together: social scenes first, then locational; within a narrative, lower level first. simultaneous scenes appear in both.
* sort units are contiguous chunks of the story. a unit's level for a narrative = mean of that narrative's features in the unit's median. a unit with level 0 is inactive for that narrative and excluded from its sort.
* precision_error = blur: mean squared distance between each scene and the unit median that stands in for it in the sorted output.
* num_comparisons falls as ~1/sort_unit_len², so the optimum bottoms out early. replaced by amount_shift: how far (in scenes) each unit moves to land in its narrative's thread, summed over units. falls as ~1/sort_unit_len. num_comparisons is still returned, not plotted.
* schematic.png: 2×2. rows: sort_unit_len = 1 vs avg_event_len. cols: processed_input (each unit replaced by its median) vs sorted_output. each panel is n_features × n_scenes.
* optimization.png: sweep sort_unit_len across seeds; sweep avg_event_len. sort_unit_len swept up to 2 × avg_event_len (not half the input): both cost curves are flat beyond that, and the min–max normalization (each cost rescaled to 0–1 over the sweep range before summing) is skewed by a long flat tail.
