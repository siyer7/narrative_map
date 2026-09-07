# Formalization

* Consider a story as an input, where each scene is a high dimensional embedding, and is primarily dominated by 2 narratives: one about the social dynamics between the 2 characters, and the other about the places they are visiting through the story. Say that social dynamics are captured by features 1-5, and places 6-10, and 11/12 are noise.
* Say that the social dynamics and locational (place) info gets built over the course of the story, in bouts (called events in code). Say that bouts are of 3 types: social, locational, simultaneous (simultaneous dropped for now; bout type is random per bout, so same type may repeat). Say that this build up is captured by an increase in the avg value of the corresponding features.
    * During social bouts, say that locational features = 0, and vice versa. Thereafter, when the locational bout resumes, locational features are an increment of the last recorded features (increment reflecting the build up).
    * Say that each feature builds up at its own random pace: per scene, each feature gets an increment ~ U(0, 2 × build_up_increment). Plus small noise on active features. Inactive features are exactly 0.

* Say that the goal is to cluster/sort inputs based on the building of each narrative: social scenes first, then locational, each in order of build up. Assume insertion sort. Assume unit of sorting can be each scene (sort_unit_len=1), or up to 2 × avg event length. In the latter case, the unit is the median of subsumed scenes; a unit whose median is 0 for a narrative is excluded from that narrative's sort. Assume costs of amount_shift and precision_error, where the former is how far (in scenes) each unit moves to reach its place in the sorted_output, summed over units, and the latter is a comparison of matched scenes between input and sorted_output (as in optimal_chunking).

## Notes

* amount_shift replaced num_comparisons: comparisons fall as ~1/sort_unit_len², so the optimum bottomed out early; shift falls as ~1/sort_unit_len. num_comparisons still returned, not plotted.
* sweep capped at 2 × avg_event_len: both costs are flat beyond that, and min–max normalizing (each cost rescaled 0–1 over the sweep before summing) gets skewed by the flat tail.
* schematic.png uses equal event lengths (sd = 0) and 6 events, so chunks land on events and within-event ramps are visible. optimization.png uses sd = 2, 10 events, 20 seeds.
