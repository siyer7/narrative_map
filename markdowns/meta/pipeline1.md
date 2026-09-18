* Recipe:
  * Inputs: contrast bayesian-surprisal vs bayesian surprise vs narr-discontinuity

    * surprisal: timeseries that spikes at every event start
    * bayesian surprisal: spikes additively at every switch *into* narrative; amplitude weighted by total_updated_narr_features, i.e., num_features across narratives that come on (are switched into)
    * narr-discontinuity: separate timeseries per narratives, which spike at every switch *out* of the corresponding narrative.
  * Model:
  * Outputs to contrast:

    * COMPOSITIONALITY
    * boundary location, event memory

      * Increased surprisal vs decreased coherence -> basically your 2 models should recover your 2 contrasted input features
    * Cued and free recall (noise-robust and diverse next-event predictions)

      * A direct readout of your model above
    * forgetting and gist memory

      * This would be cool, and would really emphasize temporal rearrangement as events gets further and further folded over time
