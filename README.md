# Abstract

How do brains process continuous information? Human cognitive neuroscience suggests that we segment information into episodes/events to make it easier to store. Why?

The leading theory is rooted in the assumption that the brain is always performing next-step prediction, and that in order to reduce the Prediction Error (PE), it makes sense to store some past information in memory. However, because of Working Memory (WM) constraints, the brain must prioritize what information to store. Given the autocorrelated structure of continuous experience, the optimal solution is to store the recent past instead of the distant past. This cleaving of recent and distant past ('segmenting') creates temporally contiguous chunks ('events'). In this view, a sudden increase in PE signifies a change in the current event, creates an event boundary or segment, offloads information from WM to long-term memory, and reinitializes the WM buffer.

Here, we want to leverage this hypothesis to build a model that parses 1) text from a dataset where humans listen to a podcast, and 2) text + visuals from a dataset where humans watch a movie, and *outputs the temporal locations of event segments/boundaries*. WM can be operationalized as a context-window, while PE can be operationalized as entropy, surprisal, embedding changes, etc.

# Papers

* [Large language models can segment narrative events similarly to humans](https://link.springer.com/article/10.3758/s13428-024-02569-z)

# Data

1. *data/podcast*: The “Podcast” ECoG dataset for modeling neural activity during natural language comprehension | Scientific Data

   * paper: [The “Podcast” ECoG dataset for modeling neural activity during natural language comprehension](https://www.nature.com/articles/s41597-025-05462-2)
   * [audio/text stimulus](https://www.thisamericanlife.org/631/transcript) (story 1)
2. *data/movie*

   * paper: [Multimodal single-neuron, intracranial EEG, and fMRI brain responses during movie watching in human patients | Scientific Data](https://www.nature.com/articles/s41597-024-03029-1.pdf)
   * [movie stimulus](https://app.box.com/s/av4ly4d01h5biuvt17o81mmwqokabzth): audio-visual data
   * *visual_frames/*: the visual data. Contains individual, 1s frames (excluding the first 10s that have the instructions on the screen)
   * *text_dialogue*: the text data. Should be easy to extract.
   * *scenecut_info.csv*: authors' documentation of when scene and camera-angle changes occur in the movie.
   * *visual_annots.csv*: our hand-annotations of which objects appear in the gaze of a sample subject (yes, gaze data was also collected). For now, we can assume this is pretty generalizable across subjects.
   * *avg_suspense.csv*: continuous suspense ratings of the movie collected from participants in a different study, with relatively low sampling rate (thus, likely needs interpolation).
   * *neural/df_spikes_processed.parquet* (1123 neurs x 479 time bins): per-neuron spike counts within 1 second bins across the movie.
   * *neural/valid_mask.npy* (1123, 479):a mask of valid/invalid time periods that varies across neurons. False = invalid = time points with noisy or missing neural recording. You can choose to retain or drop these time points for analyses.
   * *neural/neuron_regions.csv* (1123 neurons): the corresponding brain region for every neuron. Can analyze each region separately. Total 5 cool brain regions.

# Plan

0. Curate data: scrape text data for both podcast and movie.
1. Build model: a next-token predicting agent with a certain context-window. This model should be able to be prompted for memory later (see point 3).

* variables: token length, context-window length

2. Investigate PE correlates during processing: what do PE peaks coincide with? For podcast, consider event annotations from the paper linked above, etc. For movie, consider *scenecut_info.csv*, *visual_annotations.csv, *avg_suspense.csv**, etc.

* variables: PE metric

3. Investigate PE implications during recall: when prompted to freely recall, what does the model emphasize, what does it forget? How about when cued with a specific instance in podcast/movie and asked to recall what happened before/after?

* variables: something like 'forgetting rate' - we dont want the model to remember everything.

# Environment

```bash
# Create and activate (first time)
conda env create -f environment.yml
conda activate narrative_map

# After installing new packages, re-export before committing
conda env export -n narrative_map | sed 's|name: .*|name: narrative_map|' > environment.yml
```

# Notes

* Write interpretable code!
