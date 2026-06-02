# Abstract

How do brains process continuous information? Human cognitive neuroscience suggests that we segment information into episodes/events to make it easier to store. Why?

The leading theory is rooted in the assumption that the brain is always performing next-step prediction, and that in order to reduce the Prediction Error (PE), it makes sense to store some past information in memory. However, because of Working Memory (WM) constraints, the brain must prioritize what information to store. Given the autocorrelated structure of continuous experience, the optimal solution is to store the recent past instead of the distant past. This cleaving of recent and distant past ('segmenting') creates temporally contiguous chunks ('events'). In this view, a sudden increase in PE signifies a change in the current event, creates an event boundary or segment, offloads information from WM to long-term memory, and reinitializes the WM buffer.

Here, we want to leverage this hypothesis to build a model that parses 1) text from a dataset where humans listen to a podcast, and 2) text + visuals from a dataset where humans watch a movie, and *outputs the temporal locations of event segments/boundaries*. WM can be operationalized as a context-window, while PE can be operationalized as entropy, surprisal, embedding changes, etc.

# Data

For this specific purpose, the data/podcast_movie

1. *data/podcast*: The “Podcast” ECoG dataset for modeling neural activity during natural language comprehension | Scientific Data
   * d
2. *data/movie*
   * paper: [Multimodal single-neuron, intracranial EEG, and fMRI brain responses during movie watching in human patients | Scientific Data](https://www.nature.com/articles/s41597-024-03029-1.pdf)
   * [movie stimulus](https://drive.google.com/file/d/13IgC5VGRwV8OgztgX56IN9zMaOSFukdr/view?usp=share_link)
   * *frames/*: the visual data. Contains individual, 1s frames (excluding the first 10s that have the instructions on the screen)
   * *dialogue*: the text data. Should be easy to extract.
   * *scenecut_info.csv*: authors' documentation of when scene and camera-angle changes occur in the movie.
   * *visual_annotations.csv*: our hand-annotations of which objects appear in the gaze of a sample subject (yes, gaze data was also collected). For now, we can assume this is pretty generalizable across subjects.

# Plan

1. Podcast
   * d
2. Movie
   * dd
