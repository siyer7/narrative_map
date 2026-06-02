# Abstract

How do brains process continuous information? Human cognitive neuroscience suggests that we segment information into episodes/events to make it easier to store. Why? One theory is that because 1) inferring the meaning of a scene depends more strongly on recent than on distant scenes, and because of 2) working memory (WM) constraints, it is efficient to consider only the current context (‘event’ or ‘episode’) and demarcate it from previous events, resulting in event segmentation. The mechanism underlying segmentation is theorized to be prediction-error (PE) based, where the brain may encounter low next-step PEs within a context, but high next-step PEs when the context changes, thus resulting in segmentation and the conclusion of an event.
	Here, we want to test this hypothesis on text data (from a dataset where humans listen to a podcast) and text + visual data (a dataset where humans watch a movie). Specifically, given text and/or visual frames as input, we want to build a WM-constrained, next-step predicting agent. Concretely, WM can be operationalized as a context-window, while PE can be based on a myriad of metrics including next-token/sentence entropy, surprisal, NLL, etc. The output of the model, which could be a linear readout of PEs, would denote segmentation time points in the input.

## Data

input_data/podcast: The “Podcast” ECoG dataset for modeling neural activity during natural language comprehension | Scientific Data

input_data/movie: Multimodal single-neuron, intracranial EEG, and fMRI brain responses during movie watching in human patients | Scientific Data
stimulus: https://drive.google.com/file/d/13IgC5VGRwV8OgztgX56IN9zMaOSFukdr/view?usp=share_link
frames: contains individual, 1s frames (excluding the first 10s that have the instructions on the screen)
dialogue: should be easy to extract
visual_annotations.csv:
scenecut_info.csv:

## Plan
