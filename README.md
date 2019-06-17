# Tone_Analyzer
This repository contains code implementing a naive Tone Analyzer for audio. The main objective of this Tone Analyzer is to take as input a short audio recording and generate an output value that corresponds to an emotion.

## First Approach
### Dataset
The first approach was to use the The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS), which can be found in https://zenodo.org/record/1188976/?f=3#.XQZ_CYgzaUk. This database contains 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent. Speech includes calm, happy, sad, angry, fearful, surprise, and disgust expressions. Each expression is produced at two levels of emotional intensity (normal, strong), with an additional neutral expression.

The 'Audio_Speech_Actors_01-24' folder, which can be found in this repository contains 1440 files which correspond to the 60 trials of each actor x 24 actors = 1440. All files are .wav and 3-minute long. Each file follows a 7-part numerical filename convention (e.g., 02-01-06-01-02-01-12.wav), which corresponds to:

Modality (01 = full-AV, 02 = video-only, 03 = audio-only).<br />
Vocal channel (01 = speech, 02 = song).<br />
Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).<br />
Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.<br />
Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").<br />
Repetition (01 = 1st repetition, 02 = 2nd repetition).<br />
Actor (01 to 24. Odd numbered actors are male, even numbered actors are female). 
