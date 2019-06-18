# Tone_Analyzer
This repository contains code implementing a naive Tone Analyzer for audio. The main objective of this Tone Analyzer is to take as input a short audio recording and generate an output value that corresponds to an emotion.

## First Approach
### Dataset
The first approach was to use the The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS), which can be found in https://zenodo.org/record/1188976/?f=3#.XQZ_CYgzaUk. This database contains 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent. Speech includes calm, happy, sad, angry, fearful, surprise, and disgust expressions. Each expression is produced at two levels of emotional intensity (normal, strong), with an additional neutral expression.

The 'Audio_Speech_Actors_01-24' folder, which for convenience was uploaded in Google Drive and can be found in the following public link: https://drive.google.com/open?id=1Hg1R2jbn7pashf-L2vnWvxuWfgA4ypCp contains 1440 files which correspond to the 60 trials of each actor x 24 actors = 1440. All files are .wav and 3-minute long. 

Each file follows a 7-part numerical filename convention (e.g., 02-01-06-01-02-01-12.wav), which corresponds to:

Modality (01 = full-AV, 02 = video-only, 03 = audio-only).<br />
Vocal channel (01 = speech, 02 = song).<br />
Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).<br />
Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.<br />
Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").<br />
Repetition (01 = 1st repetition, 02 = 2nd repetition).<br />
Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

### Models
Two different types of Deep Learning Networks were used for the Tone Analysis task. The fist one is a Long Short-Term Memory and the second one is a Convolutional Neural Network. Python's librosa library was used for the audio processing. The Mel-frequency cepstral coefficients (MFCCs) were extracted and used to represent the audio data. The dataset of 1440 audio files was split in three: training set (80%), validation set(10%) and test set(10%). Some of the values of the train inputs were zero (these values had occured from the MFCC) and this resulted in problems in training. These values were replaced and the problems dissapeared.

Both models use ReLU as an activation function in the intermediate layers and softmax activation function in the last layer. Softmax of course is combined with the Categorical Crossentropy as a loss function. RMSProp was tested as an optimizer using different learning rates and different weight decays, but Adam seemed to outperform it, so in the end Adam was used in both models. Data were shuffled before entering the Network.

The learning procedure was not fully optimized with the meaning that there are still things that need fine-tuning. The majority of tests have been made using the CNN model, however that's not because LSTM was proven to be worse; it was a random choice to experiment more with CNN. There were several overfitting problems in the beginning. The model was facing difficulties in generalising and Dropout was introduced. The epochs were increased and the model seemed to learn more 'steadily'. Apart from that, several test have been made concerning the number of feature maps in convolutions and the respective kernel size, the need of MaxPooling, BatchNormalization and Fully Connected layers. The highest test accuracy achieved was 65%, which is not so great, but it is ok concerning the small number of training samples. Of course, this test set accuracy, as already mentioned, can be increased with some more fine-tuning.

At the end of the training procedure the model is saved, so that it can be later loaded and used. A .csv file is also created that contains both the ground truth and predicted values of the test set.

## Second Approach
The second approach had as objective to be an end-to-end procedure with the meaning that the dataset will also be created from scratch. YouTube is a top platform with millions or maybe billions of videos, so it seemed like a great idea to download videos from there and convert them to audio.

### Dataset
youtube-dl program was used to download videos with its captions and also convert them to .wav. youtube-dl has a ffmpeg dependency - consider downloading its binaries from ffmpeg.org and add its path to enviromental variables. Python's pydub library was also used in order to split the big audio file that was selected into hundreds of chunks that would compose the dataset. Splitting the big audio file was not the easiest task. The first thought was to split it uniformly, but this proved to be a bad idea, as most of the chunks contained words that were wrongly cut. The next thought was to split it there where silence (>1s) existed. This resulted in unequal chunks that also couldn't be used, as it was very difficult to match them with their respective text. The third thought, which was also the one implemented, was to cut the big audio file there where the caption file indicated. Regular Expressions were used in order to isolate the hh:mm:ss.msmsms patterns from the caption file. pydub library was again used in order to create the chunks using the spesific start and end points that were extracted from caption file. 

IBM's Watson Tone Analyzer (text-based analyzer) was afterwards used in order to identify the emotions from the caption file. These emotions were stored and used as the ground truths of the learning procedure, which follows exactly the same steps as the First Approach. Each chunk is named following th
