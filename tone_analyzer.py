#Import the necessary libraries
import librosa
import librosa.display
import numpy as np
import os
import pandas as pd
import librosa
import glob
import scipy.io.wavfile
import sys
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import tensorflow as tf
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, BatchNormalization
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from random import gauss
import json

#Define the needed variables containing the paths
dir = os.getcwd()
audio_dir = dir + '\\' + 'Audio_Speech_Actors_01-24\\'
audio_list= os.listdir(audio_dir)

#Plot the audiofile's waveform
data, sampling_rate = librosa.load(audio_dir + audio_list[0])
plt.figure(figsize=(15, 5))
librosa.display.waveplot(data, sr=sampling_rate)

##Plot the audiofile's spectogram
#sr,x = scipy.io.wavfile.read(audio_dir + audio_list[0])
#
##Parameters: 10ms step, 30ms window
#nstep = int(sr * 0.01)
#nwin  = int(sr * 0.03)
#nfft = nwin
#
#window = np.hamming(nwin)
#
##Take windows x[n1:n2]
##Generate and loop over n2 such that all frames fit within the waveform
#
#nn = range(nwin, len(x), nstep)
#
#X = np.zeros( (len(nn), nfft//2) )
#
#for i,n in enumerate(nn):
#    xseg = x[n-nwin:n]
#    z = np.fft.fft(window * xseg, nfft)
#    X[i,:] = np.log(np.abs(z[:nfft//2]))
#
#plt.imshow(X.T, interpolation='nearest',
#    origin='lower',
#    aspect='auto')
#
#plt.show()

#audio_list[0]

#Set the labels
feeling_list = []
for item in audio_list:
    if item[6:-16]=='02':
        feeling_list.append('calm')
    elif item[6:-16]=='03':
        feeling_list.append('happy')
    elif item[6:-16]=='04':
        feeling_list.append('sad')
    elif item[6:-16]=='05':
        feeling_list.append('angry')
    elif item[6:-16]=='06':
        feeling_list.append('fearful')
    elif item[6:-16]=='07':
        feeling_list.append('disgust')
    elif item[6:-16]=='08':
        feeling_list.append('surprised')
    elif item[6:-16]=='01':
        feeling_list.append('neutral')
    else:
        print('There are files that are not labeled')

#Create a dataframe holding the labels
labels = pd.DataFrame(feeling_list, columns =['labels'])
#labels[500:550]

#Extract the MFCC audio features using librosa, obtain the mean of each dimension and store them in a dataframe
df = pd.DataFrame(columns=['feature'])
bookmark=0
for index, y in enumerate(audio_list):
    y, sample_rate = librosa.load(audio_dir + y)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=40).T,axis=0)
    feature = mfccs
    #[float(i) for i in feature]
    #feature1=feature[:135]
    df.loc[bookmark] = [feature]
    bookmark=bookmark+1

#df[:5]

features = pd.DataFrame(df['feature'].values.tolist())
#features[:5]
features_and_labels = pd.concat([features,labels], axis=1)
#features_and_labels[:5]

#Shuffle data
features_and_labels = shuffle(features_and_labels)
#features_and_labels[:5]

#Replace NaNs, as they result in training problems
features_and_labels.fillna(0, inplace= True)

#Another way to split the data into training, validation and test
#random80 = np.random.rand(len(features_and_labels)) < 0.8
#train = features_and_labels[random80]
#validation = features_and_labels[~random80]
#test =

#Split the data in training, validation and test data (80%, 10%, 10%)
train = features_and_labels.sample(frac=0.8)
temp = features_and_labels[~features_and_labels.apply(tuple,1).isin(train.apply(tuple,1))]
validation = temp.sample(frac=0.5)
test = temp[~temp.apply(tuple,1).isin(validation.apply(tuple,1))]

#Split each dataframe in two different dataframes contationg the features and labels (this will be useful for the CNN later)
trainfeatures = train.iloc[:, :-1]
trainlabel = train.iloc[:, -1:]
validationfeatures = validation.iloc[:, :-1]
validationlabel = validation.iloc[:, -1:]
testfeatures = test.iloc[:, :-1]
testlabel = test.iloc[:, -1:]

#Regularize X_train and X_validation, X_test
#trainfeatures = (trainfeatures-(trainfeatures.mean().mean()))/trainfeatures.std().std()
#validationfeatures = (validationfeatures-(validationfeatures.mean().mean()))/validationfeatures.std().std()
#testfeatures = (testfeatures-(testfeatures.mean().mean()))/testfeatures.std().std()
#trainfeatures = trainfeatures/trainfeatures.max().max()
#validationfeatures = validationfeatures/validationfeatures.max().max()

#Convert data into numpy arrays
X_train = np.array(trainfeatures)
y_train = np.array(trainlabel)
X_validation = np.array(validationfeatures)
y_validation = np.array(validationlabel)
X_test = np.array(testfeatures)
y_test = np.array(testlabel)

lb = LabelEncoder()

#Encode ys in one-hot
y_train = np_utils.to_categorical(lb.fit_transform(y_train.ravel()))
y_validation = np_utils.to_categorical(lb.fit_transform(y_validation.ravel()))
y_test = np_utils.to_categorical(lb.fit_transform(y_test.ravel()))

X_train = np.expand_dims(X_train, axis=2)
X_validation = np.expand_dims(X_validation, axis=2)
X_test = np.expand_dims(X_test, axis=2)

#Define the architecture of the LSTM model
def LSTM_model():
    model = Sequential()
    model.add(LSTM(128, return_sequences=False, input_shape=(40, 1)))
    model.add(Dense(64))
    model.add(Dropout(0.4))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Dropout(0.4))
    model.add(Activation('relu'))
    model.add(Dense(8))
    model.add(Activation('softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model

#Define the architecture of the CNN model
def CNN_model():
    model = Sequential()
    model.add(Conv1D(32, 3, activation='relu', input_shape=(40,1)))
    model.add(Dropout(0.2))
    #model.add(BatchNormalization())
    #model.add(MaxPooling1D(pool_size=(2)))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(Dropout(0.2))
    #model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(2)))
    #model.add(Conv1D(128, 3, padding='same', activation='relu'))
    #model.add(Dropout(0.2))
    #model.add(BatchNormalization())
    #model.add(MaxPooling1D(pool_size=(2)))
    model.add(Conv1D(32, 3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=(2)))
    model.add(Flatten())
    model.add(Dense(8, activation='softmax'))
    model.summary()
    
    #optimizer = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)
    model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])
    return model

model = CNN_model()
history=model.fit(X_train, y_train, batch_size=20, epochs=300, validation_data=(X_validation, y_validation))

#Plot training & validation model accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

#Plot training & validation model loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

#Evaluate the model on test set
scores = model.evaluate(X_test, y_test, verbose=2)
print("Test Accuracy: {} \n Test Error: {}".format(scores[1], 1-scores[1]))

model_name = 'Tone_Analyzer_Model.h5'
save_dir = os.path.join(os.getcwd(), 'saved_models')

#Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

#Save model as JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
    
#Load json and create model
from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

#Load weights into loaded model
loaded_model.load_weights("saved_models/Tone_Analyzer_Model.h5")
print("Loaded model from disk")

#Predict emotions on test set
predictions = loaded_model.predict(X_test, batch_size=50, verbose=1)
predictions=predictions.argmax(axis=1).astype(int).flatten()
predictions = (lb.inverse_transform((predictions)))
predictions_df = pd.DataFrame({'predictedvalues': predictions})
#predictions_df[:10]

#Create a dataframe holding the ground truth values
ground_truth = y_test.argmax(axis=1).astype(int).flatten()
ground_truth = (lb.inverse_transform((ground_truth)))
ground_truth_df = pd.DataFrame({'ground_truth_values': ground_truth})
#ground_truth_df[:10]

#Create the join dataframe for comparison between values
join_df = ground_truth_df.join(predictions_df)

#Export the comparison in a CSV file
join_df.to_csv('Predictions.csv', index=False)