from pydub import AudioSegment
from ibm_watson import ToneAnalyzerV3
import re
import os
import xlsxwriter

#Use the youtube-dl program to download the video with its captions and also convert it to .wav
#youtube-dl has a ffmpeg dependecy - consider downloading its binaries from ffmpeg.org and add its path to env var
os.system('powershell.exe youtube-dl https://www.youtube.com/watch?v=WeOuKguaBec --all-subs -x --audio-format "wav"')

#Make use of IBM's Watson Tone Analyzer to identify the emotions from the caption file
tone_analyzer = ToneAnalyzerV3(
    version='2017-09-21',
    iam_apikey='0IKhnkogxIch-CLjysQfvIl9KgUCF6evDU22iewT7uih',
    url='https://gateway-lon.watsonplatform.net/tone-analyzer/api'
)

#Create a list containing all the subtitles of the video - all the captions, in a line fashion exactly as they are in the .vtt file
#It was observed that the .vtt files of YouTube captions follow the same format (the same way that are written), so this can generalise well
subs = []
count = 0
with open("Beauty In The Broken (Full HD Movie, Love, Romance, Drama, English) _full free movies-WeOuKguaBec.en.vtt") as f:
    for line in f:
        count += 1
        if count > 5 and count % 3 ==0:
            subs.append(line)

#Create a list containing the emotions of each line of subtitles - captions that occured using the IBM's Watson Analyzer
#In the case where the Analyzer cannot identify the emotion, 'Unknown' is appended
emotions = []
for i in range(len(subs)):
    subs[i] = subs[i].replace(subs[i][-1], "")
    if len(subs[i]) != 0:
        tone_analysis = tone_analyzer.tone(subs[i], content_type='text/plain').get_result()
        if len(tone_analysis['document_tone']['tones']) == 1:
            emotions.append(tone_analysis['document_tone']['tones'][0]['tone_name'])
        elif len(tone_analysis['document_tone']['tones']) == 2:
            emotions.append(tone_analysis['document_tone']['tones'][1]['tone_name'])
        elif len(tone_analysis['document_tone']['tones']) == 0:
            emotions.append('Unknown')
        elif len(tone_analysis['document_tone']['tones']) == 3:
            emotions.append(tone_analysis['document_tone']['tones'][2]['tone_name'])
        else:
            emotions.append(tone_analysis['document_tone']['tones'][3]['tone_name'])
            print('We found a sentence with more than 3 emotions')
            print(i)
    else:
        emotions.append('Unknown')

#Read the .wav file
audio = AudioSegment.from_wav("Beauty In The Broken (Full HD Movie, Love, Romance, Drama, English) _full free movies-WeOuKguaBec.wav")

#Get the current working directory
directory = os.getcwd()

#Create a RE pattern that corresponds to the way time is written in .vtt YouTube caption files
timestamp = re.compile("(\d{2}\:\d{2}\:\d{2}\.\d{3})")

#Scan through the .vtt file and find out and append to two lists the starting and ending times of each line - phrase 
start_times = []
end_times = []
count = 0
with open("Beauty In The Broken (Full HD Movie, Love, Romance, Drama, English) _full free movies-WeOuKguaBec.en.vtt") as f:
    for line in f:
        for match in re.findall(timestamp, line):
            count += 1
            if count % 2 > 0:
                start_times.append(match)
            else:
                end_times.append(match)
        #if timestamp.search(line):
            #start_times.append(line)
      
#Define a function that converts time from hh:mm:ss.msmsms format to milliseconds, which is the format that pydub library wants             
def get_millisec(time_str):
    h, m, s = time_str.split(':')
    s, ms = s.split('.')
    sec = int(h) * 3600 + int(m) * 60 + int(s)
    millisec = sec*1000 + int(ms)
    return millisec

#Duplicate emotions list in order to use it to export emotions in an Excel file and evaluate them
em = emotions    

#Encode the emotions in a way that will be more easy to read afterwards
for n, i in enumerate(emotions):
    if i == 'Confident':
        emotions[n] = '01'
    elif i == 'Tentative':
        emotions[n] = '02'
    elif i == 'Anger':
        emotions[n] = '03'
    elif i == 'Fear':
        emotions[n] = '04'
    elif i == 'Unknown':
        emotions[n] = '05'
    elif i == 'Analytical':
        emotions[n] = '06'
    elif i == 'Joy':
        emotions[n] = '07'
    elif i == 'Sadness':
        emotions[n] = '08'

#Convert all times to milliseconds and define chunks corresponding to the starting and ending times of each line
#Export those chunks and follow a syntactical formalism that includes the encoded emotion label
milli_start = []
milli_end = []
for i in range(len(start_times)):
    milli_start.append(get_millisec(start_times[i]))
    milli_end.append(get_millisec(end_times[i]))
    chunk = audio[milli_start[i]:milli_end[i]]
    chunk.export(directory + '\\Chunks\\' + 'chunk{}-{}.wav'.format(i, emotions[i]), format="wav")

#Export the names of chunks, the emotions occured from IBM's Watson Analyzer in an Excel file in order to evaluate them
chunks_directory = directory + '\\' + 'Chunks'
workbook = xlsxwriter.Workbook('IBM Tone Analyzer Evaluation.xlsx') 
worksheet = workbook.add_worksheet() 
row = 0
column = 0
content = [f for f in os.listdir(chunks_directory) if f.endswith('.wav')]

for index, item in enumerate(content): 
	worksheet.write(row, 0, item)
	worksheet.write(row, 1, item[index][-6:-4])
	row += 1
workbook.close()
	
row = 0
column = 1
for item in em:
    worksheet.write(row, column, item)
    row += 1
workbook.close()    