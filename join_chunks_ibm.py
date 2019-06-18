from pydub import AudioSegment
from ibm_watson import ToneAnalyzerV3
import re
import os

#Make use of IBM's Watson Tone Analyzer to identify the emotions from the caption file
tone_analyzer = ToneAnalyzerV3(
    version='2017-09-21',
    iam_apikey='0IKhnkogxIch-CLjysQfvIl9KgUCF6evDU22iewT7uih',
    url='https://gateway-lon.watsonplatform.net/tone-analyzer/api'
)

subs = []
count = 0
with open("Beauty In The Broken (Full HD Movie, Love, Romance, Drama, English) _full free movies-WeOuKguaBec.en.vtt") as f:
    for line in f:
        count += 1
        if count > 5 and count % 3 ==0:
            subs.append(line)

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

audio = AudioSegment.from_wav("Beauty In The Broken (Full HD Movie, Love, Romance, Drama, English) _full free movies-WeOuKguaBec.wav")

directory = os.getcwd()

timestamp = re.compile("(\d{2}\:\d{2}\:\d{2}\.\d{3})")

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
                    
def get_millisec(time_str):
    h, m, s = time_str.split(':')
    s, ms = s.split('.')
    sec = int(h) * 3600 + int(m) * 60 + int(s)
    millisec = sec*1000 + int(ms)
    return millisec

milli_start = []
milli_end = []
for i in range(len(start_times)):
    milli_start.append(get_millisec(start_times[i]))
    milli_end.append(get_millisec(end_times[i]))
    chunk = audio[milli_start[i]:milli_end[i]]
    chunk.export(directory + '\\Chunks\\' + 'chunk{}_{}.wav'.format(i, emotions[i]), format="wav")

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
    
for i in range(len(start_times)):
    chunk = audio[milli_start[i]:milli_end[i]]
    chunk.export(directory + '\\Chunks\\' + 'chunk{}-{}.wav'.format(i, emotions[i]), format="wav")
    