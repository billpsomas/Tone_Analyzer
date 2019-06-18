from pydub import AudioSegment
import re
import os

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
    chunk.export(directory + '\\Chunks\\' + 'chunk{}.wav'.format(i), format="wav")