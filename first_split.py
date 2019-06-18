from pydub import AudioSegment
from pydub.silence import split_on_silence
import os

directory = os.getcwd()
sound = AudioSegment.from_mp3("When They See Us _ Official Trailer [HD] _ Netflix-u3F9n_smGWY.wav")
chunks = split_on_silence(sound, min_silence_len=500, silence_thresh=-16)

for i, chunk in enumerate(chunks):
    chunk.export(directory + '\\' + 'chunk{0}.wav'.format(i), format="wav")
    
