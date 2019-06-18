from pydub import AudioSegment
from pydub.silence import split_on_silence
import os

directory = os.getcwd()
sound = AudioSegment.from_mp3("The Best Movie Trailers of MARCH 2019-sUgk4g-81-E.wav")
chunks = split_on_silence(
    sound,

    # split on silences longer than 1000ms (1 sec)
    min_silence_len=1000,

    # anything under -16 dBFS is considered silence
    silence_thresh=-16, 

    # keep 200 ms of leading/trailing silence
    keep_silence=200
)

# now recombine the chunks so that the parts are at least 90 sec long
target_length = 5 * 1000
output_chunks = [chunks[0]]
for chunk in chunks[1:]:
    if len(output_chunks[-1]) < target_length:
        output_chunks[-1] += chunk
    else:
        # if the last output chunk is longer than the target length,
        # we can start a new one
        output_chunks.append(chunk)

# now your have chunks that are bigger than 90 seconds (except, possibly the last one)
        
for i, chunk in enumerate(output_chunks):
    chunk.export(directory + '\\Chunks\\' + 'chunk{0}.wav'.format(i), format="wav")
    
