from audioclipextractor import AudioClipExtractor, SpecsParser
import os

directory = os.getcwd()
wav_files = [f for f in os.listdir(directory) if f.endswith('.wav')]
sub_files = [f for f in os.listdir(directory) if f.endswith('en.txt')]

# Inicialize the extractor
ext = AudioClipExtractor(wav_files[0], directory)

# Define the clips to extract
# It's possible to pass a file instead of a string
specs = sub_files[0]

# Extract the clips according to the specs and save them as a zip archive
ext.extractClips(specs, directory)