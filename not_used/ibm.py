from ibm_watson import ToneAnalyzerV3

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