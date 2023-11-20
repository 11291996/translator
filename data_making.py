import glob
import pysrt
from datetime import datetime
import pandas as pd
#first find srt paths
korean_srt = r'/Users/jaewanpark/Documents/Kigo Netflix Video Downloader/기묘한 이야기/**/**/*Korean.srt'
korean_srt_list = glob.glob(korean_srt)
english_srt = r'/Users/jaewanpark/Documents/Kigo Netflix Video Downloader/기묘한 이야기/**/**/*(CC).srt'
english_srt_list = glob.glob(english_srt)
#create empty list for paired data
new_korean_sub = []
new_english_sub = []
#create a loop for all srts
for _ in range(len(korean_srt_list)):
    korean_subs = pysrt.open(korean_srt_list[_])
    english_subs = pysrt.open(english_srt_list[_])
    #create a loop
    for korean_sub in korean_subs:
        for english_sub in english_subs:
            #checking matching timestamp
            if korean_sub.start.to_time().second == english_sub.start.to_time().second \
                and korean_sub.start.to_time().minute == english_sub.start.to_time().minute:
                #put sub in new lists
                new_korean_sub.append(korean_sub.text)
                new_english_sub.append(english_sub.text)

data = pd.DataFrame({'korean': new_korean_sub, 'english': new_english_sub})
data.to_csv('/Users/jaewanpark/Desktop/Friends_sub.csv', index = None)