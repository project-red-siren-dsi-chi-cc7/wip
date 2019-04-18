import numpy as np
import pandas as pd
import librosa
import speech_recognition as sr

files=librosa.util.find_files('/home/ubuntu/new_audios/')

r = sr.Recognizer()
values=[]

for file in files:
    try:
        hellow=sr.AudioFile(file)
        with hellow as source:
            try:
                audio = r.record(source)
                s = r.recognize_google(audio)
                values.append([file,'Good',s])
            except:
                values.append([file,'Exception',''])
    except:
        values.append([files,'Exception',''])

    df = pd.DataFrame(values, columns=(['file_name','good_exception','audio_recognition']))
    df.to_csv('df_newspeech_recognition.csv')
