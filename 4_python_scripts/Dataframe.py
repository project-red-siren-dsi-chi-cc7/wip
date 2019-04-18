#Import libraries
import pandas as pd
import numpy as np
import re
import string
import librosa

#Extract file list to process
files=librosa.util.find_files('/home/ubuntu/new_audios/')

#Review the file list (10 first elements)
files[0:10]

#Check how many files will be read
len(files)

#Loop through the file list and create a dataframe
values=[]
for file in files: 
	# y = audio time series
        # sr = sample rate of 'y'
        y, sr = librosa.load(file)
        
        # get the list of mean values extracted from different features
        stft = np.abs(librosa.stft(y))
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y, sr=sr).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
        tempogram = np.mean(librosa.feature.tempogram(y, sr=sr).T, axis=0)
        rolloff=np.mean(librosa.feature.spectral_rolloff(y, sr=sr).T, axis=0)
        chroma_cqt = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr).T, axis=0)
        chroma_cens = np.mean(librosa.feature.chroma_cens(y=y, sr=sr).T, axis=0)
        spectral_centroid=np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
        spectral_band=np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr).T, axis=0)
        spectral_flat=np.mean(librosa.feature.spectral_flatness(y=y).T, axis=0)
        spectral_contrast=np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
        
    
        # append to the list
        values.append([file,mfcc,chroma_stft,mel,contrast,tonnetz,tempogram,rolloff,chroma_cqt,chroma_cens,spectral_centroid,spectral_band,spectral_flat,spectral_contrast])

    # Create a DataFrame out of the list
        df = pd.DataFrame(values, columns=(['file_name','mfcc', 'chroma_stft','mel','contrast','tonnetz','tempogram','rolloff','chroma_cqt','chroma_cens','spectral_centroid','spectral_band','spectral_flat','spectral_contrast']))
        df.to_csv('df_testaudio_compressed.csv')

cols = df.columns[1:]
for col in cols:
    length = pd.DataFrame(df[col].tolist()).shape[1]
    col_seq = []
    for i in range(1, length+1):
        col_seq.append(f'{col}_{i}')
    dfs = [df, pd.DataFrame(df[col].tolist(), columns=col_seq)]
    df = pd.concat(dfs, axis=1).drop(col, axis=1)

    df.to_csv('df_testaudio.csv')
