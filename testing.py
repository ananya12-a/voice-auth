from voice_auth import *
import numpy as np
import pandas as pd
import openpyxl

wav_files = ["aa1.wav", "aa2.wav", "aa3.wav", "aa4.wav", "aa5.wav", "aa6.wav", "aa7.wav", "aa8.wav", "aa9.wav", "aa10.wav",  "dg1.wav", "dg2.wav", "st-61.wav", "st-62.wav", "st-63.wav", "st-64.wav", "st-65.wav", "st-66.wav", "st-67.wav", "st-68.wav", "st-69.wav", "st-70.wav"]
users_enroll = ["Ananya", "Harshi", "Dhruv", "Amy", "Collin", "Ethan"]
users_sample = ["Naveed"]
users = users_enroll + users_sample
result = np.zeros((len(users), len(wav_files)))

for user in users_enroll:
    enroll(user, "data/enroll/" + user+".wav")

for i, wav in enumerate(wav_files):
    for j,user in enumerate(users):
        result[j][i] = recognize("data/input-samples/" + wav, user)
print(result)

## convert your array into a dataframe
df = pd.DataFrame (result)

## save to xlsx file

filepath = 'testing-results.xlsx'

df.to_excel(filepath, index=False)