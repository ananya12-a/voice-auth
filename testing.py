from voice_auth import *
import numpy as np
import pandas as pd

# wav_files = ["aa1.wav", "aa2.wav", "aa3.wav", "aa4.wav", "aa5.wav", "aa6.wav", "aa7.wav", "aa8.wav", "aa9.wav", "aa10.wav",  "dg1.wav", "dg2.wav", 
#              "st-61.wav", "st-62.wav", "st-63.wav", "st-64.wav", "st-65.wav", "st-66.wav", "st-67.wav", "st-68.wav", "st-69.wav", "st-70.wav",
#              "ad-80.wav", "ad-81.wav", "ad-82.wav", "ad-83.wav", "ad-84.wav", "ad-85.wav", "ad-86.wav", "ad-87.wav", "ad-88.wav", "ad-89.wav", "ad-90.wav",
#               "rs110.wav","rs111.wav","rs112.wav","rs113.wav","rs114.wav","rs115.wav","rs116.wav","rs117.wav","rs118.wav","rs119.wav", "rs120.wav" ]
users = ["Ananya", "Harshi", "Dhruv", "Amy", "Collin", "Ethan", "Rishabh", "Adnan", "Suyash"]
wav_files = ["aa-resemble.wav", "aa-genny.wav"]
result = np.zeros((len(wav_files), len(users)))

for user in users:
    enroll(user, "data/enroll/" + user+".wav")

for i, wav in enumerate(wav_files):
    for j,user in enumerate(users):
        result[i][j] = recognize("data/input-samples/" + wav, user, is_eucl=True)

## convert your array into a dataframe
df = pd.DataFrame (result)

## save to xlsx file

filepath = 'testing-results.xlsx'

df.to_excel(filepath, index=False)