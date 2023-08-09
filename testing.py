from voice_auth import *
import numpy as np
import os
import pandas as pd
from preprocessing import convert_to_wav

# wav_files = ["aa1.wav", "aa2.wav", "aa3.wav", "aa4.wav", "aa5.wav", "aa6.wav", "aa7.wav", "aa8.wav", "aa9.wav", "aa10.wav",  "aa-resemble.wav", "aa-genny.wav",
#              "dg1.wav", "dg2.wav", 
#              "st-61.wav", "st-62.wav", "st-63.wav", "st-64.wav", "st-65.wav", "st-66.wav", "st-67.wav", "st-68.wav", "st-69.wav", "st-70.wav",
#              "ad-80.wav", "ad-81.wav", "ad-82.wav", "ad-83.wav", "ad-84.wav", "ad-85.wav", "ad-86.wav", "ad-87.wav", "ad-88.wav", "ad-89.wav", "ad-90.wav",
#               "rs110.wav","rs111.wav","rs112.wav","rs113.wav","rs114.wav","rs115.wav","rs116.wav","rs117.wav","rs118.wav","rs119.wav", "rs120.wav" ]
# users = range(2, 6)
users = range(1,32)
# wav_files = ["aa-resemble.wav", "aa-genny.wav"]
result = np.zeros((len(users), len(users)))

def rename_wav_file(old_file_path, new_file_name):
    try:
        if old_file_path.lower().endswith('.wav'):
            directory, filename = os.path.split(old_file_path)
            new_file_path = os.path.join(directory, new_file_name)

            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {filename} -> {new_file_name}")
        else:
            print("The provided file is not a WAV file.")

    except Exception as e:
        print(f"An error occurred: {e}")

to_convert_rename = False    
if to_convert_rename:
    for user in users:
        path = f"data/va-samples/{user}"
        dir_list = os.listdir(path)
        if get_extension(path + '/'+dir_list[0]) != 'wav':
            convert_to_wav(path + '/'+dir_list[0], os.path.splitext(path + '/'+dir_list[0])[0]+'.wav')
        rename_wav_file(path + '/'+ dir_list[0], "enroll.wav")
        if get_extension(path + '/'+dir_list[1]) != 'wav':
            convert_to_wav(path + '/'+dir_list[1], os.path.splitext(path + '/'+dir_list[1])[0]+'.wav')
        rename_wav_file(path + '/'+ dir_list[1], "sample.wav")


to_rename = False
if to_rename:
    for user in users:
        path = f"data/va-samples/{user}"
        dir_list = os.listdir(path)
        print(dir_list)
        if dir_list[0] != 'enroll.wav':
            rename_wav_files(path + '/'+ dir_list[0], "enroll")
        if dir_list[1] != 'sample.wav':
            rename_wav_files(path + '/'+dir_list[1], "sample")

for user in users:
    # path = f"data/va-samples/{user}"
    # dir_list = os.listdir(path)
    # print(path + "/" + dir_list[0])
    enroll(str(user), f"data/va-samples/{user}/enroll.wav")

for i in range(len(users)):
    for j,user in enumerate(users):
        # path = f"data/va-samples/{user}"
        # dir_list = os.listdir(path)
        # print(path + "/" + dir_list[1])
        result[i][j] = recognize(f"data/va-samples/{user}/sample.wav", str(user), is_eucl=False)

## convert the array into a dataframe
df = pd.DataFrame (result)

## save to xlsx file

filepath = 'testing-results.xlsx'

df.to_excel(filepath, index=False)