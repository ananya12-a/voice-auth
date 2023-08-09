#IMPORT SYSTEM FILES
import argparse
import scipy.io.wavfile as wavfile
import traceback as tb
import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, euclidean
import warnings
from keras.models import load_model
import logging
logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
#IMPORT USER-DEFINED FUNCTIONS
from feature_extraction import get_embedding, get_embeddings_from_list_file
from preprocessing import *
from preprocess import get_fft_spectrum
import parameters as p

# args() returns the args passed to the script
def args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--task',
                       help='Task to do. Either "enroll" or "recognize"',
                       required=True)
    parser.add_argument( '-n', '--name',
                        help='Specify the name of the person you want to enroll',
                        required=True)
    parser.add_argument('-f', '--file',
                        help='Specify the audio file you want to enroll',
                        type=lambda fn:file_choices(("csv","wav","flac"),fn),
                       required=True)
    ret = parser.parse_args()
    return ret


def enroll(name,file, change_sr: Optional[bool] = False):
    """Enroll a user with an audio file
        inputs: str (Name of the person to be enrolled and registered)
                str (Path to the audio file of the person to enroll)
        outputs: None"""

    print("Loading model weights from [{}]....".format(p.MODEL_FILE))
    # try:
    # model = load_model(p.MODEL_FILE)
    # except:
    #     print("Failed to load weights from the weights file, please ensure *.pb file is present in the MODEL_FILE directory")
    #     exit()
    
    # try:
    print("Processing enroll sample....")
    preprocess_wav(file)
    enroll_result = get_embedding("processed.wav", p.MAX_SEC)
    enroll_embs = np.array(enroll_result.tolist())
    speaker = name
    # except:
    #     print("Error processing the input audio file. Make sure the path.")
    # try:
    np.save(os.path.join(p.EMBED_LIST_FILE,speaker +".npy"), enroll_embs)
    print(f"Succesfully enrolled {name}")
    # except:
    #     print("Unable to save the user into the database.")

def enroll_csv(csv_file):
    """Enroll a list of users using csv file
        inputs:  str (Path to comma seperated file for the path to voice & person to enroll)
        outputs: None"""

    print("Getting the model weights from [{}]".format(p.MODEL_FILE))
    try:
        model = load_model(p.MODEL_FILE)

    except:
        print("Failed to load weights from the weights file, please ensure *.pb file is present in the MODEL_FILE directory")
        exit()
    print("Processing enroll samples....")
    try:
        #TODO: add preprocessing before getting embeddings
        enroll_results = get_embeddings_from_list_file(model, csv_file, p.MAX_SEC)
        enroll_embs = np.array([emb.tolist() for emb in enroll_results['embedding']])
        speakers = enroll_results['speaker']
    except:
        print("Error processing the input audio files. Make sure the csv file has two columns (path to file,name of the person).")

    i=0
    try:
        for i in range(len(speakers)):
            np.save(os.path.join(p.EMBED_LIST_FILE,str(speakers[i]) +".npy"), enroll_embs[i])
            print("Succesfully enrolled the user")
    except:
        print("Unable to save the user into the database.")

def get_users():
    if os.path.exists(p.EMBED_LIST_FILE):
        embeds = os.listdir(p.EMBED_LIST_FILE)
        for i, embed in enumerate(embeds):
            embeds[i] = embed[:-4]
        return embeds
    else:
        print("Path doesn't exist")

def recognize(file, name, is_eucl: Optional[bool] = False, change_sr: Optional[bool] = False):
    """Recognize the input audio file by comparing to saved users' voice prints
        inputs: str (Path to audio file of unknown person to recognize)
        outputs: str (Name of the person recognized)"""
    
    if os.path.exists(p.EMBED_LIST_FILE):
        embeds = os.listdir(p.EMBED_LIST_FILE)
    if len(embeds) == 0:
        print("No enrolled users found")
        exit()
    print("Loading model weights from [{}]....".format(p.MODEL_FILE))
    # try:
    #     model = load_model(p.MODEL_FILE)

    # except:
    #     print("Failed to load weights from the weights file, please ensure *.pb file is present in the MODEL_FILE directory")
    #     exit()
        
    # distances = {}
    print("Processing test sample....")
    preprocess_wav(file)
    test_result = get_embedding("processed.wav", p.MAX_SEC)
    test_embs = np.array(test_result.tolist())
    emb = name + '.npy'
    enroll_embs = np.load(os.path.join(p.EMBED_LIST_FILE,emb))
    if is_eucl:
        distance = euclidean(test_embs, enroll_embs)
    else:
        distance = cdist(test_embs, enroll_embs, metric="cosine")[0][0]
    # print(p.THRESHOLD)
    if distance<0.03:
        print("Authenticated: True")
        print("Score: ",distance)
    else:
        print("Authenticated: False")
        print("Score: ",distance)
    return distance
        
#Helper functions
def file_choices(choices,filename):
    ext = os.path.splitext(filename)[1][1:]
    if ext not in choices:
        parser.error("file doesn't end with one of {}".format(choices))
    return filename

def get_extension(filename):
    return os.path.splitext(filename)[1][1:]

if __name__ == '__main__':
    try:
        args = args()
    except Exception as e:
        print('An Exception occured, make sure the file format is .wav or .flac')
        exit()
    task = args.task
    file = args.file
    try:
        name = args.name
    except:
        if task =="enroll" and get_extension(file)!= 'csv':
            print("Missing Arguement, -n name is required for the user name")
            exit()
    if get_extension(file)=='csv':
        if task == 'enroll':
            enroll_csv(file)
        if task == 'recognize':
            print("Recognize arguement cannot process a comma-seperated file. Please specify an auido file")
    else:
        if task == 'enroll':
            enroll(name, file)
        if task == 'recognize':
            recognize(file, name)