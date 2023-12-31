from scipy.ndimage import binary_dilation
from pathlib import Path
from typing import Optional, Union
from warnings import warn
import numpy as np
import librosa
import struct
import os
import soundfile as sf
from scipy.io import wavfile
from pydub import AudioSegment
from pydub.silence import split_on_silence
import noisereduce as nr
import parameters as p

try:
    import webrtcvad
except:
    warn("Unable to import 'webrtcvad'. This package enables noise removal and is recommended.")
    webrtcvad=None

int16_max = (2 ** 15) - 1
audio_norm_target_dBFS = -30
vad_window_length = 30
vad_moving_average_width = 8
vad_max_silence_length = 6

def get_extension(filename):
    return os.path.splitext(filename)[1][1:]

def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    dBFS_change = target_dBFS - 10 * np.log10(np.mean(wav ** 2))
    if (dBFS_change < 0 and increase_only) or (dBFS_change > 0 and decrease_only):
        return wav
    return wav * (10 ** (dBFS_change / 20))

def trim_long_silences(fpath):
    # """
    # Ensures that segments without voice in the waveform remain no longer than a 
    # threshold determined by the VAD parameters in params.py.

    # :param wav: the raw waveform as a numpy array of floats 
    # :return: the same waveform with silences trimmed away (length <= original wav length)
    # """
    # # Compute the voice detection window size
    # samples_per_window = (vad_window_length * sampling_rate) // 1000
    
    # # Trim the end of the audio to have a multiple of the window size
    # wav = wav[:len(wav) - (len(wav) % samples_per_window)]
    
    # # Convert the float waveform to 16-bit mono PCM
    # pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))
    
    # # Perform voice activation detection
    # voice_flags = []
    # vad = webrtcvad.Vad(mode=3)
    # for window_start in range(0, len(wav), samples_per_window):
    #     print(voice_flags)
    #     window_end = window_start + samples_per_window
    #     voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
    #                                      sample_rate=sampling_rate))
    # voice_flags = np.array(voice_flags)
    
    # # Smooth the voice detection with a moving average
    # def moving_average(array, width):
    #     array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
    #     ret = np.cumsum(array_padded, dtype=float)
    #     ret[width:] = ret[width:] - ret[:-width]
    #     return ret[width - 1:] / width
    
    # audio_mask = moving_average(voice_flags, vad_moving_average_width)
    # audio_mask = np.round(audio_mask).astype(np.bool)
    
    # # Dilate the voiced regions
    # audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
    # audio_mask = np.repeat(audio_mask, samples_per_window)
    
    # return wav[audio_mask == True]
    #import required libraries
    # path to audio file
    # below method returns the active / non silent segments of the audio file 
    sound = AudioSegment.from_file(fpath, format='wav') 
    audio_chunks = split_on_silence(sound
                                ,min_silence_len = 100
                                ,silence_thresh = -45
                                ,keep_silence = 50
                            )

    # Putting the file back together
    combined = AudioSegment.empty()
    for chunk in audio_chunks:
        combined += chunk
    combined.export(fpath, format = "wav")

# def convert_to_wav(fpath):
#     ext = get_extension(fpath)
#     if ext == 'm4a':
#         track = AudioSegment.from_file(fpath,  format= 'm4a')
#         res = track.export(os.path.splitext(fpath)[0]+'.wav', format='wav')
#         print(res)
#     elif ext == 'mp3':
#         sound = AudioSegment.from_mp3(fpath)
#         res = sound.export(os.path.splitext(fpath)[0]+'.wav', format="wav")
#         print(res)
#     elif ext == 'ogg':
#         song = AudioSegment.from_ogg(fpath)
#         res = song.export(os.path.splitext(fpath)[0]+'.wav', format="wav")
#         print(res)
#     elif ext == 'opus':
#         return
#     elif ext == 'mpeg':
#         return
#     elif ext == 'mp4':
#         return
#     elif ext == 'aac':
#         return

def convert_to_wav(input_file_path, output_file_path):
    try:
        # Load the audio file
        audio = AudioSegment.from_file(input_file_path)

        # Ensure the output file has a .wav extension
        if not output_file_path.lower().endswith('.wav'):
            output_file_path = os.path.splitext(output_file_path)[0] + '.wav'

        # Export the audio as a WAV file
        audio.export(output_file_path, format='wav')

        print(f"Conversion successful. Output WAV file: {output_file_path}")

        # Delete the original input file
        os.remove(input_file_path)
        print(f"Original file '{input_file_path}' deleted.")
    except Exception as e:
        print(f"An error occurred: {e}")

def preprocess_wav(fpath_or_wav: Union[str, Path, np.ndarray],
                   source_sr: Optional[int] = None,
                   normalize: Optional[bool] = True,
                   trim_silence: Optional[bool] = True,
                   reduce_noise: Optional[bool] = False,
                   change_sr: Optional[float] = None):
    """
    Applies the preprocessing operations used in training the Speaker Encoder to a waveform 
    either on disk or in memory. The waveform will be resampled to match the data hyperparameters.

    :param fpath_or_wav: either a filepath to an audio file (many extensions are supported, not 
    just .wav), either the waveform as a numpy array of floats.
    :param source_sr: if passing an audio waveform, the sampling rate of the waveform before 
    preprocessing. After preprocessing, the waveform's sampling rate will match the data 
    hyperparameters. If passing a filepath, the sampling rate will be automatically detected and 
    this argument will be ignored.
    :change_sr: bool to signify if sr should be set to the one in parameters.py
    """
    if get_extension(fpath_or_wav) != 'wav':
        convert_to_wav(fpath_or_wav, os.path.splitext(fpath_or_wav)[0]+'.wav')
        fpath_or_wav = os.path.splitext(fpath_or_wav)[0]+'.wav'
    # Load the wav from disk if needed
    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
        wav, source_sr = librosa.load(str(fpath_or_wav), sr=None)
    else:
        wav = fpath_or_wav
    sampling_rate,_ = wavfile.read(fpath_or_wav)
    # Resample the wav if needed
    if source_sr is not None and source_sr != sampling_rate:
        wav = librosa.resample(wav, source_sr, sampling_rate)
    if change_sr is not None:
        wav = librosa.resample(wav, orig_sr=source_sr, target_sr=change_sr)
    # Apply the preprocessing: normalize volume and shorten long silences 
    if normalize:
        wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)
    sf.write("processed.wav", wav.astype(np.float32), sampling_rate)
    if webrtcvad and trim_silence:
        trim_long_silences("processed.wav")
    if reduce_noise:
        # load data
        rate, data = wavfile.read("processed.wav")
        # last second cuts off so padding with 1 second
        data = np.pad(data, (0, rate), mode="constant")
        # perform noise reduction
        reduced_noise = nr.reduce_noise(y=data, sr=rate)
        wavfile.write("processed.wav", rate, reduced_noise)