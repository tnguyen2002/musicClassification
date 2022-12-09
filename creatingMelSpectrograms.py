import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from glob import glob
# import skimage.io
import warnings

warnings.filterwarnings("ignore")
base_path = "data/fma_medium/"
files = sorted(list(glob(base_path + "/*/*.mp3")))

# hl = 512  # number of samples per time-step in spectrogram
# hi = 128  # Height of image
# wi = 384  # Width of image

warnings.filterwarnings("ignore")
base_path = "data/fma_medium/"
files = sorted(list(glob(base_path + "/*/*.mp3")))

# hl = 512  # number of samples per time-step in spectrogram
# hi = 128  # Height of image
# wi = 384  # Width of image

# print(len(files))
# base_out = "images/"
base_out = "melSpectrogramsColored/"
for i in range(len(files)):
    #     print(i)
    file = files[i]
    print(str(i) + " = " + file)
    filename = os.path.basename(file)[:-4]
    try:
        x, sr = librosa.load(file, sr=None, mono=True)
        stft = np.abs(librosa.stft(y=x, n_fft=2048, hop_length=512))
        mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
        mel = librosa.power_to_db(mel, ref=np.max)
        librosa.display.specshow(
            mel, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
        plt.axis('off')
        plt.savefig(base_out + filename + ".png")
    except:
        print("Corrupted:" + str(file))
