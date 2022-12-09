import os
import IPython.display as ipd
import numpy as np
import pandas as pd
# import tensorflow as tf
from glob import glob
import PIL
from PIL import Image
import skimage.io


mfccFiles = sorted(list(glob("mfcc" + "/*.npy")))
melSpectrograms = sorted(list(glob("spectrogramDirectory" + '/*.png')))

base_out = "combinedImages/"
for i in range(len(mfccFiles)):
    print(i)
    print(mfccFiles[i])
    file = mfccFiles[i]
    filename = os.path.basename(file)[:-4]
    mfcc = np.load(mfccFiles[0])
    mfcc = mfcc.astype(np.uint8)
    melSpectrogram = Image.open(str(melSpectrograms[0]))
    melSpectrogram = np.array(melSpectrogram).astype(np.uint8)
    combined = np.vstack((melSpectrogram, mfcc))
    skimage.io.imsave(base_out + filename + ".png", combined)
