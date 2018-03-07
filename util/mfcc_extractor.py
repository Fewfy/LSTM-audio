# coding: utf-8
import librosa
import numpy as np

class audiofeature_extractor(object):
    def __init__(self):
        pass
    
    def extract(self, wave, sample_rate):
        mfcc = librosa.feature.mfcc(wave, sample_rate)
        mfcc = np.pad(mfcc, ((0, 0), (0, 80-len(mfcc[0]))), mode='constant', constant_values=0)
        return np.array(mfcc)