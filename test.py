import os
import argparse

import numpy as np
import tensorflow as tf
import pandas as pd
from pandas import Series, DataFrame

from model.evqa import EVQA
import config as cfg
import util.dataset as dt

if __name__ == '__main__':
    video_ids = pd.read_csv('data/msrvtt_qa/answer_set.txt', header=None)[0]
    print video_ids
    pass