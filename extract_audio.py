# coding: utf-8
import subprocess
import config as cfg
import csv
import os
import sys
from pandas import Series, DataFrame

print sys.argv[0]
msrvtt_dir = os.path.join('data', 'msrvtt')

train_dir = os.path.join(msrvtt_dir, 'train')
test_dir = os.path.join(msrvtt_dir, 'test')
validate_dir = os.path.join(msrvtt_dir, 'validate')

count = 0
'''extract audio from training videos'''
'''
for v in os.listdir(train_dir):
    cmd = 'ffmpeg -i ' + os.path.join(train_dir, v) + ' -ab 160k -ac 2 -ar 44100 -vn ' + os.path.join(msrvtt_dir,
                                                                                                      'train_audio',
                                                                                                      v.split('.')[
                                                                                                          0] + '.wav')
    subprocess.call(cmd, shell=True)
'''    
    
'''extract audio from test videos'''
'''
for v in os.listdir(test_dir):
    cmd = 'ffmpeg -i ' + os.path.join(test_dir, v) + ' -ab 160k -ac 2 -ar 44100 -vn ' + os.path.join(msrvtt_dir,
                                                                                                      'test_audio',
                                                                                                      v.split('.')[
                                                                                                          0] + '.wav')
    subprocess.call(cmd, shell=True)
'''
'''extract audio from validate videos'''
'''
for v in os.listdir(validate_dir):
    cmd = 'ffmpeg -i ' + os.path.join(validate_dir, v) + ' -ab 160k -ac 2 -ar 44100 -vn ' + os.path.join(msrvtt_dir,
                                                                                                      'validate_audio',
                                                                                                      v.split('.')[
                                                                                                          0] + '.wav')
    subprocess.call(cmd, shell=True)
'''

'''there is some videos do not contains audio signal'''
count = 0

video_id = []
for v in os.listdir(os.path.join(msrvtt_dir, 'train_audio')):
    id = v.split('.')[0][5:]
    video_id.append(id)
    
for v in os.listdir(os.path.join(msrvtt_dir, 'test_audio')):
    id = v.split('.')[0][5:]
    video_id.append(id)
    
for v in os.listdir(os.path.join(msrvtt_dir, 'validate_audio')):
    id = v.split('.')[0][5:]
    video_id.append(id)
    
with open('video_id.txt', 'w')as f:
    for id in video_id:
        f.write(str(id) + '\n')
    