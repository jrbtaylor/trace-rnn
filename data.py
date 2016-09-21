# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 14:51:24 2016

@author: jrbtaylor
"""

#%% test dataloading across the network
import os
import cv2
from scipy.misc import imresize
import numpy
import math

class loader(object):
    def __init__(self,framerate=25./3., # all UCF videos are 25 fps
                 framesize=[224,224], # all UCF videos are 240x320
                 path='/home/ml/jtaylo55/Documents/data/UCF101/UCF-101'):
        """
        Make a dataloader for UCF101 dataset
        param: framerate  -- frames per second
        param: framesize  -- frame size in rows x cols
        """
        # UCF videos are 25 fps, force integer sample rate
        self.framerate = 25./numpy.round(25./framerate)
        self.framesize = framesize
        self.filenames = []
        self.labelnames = []
        self.labels = []
        self.groups = []
        self.clipidx = []
        for folder_idx,folder in enumerate(os.listdir(path)):
            subfolder = os.path.join(path,folder)
            if os.path.isdir(subfolder):
                self.labelnames.append(folder)
                for f in os.listdir(subfolder):
                    filename = os.path.join(subfolder,f)
                    if os.path.isfile(filename):
                        self.labels.append(folder_idx)
                        self.filenames.append(filename)
                        # filename is v_LabelName_gX_cY where X is the video ID
                        # to record to make sure clips from the same video end
                        # up in the same split
                        self.groups.append(int(f.split('_')[2].split('g')[1]))
                        self.clipidx.append(int(f.split('_')[3].split('c')[1].split('.')[0]))

    def get(self,idx):
        label = self.labels[idx]
        group = self.groups[idx]
        clip = self.clipidx[idx]
        videofile = cv2.VideoCapture(self.filenames[idx])
        fps = videofile.get(cv2.cv.CV_CAP_PROP_FPS)
        if math.isnan(fps): # sometimes fps info is missing from file
            fps = 25
        downsamplerate = numpy.floor(fps/float(self.framerate))
        nframes_orig = int(videofile.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        nframes = int(numpy.floor(nframes_orig/downsamplerate)+1)
        video = numpy.zeros([nframes]+[3]+self.framesize,dtype='float32')
        loading = True
        for f in range(nframes_orig):
            loading,frame = videofile.read()
            if loading and (f % downsamplerate)==0:
                frame = imresize(frame,self.framesize,interp='bilinear')
                frame = numpy.ndarray.transpose(frame,(2,1,0)) # colour x height x width
                video[int(numpy.floor(f/downsamplerate))] = frame/255.
        videofile.release()
        return video,label,group,clip





