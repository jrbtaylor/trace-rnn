# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 14:53:18 2016

@author: jrbtaylor
"""

#%% Extract features for all videos
import os
import cPickle as pickle
import numpy

import lasagne

import data

# import pretrained model off-path
import sys
pretrained_dir = '/home/ml/jtaylo55/Documents/data/pretrained-nets'
sys.path.append(pretrained_dir)
import vgg19

# Build the model
cnn = vgg19.build_model()
pretrained = pickle.load(open(os.path.join(pretrained_dir,'vgg19.pkl')))
lasagne.layers.set_all_param_values(cnn['prob'],pretrained['param values'])

# Extract the features
def frame_feature(x):
    return numpy.array(lasagne.layers.get_output(cnn['fc7'],x,
                                              deterministic=True).eval())
dataloader = data.loader()
nfiles = len(dataloader.filenames)
features_path = os.path.join(os.curdir,os.path.join('framefeatures'))
if not os.path.isdir(features_path):
    os.mkdir(features_path)
for idx in range(nfiles):
    print(('idx %i / %i') % (idx,nfiles))
    video,label,group,clip = dataloader.get(idx)
    filename = 'l'+str(label)+'_g'+str(group)+'_c'+str(clip)+'.pkl'
    pickle.dump(frame_feature(video),
                open(os.path.join(features_path,filename),'wb'))


























