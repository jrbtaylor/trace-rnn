# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 16:11:15 2016

@author: jrbtaylor
"""

#%%
import os
import numpy
import csv
import matplotlib
matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.
import matplotlib.pyplot as plt

def _hasdigits(string):
        return any(char.isdigit() for char in string)

# check which columns are plottable data
def _isplottable(csvfile):
    if not csvfile[-4:]=='.csv':
        csvfile = csvfile+'.csv'
    file = open(csvfile,'r')
    reader = csv.reader(file,delimiter=',')
    
    # first row is category names
    categories = reader.next()
    
    isdata = [False]*len(categories)
    for row in reader:
        for col in range(len(categories)):
            # data is stored in string like "['2.34','3.45']"...
            if '[' in row[col]:
                isdata[col] = True
    return isdata

def make_graph(csvfile,metric,title,saveto,rows_to_plot=[],hyperparam_idx=0):
    # open the file
    if not csvfile[-4:]=='.csv':
        csvfile = csvfile+'.csv'
    file = open(csvfile,'r')
    reader = csv.reader(file,delimiter=',')
    
    # read the labels from the first row
    categories = reader.next()
    assert(metric in categories)
    
    # read the desired metric
    col = categories.index(metric)
    data = []
    hyperparam = []
    for row in reader:
        # if rows_to_plot is not given, plot all
        if row in rows_to_plot or rows_to_plot==[]:
            # note: data is stored in string like "['2.34', '3.45']"
            datastring = row[col]
            datastring = datastring[1:-1] # strip '[' and ']'
            datastring = datastring.split(', ') # split into numbers
            if all([_hasdigits(d) for d in datastring]):
                data.append([float(d) for d in datastring])
                hyperparam.append(row[hyperparam_idx])

    # plot it
    if len(data)>1: # don't want to save blank graphs
        print('making graph for '+metric)
        # loop through each row and plot
        for idx,d in enumerate(data):
            plt.plot(range(1,len(d)+1),d,label=hyperparam[idx])
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.title(title)
        
        # basic outlier detection for setting graph limits
        ymaxs = [numpy.nanmax(d) for d in data]
        ylim = numpy.nanmax(ymaxs)
        everythingelse = [ymaxs[i] for i in range(len(ymaxs)) \
                            if i!=numpy.nanargmax(ymaxs)]
        if ylim > 10*numpy.nanmax(everythingelse):
            ylim = numpy.nanmax(everythingelse)
        plt.ylim(0,ylim)
        plt.legend(title=categories[hyperparam_idx])
        
        # save it
        subfolder = os.path.join(os.getcwd(),'results')
        if not os.path.isdir(subfolder):
            os.mkdir(subfolder)
        subfolder = os.path.join(subfolder,csvfile.split('.')[0])
        if not os.path.isdir(subfolder):
            os.mkdir(subfolder)
        if not saveto[-4:]=='.png':
            saveto = saveto+'.png'
        plt.savefig(os.path.join(subfolder,saveto))
        plt.close()
    else:
        print('Empty graph not saved for '+csvfile+', '+metric)
    file.close()


def make_all(csvfile,hyperparam_idx=0):    
    # open the file
    if not csvfile[-4:]=='.csv':
        csvfile = csvfile+'.csv'
    file = open(csvfile,'r')
    reader = csv.reader(file,delimiter=',')
    
    # read the labels from the first row
    categories = reader.next()
    file.close()
    
    # check which columns are plottable data
    is_data = _isplottable(csvfile)
    categories = [c for i,c in enumerate(categories) if is_data[i]]

    # loop through everything and make all the graphs
    for metric in categories:
        make_graph(csvfile,metric,title=metric,saveto=csvfile[:-4]+'_'+metric,
                   hyperparam_idx=hyperparam_idx)


















