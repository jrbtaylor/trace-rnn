# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 16:34:41 2016

Test the generally implemented RNN-DNI class in recurrent.py
    - track loss, dni_error, update magnitudes from both loss & dni components
    - compare with preliminary eligibility trace idea

@author: jrbtaylor
"""

import numpy
import timeit
import sys

import recurrent

# -----------------------------------------------------------------------------
# Common copy task
# n_in is the number of words + 2 (one for pause, one for copy)
# the n_in-2 words are randomly chosen and 1-hot encoded sequence_length times
# then the blank character is input pause times
# then the copy character is input (once)
# then the output repeats the original sequence (minus the pause & copy words)
# the input during the copy is the blank character again
# -----------------------------------------------------------------------------

def data(n_in,n_train,n_val,sequence_length,pause):
    rng = numpy.random.RandomState(1)
    def generate_data(examples):
        x = numpy.zeros((examples,2*sequence_length+pause+1,n_in),dtype='float32')
        y = numpy.zeros((examples,2*sequence_length+pause+1,n_in-2),dtype='float32')
        for ex in range(examples):
            # original sequence
            oneloc = rng.randint(0,n_in-2,size=(sequence_length))
            x[ex,numpy.arange(sequence_length),oneloc] = 1
            # blank characters before copy
            x[ex,sequence_length+numpy.arange(pause),n_in-2] = 1
            # copy character
            x[ex,sequence_length+pause,n_in-1] = 1
            # blank characters during copy
            x[ex,sequence_length+pause+1+numpy.arange(sequence_length),n_in-2] = 1
            # output
            y[ex,sequence_length+pause+1+numpy.arange(sequence_length),oneloc] = 1
        return x,y
    x_train,y_train = generate_data(n_train)
    x_val,y_val = generate_data(n_val)
    return [x_train,y_train,x_val,y_val]


# -----------------------------------------------------------------------------
# Experiments
# -----------------------------------------------------------------------------

def experiment(train_fcn,x_train,y_train,lr,lr_decay,batch_size,
               test_fcn,x_val,y_val,n_epochs,patience):
    loss = []
    dni_err = []
    dldp_l2 = []
    dniJ_l2 = []
    val_loss = []
    
    train_idx = range(x_train.shape[0])
    
    best_val = numpy.inf
    epoch = 0
    init_patience = patience
    while epoch<n_epochs and patience>0:
        start_time = timeit.default_timer()
        
        # train
        loss_epoch = 0
        dni_err_epoch = 0
        dldp_l2_epoch = 0
        dniJ_l2_epoch = 0
        numpy.random.shuffle(train_idx)
        n_train_batches = int(numpy.floor(x_train.shape[0]/batch_size))
        for batch in range(n_train_batches):
            batch_idx = train_idx[batch*batch_size:(batch+1)*batch_size]
            x_batch = x_train[batch_idx]
            y_batch = y_train[batch_idx]
            loss_batch,dni_err_batch,dldp_l2_batch,dniJ_l2_batch = train_fcn(x_batch,y_batch,lr)
            loss_epoch += loss_batch
            dni_err_epoch += dni_err_batch
            dldp_l2_epoch += dldp_l2_batch
            dniJ_l2_epoch += dniJ_l2_batch
        loss_epoch = loss_epoch/n_train_batches
        end_time = timeit.default_timer()
        print('Epoch %d  -----  time per example (msec): %f' \
             % (epoch,1000*(end_time-start_time)/x_train.shape[0]))
        print('Training loss  =  %f,   DNI error = %f, |dLdp| = %f, |dniJ| = %f' \
              % (loss_epoch,dni_err_epoch,dldp_l2_epoch,dniJ_l2_epoch))
        sys.stdout.flush() # force print to appear
        loss.append(loss_epoch)
        dni_err.append(dni_err_epoch)
        dldp_l2.append(dldp_l2_epoch)
        dniJ_l2.append(dniJ_l2_epoch)
        
        # validate
        val_loss_epoch = 0
        n_val_batches = int(numpy.floor(x_val.shape[0]/batch_size))
        for batch in range(n_val_batches):
            x_batch = x_val[batch*batch_size:(batch+1)*batch_size]
            y_batch = y_val[batch*batch_size:(batch+1)*batch_size]
            val_loss_epoch += test_fcn(x_batch,y_batch)
        val_loss_epoch = val_loss_epoch/n_val_batches
        print('Validation loss = %f' % val_loss_epoch)
        sys.stdout.flush() # force print to appear
        val_loss.append(val_loss_epoch)
        
        # early stopping
        if val_loss_epoch<best_val:
            best_val = val_loss_epoch
            patience = init_patience
        else:
            patience -= 1
        
        # or stop once it gets good enough
        # DNI paper stops <0.15 bits error
        if val_loss_epoch<0.15*numpy.log(2):
            patience = 0
        
        # set up next epoch
        epoch += 1
        lr = lr*lr_decay
    
    return loss, dni_err, dldp_l2, dniJ_l2, val_loss

def log_results(filename,line,sequence_length,steps,dni_scale,n_in,n_hidden,n_out,
                loss,dni_err,dldp_l2,dniJ_l2,val_loss,overwrite=False):
        import csv
        import os
        if not filename[-4:]=='.csv':
            filename = filename+'.csv'
        if line==0 and overwrite:
            # check if old log exists and delete
            if os.path.isfile(filename):
                os.remove(filename)
        file = open(filename,'a')
        writer = csv.writer(file)
        if line==0:
            writer.writerow(('sequence_length','DNI_steps','DNI_scale',
                             'n_in','n_hidden','n_out',
                             'Training_loss','DNI_err',
                             '|dldp|','|dniJ|','Validation_loss'))
        writer.writerow((sequence_length,steps,dni_scale,n_in,n_hidden,n_out,
                         loss,dni_err,dldp_l2,dniJ_l2,val_loss))

def test_dni(x_train,y_train,x_val,y_val,
             n_in,n_hidden,n_out,steps,dni_scale,
             lr,lr_decay,batch_size,n_epochs,patience):
    model = recurrent.rnn_dni(n_in,n_hidden,n_out,steps)
    train = lambda x,y,l: model.train()(x,y,l,dni_scale)
    test = model.test()
    return experiment(train,x_train,y_train,lr,lr_decay,batch_size,
                      test,x_val,y_val,n_epochs,patience)

def test_dni_trace(x_train,y_train,x_val,y_val,
             n_in,n_hidden,n_out,steps,dni_scale,
             lr,lr_decay,batch_size,n_epochs,patience):
    model = recurrent.rnn_trace(n_in,n_hidden,n_out,steps)
    trace_decay = 0.9
    train = lambda x,y,l: model.train()(x,y,l,dni_scale,trace_decay)
    test = model.test()
    return experiment(train,x_train,y_train,lr,lr_decay,batch_size,
                      test,x_val,y_val,n_epochs,patience)

def test_lstm_dni(x_train,y_train,x_val,y_val,
             n_in,n_hidden,n_out,steps,dni_scale,
             lr,lr_decay,batch_size,n_epochs,patience):
    model = recurrent.lstm_dni(n_in,n_hidden,n_out,steps)
    train = lambda x,y,l: model.train()(x,y,l,dni_scale)
    test = model.test()
    return experiment(train,x_train,y_train,lr,lr_decay,batch_size,
                      test,x_val,y_val,n_epochs,patience)

def test_lstm_trace(x_train,y_train,x_val,y_val,
             n_in,n_hidden,n_out,steps,dni_scale,
             lr,lr_decay,batch_size,n_epochs,patience):
    model = recurrent.lstm_trace(n_in,n_hidden,n_out,steps)
    trace_decay = 0.9
    train = lambda x,y,l: model.train()(x,y,l,dni_scale,trace_decay)
    test = model.test()
    return experiment(train,x_train,y_train,lr,lr_decay,batch_size,
                      test,x_val,y_val,n_epochs,patience)


if __name__ == "__main__":
    import graph
    import itertools
    import argparse
    parser = argparse.ArgumentParser(description='Run DNI experiments')
    parser.add_argument('--sequence_length',nargs='*',type=int,
                        default=[5,10,15])
    parser.add_argument('--pause',nargs='*',type=int,
                        default=[-1])
    parser.add_argument('--dni_steps',nargs='*',type=int,
                        default=[2])
    parser.add_argument('--learnrate',nargs='*',type=float,
                        default=[7e-5])
    parser.add_argument('--model',nargs='*',type=str,
                        default=['lstm'])
    sequence_lengths = parser.parse_args().sequence_length
    pause = parser.parse_args().pause[0]
    dni_steps = parser.parse_args().dni_steps
    lr = parser.parse_args().learnrate[0]
    model = parser.parse_args().model[0]
    
    for steps,sequence_length in itertools.product(dni_steps,sequence_lengths):
        if type(steps)==list:
            steps = steps[0]
        if type(sequence_length)==list:
            sequence_length = sequence_length[0]
        
	   # make some data
        n_in = 4 # one-hot encoding, n_in-2 words + pause + copy
        n_out = n_in-2
        n_train = 20*256
        n_val = 256
        # note: total sequence length is 2*sequence_length+pause+1
        #       must be divisible by steps
        if pause==-1: # default setting is lowest that will make it reshapeable
            pause = (2*sequence_length+1)%steps
        x_train,y_train,x_val,y_val = data(n_in,n_train,n_val,
                                           sequence_length,pause)
        
        # test dni
        lr_decay = 1
        n_epochs = 1000
        patience = 10
        batch_size = 256 # from paper
        n_hidden = 256 # from paper
        dni_scales = [0,0.1]
        
        final_results = []
        for dni_scale in dni_scales:
            # run the experiment
            if model == 'rnn':
                loss,dni_err,dldp_l2,dniJ_l2,val_loss = \
                    test_dni(x_train,y_train,x_val,y_val,n_in,n_hidden,n_out,steps,dni_scale,
                             lr,lr_decay,batch_size,n_epochs,patience)
            elif model == 'rnn_trace':
                loss,dni_err,dldp_l2,dniJ_l2,val_loss = \
                    test_dni_trace(x_train,y_train,x_val,y_val,n_in,n_hidden,n_out,steps,dni_scale,
                             lr,lr_decay,batch_size,n_epochs,patience)
            elif model == 'lstm':
                loss,dni_err,dldp_l2,dniJ_l2,val_loss = \
                    test_lstm_dni(x_train,y_train,x_val,y_val,n_in,n_hidden,n_out,steps,dni_scale,
                             lr,lr_decay,batch_size,n_epochs,patience)
            elif model == 'lstm_trace':
                loss,dni_err,dldp_l2,dniJ_l2,val_loss = \
                    test_lstm_trace(x_train,y_train,x_val,y_val,n_in,n_hidden,n_out,steps,dni_scale,
                             lr,lr_decay,batch_size,n_epochs,patience)
            else:
                print('unknown model type')
            
            # log the result
            filename = model+'seqlen'+str(sequence_length)+'_DNI'+str(steps)
            log_results(filename,0,sequence_length,steps,dni_scale,n_in,n_hidden,n_out,
                    loss,dni_err,dldp_l2,dniJ_l2,val_loss)
            
            # make graphs
            graph.make_all(filename,2)






















