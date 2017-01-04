# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 23:04:28 2016

Copy task for DNI experiments

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
        y = numpy.zeros((examples,2*sequence_length+pause+1,n_in-1),dtype='float32')
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
            # output is blank character until copy character is input
            y[ex,numpy.arange(sequence_length+pause+1),n_in-2] = 1
            # repeat the original sequence
            y[ex,sequence_length+pause+1+numpy.arange(sequence_length),oneloc] = 1
        return x,y
    x_train,y_train = generate_data(n_train)
    x_val,y_val = generate_data(n_val)
    return [x_train,y_train,x_val,y_val]


# -----------------------------------------------------------------------------
# Experiments
# -----------------------------------------------------------------------------

def experiment(train_fcn,lr,lr_decay,dni_scale,batch_size,
               test_fcn,n_train,n_val,patience):
    seqlens = []
    loss = []
    dni_err = []
    dldp_l2 = []
    dniJ_l2 = []
    val_loss = []
    
    best_val = numpy.inf
    epoch = 0
    train_idx = range(n_train)
    
    seqlen = 1
    init_patience = patience
    while patience>0:
        start_time = timeit.default_timer()
        
        # turn the dni off for the start of training
        if epoch<4:
            dni_scale_epoch = 0
        else:
            dni_scale_epoch = dni_scale
        
        # re-generate data at each epoch (essential as seqlen>10)
        x_train,y_train,x_val,y_val = data(n_in,n_train,n_val,seqlen,0)   
        
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
            loss_batch,dni_err_batch, \
                dldp_l2_batch,dniJ_l2_batch = train_fcn(x_batch,
                                                        y_batch,
                                                        lr,
                                                        dni_scale_epoch)
            loss_epoch += loss_batch
            dni_err_epoch += dni_err_batch
            dldp_l2_epoch += dldp_l2_batch
            dniJ_l2_epoch += dniJ_l2_batch
        loss_epoch = loss_epoch/n_train_batches
        end_time = timeit.default_timer()
        print('Epoch %d  -----  sequence: %i  -----  time per example (msec): %f' \
             % (epoch,seqlen,1000*(end_time-start_time)/x_train.shape[0]))
        print('Training loss   = %f,   DNI error = %f, |dLdp| = %f, |dniJ| = %f' \
              % (loss_epoch,dni_err_epoch,dldp_l2_epoch,dniJ_l2_epoch))
        sys.stdout.flush() # force print to appear
        seqlens.append(seqlen)
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
        
        if val_loss_epoch<best_val:
            best_val = val_loss_epoch
            patience = init_patience
        else:
            patience -= 1
        
        # increase seqlen once it gets good enough
        if val_loss_epoch<0.15:
            patience = init_patience
            best_val = numpy.inf
            seqlen += 1
            x_train,y_train,x_val,y_val = data(n_in,n_train,n_val,seqlen,0)
            print('==========================================================')
            print('              Increasing sequence length')
            print('==========================================================')
        
        # set up next epoch
        epoch += 1
        lr = lr*lr_decay
    
    return seqlens, loss, dni_err, dldp_l2, dniJ_l2, val_loss

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

def test_lstm_dni(n_in,n_hidden,n_out,dni_steps,dni_scale,
                  lr,lr_decay,n_train,n_val,batch_size,patience):
    model = recurrent.lstm_dni(n_in,n_hidden,n_out,dni_steps)
    train = model.train()
    test = model.test()
    return experiment(train,lr,lr_decay,dni_scale,batch_size,
                      test,n_train,n_val,patience)

def test_lstm_trace(n_in,n_hidden,n_out,dni_steps,dni_scale,
                  lr,lr_decay,n_train,n_val,batch_size,patience,
                  trace_incr,trace_norm,trace_decay):
    model = recurrent.lstm_trace(n_in,n_hidden,n_out,dni_steps,
                                 trace_incr=trace_incr,trace_norm=trace_norm)
    model_train = model.train()
    train = lambda x,y,l,d: model_train(x,y,l,d,trace_decay)
    test = model.test()
    return experiment(train,lr,lr_decay,dni_scale,batch_size,
                      test,n_train,n_val,patience)

if __name__ == "__main__":
    import graph
    import itertools
    import argparse
    parser = argparse.ArgumentParser(description='Run DNI experiments')
    parser.add_argument('--dni_steps',nargs='*',type=int,
                        default=[1])
    parser.add_argument('--learnrate',nargs='*',type=float,
                        default=[7e-5])
    parser.add_argument('--models',nargs='*',type=str,
                        default=['lstm'])
    dni_steps = parser.parse_args().dni_steps
    lr = parser.parse_args().learnrate[0]
    models = parser.parse_args().models
    
    # make some data
    n_in = 4 # one-hot encoding, n_in-2 words + pause + copy
    n_out = n_in-1 # n_in-2 words + blank
    n_train = 20*256
    n_val = 256
    
    # test dni
    lr_decay = 0.995
    patience = 500
    batch_size = 256 # from paper
    n_hidden = 256 # from paper
    dni_scales = [0,0.1,1]
    
    for steps,model in itertools.product(dni_steps,models):
        print('model: %s' % model)
        print('dni_steps = %i' % steps)
        for dni_scale in dni_scales:
            # run the experiment
            if model == 'lstm':
                seqlen,loss,dni_err,dldp_l2,dniJ_l2,val_loss = \
                    test_lstm_dni(n_in,n_hidden,n_out,steps,dni_scale,
                                 lr,lr_decay,n_train,n_val,batch_size,patience)
                # log the result
                filename = model+'_copytest_DNI'+str(steps)
                log_results(filename,0,seqlen,steps,dni_scale,n_in,n_hidden,
                            n_out,loss,dni_err,dldp_l2,dniJ_l2,val_loss)
                # make graphs
                graph.make_all(filename,2)
            elif model == 'lstm_trace':
                trace_incrs = ['l1','l2','l1_stepnorm','l2_stepnorm',
                               'l1','l2','l1_stepnorm','l2_stepnorm']
                trace_norms = ['l1','l2','l1_inv','l2_inv',
                               'l1_inv','l2_inv','l1','l2']
                for trace_incr,trace_norm in zip(trace_incrs,trace_norms):
                    for trace_decay in [0.5,0.9,0.99]:
                        seqlen,loss,dni_err,dldp_l2,dniJ_l2,val_loss = \
                            test_lstm_trace(n_in,n_hidden,n_out,steps,dni_scale,
                                         lr,lr_decay,n_train,n_val,batch_size,patience,
                                         trace_incr,trace_norm,trace_decay)
                        # log the result
                        filename = model+'_copytest_DNI'+str(steps) \
                                   +'_incr'+trace_incr+'_norm'+trace_norm \
                                   +'_decay'+str(trace_decay).replace('.','p')
                        log_results(filename,0,seqlen,steps,dni_scale,n_in,n_hidden,
                                    n_out,loss,dni_err,dldp_l2,dniJ_l2,val_loss)
                        # make graphs
                        graph.make_all(filename,2)
            else:
                print('unknown model type')
            
            
            
            






















