# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 16:34:41 2016

Test the generally implemented RNN-DNI class in recurrent.py
    - implement momentum
    - track loss, dni_error, update magnitudes from both loss & dni components
    - compare with preliminary eligibility trace idea

@author: jrbtaylor
"""

import numpy
import timeit

import recurrent

# -----------------------------------------------------------------------------
# Simple recall data
# -----------------------------------------------------------------------------

def data(n_in,n_train,n_val,sequence_length,delay):
    rng = numpy.random.RandomState(1)
    def generate_data(examples):
        x = rng.uniform(low=0,high=1,
                        size=(examples,sequence_length,n_in)).astype('float32')
        y = numpy.zeros((x.shape[0],sequence_length,n_in),dtype='float32')
        for t in range(sequence_length):
            y[:,t,:] = x[:,t-delay,:]
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
        val_loss.append(val_loss_epoch)
        
        # early stopping
        if val_loss_epoch<0.995*best_val:
            best_val = val_loss_epoch
            patience = init_patience
        else:
            patience -= 1
        
        # set up next epoch
        epoch += 1
        lr = lr*lr_decay
    
    return loss, dni_err, dldp_l2, dniJ_l2, val_loss

def log_results(filename,line,delay,steps,dni_scale,n_in,n_hidden,n_out,
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
            writer.writerow(('Delay','DNI_steps','DNI_scale',
                             'n_in','n_hidden','n_out',
                             'Training_loss','DNI_err',
                             '|dldp|','|dniJ|','Validation_loss'))
        writer.writerow((delay,steps,dni_scale,n_in,n_hidden,n_out,
                         loss,dni_err,dldp_l2,dniJ_l2,val_loss))

def test_dni(x_train,y_train,x_val,y_val,
             n_in,n_hidden,n_out,steps,dni_scale,
             lr,lr_decay,momentum,batch_size,n_epochs,patience):
    model = recurrent.rnn_dni(n_in,n_hidden,n_out,steps)
    train = lambda x,y,l: model.train()(x,y,l,momentum,dni_scale)
    test = model.test()
    return experiment(train,x_train,y_train,lr,lr_decay,batch_size,
                      test,x_val,y_val,n_epochs,patience)


if __name__ == "__main__":
    import graph
    import itertools
    import argparse
    parser = argparse.ArgumentParser(description='Run DNI experiments')
    parser.add_argument('--delay',nargs='*',type=int,
                        default=[1,2,3,4,5])
    parser.add_argument('--dni_steps',nargs='*',type=int,
                        default=[3])
    parser.add_argument('--learnrate',nargs='*',type=float,
                        default=[1e-3])
    delays = parser.parse_args().delay,
    dni_steps = parser.parse_args().dni_steps,
    lr = parser.parse_args().learnrate[0]
    
    for steps,delay in itertools.product(dni_steps,delays):
        if type(steps)==list:
            steps = steps[0]
        if type(delay)==list:
            delay = delay[0]
	# make some data
        sequence_length = steps*(500//steps)
        n_in = 32
        n_train = 1000
        n_val = 100
        x_train,y_train,x_val,y_val = data(n_in,n_train,n_val,
                                           sequence_length,delay)
        # test dni
        lr_decay = 0.99
        momentum = 0.9
        n_epochs = 500
        patience = 50
        batch_size = 100
        n_hidden = 2*n_in
        n_out = n_in
        
        final_results = []
        for dni_scale in [0,1]:
            # run the experiment
            loss,dni_err,dldp_l2,dniJ_l2,val_loss = \
                test_dni(x_train,y_train,x_val,y_val,n_in,n_hidden,n_out,steps,dni_scale,
                         lr,lr_decay,momentum,batch_size,n_epochs,patience)
            
            # log the result
            filename = 'Delay'+str(delay)+'_DNI'+str(steps)
            log_results(filename,0,delay,steps,dni_scale,n_in,n_hidden,n_out,
                    loss,dni_err,dldp_l2,dniJ_l2,val_loss)
            
            # make graphs
            graph.make_all(filename,2)






















