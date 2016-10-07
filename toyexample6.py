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
# Make some simple recursive data
# -----------------------------------------------------------------------------

def data(n_in,n_hidden,n_out,n_train,n_val,sequence_length,period):
    rng = numpy.random.RandomState(1)
    # generate data from a pre-fixed RNN + noise
    def ortho_weight(ndim,rng):
        W = rng.randn(ndim, ndim)
        u, s, v = numpy.linalg.svd(W)
        return u.astype('float32')
    Wx_true = rng.uniform(low=-0.1,high=0.1,size=(n_in,n_hidden)).astype('float32')
    Wh_true = rng.uniform(low=-0.1,high=0.1,size=(n_hidden,n_hidden)).astype('float32')
    bh_true = rng.uniform(low=0,high=0.1,size=(n_hidden)).astype('float32')
    Wy_true = rng.uniform(low=-0.1,high=0.1,size=(n_hidden,n_out)).astype('float32')
    by_true = rng.uniform(low=-0.02,high=0.1,size=(n_out)).astype('float32')
    def np_relu(x):
        return numpy.maximum(numpy.zeros_like(x),x)
    def generate_data(examples):
        x = rng.uniform(low=-1,high=1,
                        size=(examples,sequence_length,n_in)).astype('float32')
        y = numpy.zeros((x.shape[0],sequence_length,n_out),dtype='float32')
        h_tm1 = numpy.zeros((n_hidden),dtype='float32')
        for t in range(sequence_length):
            h = np_relu(numpy.dot(x[:,t,:],Wx_true) \
                +numpy.dot(h_tm1,Wh_true)+bh_true)
            h_tm1 = h
            # artificially introduce long-term dependencies
            y[:,t,:] = 0.5*np_relu(numpy.dot(h[0],Wy_true)+by_true) \
                       +0.5*y[:,t-period,:]
        noise = rng.normal(loc=0.,
                           scale=0.1*numpy.std(y),
                           size=(x.shape[0],sequence_length,n_out))
        y += noise
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
        if val_loss_epoch<best_val:
            best_val = val_loss_epoch
            patience = init_patience
        else:
            patience -= 1
        
        # set up next epoch
        epoch += 1
        lr = lr*lr_decay
    
    return loss, dni_err, dldp_l2, dniJ_l2, val_loss

def test_dni(x_train,y_train,x_val,y_val,
             n_in,n_hidden,n_out,steps,dni_scale,
             lr,lr_decay,batch_size,n_epochs,patience):
    model = recurrent.rnn_dni(n_in,n_hidden,n_out,steps)
    train = lambda x,y,l: model.train()(x,y,l,dni_scale)
    test = model.test()
    return experiment(train,x_train,y_train,lr,lr_decay,batch_size,
                      test,x_val,y_val,n_epochs,patience)


if __name__ == "__main__":
    # make some data
    n_in = 256
    n_hidden = 512
    n_out = 10
    sequence_length = 555
    n_train = 1000
    n_val = 100
    steps = 5
    period = 6
    x_train,y_train,x_val,y_val = data(n_in,n_hidden,n_out,n_train,n_val,
                                       sequence_length,period)
    # test dni
    dni_scale = 1
    lr = 1e-2
    lr_decay = 0.99
    n_epochs = 1000
    patience = 20
    batch_size = 100
    test_dni(x_train,y_train,x_val,y_val,n_in,n_hidden,n_out,steps,dni_scale,
             lr,lr_decay,batch_size,n_epochs,patience)






















