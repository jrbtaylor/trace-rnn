# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 10:55:35 2016

Fix the backprop issue in toyexample2 (no backprop more than 1 step)

@author: jrbtaylor
"""

import numpy
import theano
from theano import tensor as T

from collections import OrderedDict

# -----------------------------------------------------------------------------
# Make some exceedingly simple data
# -----------------------------------------------------------------------------

n_in = 3
sequence_length = 10
n_out = n_in
n_examples = 10000
rng = numpy.random.RandomState(1)
# inputs are vectors of uniform random numbers
x_train = rng.uniform(low=-1,high=1,
                      size=(n_examples,sequence_length,n_in)
                      ).astype('float32')
# output is the cumulative sums of inputs temporally and across the input dim
y_train = numpy.cumsum(numpy.cumsum(x_train,axis=2),axis=1).astype('float32')

x_train = theano.shared(x_train)
y_train = theano.shared(y_train)


online_training = True

if not online_training:
    # -----------------------------------------------------------------------------
    # Make a recursive linear model
    # -----------------------------------------------------------------------------
    
    x = T.tensor3('x')
    y = T.tensor3('y')
    
    # model is y_t = A*x_t + B*y_tm1
    A = theano.shared(rng.uniform(low=0,high=1,size=(n_in,n_out)).astype('float32'),
                      name='A')
    B = theano.shared(rng.uniform(low=0,high=1,size=(n_out,n_out)).astype('float32'),
                      name='B')
    def step(x_t,y_tm1,A_t,B_t):
        y_t = T.dot(x_t,A_t)+T.dot(y_tm1,B_t)
        return y_t
    
    # scan
    y0 = T.zeros((n_out,),dtype=theano.config.floatX)
    output, updates = theano.scan(fn=step,
                                  sequences=x.dimshuffle([1,0,2]),
                                  outputs_info=[T.alloc(y0,x.shape[0],n_out)],
                                  non_sequences=[A,B],
                                  strict=True,                                  
                                  truncate_gradient=1)
                                  
    
    # -----------------------------------------------------------------------------
    # Stochastic gradient descent
    # -----------------------------------------------------------------------------
    
    # Backprop
    cost = T.mean(T.sum(T.square(y.dimshuffle([1,0,2])-output),axis=1))
    gA = T.grad(cost,A)
    gB = T.grad(cost,B)
    
    # SGD
    learning_rate = 1e-5
    batch_size = 100
    updates = OrderedDict()
    updates[A] = A-learning_rate*gA
    updates[B] = B-learning_rate*gB
    index = T.lscalar()  # index to a [mini]batch
    train_step = theano.function(inputs=[index],
                                 outputs=(cost,output),
                                 updates=updates,
                                 givens={x:x_train[index*batch_size:(index+1)*batch_size],
                                         y:y_train[index*batch_size:(index+1)*batch_size]})
    
    # Loop through and train
    n_batches = int(numpy.floor(n_examples/batch_size))
    n_epochs = 100
    for epoch in range(n_epochs):
        train_loss = 0
        for batch in range(n_batches):
            batch_loss,output = train_step(batch)
            train_loss += batch_loss
            if numpy.any(numpy.isnan(output)):
                break
        print('Epoch %d loss: %f' % (epoch,train_loss))
    
else:
    # -----------------------------------------------------------------------------
    # With online updating
    # -----------------------------------------------------------------------------
    
    learning_rate = 1e-5
    bptt_limit = 1
    
    x = T.tensor3('x')
    y = T.tensor3('y')
    
    # shared variables for updates
    A = theano.shared(rng.uniform(low=0,high=1,size=(n_in,n_out)).astype('float32'))
    B = theano.shared(rng.uniform(low=0,high=1,size=(n_out,n_out)).astype('float32'))
    dodA = theano.shared(numpy.zeros((bptt_limit,n_in,n_out),dtype=theano.config.floatX))
    dodB = theano.shared(numpy.zeros((bptt_limit,n_out,n_out),dtype=theano.config.floatX))
    dodotm1 = theano.shared(numpy.ones((bptt_limit,n_out,n_out),dtype=theano.config.floatX))
    
    # model is y_t = x_t*A + y_tm1*B
    def step(x_t,y_t,o_tm1,A,B,dodA,dodB,dodotm1):
        # the model itself
        o_t = T.dot(x_t,A)+T.dot(o_tm1,B)
        mse = T.mean(T.sum(T.square(y_t-o_t),axis=0))
        
        # gradient of o_t w.r.t. A is x_t, w.r.t. B is o_tm1, w.r.t. o_tm1 is B
        dodA_t = T.repeat(T.shape_padright(T.mean(x_t,axis=0)),repeats=n_out,axis=1)
        dodA_up = T.concatenate([T.shape_padleft(dodA_t),dodA[:-1]],axis=0)
        dodB_t = T.repeat(T.shape_padright(T.mean(o_tm1,axis=0)),repeats=n_out,axis=1)
        dodB_up = T.concatenate([T.shape_padleft(dodB_t),dodB[:-1]],axis=0)
        dodotm1_t = B
        dodotm1_up = T.concatenate([T.shape_padleft(dodotm1_t),dodotm1[:-1]],axis=0)
        
        # deltaE:   update component from current error
        #           take mean over batch index
        #           and padleft so size is 1 x n_out
        deltaE = T.shape_padleft(T.mean(T.grad(mse,o_t),axis=0)) # mean over the batch index
        
        # deltaR:   update components over time from recurrence
        #           cumulative product effectively does backprop
        #           size is bptt_limit x n_out x n_out
        deltaR = T.cumprod(dodotm1_up,axis=0)
        
        # updates
#        dA = T.dot(deltaE,T.sum(T.batched_dot(deltaR,dodA_up),axis=0))
        dA = deltaE*T.sum(T.batched_dot(deltaR,dodA_up),axis=0)
        dB = deltaE*T.sum(T.batched_dot(deltaR,dodB_up),axis=0)

        updates = OrderedDict()
        updates[A] = A-learning_rate*dA
        updates[B] = B-learning_rate*dB
        updates[dodA] = dodA_up
        updates[dodB] = dodB_up
        updates[dodotm1] = dodotm1_up
        return [o_t,mse,dA],updates
    
    # scan
    y0 = T.zeros((n_out,),dtype=theano.config.floatX)
    [output,cost,monitor], updates = theano.scan(fn=step,
                                    sequences=[x.dimshuffle([1,0,2]),
                                               y.dimshuffle([1,0,2])],
                                    outputs_info=[T.alloc(y0,x.shape[0],n_out),
                                                  None,None],
                                    non_sequences=[A,B,dodA,dodB,dodotm1])
    
# NOTE: truncate_gradient has no effect on the returned gradients for each time step
#       meaning that using T.grad() within the step function is limiting it to 0-step backprop
    
    batch_size = 100
    index = T.lscalar()  # index to a [mini]batch
    train_step = theano.function(inputs=[index],
                                 outputs=(output,T.mean(cost),monitor),
                                 updates=updates,
                                 givens={x:x_train[index*batch_size:(index+1)*batch_size],
                                         y:y_train[index*batch_size:(index+1)*batch_size]})
    
    # Loop through and train
    n_batches = int(numpy.floor(n_examples/batch_size))
    n_epochs = 100
    for epoch in range(n_epochs):
        train_loss = 0
        for batch in range(n_batches):
            output,batch_loss,monitor = train_step(batch)
#            print(monitor.shape)
            train_loss += batch_loss
            if numpy.any(numpy.isnan(output)):
                break
        print('Epoch %d loss: %f' % (epoch,train_loss))
#        print(A_val)

















