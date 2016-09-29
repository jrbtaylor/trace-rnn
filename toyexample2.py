# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 20:04:57 2016

Figure out how to update shared variables

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
    
    x = T.tensor3('x')
    y = T.tensor3('y')
    A = theano.shared(rng.uniform(low=0,high=1,size=(n_in,n_out)).astype('float32'))
    B = theano.shared(rng.uniform(low=0,high=1,size=(n_out,n_out)).astype('float32'))
    
    learning_rate = 1e-5
    
    # model is y_t = A*x_t + B*y_tm1
    def step(x_t,y_t,o_tm1,A,B):
        o_t = T.dot(x_t,A)+T.dot(o_tm1,B)
        mse = T.mean(T.sum(T.square(y_t-o_t),axis=0))
        updates = OrderedDict()
        updates[A] = A-learning_rate*T.grad(mse,A)
        updates[B] = B-learning_rate*T.grad(mse,B)
        return [o_t,mse],updates
    
    # scan
    y0 = T.zeros((n_out,),dtype=theano.config.floatX)
    [output,cost], updates = theano.scan(fn=step,
                                  sequences=[x.dimshuffle([1,0,2]),
                                             y.dimshuffle([1,0,2])],
                                  outputs_info=[T.alloc(y0,x.shape[0],n_out),
                                                None],
                                  non_sequences=[A,B])
    
# NOTE: truncate_gradient has no effect on the returned gradients for each time step
#       meaning that using T.grad() within the step function is limiting it to 0-step backprop
    
    batch_size = 100
    index = T.lscalar()  # index to a [mini]batch
    train_step = theano.function(inputs=[index],
                                 outputs=(output,T.mean(cost)),
                                 updates=updates,
                                 givens={x:x_train[index*batch_size:(index+1)*batch_size],
                                         y:y_train[index*batch_size:(index+1)*batch_size]})
    
    # Loop through and train
    n_batches = int(numpy.floor(n_examples/batch_size))
    n_epochs = 100
    for epoch in range(n_epochs):
        train_loss = 0
        for batch in range(n_batches):
            output,batch_loss = train_step(batch)
            train_loss += batch_loss
            if numpy.any(numpy.isnan(output)):
                break
        print('Epoch %d loss: %f' % (epoch,train_loss))
#        print(A_val)

















