# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 23:49:14 2016

@author: jrbtaylor
"""
#%%
from __future__ import print_function

import numpy
import theano
from theano import tensor as T
from theano.tensor.nnet import relu

from collections import OrderedDict

# -----------------------------------------------------------------------------
# Make some exceedingly simple data
# -----------------------------------------------------------------------------

# inputs are randomly 0-1
n_in = 256
n_examples = 10000
rng = numpy.random.RandomState(1)
x_train = rng.uniform(low=0,high=1,size=(n_examples,100,n_in)).astype('float32')
# output is the cumulative sum of inputs
y_train = numpy.cumsum(numpy.sum(x_train,axis=2),axis=1).astype('float32')
# with some scaling for numerical stability
y_train = y_train/numpy.max(y_train)
y_train = y_train[:,:,numpy.newaxis]

x_train = theano.shared(x_train)
y_train = theano.shared(y_train)

# -----------------------------------------------------------------------------
# Make some exceedingly simple model
# -----------------------------------------------------------------------------

# from http://deeplearning.net/tutorial/code/lstm.py
def ortho_weight(ndim,rng):
    W = rng.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(theano.config.floatX)

#class rnn(object):
#    def __init__(self,x,n_in,n_hidden,n_out,bptt_limit):
#        # assign variables
#        self.x = x
#        self.n_in = n_in
#        self.n_hidden = n_hidden
#        self.n_out = n_out
#        self.bptt_limit = bptt_limit
#        # initialize weights
#        rng = numpy.random.RandomState(1)
#        self.Wx = theano.shared((rng.uniform(low=-numpy.sqrt(6./(self.n_in+self.n_hidden)),
#                                             high=numpy.sqrt(6./(self.n_in+self.n_hidden)),
#                                             size=(self.n_in,self.n_hidden))).astype(theano.config.floatX),
#                                             borrow=True,
#                                             name='Wx')
#        self.Wh = theano.shared(ortho_weight(self.n_hidden,rng),borrow=True,name='Wh')
#        self.Wo = theano.shared((rng.uniform(low=-numpy.sqrt(6./(self.n_hidden+self.n_out)),
#                                             high=numpy.sqrt(6./(self.n_hidden+self.n_out)),
#                                             size=(self.n_hidden,self.n_out))).astype(theano.config.floatX),
#                                             borrow=True,
#                                             name='Wo')
#        # biases
#        self.bh = theano.shared(numpy.zeros((self.n_hidden,),dtype=theano.config.floatX),
#                                borrow=True,name='bh')
#        self.bo = theano.shared(numpy.zeros((self.n_out,),dtype=theano.config.floatX),
#                                borrow=True,name='bo')
#        self.params = [self.Wx,self.Wh,self.Wo,self.bh,self.bo] # package for gradient calc
#        self.W = [self.Wx,self.Wh,self.Wo] # package for weight regularization
#        self.L1 = numpy.sum([abs(w).sum() for w in self.W])
#        self.L2 = numpy.sum([(w**2).sum() for w in self.W])
#        # forward function
#        def forward(x_t,h_tm1,Wx,Wh,Wo,bh,bo):
#            h_t = relu(T.dot(x_t,Wx)+T.dot(h_tm1,Wh)+bh)
#            y_t = relu(T.dot(h_t,Wo)+bo)
#            return [h_t,y_t]
#        h0 = T.zeros((self.n_hidden,),dtype=theano.config.floatX)
#        ([h,self.output],updates) = theano.scan(fn=forward,
#                                      sequences=x.dimshuffle([1,0,2]),
#                                      outputs_info=[T.alloc(h0,x.shape[0],n_hidden),None],#outputs_info=[dict(initial=h0,tapes=[-1]),None],
#                                      non_sequences=[self.Wx,self.Wh,self.Wo,self.bh,self.bo],
#                                      strict=True,
#                                      truncate_gradient=self.bptt_limit)
#    
#    def mse(self,y):
#        return T.mean(T.square(y.dimshuffle([1,0,2])-self.output))
#    
#
## -----------------------------------------------------------------------------
## Train a standard RNN with no online updates
## -----------------------------------------------------------------------------
#
#n_hidden = 64
#learning_rate = 1e-2
#momentum = 0.9
#batch_size = 1000
#bptt_limit = -1 # -1 means no truncation
#n_epochs = 100
#
#index = T.lscalar()  # index to a [mini]batch
#x = T.tensor3('x')
#y = T.tensor3('y')
#model = rnn(x,n_in,n_hidden,1,bptt_limit)
#cost = model.mse(y)
#
## SGD w/ momentum
## Initialize momentum
#gparams_mom = []
#for param in model.params:
#    gparam_mom = theano.shared(numpy.zeros(param.get_value(borrow=True).shape,
#                                           dtype=theano.config.floatX))
#    gparams_mom.append(gparam_mom)
## Setup backprop
#gparams = T.grad(cost,model.params)
#updates = OrderedDict()
## Momentum update
#for gparam_mom, gparam in zip(gparams_mom,gparams):
#    updates[gparam_mom] = momentum*gparam_mom-(1.-momentum)*learning_rate*gparam
## Parameter update
#for param,gparam_mom in zip(model.params,gparams_mom):
#    updates[param] = param+updates[gparam_mom]
#
## Training function
#train_model = theano.function(inputs=[index],
#                              outputs=cost,
#                              updates=updates,
#                              givens={
#                                  x: x_train[index*batch_size:(index+1)*batch_size],
#                                  y: y_train[index*batch_size:(index+1)*batch_size]})
#
#n_batches = int(numpy.floor(n_examples/batch_size))
#offline_loss = []
#print('Training RNN offline')
#for epoch in range(n_epochs):
#    loss_epoch = 0
#    for batch in range(n_batches):
#        loss_batch = train_model(batch)
#        loss_epoch += loss_batch
#        offline_loss.append(loss_epoch)
#    print('Epoch %d: %f' % (epoch,loss_epoch))


# -----------------------------------------------------------------------------
# Make the online-updating equivalent
# -----------------------------------------------------------------------------

class rnn_onlineSGD(object):
    def __init__(self,x,n_in,n_hidden,n_out,lr,momentum,bptt_limit):
        # assign variables
        self.x = x
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.lr = lr
        self.momentum = momentum
        self.bptt_limit = bptt_limit
        # initialize weights
        rng = numpy.random.RandomState(1)
        self.Wx = theano.shared((rng.uniform(low=-numpy.sqrt(6./(self.n_in+self.n_hidden)),
                                             high=numpy.sqrt(6./(self.n_in+self.n_hidden)),
                                             size=(self.n_in,self.n_hidden))).astype(theano.config.floatX),
                                             borrow=True,
                                             name='Wx')
        self.Wh = theano.shared(ortho_weight(self.n_hidden,rng),borrow=True,name='Wh')
        self.Wo = theano.shared((rng.uniform(low=-numpy.sqrt(6./(self.n_hidden+self.n_out)),
                                             high=numpy.sqrt(6./(self.n_hidden+self.n_out)),
                                             size=(self.n_hidden,self.n_out))).astype(theano.config.floatX),
                                             borrow=True,
                                             name='Wo')
        # biases
        self.bh = theano.shared(numpy.zeros((self.n_hidden,),dtype=theano.config.floatX),
                                borrow=True,name='bh')
        self.bo = theano.shared(numpy.zeros((self.n_out,),dtype=theano.config.floatX),
                                borrow=True,name='bo')
        self.params = [self.Wx,self.Wh,self.Wo,self.bh,self.bo] # package for gradient calc
        self.W = [self.Wx,self.Wh,self.Wo] # package for weight regularization
        self.L1 = numpy.sum([abs(w).sum() for w in self.W])
        self.L2 = numpy.sum([(w**2).sum() for w in self.W])
        # Initialize momentum
        self.gparams_mom = []
        for param in self.params:
            gparam_mom = theano.shared(numpy.zeros(param.get_value(borrow=True).shape,
                                                   dtype=theano.config.floatX))
            self.gparams_mom.append(gparam_mom)
        
    def train_batch(self,x_data,y_data,index):
        y = T.tensor3('y')
        idx = T.lscalar('idx')
        # step function
        def step(x_t,y_t,h_tm1,Wx,Wh,Wo,bh,bo,lr):
            # Calculate the output at time t ----------------
            h_t = relu(T.dot(x_t,Wx)+T.dot(h_tm1,Wh)+bh)
            o_t = relu(T.dot(h_t,Wo)+bo)
            
            # Online updates --------------------------------
            mse = T.mean(T.square(y_t-o_t))
            # backprop
            gparams = T.grad(mse,self.params)
            updates = OrderedDict()
            # Momentum update
            for gparam_mom, gparam in zip(self.gparams_mom,gparams):
                updates[gparam_mom] = self.momentum*gparam_mom \
                                      -(1.-self.momentum)*lr*gparam
            # Parameter update
            for param,gparam_mom in zip(self.params,self.gparams_mom):
                updates[param] = param+updates[gparam_mom]
            return [h_t,o_t,mse]
        h0 = T.zeros((self.n_hidden,),dtype=theano.config.floatX)
        ([h,self.output,mse],updates) = theano.scan(fn=step,
                                          sequences=[x.dimshuffle([1,0,2]),
                                                     y.dimshuffle([1,0,2])],
                                          outputs_info=[T.alloc(h0,x.shape[0],n_hidden),
                                                        None,None],
                                          non_sequences=[self.Wx,self.Wh,self.Wo,
                                                         self.bh,self.bo,self.lr],
                                          strict=True,
                                          truncate_gradient=self.bptt_limit)
        train_fn = theano.function(inputs=[idx],
                                   outputs=T.mean(mse),
                                   updates=updates,
                                   givens={
                                       x:x_data[idx*batch_size:(idx+1)*batch_size],
                                       y:y_data[idx*batch_size:(idx+1)*batch_size]})
        return train_fn(index)
    

# -----------------------------------------------------------------------------
# Train a standard RNN with no online updates
# -----------------------------------------------------------------------------

n_hidden = 64
lr_init = 1e-2
lr_decay = 0.99
momentum = 0.9
batch_size = 1000
bptt_limit = -1 # -1 means no truncation
n_epochs = 100

index = T.lscalar()  # index to a [mini]batch
epoch = T.lscalar() # epoch number
x = T.tensor3('x')

lr = theano.shared(numpy.array(lr_init,
                               dtype = theano.config.floatX))
model = rnn_onlineSGD(x,n_in,n_hidden,1,lr,momentum,bptt_limit)

n_batches = int(numpy.floor(n_examples/batch_size))
offline_loss = []
print('Training RNN online')
for ep in range(n_epochs):
    loss_epoch = 0
    for batch in range(n_batches):
        loss_batch = model.train_batch(x_train,y_train,batch)
        loss_epoch += loss_batch
        offline_loss.append(loss_epoch)
    print('Epoch %d: %f' % (ep,loss_epoch))
    lr *= lr_decay







































