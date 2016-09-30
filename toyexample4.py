# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 10:54:16 2016

Try synthetic gradients on a simple example

@author: jrbtaylor
"""


import numpy
import theano
from theano import tensor as T
from theano.tensor.nnet import relu

from collections import OrderedDict


# -----------------------------------------------------------------------------
# Make some simple recursive data
# -----------------------------------------------------------------------------

n_in = 5
n_hidden = 10
n_out = 3
sequence_length = 20
n_examples = 100000
rng = numpy.random.RandomState(1)
# inputs are vectors of uniform random numbers
x_train = rng.uniform(low=-1,high=1,
                      size=(n_examples,sequence_length,n_in)
                      ).astype('float32')
# outputs are from a prefixed RNN plus noise
def ortho_weight(ndim,rng):
    W = rng.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')
y_train = numpy.zeros((n_examples,sequence_length,n_out),dtype='float32')
Wx_true = rng.uniform(low=-0.1,high=0.1,size=(n_in,n_hidden)).astype('float32')
Wr_true = ortho_weight(n_hidden,rng)
bh_true = rng.uniform(low=0,high=0.1,size=(n_hidden)).astype('float32')
Wy_true = rng.uniform(low=-0.1,high=0.1,size=(n_hidden,n_out)).astype('float32')
by_true = rng.uniform(low=-0.02,high=0.1,size=(n_out)).astype('float32')
h = numpy.zeros((n_examples,n_hidden),dtype='float32')
def np_relu(x):
    return numpy.maximum(numpy.zeros_like(x),x)
for t in range(x_train.shape[1]):
    h = np_relu(numpy.dot(x_train[:,t,:],Wx_true) \
               +numpy.dot(h,Wr_true)+bh_true)
    noise = rng.normal(loc=0.,scale=0.01,size=(n_out,))
    y_train[:,t,:] = np_relu(numpy.dot(h,Wy_true)+by_true+noise)
x_train = theano.shared(x_train)
y_train = theano.shared(y_train)


# -----------------------------------------------------------------------------
# Vanilla-RNN model
# -----------------------------------------------------------------------------

def _uniform_weight(n1,n2,rng=rng):
    limit = numpy.sqrt(6./(n1+n2))
    return theano.shared((rng.uniform(low=-limit,
                                      high=limit,
                                      size=(n1,n2))
                         ).astype(theano.config.floatX),
                         borrow=True)

def _ortho_weight(n,rng=rng):
    W = rng.randn(n, n)
    u, s, v = numpy.linalg.svd(W)
    return theano.shared(u.astype(theano.config.floatX),
                         borrow=True)

def _zero_bias(n):
    return theano.shared(numpy.zeros((n,),dtype=theano.config.floatX),
                         borrow=True)

class rnn(object):
    def __init__(self,x,n_in,n_hidden,n_out,rng=rng):
        """
        Initialize a basic single-layer RNN
        
        x:    symbolic input tensor
        n_in:    input dimensionality
        n_hidden:    # of hidden units
        hidden_activation:    non-linearity at hidden units (e.g. relu)
        n_out:    # of output units
        output_activation:    non-linearity at output units (e.g. softmax)
        """
        self.Wx = _uniform_weight(n_in,n_hidden,rng)
        self.Wh = _ortho_weight(n_hidden,rng)
        self.Wy = _uniform_weight(n_hidden,n_out,rng)
        self.bh = _zero_bias(n_hidden)
        self.by = _zero_bias(n_out)
        self.params = [self.Wx,self.Wh,self.Wy,self.bh,self.by]
        
        def step(x_t,h_tm1,Wx,Wh,Wy,bh,by):
            h_t = relu(T.dot(x_t,Wx)+T.dot(h_tm1,Wh)+bh)
            y_t = relu(T.dot(h_t,Wy)+by)
            return [h_t,y_t]
        h0 = T.zeros((n_hidden,),dtype=theano.config.floatX)
        ([h,self.output],_) = theano.scan(fn=step, 
                                sequences=x.dimshuffle([1,0,2]),
                                outputs_info=[T.alloc(h0,x.shape[0],n_hidden),
                                              None],
                                non_sequences=[self.Wx,self.Wh,self.Wy,
                                               self.bh,self.by],
                                strict=True)
        self.orthogonality = T.sum(T.sqr(T.dot(self.Wh,self.Wh.T)-T.identity_like(self.Wh)))
    def square_error(self,y):
        return T.mean(T.square(self.output-y.dimshuffle([1,0,2])),axis=(1,2))
    def mse(self,y):
        return T.mean(self.square_error(y))

x = T.tensor3('x')
y = T.tensor3('y')
model = rnn(x,n_in,n_hidden,n_out)

index = T.lscalar()  # index to a [mini]batch
batch_size = 100
lr = 1e-4
n_epochs = 500

run = 'synth'

# -----------------------------------------------------------------------------
# Classic SGD
# -----------------------------------------------------------------------------

if run=='sgd':
    cost = model.mse(y)
    gparams = T.grad(cost,model.params)
    updates = OrderedDict()
    for param,gparam in zip(model.params,gparams):
        updates[param] = param-lr*gparam
    train = theano.function(inputs=[index],
                            outputs=cost,
                            updates=updates,
                            givens={
                              x: x_train[index*batch_size:(index+1)*batch_size],
                              y: y_train[index*batch_size:(index+1)*batch_size]})
    for epoch in range(n_epochs):
        loss = 0
        for batch in range(int(numpy.floor(n_examples/batch_size))):
            loss = train(batch)
            print(loss.shape)
        print('Epoch %d loss: %f' % (epoch,loss))


# -----------------------------------------------------------------------------
# SGD w/ eligiblity trace scaling
# -----------------------------------------------------------------------------

elif run=='synth':
    








































