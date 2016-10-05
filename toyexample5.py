# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 10:54:16 2016

Try synthetic gradients w/ stepsize of 1, then go back and finish toyexample4
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
n_hidden = 11
n_out = 4
sequence_length = 120
n_examples = 10000
dependency = 10 # artificially introduce long-term dependencies
truncate = 1 # truncate BPTT

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
Wr1_true = rng.uniform(low=-0.1,high=0.1,size=(n_hidden,n_hidden)).astype('float32')
Wr2_true = ortho_weight(n_hidden,rng)
bh_true = rng.uniform(low=0,high=0.1,size=(n_hidden)).astype('float32')
Wy_true = rng.uniform(low=-0.1,high=0.1,size=(n_hidden,n_out)).astype('float32')
by_true = rng.uniform(low=-0.02,high=0.1,size=(n_out)).astype('float32')
h = numpy.zeros((dependency+1,n_examples,n_hidden),dtype='float32')
def np_relu(x):
    return numpy.maximum(numpy.zeros_like(x),x)
for t in range(x_train.shape[1]):
    # roll the memory back
    h[1:] = h[:-1]
    h[0] = np_relu(numpy.dot(x_train[:,t,:],Wx_true) \
               +0.5*(numpy.dot(h[1],Wr1_true)+numpy.dot(h[-1],Wr2_true)) \
               +bh_true)
    noise = rng.normal(loc=0.,scale=0.01,size=(n_out,))
    y_train[:,t,:] = np_relu(numpy.dot(h[0],Wy_true)+by_true+noise)



# -----------------------------------------------------------------------------
# RNN helper functions
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

run = 'dni'
batch_size = 100
lr = 1e-4
n_epochs = 500

x = T.tensor3('x')
y = T.tensor3('y')
index = T.lscalar()  # index to a [mini]batch


# -----------------------------------------------------------------------------
# Classic Backprop
# -----------------------------------------------------------------------------

if run=='bptt':
    x_train = theano.shared(x_train)
    y_train = theano.shared(y_train)
    
    class rnn(object):
        def __init__(self,x,n_in,n_hidden,n_out,steps,rng=rng):
            """
            Initialize a basic single-layer RNN
            
            x:    symbolic input tensor
            n_in:    input dimensionality
            n_hidden:    # of hidden units
            hidden_activation:    non-linearity at hidden units (e.g. relu)
            n_out:    # of output units
            steps:    # of time steps to truncate BPTT at
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
                                    strict=True,
                                    truncate_gradient=steps)
            self.orthogonality = T.sum(T.sqr(T.dot(self.Wh,self.Wh.T)-T.identity_like(self.Wh)))
        def square_error(self,y):
            return T.mean(T.square(self.output-y.dimshuffle([1,0,2])),axis=(1,2))
        def mse(self,y):
            return T.mean(self.square_error(y))
    model = rnn(x,n_in,n_hidden,n_out,truncate)    
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
        print('Epoch %d loss: %f' % (epoch,loss))


# -----------------------------------------------------------------------------
# Synthetic Gradient
# -----------------------------------------------------------------------------

elif run=='dni':
    
    lr = 1e-7
    
    class dni(object):
        def __init__(self,n_feat,n_layers,rng=rng):
            """
            Simple fully-connected neural net
            """
            self.n_feat = n_feat
            self.n_layers = n_layers
            self.W = []
            self.b = []
            self.params = []
            for n in range(n_layers):
                if n<n_layers-1:
                    W = _ortho_weight(n_feat,rng)
                else: # last layer weights should be zero to not wreck shit
                    W = theano.shared(numpy.zeros((n_feat,n_feat),
                                                  dtype=theano.config.floatX),
                                                  borrow=True)
                self.W.append(W)
                self.params.append(W)
                b = _zero_bias(n_feat)
                self.b.append(b)
                self.params.append(b)
        
        def output(self,x):
            next_input = x
            for n in range(self.n_layers):
                next_input = relu(T.dot(next_input,self.W[n])+self.b[n])
            return next_input
    
    class rnn_dni(object):
        def __init__(self,n_in,n_hidden,n_out,rng=rng):
            """
            Build a simple RNN with a DNI every time-step
            """
            self.n_in = n_in
            self.n_hidden = n_hidden
            self.n_out = n_out
            
            self.Wx = _uniform_weight(n_in,n_hidden)
            self.Wh = _ortho_weight(n_hidden)
            self.bh = _zero_bias(n_hidden)
            self.Wy = _uniform_weight(n_hidden,n_out)
            self.by = _zero_bias(n_out)
            self.params = [self.Wx,self.Wh,self.bh,self.Wy,self.by]
            
            self.dni = dni(n_hidden,2)
            
        def train(self):
            x = T.tensor3('x')
            y = T.tensor3('y')
            learning_rate = T.scalar('learning_rate')
            dni_switch = T.scalar('dni_switch')
            def step(x_t,y_t,h_tm1,Wx,Wh,bh,Wy,by,lr,switch):
                h_t = relu(T.dot(x_t,Wx)+T.dot(h_tm1,Wh)+bh)
                yo_t = relu(T.dot(h_t,Wy)+by)
                
                updates = OrderedDict()
                
                # Train the RNN: backprop (loss + DNI output)
                loss = T.mean(T.square(yo_t-y_t))
                dni_out = self.dni.output(h_t)
#                for param in self.params:
                for param in [self.Wx,self.bh,self.Wy,self.by,self.Wh]:
                    dlossdparam = T.grad(loss,param)
                    dniJ = T.Lop(h_t,param,dni_out,disconnected_inputs='ignore')
                    J = dniJ
                    updates[param] = param-lr*T.switch(T.gt(switch,0),
                                                       dlossdparam+dniJ,
                                                       dlossdparam)
                                
                # Update the DNI (from the last step)
                # re-calculate the DNI prediction from the last step
                # note: can't be passed through scan or T.grad won't work
                dni_out_old = self.dni.output(h_tm1)
                # dni_target: current loss backprop'ed + new dni backprop'ed
                dni_target = T.grad(loss,h_tm1) \
                             +T.Lop(h_t,h_tm1,dni_out)
                dni_error = T.sum(T.square(dni_out_old-dni_target))
                for param in self.dni.params:
                    gparam = T.grad(dni_error,param)
                    updates[param] = param-lr*gparam
                
                return [h_t,loss,J],updates
            h0 = T.zeros((n_hidden,),dtype=theano.config.floatX)
            [h,seq_loss,J_out],updates = theano.scan(fn=step, 
                                 sequences=[x.dimshuffle([1,0,2]),
                                            y.dimshuffle([1,0,2])],
                                 outputs_info=[T.alloc(h0,x.shape[0],n_hidden),
                                               None,None],
                                 non_sequences=[self.Wx,self.Wh,self.bh,
                                                self.Wy,self.by,
                                                learning_rate,dni_switch])
            return theano.function(inputs=[x,y,learning_rate,dni_switch],
                                   outputs=[T.mean(seq_loss),J_out],
                                   updates=updates)
    
    model = rnn_dni(n_in,n_hidden,n_out)
    train = model.train()
    for epoch in range(n_epochs):
        if epoch>0:
            dni_on = 1
        else:
            dni_on = 0
        loss = 0
        for batch in range(int(numpy.floor(n_examples/batch_size))):
            x_batch = x_train[batch*batch_size:(batch+1)*batch_size]
            y_batch = y_train[batch*batch_size:(batch+1)*batch_size]
            loss,debugJ = train(x_batch,y_batch,lr,dni_on)
#            print(debugJ)
        print('Epoch %d loss: %f' % (epoch,loss))





















