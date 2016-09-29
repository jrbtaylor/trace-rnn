# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 17:31:59 2016

@author: jrbtaylor
"""

from __future__ import print_function

import numpy

import theano
from theano import tensor as T
from theano.tensor import tanh
from theano.tensor.nnet import sigmoid
from theano.tensor.nnet.nnet import softmax, categorical_crossentropy

rng = numpy.random.RandomState(1)

# from http://deeplearning.net/tutorial/code/lstm.py
def ortho_weight(ndim,rng):
    W = rng.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(theano.config.floatX)

class lstm(object):
    def __init__(self,x,n_in,n_hidden,n_out,bptt_limit):
        self.x = x
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.bptt_limit = bptt_limit
        # initialize weights
        rng=numpy.random.RandomState(1)
        def uniform_weight(m,n):
            return theano.shared((rng.uniform(low=-numpy.sqrt(6./(m+n)),
                                              high=numpy.sqrt(6./(m+n)),
                                              size=(m,n))).astype(theano.config.floatX),
                                              borrow=True)
        self.Wxi = uniform_weight(n_in,n_hidden)
        self.Wsi = theano.shared(ortho_weight(n_hidden,rng),borrow=True)
        self.Wxf = uniform_weight(n_in,n_hidden)
        self.Wsf = theano.shared(ortho_weight(n_hidden,rng),borrow=True)
        self.Wxo = uniform_weight(n_in,n_hidden)
        self.Wso = theano.shared(ortho_weight(n_hidden,rng),borrow=True)
        self.Wxg = uniform_weight(n_in,n_hidden)
        self.Wsg = theano.shared(ortho_weight(n_hidden,rng),borrow=True)
        self.Wsy = uniform_weight(n_hidden,n_out)
        self.bi = theano.shared(numpy.zeros((self.n_hidden,),dtype=theano.config.floatX),borrow=True)
        self.bf = theano.shared(numpy.zeros((self.n_hidden,),dtype=theano.config.floatX),borrow=True)
        self.bo = theano.shared(numpy.zeros((self.n_hidden,),dtype=theano.config.floatX),borrow=True)
        self.bg = theano.shared(numpy.zeros((self.n_hidden,),dtype=theano.config.floatX),borrow=True)
        self.by = theano.shared(numpy.zeros((self.n_out,),dtype=theano.config.floatX),borrow=True)
        self.params = [self.Wxi,self.Wsi,self.Wxf,self.Wsf,self.Wxo,self.Wso,self.Wxg,self.Wsg,self.Wsy,self.bi,self.bf,self.bo,self.bg,self.by]
        self.W = [self.Wxi,self.Wsi,self.Wxf,self.Wsf,self.Wxo,self.Wso,self.Wxg,self.Wsg,self.Wsy]
        self.L1 = numpy.sum([abs(w).sum() for w in self.W])
        self.L2 = numpy.sum([(w**2).sum() for w in self.W])
        # forward function
        def forward(x_t,c_tm1,s_tm1,Wxi,Wsi,Wxf,Wsf,Wxo,Wso,Wxg,Wsg,Wsy,bi,bf,bo,bg,by):
            i = sigmoid(T.dot(x_t,Wxi)+T.dot(s_tm1,Wsi)+bi)
            f = sigmoid(T.dot(x_t,Wxf)+T.dot(s_tm1,Wsf)+bf)
            o = sigmoid(T.dot(x_t,Wxo)+T.dot(s_tm1,Wso)+bo)
            g = tanh(T.dot(x_t,Wxg)+T.dot(s_tm1,Wsg)+bg)
            c = c_tm1*f+g*i
            s = tanh(c)*o
            y = softmax(T.dot(s,Wsy)+by)
            return [c,s,y]
        c0 = T.alloc(T.zeros((self.n_hidden,),dtype=theano.config.floatX),x.shape[0],self.n_hidden)
        s0 = T.alloc(T.zeros((self.n_hidden,),dtype=theano.config.floatX),x.shape[0],self.n_hidden)
        ([c,s,y],updates) = theano.scan(fn=forward,
                                      sequences=x.dimshuffle([1,0,2]),
                                      outputs_info=[dict(initial=c0,taps=[-1]),
                                                    dict(initial=s0,taps=[-1]),
                                                    None],
                                      non_sequences=[self.Wxi,self.Wsi,self.Wxf,self.Wsf,self.Wxo,self.Wso,self.Wxg, \
                                                     self.Wsg,self.Wsy,self.bi,self.bf,self.bo,self.bg,self.by],
                                      strict=True)
        self.output = y[-1] # train on last output only? (play with this later)
        self.pred = T.argmax(self.output,axis=1)
    
    # ----- Classification -----
    def crossentropy(self,y):
        return T.mean(categorical_crossentropy(self.output,y))
    
    def errors(self,y):
        return T.mean(T.neq(self.pred,y))
    
    # ----- Regression -----
    def mse(self,y):
        return T.mean(T.sqr(T.sub(self.output,y)))








