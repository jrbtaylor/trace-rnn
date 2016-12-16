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
from theano.tensor.nnet import sigmoid, relu
from theano.tensor.nnet.nnet import softmax, categorical_crossentropy

from optim import adam

rng = numpy.random.RandomState(1)

# from http://deeplearning.net/tutorial/code/lstm.py
def ortho_weight(ndim,rng=rng):
    W = rng.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return theano.shared(u.astype(theano.config.floatX),borrow=True)

def const_bias(n,value=0):
    return theano.shared(value*numpy.ones((n,),dtype=theano.config.floatX),
                         borrow=True)

def uniform_weight(n1,n2,rng=rng):
    limit = numpy.sqrt(6./(n1+n2))
    return theano.shared((rng.uniform(low=-limit,
                                      high=limit,
                                      size=(n1,n2))
                         ).astype(theano.config.floatX),borrow=True)

def layer_norm(h,scale=1,shift=0,eps=1e-5):
    mean = T.mean(h,axis=1,keepdims=True,dtype=theano.config.floatX)
    std = T.std(h,axis=1,keepdims=True)
    normed = (h-mean)/(eps+std)
    return scale*normed+shift
    
def dropout(h,p,rng=rng):
    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(99999))
    mask = T.cast(srng.binomial(n=1,p=1-p,size=h.shape),theano.config.floatX)
    # rescale activations at train time to avoid rescaling weights at test
    h = h/(1-p)
    return h*mask

def zoneout(h_t,h_tm1,p,rng=rng):
    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(99999))
    mask = T.cast(srng.binomial(n=1,p=1-p,size=h_t.shape),theano.config.floatX)
    return h_t*mask+h_tm1*(1-mask)

class lstm(object):
    def __init__(self,x,n_in,n_hidden,n_out):
        self.x = x
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        
        # initialize weights
        def ortho_weight(ndim,rng=rng):
            W = rng.randn(ndim, ndim)
            u, s, v = numpy.linalg.svd(W)
            return u.astype(theano.config.floatX)
        def uniform_weight(n1,n2,rng=rng):
            limit = numpy.sqrt(6./(n1+n2))
            return rng.uniform(low=-limit,high=limit,size=(n1,n2)).astype(theano.config.floatX)
        def const_bias(n,value=0):
            return value*numpy.ones((n,),dtype=theano.config.floatX)
        self.Wx = theano.shared(numpy.concatenate(
                    [uniform_weight(n_in,n_hidden) for i in range(4)],axis=1),
                     borrow=True)
        self.Wh = theano.shared(numpy.concatenate(
                    [ortho_weight(n_hidden,rng) for i in range(4)],axis=1),
                     borrow=True)
        self.Wy = theano.shared(uniform_weight(n_hidden,n_out))
        self.b = theano.shared(numpy.concatenate(
                    [const_bias(n_hidden,0) for i in range(4)],axis=0),
                     borrow=True)
        self.by = theano.shared(const_bias(n_out,0))
        
        self.params = [self.Wx,self.Wh,self.Wy,self.b,self.by]
        self.W = [self.Wx,self.Wh,self.Wy]
        self.L1 = numpy.sum([abs(w).sum() for w in self.W])
        self.L2 = numpy.sum([(w**2).sum() for w in self.W])
        
        # slice for doing step calculations in parallel
        def _slice(x,n):
            return x[:,n*self.n_hidden:(n+1)*self.n_hidden]        
        
        # forward function
        def forward(x_t,c_tm1,h_tm1,Wx,Wh,Wy,b,by):
            preact = T.dot(x_t,Wx)+T.dot(h_tm1,Wh)+b
            i = sigmoid(_slice(preact,0))
            f = sigmoid(_slice(preact,1))
            o = sigmoid(_slice(preact,2))
            g = tanh(_slice(preact,3))
            c = c_tm1*f+g*i
            h = tanh(c)*o
            y = softmax(T.dot(h,Wy)+by)
            return [c,h,y]
        c0 = T.alloc(T.zeros((self.n_hidden,),dtype=theano.config.floatX),x.shape[0],self.n_hidden)
        h0 = T.alloc(T.zeros((self.n_hidden,),dtype=theano.config.floatX),x.shape[0],self.n_hidden)
        ([c,h,y],updates) = theano.scan(fn=forward,
                                      sequences=x.dimshuffle([1,0,2]),
                                      outputs_info=[dict(initial=c0,taps=[-1]),
                                                    dict(initial=h0,taps=[-1]),
                                                    None],
                                      non_sequences=[self.Wx,self.Wh,self.Wy,
                                                     self.b,self.by],
                                      strict=True)
        self.output = y
        self.pred = T.argmax(self.output,axis=1)
    
    # ----- Classification -----
    def crossentropy(self,y):
        return T.mean(categorical_crossentropy(self.output,y.dimshuffle([1,0,2])))
    
    def errors(self,y):
        return T.mean(T.neq(self.pred,y.dimshuffle([1,0,2])))
    
    # ----- Regression -----
    def mse(self,y):
        return T.mean(T.sqr(T.sub(self.output,y.dimshuffle([1,0,2]))))


class gru(object):
    def __init__(self,x,n_in,n_hidden,n_out):
        self.x = x
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        
        # initialize weights
        def ortho_weight(ndim,rng=rng):
            W = rng.randn(ndim, ndim)
            u, s, v = numpy.linalg.svd(W)
            return u.astype(theano.config.floatX)
        def uniform_weight(n1,n2,rng=rng):
            limit = numpy.sqrt(6./(n1+n2))
            return rng.uniform(low=-limit,high=limit,size=(n1,n2)).astype(theano.config.floatX)
        def const_bias(n,value=0):
            return value*numpy.ones((n,),dtype=theano.config.floatX)
        self.Wx = theano.shared(numpy.concatenate(
                    [uniform_weight(n_in,n_hidden) for i in range(2)],axis=1),
                     borrow=True)
        self.Wh = theano.shared(numpy.concatenate(
                    [ortho_weight(n_hidden,rng) for i in range(2)],axis=1),
                     borrow=True)
        self.Wxg = theano.shared(uniform_weight(n_in,n_hidden),borrow=True)
        self.Whg = theano.shared(ortho_weight(n_hidden),borrow=True)
        self.Wy = theano.shared(uniform_weight(n_hidden,n_out),borrow=True)
        self.b = theano.shared(numpy.concatenate(
                    [const_bias(n_hidden,0) for i in range(2)],axis=0),
                     borrow=True)
        self.by = theano.shared(const_bias(n_out,0),borrow=True)
        self.bg = theano.shared(const_bias(n_hidden,0),borrow=True)
        
        self.params = [self.Wx,self.Wh,self.Wxg,self.Whg,self.Wy,
                       self.b,self.by,self.bg]
        self.W = [self.Wx,self.Wh,self.Wy]
        self.L1 = numpy.sum([abs(w).sum() for w in self.W])
        self.L2 = numpy.sum([(w**2).sum() for w in self.W])
        
        # slice for doing step calculations in parallel
        def _slice(x,n):
            return x[:,n*self.n_hidden:(n+1)*self.n_hidden]        
        
        # forward function
        def forward(x_t,h_tm1,Wx,Wh,Wxg,Whg,Wy,b,bg,by):
            preact = T.dot(x_t,Wx)+T.dot(h_tm1,Wh)+b
            z = sigmoid(_slice(preact,0))
            r = sigmoid(_slice(preact,1))
            g = tanh(T.dot(x_t,Wxg)+T.dot(r*h_tm1,Whg)+bg)
            h = (1-z)*h_tm1+z*g
            y = softmax(T.dot(h,Wy)+by)
            return [h,y]
        h0 = T.alloc(T.zeros((self.n_hidden,),dtype=theano.config.floatX),x.shape[0],self.n_hidden)
        ([h,y],updates) = theano.scan(fn=forward,
                                      sequences=x.dimshuffle([1,0,2]),
                                      outputs_info=[dict(initial=h0,taps=[-1]),
                                                    None],
                                      non_sequences=[self.Wx,self.Wh,self.Wxg,
                                                     self.Whg,self.Wy,
                                                     self.b,self.bg,self.by],
                                      strict=True)
        self.output = y
        self.pred = T.argmax(self.output,axis=1)
    
    # ----- Classification -----
    def crossentropy(self,y):
        return T.mean(categorical_crossentropy(self.output,y.dimshuffle([1,0,2])))
    
    def errors(self,y):
        return T.mean(T.neq(self.pred,y.dimshuffle([1,0,2])))
    
    # ----- Regression -----
    def mse(self,y):
        return T.mean(T.sqr(T.sub(self.output,y.dimshuffle([1,0,2]))))


class dni(object):
        def __init__(self,n_in,n_out,n_layers,rng=rng):
            """
            Simple fully-connected neural net
            """
            self.n_in = n_in
            self.n_out = n_out
            self.n_layers = n_layers
            self.W = []
            self.b = []
            self.params = []
            for n in range(n_layers):
                if n==0:
                    if n_in==n_out:
                        W = ortho_weight(n_out,rng)
                    else:
                        W = uniform_weight(n_in,n_out,rng)
                elif n<n_layers-1:
                    W = ortho_weight(n_out,rng)
                else: # last layer weights should be zero to not wreck things
                    W = theano.shared(numpy.zeros((n_out,n_out),
                                                  dtype=theano.config.floatX),
                                                  borrow=True)
                self.W.append(W)
                self.params.append(W)
                b = const_bias(n_out,0)
                self.b.append(b)
                self.params.append(b)
        
        def output(self,x):
            next_input = x
            for n in range(self.n_layers-1):
                next_input = relu(T.dot(next_input,self.W[n])+self.b[n])
            # last layer is linear
            next_input = T.dot(next_input,self.W[-1])+self.b[-1]
            return next_input
    
class rnn_dni(object):
    def __init__(self,n_in,n_hidden,n_out,steps,rng=rng):
        """
        Build a simple RNN with a DNI every 'steps' timesteps
        """
        # parameters
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.steps = steps
        
        # initialize weights
        self.Wx = uniform_weight(n_in,n_hidden)
        self.Wh = ortho_weight(n_hidden)
        self.bh = const_bias(n_hidden,0)
        self.Wy = uniform_weight(n_hidden,n_out)
        self.by = const_bias(n_out,0)
        self.params = [self.Wx,self.Wh,self.bh,self.Wy,self.by]
        
        # initialize DNI
        self.dni = dni(n_hidden,n_hidden,2)
        
    def train(self):
        x = T.tensor3('x')
        y = T.tensor3('y')
        learning_rate = T.scalar('learning_rate')
        dni_scale = T.scalar('dni_scale')
        
        # reshape the inputs
        # batch x time x n -> time//steps x steps x batch x n
        def shufflereshape(r):
            r = r.dimshuffle([1,0,2])
            r = r.reshape((r.shape[0]//self.steps,
                           self.steps,
                           r.shape[1],
                           r.shape[2]))
            return r
        
        # step takes a set of inputs over self.steps time-steps
        def step(x_t,y_t,h_tmT,Wx,Wh,bh,Wy,by,lr,scale):
            
            # manually build the graph for the inner loop...
            # passing correct h_tm1 is impossible in nested scans
            yo_t = []
            h_tm1 = h_tmT
            loss = 0
            for t in range(self.steps):
                h_t = relu(T.dot(x_t[t],Wx)+T.dot(h_tm1,Wh)+bh)
                output = softmax(T.dot(h_t,Wy)+by)
                yo_t.append(output)
                h_tm1 = h_t
                loss += T.mean(categorical_crossentropy(output,y_t[t]))
            
            loss = loss/self.steps # to take mean
            up_dldp_l2 = 0
            up_dni_l2 = 0
            # Train the LSTM: backprop (loss + DNI output)
            dni_out = self.dni.output(h_t)
            grads = []
            for param in self.params:
                dlossdparam = T.grad(loss,param)
                dniJ = T.Lop(h_t,param,dni_out,disconnected_inputs='ignore')
                up_dldp_l2 += T.sum(T.square(dlossdparam))
                up_dni_l2 += T.sum(T.square(dniJ))
                grads.append(dlossdparam+scale*dniJ)
            
            # Update the DNI (from the last step)
            # recalculate the DNI prediction since it can't be passed
            dni_out_old =self.dni.output(h_tmT)
            # dni target: current loss backprop'ed + new dni backprop'ed
            dni_target = T.grad(loss,h_tmT) \
                         +T.Lop(h_t,h_tmT,dni_out)
            dni_error = T.sum(T.square(dni_out_old-dni_target))
            for param in self.dni.params:
                grads.append(T.grad(dni_error,param))
            
            updates = adam(lr,self.params+self.dni.params,grads)
            
            return [h_t,loss,dni_error,T.sqrt(up_dldp_l2),T.sqrt(up_dni_l2)],updates
            
        h0 = T.zeros((self.n_hidden,),dtype=theano.config.floatX)
        [h,seq_loss,seq_dni_error,up_dldp,up_dni],updates = theano.scan(fn=step, 
                             sequences=[shufflereshape(x),
                                        shufflereshape(y)],
                             outputs_info=[T.alloc(h0,x.shape[0],self.n_hidden),
                                           None,None,None,None],
                             non_sequences=[self.Wx,self.Wh,self.bh,
                                            self.Wy,self.by,
                                            learning_rate,dni_scale])
        return theano.function(inputs=[x,y,learning_rate,dni_scale],
                               outputs=[T.mean(seq_loss),
                                        T.mean(seq_dni_error),
                                        T.mean(up_dldp),
                                        T.mean(up_dni)],
                               updates=updates)
    def test(self):
        x = T.tensor3('x')
        y = T.tensor3('y')
        
        def step(x_t,y_t,h_tm1,Wx,Wh,bh,Wy,by):
            h_t = relu(T.dot(x_t,Wx)+T.dot(h_tm1,Wh)+bh)
            yo_t = softmax(T.dot(h_t,Wy)+by)
            loss_t = T.mean(categorical_crossentropy(yo_t,y_t))
            return h_t,yo_t,loss_t
        h0 = T.zeros((self.n_hidden,),dtype=theano.config.floatX)
        [h,yo,loss],_ = theano.scan(fn=step, 
                             sequences=[x.dimshuffle([1,0,2]),
                                        y.dimshuffle([1,0,2])],
                             outputs_info=[T.alloc(h0,x.shape[0],self.n_hidden),
                                           None,None],
                             non_sequences=[self.Wx,self.Wh,self.bh,
                                            self.Wy,self.by])
        loss = T.mean(loss)
        return theano.function(inputs=[x,y],
                               outputs=loss)


class gru_dni(object):
    def __init__(self,n_in,n_hidden,n_out,steps,
                 norm=True,p_zoneout=-1,p_dropout=-1,gradclip=10,rng=rng):
        """
        GRU with a DNI every 'steps' timesteps
        """
        # parameters
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.steps = steps
        self.norm = norm
        self.p_zoneout = p_zoneout
        self.p_dropout = p_dropout
        self.gradclip = gradclip
        
        # initialize weights
        def ortho_weight(ndim,rng=rng):
            W = rng.randn(ndim, ndim)
            u, s, v = numpy.linalg.svd(W)
            return u.astype(theano.config.floatX)
        def uniform_weight(n1,n2,rng=rng):
            limit = numpy.sqrt(6./(n1+n2))
            return rng.uniform(low=-limit,high=limit,size=(n1,n2)).astype(theano.config.floatX)
        def const_bias(n,value=0):
            return value*numpy.ones((n,),dtype=theano.config.floatX)
        
        self.Wxz = theano.shared(uniform_weight(n_in,n_hidden),borrow=True)
        self.Wxr = theano.shared(uniform_weight(n_in,n_hidden),borrow=True)
        self.Wxg = theano.shared(uniform_weight(n_in,n_hidden),borrow=True)
        self.Whz = theano.shared(ortho_weight(n_hidden),borrow=True)
        self.Whr = theano.shared(ortho_weight(n_hidden),borrow=True)
        self.Whg = theano.shared(ortho_weight(n_hidden),borrow=True)
        self.Wy = theano.shared(uniform_weight(n_hidden,n_out),borrow=True)

        self.bz = theano.shared(const_bias(n_hidden,0),borrow=True)
        self.br = theano.shared(const_bias(n_hidden,1),borrow=True)
        self.bg = theano.shared(const_bias(n_hidden,0),borrow=True)
        self.by = theano.shared(const_bias(n_out,0),borrow=True)
        
        
        self.params = [self.Wxz,self.Wxr,self.Wxg,self.Whz,self.Whr,self.Whg,
                       self.Wy,self.bz,self.br,self.bg,self.by]
        # train some parameters without the dni backprop
        self.use_dni = [True]*len(self.params)
        
        if self.norm:
            # because of where layer normalization is used, no need for shifts
            self.scale1 = theano.shared(const_bias(n_hidden,0.5),borrow=True)
            self.scale2 = theano.shared(const_bias(n_hidden,0.5),borrow=True)
            self.scale3 = theano.shared(const_bias(n_hidden,0.5),borrow=True)
            self.scale4 = theano.shared(const_bias(n_hidden,0.5),borrow=True)
            self.scale5 = theano.shared(const_bias(n_hidden,0.5),borrow=True)
            self.scale6 = theano.shared(const_bias(n_hidden,0.5),borrow=True)
            self.params = self.params+[self.scale1,self.scale2,self.scale3,
                                       self.scale4,self.scale5,self.scale6]
            self.use_dni = self.use_dni+[True]*6
            
            # if using layer norm, initial hidden state h0 needs var>0
            # using a stable value rather than random helps w/ training
            h0 = numpy.zeros((n_hidden,),dtype=theano.config.floatX)
            h0[0] = 1
            self.h0 = theano.shared(h0,borrow=True)
        else:
            self.h0 = T.zeros((self.n_hidden,),dtype=theano.config.floatX)
        
        # initialize DNI
        self.dni = dni(n_hidden,n_hidden,2)
        
    # slice for doing step calculations in parallel
    def _slice(self,x,n):
        return x[:,n*self.n_hidden:(n+1)*self.n_hidden]
        
    def train(self):
        x = T.tensor3('x')
        y = T.tensor3('y')
        learning_rate = T.scalar('learning_rate')
        dni_scale = T.scalar('dni_scale')
        
        # reshape the inputs
        # batch x time x n -> time//steps x steps x batch x n
        def shufflereshape(r):
            r = r.dimshuffle([1,0,2])
            r = r.reshape((r.shape[0]//self.steps,
                           self.steps,
                           r.shape[1],
                           r.shape[2]))
            return r
        
        # step takes a set of inputs over self.steps time-steps
        def step(x_t,y_t,h_tmT,
                 Wxz,Whz,Wxr,Whr,Wxg,Whg,Wy,bz,br,bg,by,
                 lr,scale):
            
            # manually build the graph for the inner loop
            # note: nested scans are way slower unless (maybe) T is huge
            yo_t = []
            h_tm1 = h_tmT
            loss = 0
            for t in range(self.steps):
                if self.norm:
                    z = sigmoid(layer_norm(T.dot(x_t[t],Wxz),self.scale1) \
                                +layer_norm(T.dot(h_tm1,Whz),self.scale2)+bz)
                    r = sigmoid(layer_norm(T.dot(x_t[t],Wxr),self.scale3) \
                                +layer_norm(T.dot(h_tm1,Whr),self.scale4)+br)
                    g = tanh(layer_norm(T.dot(x_t[t],Wxg),self.scale5) \
                            +layer_norm(T.dot(r*h_tm1,Whg),self.scale6)+bg)
                else:
                    z = sigmoid(T.dot(x_t[t],Wxz)+T.dot(h_tm1,Whz)+bz)
                    r = sigmoid(T.dot(x_t[t],Wxr)+T.dot(h_tm1,Whr)+br)
                    g = tanh(T.dot(x_t[t],Wxg)+T.dot(r*h_tm1,Whg)+bg)
                h_t = (1-z)*h_tm1+z*g
                if self.p_zoneout>0:
                    h_t = zoneout(h_t,h_tm1,self.p_zoneout)
                h_t = theano.gradient.grad_clip(h_t,
                                                -self.gradclip,
                                                self.gradclip)
                if self.p_dropout>0:
                    output = softmax(T.dot(dropout(h_t,self.p_dropout),Wy)+by)
                else:
                    output = softmax(T.dot(h_t,Wy)+by)
                yo_t.append(output)
                h_tm1 = h_t
                loss += T.mean(categorical_crossentropy(output,y_t[t]))
            
            loss = loss/self.steps # to take mean
            up_dldp_l2 = 0
            up_dni_l2 = 0
            # Train the GRU: backprop (loss + DNI output)
            dni_out = self.dni.output(h_t)
            grads = []
            for param,use_dni in zip(self.params,self.use_dni):
                dlossdparam = T.grad(loss,param)
                up_dldp_l2 += T.sum(T.square(dlossdparam))
                if use_dni:
                    dniJ = T.Lop(h_t,param,dni_out,disconnected_inputs='ignore')
                    up_dni_l2 += T.sum(T.square(dniJ))
                    dlossdparam = dlossdparam+scale*dniJ
                grads.append(dlossdparam)
            
            # Update the DNI (from the last step)
            # recalculate the DNI prediction since it can't be passed
            dni_out_old =self.dni.output(h_tmT)
            # dni target: current loss backprop'ed + new dni backprop'ed
            dni_target = T.grad(loss,h_tmT) \
                         +T.Lop(h_t,h_tmT,dni_out)
            dni_error = T.sum(T.square(dni_out_old-dni_target))
            for param in self.dni.params:
                grads.append(T.grad(dni_error,param))
            
            updates = adam(lr,self.params+self.dni.params,grads)
            
            return [h_t,loss,dni_error,T.sqrt(up_dldp_l2),T.sqrt(up_dni_l2)],updates
        
        [h,seq_loss,seq_dni_error,up_dldp,up_dni],updates = theano.scan(fn=step, 
                             sequences=[shufflereshape(x),
                                        shufflereshape(y)],
                             outputs_info=[T.alloc(self.h0,x.shape[0],self.n_hidden),
                                           None,None,None,None],
                             non_sequences=[self.Wxz,self.Whz,self.Wxr,self.Whr,self.Wxg,self.Whg,
                                            self.Wy,self.bz,self.br,self.bg,self.by,
                                            learning_rate,dni_scale])
        return theano.function(inputs=[x,y,learning_rate,dni_scale],
                               outputs=[T.mean(seq_loss),
                                        T.mean(seq_dni_error),
                                        T.mean(up_dldp),
                                        T.mean(up_dni)],
                               updates=updates)
    def test(self):
        x = T.tensor3('x')
        y = T.tensor3('y')
        
        def step(x_t,y_t,h_tm1,Wxz,Whz,Wxr,Whr,Wxg,Whg,Wy,bz,br,bg,by):
            if self.norm:
                z = sigmoid(layer_norm(T.dot(x_t,Wxz),self.scale1) \
                            +layer_norm(T.dot(h_tm1,Whz),self.scale2)+bz)
                r = sigmoid(layer_norm(T.dot(x_t,Wxr),self.scale3) \
                            +layer_norm(T.dot(h_tm1,Whr),self.scale4)+br)
                g = tanh(layer_norm(T.dot(x_t,Wxg),self.scale5) \
                        +layer_norm(T.dot(r*h_tm1,Whg),self.scale6)+bg)
            else:
                z = sigmoid(T.dot(x_t,Wxz)+T.dot(h_tm1,Whz)+bz)
                r = sigmoid(T.dot(x_t,Wxr)+T.dot(h_tm1,Whr)+br)
                g = tanh(T.dot(x_t,Wxg)+T.dot(r*h_tm1,Whg)+bg)
            h_t = (1-z)*h_tm1+z*g
            yo_t = softmax(T.dot(h_t,Wy)+by)
            loss_t = T.mean(categorical_crossentropy(yo_t,y_t))
            return h_t,yo_t,loss_t
        [h,yo,loss],_ = theano.scan(fn=step, 
                             sequences=[x.dimshuffle([1,0,2]),
                                        y.dimshuffle([1,0,2])],
                             outputs_info=[T.alloc(self.h0,x.shape[0],self.n_hidden),
                                           None,None],
                             non_sequences=[self.Wxz,self.Whz,self.Wxr,self.Whr,self.Wxg,self.Whg,
                                            self.Wy,self.bz,self.br,self.bg,self.by])
        loss = T.mean(loss)
        return theano.function(inputs=[x,y],
                               outputs=loss)


class lstm_dni(object):
    def __init__(self,n_in,n_hidden,n_out,steps,
                 norm=True,rng=rng):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.steps = steps
        self.norm = norm
        
        # initialize weights
        def ortho_weight(ndim,rng=rng):
            W = rng.randn(ndim, ndim)
            u, s, v = numpy.linalg.svd(W)
            return u.astype(theano.config.floatX)
        def uniform_weight(n1,n2,rng=rng):
            limit = numpy.sqrt(6./(n1+n2))
            return rng.uniform(low=-limit,high=limit,size=(n1,n2)).astype(theano.config.floatX)
        def const_bias(n,value=0):
            return value*numpy.ones((n,),dtype=theano.config.floatX)
        self.Wxi = theano.shared(uniform_weight(n_in,n_hidden),borrow=True)
        self.Wxf = theano.shared(uniform_weight(n_in,n_hidden),borrow=True)
        self.Wxo = theano.shared(uniform_weight(n_in,n_hidden),borrow=True)
        self.Wxg = theano.shared(uniform_weight(n_in,n_hidden),borrow=True)
        self.Wx = T.concatenate([self.Wxi,self.Wxf,self.Wxo,self.Wxg],
                                axis=1)
        self.Whi = theano.shared(ortho_weight(n_hidden,rng),borrow=True)
        self.Whf = theano.shared(ortho_weight(n_hidden,rng),borrow=True)
        self.Who = theano.shared(ortho_weight(n_hidden,rng),borrow=True)
        self.Whg = theano.shared(ortho_weight(n_hidden,rng),borrow=True)
        self.Wh = T.concatenate([self.Whi,self.Whf,self.Who,self.Whg],
                                axis=1)
        self.Wy = theano.shared(uniform_weight(n_hidden,n_out),borrow=True)
        
        self.bi = theano.shared(const_bias(n_hidden,-1),borrow=True)
        self.bf = theano.shared(const_bias(n_hidden,1),borrow=True)
        self.bo = theano.shared(const_bias(n_hidden,1),borrow=True)
        self.bg = theano.shared(const_bias(n_hidden,0),borrow=True)
        self.b = T.concatenate([self.bi,self.bf,self.bo,self.bg],axis=0)
        self.by = theano.shared(const_bias(n_out,0),borrow=True)
        
        self.params = [self.Wx,self.Wh,self.Wy,self.b,self.by]
        self.W = [self.Wx,self.Wh,self.Wy]
        self.L1 = numpy.sum([abs(w).sum() for w in self.W])
        self.L2 = numpy.sum([(w**2).sum() for w in self.W])
        
        if self.norm:
            # because of bias right after, no need for shifts 1 and 2
            self.scaleh = theano.shared(const_bias(4*n_hidden,0.5),borrow=True)
            self.scalex = theano.shared(const_bias(4*n_hidden,0.5),borrow=True)
            self.scalec = theano.shared(const_bias(n_hidden,1),borrow=True)
            self.shiftc = theano.shared(const_bias(n_hidden,0),borrow=True)
            self.params = self.params+[self.scaleh,self.scalex,
                                       self.scalec,self.shiftc]
            
            # if using layer norm, initial hidden state h0 needs var>0
            # using a stable value rather than random helps w/ training
            h0 = numpy.zeros((n_hidden,),dtype=theano.config.floatX)
            h0[0] = 1
            self.h0 = theano.shared(h0,borrow=True)
        else:
            self.h0 = T.zeros((self.n_hidden,),dtype=theano.config.floatX)
        
        # initialize DNI
        self.dni = dni(n_hidden,2*n_hidden,2)
    
    # slice for doing step calculations in parallel
    def _slice(self,x,n):
        return x[:,n*self.n_hidden:(n+1)*self.n_hidden]
        
    def train(self):
        x = T.tensor3('x')
        y = T.tensor3('y')
        learning_rate = T.scalar('learning_rate')
        dni_scale = T.scalar('dni_scale')
        
        # reshape the inputs
        # batch x time x n -> time//steps x steps x batch x n
        def shufflereshape(r):
            r = r.dimshuffle([1,0,2])
            r = r.reshape((r.shape[0]//self.steps,
                           self.steps,
                           r.shape[1],
                           r.shape[2]))
            return r
        
        # step takes a set of inputs over self.steps time-steps
        def step(x_t,y_t,c_tmT,h_tmT,
                 Wx,Wh,Wy,b,by,
                 lr,scale):
            
            # manually build the graph for the inner loop
            yo_t = []
            c_tm1 = c_tmT
            h_tm1 = h_tmT
            loss = 0
            for t in range(self.steps):
                if self.norm:
                    preact = layer_norm(T.dot(x_t[t],Wx),self.scalex) \
                             +layer_norm(T.dot(h_tm1,Wh),self.scaleh)+b
                else:
                    preact = T.dot(x_t[t],Wx)+T.dot(h_tm1,Wh)+b
                i = sigmoid(self._slice(preact,0))
                f = sigmoid(self._slice(preact,1))
                o = sigmoid(self._slice(preact,2))
                g = tanh(self._slice(preact,3))
                c_t = c_tm1*f+g*i
                if self.norm:
                    c_t = layer_norm(c_t,self.scalec,self.shiftc)
                h_t = o*tanh(c_t)
                output = softmax(T.dot(h_t,Wy)+by)
                yo_t.append(output)
                # update for next step
                c_tm1 = c_t
                h_tm1 = h_t
                loss += T.mean(categorical_crossentropy(output,y_t[t]))
            
            loss = loss/self.steps # to take mean
            up_dldp_l2 = 0
            up_dni_l2 = 0
            # Train the LSTM: backprop (loss + DNI output)
            dni_out = self.dni.output(h_t)
            dni_out_h = self._slice(dni_out,0)
            dni_out_c = self._slice(dni_out,1)
            grads = []
            params = [] # params to Adam need to be in same order as grads 
            # output weights don't have dni feedback
            for param in [self.Wy,self.by]:
                dlossdparam = T.grad(loss,param)
                grads.append(dlossdparam)
                params.append(param)
            # output-gate weights get dni_h
            dni_h_params = [self.Wxo,self.Who,self.bo]
            for param in dni_h_params:
                dlossdparam = T.grad(loss,param)
                dniJ = T.Lop(h_t,param,dni_out_h)
                grads.append(dlossdparam+scale*dniJ)
                params.append(param)
                up_dldp_l2 += T.sum(T.square(dlossdparam))
                up_dni_l2 += T.sum(T.square(dniJ))
            # all other gates get dni_c
            dni_c_params = [self.Wxi,self.Wxf,self.Wxg,
                            self.Whi,self.Whf,self.Whg,
                            self.bi,self.bf,self.bg]
            if self.norm:
                dni_c_params = dni_c_params + [self.scalex,self.scaleh,
                                               self.scalec,self.shiftc]
            for param in dni_c_params:
                dlossparam = T.grad(loss,param)
                dniJ = T.Lop(c_t,param,dni_out_c)
                grads.append(dlossparam+scale*dniJ)
                params.append(param)
                up_dldp_l2 += T.sum(T.square(dlossdparam))
                up_dni_l2 += T.sum(T.square(dniJ))
            
            # Update the DNI (from the last step)
            # recalculate the DNI prediction since it can't be passed
            dni_out_old = self.dni.output(h_tmT)
            dni_old_h = self._slice(dni_out_old,0)
            dni_old_c = self._slice(dni_out_old,1)
            # dni target: current loss backprop'ed + new dni backprop'ed
            dni_target_h = T.grad(loss,h_tmT) \
                           +T.Lop(h_t,h_tmT,dni_out_h)
            dni_target_c = T.grad(loss,c_tmT) \
                           +T.Lop(c_t,c_tmT,dni_out_c)
            dni_error = T.sum(T.square(dni_old_h-dni_target_h)) \
                        +T.sum(T.square(dni_old_c-dni_target_c))
            for param in self.dni.params:
                grads.append(T.grad(dni_error,param))
            
            updates = adam(lr,params+self.dni.params,grads)
            
            return [c_t,h_t,loss,dni_error,
                    T.sqrt(up_dldp_l2),T.sqrt(up_dni_l2)],updates
        
        c0 = T.zeros((self.n_hidden,),dtype=theano.config.floatX)
        [c,h,
         seq_loss,seq_dni_error,
         up_dldp,up_dni],updates = theano.scan(fn=step,
                                sequences=[shufflereshape(x),
                                           shufflereshape(y)],
                                outputs_info=[T.alloc(c0,x.shape[0],self.n_hidden),
                                              T.alloc(self.h0,x.shape[0],self.n_hidden),
                                              None,None,None,None],
                                non_sequences=[self.Wx,self.Wh,self.Wy,
                                               self.b,self.by,
                                               learning_rate,dni_scale])
        return theano.function(inputs=[x,y,learning_rate,dni_scale],
                               outputs=[T.mean(seq_loss),
                                        T.mean(seq_dni_error),
                                        T.mean(up_dldp),
                                        T.mean(up_dni)],
                                updates=updates)
    
    def test(self):
        x = T.tensor3('x')
        y = T.tensor3('y')
        
        def step(x_t,y_t,c_tm1,h_tm1,
                 Wx,Wh,Wy,b,by):
            if self.norm:
                preact = layer_norm(T.dot(x_t,Wx),self.scalex) \
                         +layer_norm(T.dot(h_tm1,Wh),self.scaleh)+b
            else:
                preact = T.dot(x_t,Wx)+T.dot(h_tm1,Wh)+b
            i = sigmoid(self._slice(preact,0))
            f = sigmoid(self._slice(preact,1))
            o = sigmoid(self._slice(preact,2))
            g = tanh(self._slice(preact,3))
            c_t = c_tm1*f+g*i
            if self.norm:
                    c_t = layer_norm(c_t,self.scalec,self.shiftc)
            h_t = o*tanh(c_t)
            yo_t = softmax(T.dot(h_t,Wy)+by)
            loss_t = T.mean(categorical_crossentropy(yo_t,y_t))
            return c_t,h_t,yo_t,loss_t
        c0 = T.alloc(T.zeros((self.n_hidden,),dtype=theano.config.floatX),
                     x.shape[0],self.n_hidden)
        h0 = T.alloc(self.h0,x.shape[0],self.n_hidden)
        [c,s,yo,loss],_ = theano.scan(fn=step,
                              sequences=[x.dimshuffle([1,0,2]),
                                         y.dimshuffle([1,0,2])],
                              outputs_info=[c0,h0,None,None],
                              non_sequences=[self.Wx,self.Wh,self.Wy,
                                             self.b,self.by])
        loss = T.mean(loss)
        return theano.function(inputs=[x,y],
                               outputs=loss)


class rnn_trace(object):
    def __init__(self,n_in,n_hidden,n_out,steps,rng=rng):
        """
        Build a simple RNN with a DNI every 'steps' timesteps
        """
        # parameters
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.steps = steps
        
        # initialize weights
        self.Wx = uniform_weight(n_in,n_hidden)
        self.Wh = ortho_weight(n_hidden)
        self.bh = const_bias(n_hidden,0)
        self.Wy = uniform_weight(n_hidden,n_out)
        self.by = const_bias(n_out,0)
        self.params = [self.Wx,self.Wh,self.bh,self.Wy,self.by]
        
        self.non_trace_params = [self.Wx,self.Wy,self.bh,self.by]
        
        # initialize DNI
        self.dni = dni(n_hidden,n_hidden,2)
        
    def train(self):
        x = T.tensor3('x')
        y = T.tensor3('y')
        learning_rate = T.scalar('learning_rate')
        trace_decay = T.scalar('trace_decay')
        dni_scale = T.scalar('dni_scale')
        
        # re-initialize activation trace
        def floatX(data):
            return numpy.asarray(data,dtype=theano.config.floatX)
        self.trace = theano.shared(self.Wh.get_value()*floatX(0.))
        
        # small constant to avoid dividing by zero
        eps = 1e-8
        
        # reshape the inputs
        # batch x time x n -> time//steps x steps x batch x n
        def shufflereshape(r):
            r = r.dimshuffle([1,0,2])
            r = r.reshape((r.shape[0]//self.steps,
                           self.steps,
                           r.shape[1],
                           r.shape[2]))
            return r
        
        # step takes a set of inputs over self.steps time-steps
        def step(x_t,y_t,h_tmT,Wx,Wh,bh,Wy,by,lr,scale,trace,decay):
            
            # manually build the graph for the inner loop...
            # passing correct h_tm1 is impossible in nested scans
            yo_t = []
            h_tm1 = h_tmT
            old_trace = trace
            loss = 0
            for t in range(self.steps):
                preact_h = T.dot(h_tm1,Wh)
                h_t = relu(T.dot(x_t[t],Wx)+preact_h+bh)
                output = softmax(T.dot(h_t,Wy)+by)
                yo_t.append(output)
                h_tm1 = h_t
                loss += T.mean(categorical_crossentropy(output,y_t[t]))
                # update traces: RMS taken over the batch dimension
                trace = decay*trace \
                        +T.sqrt(T.mean(T.sqr(preact_h),axis=0,keepdims=True))
            
            loss = loss/self.steps # to take mean
            up_dldp_l2 = 0
            up_dni_l2 = 0
            # Train the LSTM: backprop (loss + DNI output)
            dni_out = self.dni.output(h_t)
            grads = []
            # recurrent weights incorporate traces
            for param in [self.Wh]:
                trace_scale = (trace+eps)/ \
                                (T.sqrt(T.mean(T.sqr(trace)))+eps)
                dlossdparam = T.grad(loss,param)
                dniJ = T.Lop(h_t,param,dni_out,disconnected_inputs='ignore')
                up_dldp_l2 += T.sum(T.square(dlossdparam))
                up_dni_l2 += T.sum(T.square(dniJ))
                grads.append(trace_scale*(dlossdparam+scale*dniJ))
            # non-trace parameters get normal DNI updates
            for param in self.non_trace_params:
                dlossdparam = T.grad(loss,param)
                dniJ = T.Lop(h_t,param,dni_out,disconnected_inputs='ignore')
                up_dldp_l2 += T.sum(T.square(dlossdparam))
                up_dni_l2 += T.sum(T.square(dniJ))
                grads.append(dlossdparam+scale*dniJ)
            
            # Update the DNI (from the last step)
            # recalculate the DNI prediction since it can't be passed
            dni_out_old =self.dni.output(h_tmT)
            # dni target: current loss backprop'ed + new dni backprop'ed
            dni_target = T.grad(loss,h_tmT) \
                         +T.Lop(h_t,h_tmT,dni_out)
            dni_error = T.sum(T.square(dni_out_old-dni_target))
            for param in self.dni.params:
                grads.append(T.grad(dni_error,param))
            
            # params need to be passed to adam in same order as their grads
            updates = adam(lr,[self.Wh]+self.non_trace_params+self.dni.params,grads)
            updates.append((old_trace,trace))
            
            return [h_t,loss,dni_error,T.sqrt(up_dldp_l2),T.sqrt(up_dni_l2)],updates
            
        h0 = T.zeros((self.n_hidden,),dtype=theano.config.floatX)
        [h,seq_loss,seq_dni_error,up_dldp,up_dni],updates = theano.scan(fn=step, 
                             sequences=[shufflereshape(x),
                                        shufflereshape(y)],
                             outputs_info=[T.alloc(h0,x.shape[0],self.n_hidden),
                                           None,None,None,None],
                             non_sequences=[self.Wx,self.Wh,self.bh,
                                            self.Wy,self.by,
                                            learning_rate,dni_scale,
                                            self.trace,trace_decay])
        return theano.function(inputs=[x,y,learning_rate,
                                       dni_scale,trace_decay],
                               outputs=[T.mean(seq_loss),
                                        T.mean(seq_dni_error),
                                        T.mean(up_dldp),
                                        T.mean(up_dni)],
                               updates=updates)
    def test(self):
        x = T.tensor3('x')
        y = T.tensor3('y')
        
        def step(x_t,y_t,h_tm1,Wx,Wh,bh,Wy,by):
            h_t = relu(T.dot(x_t,Wx)+T.dot(h_tm1,Wh)+bh)
            yo_t = softmax(T.dot(h_t,Wy)+by)
            loss_t = T.mean(categorical_crossentropy(yo_t,y_t))
            return h_t,yo_t,loss_t
        h0 = T.zeros((self.n_hidden,),dtype=theano.config.floatX)
        [h,yo,loss],_ = theano.scan(fn=step, 
                             sequences=[x.dimshuffle([1,0,2]),
                                        y.dimshuffle([1,0,2])],
                             outputs_info=[T.alloc(h0,x.shape[0],self.n_hidden),
                                           None,None],
                             non_sequences=[self.Wx,self.Wh,self.bh,
                                            self.Wy,self.by])
        loss = T.mean(loss)
        return theano.function(inputs=[x,y],
                               outputs=loss)


class gru_trace(object):
    def __init__(self,n_in,n_hidden,n_out,steps,
                 norm=True,p_zoneout=-1,p_dropout=-1,gradclip=10,rng=rng):
        """
        GRU with a DNI every 'steps' timesteps and eligibility trace updates
        """
        # parameters
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.steps = steps
        self.norm = norm
        self.p_zoneout = p_zoneout
        self.p_dropout = p_dropout
        self.gradclip = gradclip
        
        # initialize weights
        def ortho_weight(ndim,rng=rng):
            W = rng.randn(ndim, ndim)
            u, s, v = numpy.linalg.svd(W)
            return u.astype(theano.config.floatX)
        def uniform_weight(n1,n2,rng=rng):
            limit = numpy.sqrt(6./(n1+n2))
            return rng.uniform(low=-limit,high=limit,size=(n1,n2)).astype(theano.config.floatX)
        def const_bias(n,value=0):
            return value*numpy.ones((n,),dtype=theano.config.floatX)
        
        self.Wxz = theano.shared(uniform_weight(n_in,n_hidden),borrow=True)
        self.Wxr = theano.shared(uniform_weight(n_in,n_hidden),borrow=True)
        self.Wxg = theano.shared(uniform_weight(n_in,n_hidden),borrow=True)
        self.Whz = theano.shared(ortho_weight(n_hidden),borrow=True)
        self.Whr = theano.shared(ortho_weight(n_hidden),borrow=True)
        self.Whg = theano.shared(ortho_weight(n_hidden),borrow=True)
        self.Wy = theano.shared(uniform_weight(n_hidden,n_out),borrow=True)

        self.bz = theano.shared(const_bias(n_hidden,0),borrow=True)
        self.br = theano.shared(const_bias(n_hidden,1),borrow=True)
        self.bg = theano.shared(const_bias(n_hidden,0),borrow=True)
        self.by = theano.shared(const_bias(n_out,0),borrow=True)
        
        self.params = [self.Wxz,self.Wxr,self.Wxg,self.Whz,self.Whr,self.Whg,
                       self.Wy,self.bz,self.br,self.bg,self.by]
        
        if self.norm:
            # because of where layer normalization is used, no need for shifts
            self.scale1 = theano.shared(const_bias(n_hidden,0.5),borrow=True)
            self.scale2 = theano.shared(const_bias(n_hidden,0.5),borrow=True)
            self.scale3 = theano.shared(const_bias(n_hidden,0.5),borrow=True)
            self.scale4 = theano.shared(const_bias(n_hidden,0.5),borrow=True)
            self.scale5 = theano.shared(const_bias(n_hidden,0.5),borrow=True)
            self.scale6 = theano.shared(const_bias(n_hidden,0.5),borrow=True)
            self.params = self.params+[self.scale1,self.scale2,self.scale3,
                                       self.scale4,self.scale5,self.scale6]
            
            # if using layer norm, initial hidden state h0 needs var>0
            # using a stable value rather than random helps w/ training
            h0 = numpy.zeros((n_hidden,),dtype=theano.config.floatX)
            h0[0] = 1
            self.h0 = theano.shared(h0,borrow=True)
        else:
            self.h0 = T.zeros((self.n_hidden,),dtype=theano.config.floatX)
        
        # initialize DNI
        self.dni = dni(n_hidden,n_hidden,2)
        
        # not all variables get eligiblity trace updates
        self.trace_params = [self.Wxz,self.Wxr,self.Wxg,self.Whz,self.Whr,self.Whg]
        self.non_trace_params = [p for p in self.params if p not in self.trace_params]
        
    # slice for doing step calculations in parallel
    def _slice(self,x,n):
        return x[:,n*self.n_hidden:(n+1)*self.n_hidden]
        
    def train(self):
        x = T.tensor3('x')
        y = T.tensor3('y')
        learning_rate = T.scalar('learning_rate')
        trace_decay = T.scalar('trace_decay')
        dni_scale = T.scalar('dni_scale')
        
        # re-initialize activation traces
        def floatX(data):
            return numpy.asarray(data,dtype=theano.config.floatX)
        trace_xz = theano.shared(self.Wxz.get_value()*floatX(0.))
        trace_xr = theano.shared(self.Wxr.get_value()*floatX(0.))
        trace_xg = theano.shared(self.Wxg.get_value()*floatX(0.))
        trace_hz = theano.shared(self.Whz.get_value()*floatX(0.))
        trace_hr = theano.shared(self.Whr.get_value()*floatX(0.))
        trace_hg = theano.shared(self.Whg.get_value()*floatX(0.))
        
        # small constant to avoid dividing by zero
        eps = 1e-8
        
        # reshape the inputs
        # batch x time x n -> time//steps x steps x batch x n
        def shufflereshape(r):
            r = r.dimshuffle([1,0,2])
            r = r.reshape((r.shape[0]//self.steps,
                           self.steps,
                           r.shape[1],
                           r.shape[2]))
            return r
        
        # step takes a set of inputs over self.steps time-steps
        def step(x_t,y_t,h_tmT,
                 Wxz,Whz,Wxr,Whr,Wxg,Whg,Wy,bz,br,bg,by,
                 lr,scale,Txz,Txr,Txg,Thz,Thr,Thg,decay):
            
            # manually build the graph for the inner loop
            # note: nested scans are way slower unless (maybe) T is huge
            yo_t = []
            h_tm1 = h_tmT
            loss = 0
            oldTxz = Txz
            oldTxr = Txr
            oldTxg = Txg
            oldThz = Thz
            oldThr = Thr
            oldThg = Thg
            for t in range(self.steps):
                if self.norm:
                    xWxz = layer_norm(T.dot(x_t[t],Wxz),self.scale1)
                    hWhz = layer_norm(T.dot(h_tm1,Whz),self.scale2)
                    z = sigmoid(xWxz \
                                +hWhz+bz)
                    xWxr = layer_norm(T.dot(x_t[t],Wxr),self.scale3)
                    hWhr = layer_norm(T.dot(h_tm1,Whr),self.scale4)
                    r = sigmoid(xWxr \
                                +hWhr+br)
                    xWxg = layer_norm(T.dot(x_t[t],Wxg),self.scale5)
                    hWhg = layer_norm(T.dot(r*h_tm1,Whg),self.scale6)
                    g = tanh(xWxg \
                            +hWhg+bg)
                else:
                    xWxz = T.dot(x_t[t],Wxz)
                    hWhz = T.dot(h_tm1,Whz)
                    z = sigmoid(xWxz+hWhz+bz)
                    xWxr = T.dot(x_t[t],Wxr)
                    hWhr = T.dot(h_tm1,Whr)
                    r = sigmoid(xWxr+hWhr+br)
                    xWxg = T.dot(x_t[t],Wxg)
                    hWhg = T.dot(r*h_tm1,Whg)
                    g = tanh(xWxg+hWhg+bg)
                h_t = (1-z)*h_tm1+z*g
                if self.p_zoneout>0:
                    h_t = zoneout(h_t,h_tm1,self.p_zoneout)
                h_t = theano.gradient.grad_clip(h_t,
                                                -self.gradclip,
                                                self.gradclip)
                if self.p_dropout>0:
                    output = softmax(T.dot(dropout(h_t,self.p_dropout),Wy)+by)
                else:
                    output = softmax(T.dot(h_t,Wy)+by)
                yo_t.append(output)
                h_tm1 = h_t
                loss += T.mean(categorical_crossentropy(output,y_t[t]))
                
#                # update traces: variance over batch dim
#                Txz = decay*Txz+T.var(xWxz,axis=0,keepdims=True)
#                Txr = decay*Txr+T.var(xWxr,axis=0,keepdims=True)
#                Txg = decay*Txg+T.var(xWxg,axis=0,keepdims=True)
#                Thz = decay*Thz+T.var(hWhz,axis=0,keepdims=True)
#                Thr = decay*Thr+T.var(hWhr,axis=0,keepdims=True)
#                Thg = decay*Thg+T.var(hWhg,axis=0,keepdims=True)
                
#                # update traces: variance over batch divided by total variance
#                Txz = decay*Txz+T.var(xWxz,axis=0,keepdims=True)/T.var(xWxz)
#                Txr = decay*Txr+T.var(xWxr,axis=0,keepdims=True)/T.var(xWxr)
#                Txg = decay*Txg+T.var(xWxg,axis=0,keepdims=True)/T.var(xWxg)
#                Thz = decay*Thz+T.var(hWhz,axis=0,keepdims=True)/T.var(hWhz)
#                Thr = decay*Thr+T.var(hWhr,axis=0,keepdims=True)/T.var(hWhr)
#                Thg = decay*Thg+T.var(hWhg,axis=0,keepdims=True)/T.var(hWhg)
                
                # update traces: mean abs over batch divided by total mean
                Txz = decay*Txz+T.mean(T.abs_(xWxz),axis=0,keepdims=True)/T.mean(T.abs_(xWxz))
                Txr = decay*Txr+T.mean(T.abs_(xWxr),axis=0,keepdims=True)/T.mean(T.abs_(xWxr))
                Txg = decay*Txg+T.mean(T.abs_(xWxg),axis=0,keepdims=True)/T.mean(T.abs_(xWxg))
                Thz = decay*Thz+T.mean(T.abs_(hWhz),axis=0,keepdims=True)/T.mean(T.abs_(hWhz))
                Thr = decay*Thr+T.mean(T.abs_(hWhr),axis=0,keepdims=True)/T.mean(T.abs_(hWhr))
                Thg = decay*Thg+T.mean(T.abs_(hWhg),axis=0,keepdims=True)/T.mean(T.abs_(hWhg))
                
#                # update traces: mean abs over batch dim
#                Txz = decay*Txz+T.mean(T.abs_(xWxz),axis=0,keepdims=True)
#                Txr = decay*Txr+T.mean(T.abs_(xWxr),axis=0,keepdims=True)
#                Txg = decay*Txg+T.mean(T.abs_(xWxg),axis=0,keepdims=True)
#                Thz = decay*Thz+T.mean(T.abs_(hWhz),axis=0,keepdims=True)
#                Thr = decay*Thr+T.mean(T.abs_(hWhr),axis=0,keepdims=True)
#                Thg = decay*Thg+T.mean(T.abs_(hWhg),axis=0,keepdims=True)
            
            loss = loss/self.steps # to take mean
            up_dldp_l2 = 0
            up_dni_l2 = 0
            # Train the GRU: backprop (loss + DNI output)
            dni_out = self.dni.output(h_t)
            grads = []
            params = []
            traces = [Txz,Txr,Txg,Thz,Thr,Thg]
            for param,trace in zip(self.trace_params,traces):
#                trace_scale = (trace+eps)/(T.sqrt(T.mean(T.sqr(trace)))+eps)
                trace_scale = (trace+eps)/(T.mean(trace)+eps)
#                trace_scale = (T.mean(trace)+eps)/(trace+eps)
#                trace_scale = (T.sqrt(trace)+eps)/(T.sqrt(T.mean(trace))+eps)
                dlossdparam = T.grad(loss,param)
                up_dldp_l2 += T.sum(T.square(dlossdparam))
                dniJ = T.Lop(h_t,param,dni_out,disconnected_inputs='ignore')
                up_dni_l2 += T.sum(T.square(dniJ))
                dlossdparam = dlossdparam+scale*dniJ
                grads.append(trace_scale*dlossdparam)
                params.append(param)
            for param in self.non_trace_params:
                dlossdparam = T.grad(loss,param)
                up_dldp_l2 += T.sum(T.square(dlossdparam))
                dniJ = T.Lop(h_t,param,dni_out,disconnected_inputs='ignore')
                up_dni_l2 += T.sum(T.square(dniJ))
                dlossdparam = dlossdparam+scale*dniJ
                grads.append(dlossdparam)
                params.append(param)
            
            # Update the DNI (from the last step)
            # recalculate the DNI prediction since it can't be passed
            dni_out_old =self.dni.output(h_tmT)
            # dni target: current loss backprop'ed + new dni backprop'ed
            dni_target = T.grad(loss,h_tmT) \
                         +scale*T.Lop(h_t,h_tmT,dni_out)
            dni_error = T.sum(T.square(dni_out_old-dni_target))
            for param in self.dni.params:
                grads.append(T.grad(dni_error,param))
                params.append(param)
            
            updates = adam(lr,params,grads)
            updates.append((oldTxz,Txz))
            updates.append((oldTxr,Txr))
            updates.append((oldTxg,Txg))
            updates.append((oldThz,Thz))
            updates.append((oldThr,Thr))
            updates.append((oldThg,Thg))
            
            return [h_t,loss,dni_error,T.sqrt(up_dldp_l2),T.sqrt(up_dni_l2)],updates
        
        [h,seq_loss,seq_dni_error,up_dldp,up_dni],updates = theano.scan(fn=step, 
                             sequences=[shufflereshape(x),
                                        shufflereshape(y)],
                             outputs_info=[T.alloc(self.h0,x.shape[0],self.n_hidden),
                                           None,None,None,None],
                             non_sequences=[self.Wxz,self.Whz,self.Wxr,self.Whr,
                                            self.Wxg,self.Whg,self.Wy,
                                            self.bz,self.br,self.bg,self.by,
                                            learning_rate,dni_scale,
                                            trace_xz,trace_xr,trace_xg,
                                            trace_hz,trace_hr,trace_hg,
                                            trace_decay])
        return theano.function(inputs=[x,y,learning_rate,dni_scale,trace_decay],
                               outputs=[T.mean(seq_loss),
                                        T.mean(seq_dni_error),
                                        T.mean(up_dldp),
                                        T.mean(up_dni)],
                               updates=updates,
                               name='gru_trace_train')
    def test(self):
        x = T.tensor3('x')
        y = T.tensor3('y')
        
        def step(x_t,y_t,h_tm1,Wxz,Whz,Wxr,Whr,Wxg,Whg,Wy,bz,br,bg,by):
            if self.norm:
                z = sigmoid(layer_norm(T.dot(x_t,Wxz),self.scale1) \
                            +layer_norm(T.dot(h_tm1,Whz),self.scale2)+bz)
                r = sigmoid(layer_norm(T.dot(x_t,Wxr),self.scale3) \
                            +layer_norm(T.dot(h_tm1,Whr),self.scale4)+br)
                g = tanh(layer_norm(T.dot(x_t,Wxg),self.scale5) \
                        +layer_norm(T.dot(r*h_tm1,Whg),self.scale6)+bg)
            else:
                z = sigmoid(T.dot(x_t,Wxz)+T.dot(h_tm1,Whz)+bz)
                r = sigmoid(T.dot(x_t,Wxr)+T.dot(h_tm1,Whr)+br)
                g = tanh(T.dot(x_t,Wxg)+T.dot(r*h_tm1,Whg)+bg)
            h_t = (1-z)*h_tm1+z*g
            yo_t = softmax(T.dot(h_t,Wy)+by)
            loss_t = T.mean(categorical_crossentropy(yo_t,y_t))
            return h_t,yo_t,loss_t
        [h,yo,loss],_ = theano.scan(fn=step, 
                             sequences=[x.dimshuffle([1,0,2]),
                                        y.dimshuffle([1,0,2])],
                             outputs_info=[T.alloc(self.h0,x.shape[0],self.n_hidden),
                                           None,None],
                             non_sequences=[self.Wxz,self.Whz,self.Wxr,self.Whr,self.Wxg,self.Whg,
                                            self.Wy,self.bz,self.br,self.bg,self.by])
        loss = T.mean(loss)
        return theano.function(inputs=[x,y],
                               outputs=loss)


class lstm_trace(object):
    def __init__(self,n_in,n_hidden,n_out,steps,
                 norm=True,rng=rng):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.steps = steps
        self.norm = norm
        
        # initialize weights
        def ortho_weight(ndim,rng=rng):
            W = rng.randn(ndim, ndim)
            u, s, v = numpy.linalg.svd(W)
            return u.astype(theano.config.floatX)
        def uniform_weight(n1,n2,rng=rng):
            limit = numpy.sqrt(6./(n1+n2))
            return rng.uniform(low=-limit,high=limit,size=(n1,n2)).astype(theano.config.floatX)
        def const_bias(n,value=0):
            return value*numpy.ones((n,),dtype=theano.config.floatX)
        self.Wxi = theano.shared(uniform_weight(n_in,n_hidden),borrow=True)
        self.Wxf = theano.shared(uniform_weight(n_in,n_hidden),borrow=True)
        self.Wxo = theano.shared(uniform_weight(n_in,n_hidden),borrow=True)
        self.Wxg = theano.shared(uniform_weight(n_in,n_hidden),borrow=True)
        self.Wx = T.concatenate([self.Wxi,self.Wxf,self.Wxo,self.Wxg],
                                axis=1)
        self.Whi = theano.shared(ortho_weight(n_hidden,rng),borrow=True)
        self.Whf = theano.shared(ortho_weight(n_hidden,rng),borrow=True)
        self.Who = theano.shared(ortho_weight(n_hidden,rng),borrow=True)
        self.Whg = theano.shared(ortho_weight(n_hidden,rng),borrow=True)
        self.Wh = T.concatenate([self.Whi,self.Whf,self.Who,self.Whg],
                                axis=1)
        self.Wy = theano.shared(uniform_weight(n_hidden,n_out),borrow=True)
        
        self.bi = theano.shared(const_bias(n_hidden,-1),borrow=True)
        self.bf = theano.shared(const_bias(n_hidden,1),borrow=True)
        self.bo = theano.shared(const_bias(n_hidden,1),borrow=True)
        self.bg = theano.shared(const_bias(n_hidden,0),borrow=True)
        self.b = T.concatenate([self.bi,self.bf,self.bo,self.bg],axis=0)
        self.by = theano.shared(const_bias(n_out,0),borrow=True)
        
        self.params = [self.Wx,self.Wh,self.Wy,self.b,self.by]
        self.W = [self.Wx,self.Wh,self.Wy]
        self.L1 = numpy.sum([abs(w).sum() for w in self.W])
        self.L2 = numpy.sum([(w**2).sum() for w in self.W])
        
        if self.norm:
            # because of bias right after, no need for shifts 1 and 2
            self.scaleh = theano.shared(const_bias(4*n_hidden,0.5),borrow=True)
            self.scalex = theano.shared(const_bias(4*n_hidden,0.5),borrow=True)
            self.scalec = theano.shared(const_bias(n_hidden,1),borrow=True)
            self.shiftc = theano.shared(const_bias(n_hidden,0),borrow=True)
            self.params = self.params+[self.scaleh,self.scalex,
                                       self.scalec,self.shiftc]
            
            # if using layer norm, initial hidden state h0 needs var>0
            # using a stable value rather than random helps w/ training
            h0 = numpy.zeros((n_hidden,),dtype=theano.config.floatX)
            h0[0] = 1
            self.h0 = theano.shared(h0,borrow=True)
        else:
            self.h0 = T.zeros((self.n_hidden,),dtype=theano.config.floatX)
        
        # initialize DNI
        self.dni = dni(n_hidden,2*n_hidden,2)
    
    # slice for doing step calculations in parallel
    def _slice(self,x,n):
        return x[:,n*self.n_hidden:(n+1)*self.n_hidden]
        
    def train(self):
        x = T.tensor3('x')
        y = T.tensor3('y')
        learning_rate = T.scalar('learning_rate')
        dni_scale = T.scalar('dni_scale')
        
        # reshape the inputs
        # batch x time x n -> time//steps x steps x batch x n
        def shufflereshape(r):
            r = r.dimshuffle([1,0,2])
            r = r.reshape((r.shape[0]//self.steps,
                           self.steps,
                           r.shape[1],
                           r.shape[2]))
            return r
        
        # step takes a set of inputs over self.steps time-steps
        def step(x_t,y_t,c_tmT,h_tmT,
                 Wx,Wh,Wy,b,by,
                 lr,scale):
            
            # manually build the graph for the inner loop
            yo_t = []
            c_tm1 = c_tmT
            h_tm1 = h_tmT
            loss = 0
            for t in range(self.steps):
                if self.norm:
                    preact = layer_norm(T.dot(x_t[t],Wx),self.scalex) \
                             +layer_norm(T.dot(h_tm1,Wh),self.scaleh)+b
                else:
                    preact = T.dot(x_t[t],Wx)+T.dot(h_tm1,Wh)+b
                i = sigmoid(self._slice(preact,0))
                f = sigmoid(self._slice(preact,1))
                o = sigmoid(self._slice(preact,2))
                g = tanh(self._slice(preact,3))
                c_t = c_tm1*f+g*i
                if self.norm:
                    c_t = layer_norm(c_t,self.scalec,self.shiftc)
                h_t = o*tanh(c_t)
                output = softmax(T.dot(h_t,Wy)+by)
                yo_t.append(output)
                # update for next step
                c_tm1 = c_t
                h_tm1 = h_t
                loss += T.mean(categorical_crossentropy(output,y_t[t]))
            
            loss = loss/self.steps # to take mean
            up_dldp_l2 = 0
            up_dni_l2 = 0
            # Train the LSTM: backprop (loss + DNI output)
            dni_out = self.dni.output(h_t)
            dni_out_h = self._slice(dni_out,0)
            dni_out_c = self._slice(dni_out,1)
            grads = []
            params = [] # params to Adam need to be in same order as grads 
            # output weights don't have dni feedback
            for param in [self.Wy,self.by]:
                dlossdparam = T.grad(loss,param)
                grads.append(dlossdparam)
                params.append(param)
            # output-gate weights get dni_h
            dni_h_params = [self.Wxo,self.Who,self.bo]
            for param in dni_h_params:
                dlossdparam = T.grad(loss,param)
                dniJ = T.Lop(h_t,param,dni_out_h)
                grads.append(dlossdparam+scale*dniJ)
                params.append(param)
                up_dldp_l2 += T.sum(T.square(dlossdparam))
                up_dni_l2 += T.sum(T.square(dniJ))
            # all other gates get dni_c
            dni_c_params = [self.Wxi,self.Wxf,self.Wxg,
                            self.Whi,self.Whf,self.Whg,
                            self.bi,self.bf,self.bg]
            if self.norm:
                dni_c_params = dni_c_params + [self.scalex,self.scaleh,
                                               self.scalec,self.shiftc]
            for param in dni_c_params:
                dlossparam = T.grad(loss,param)
                dniJ = T.Lop(c_t,param,dni_out_c)
                grads.append(dlossparam+scale*dniJ)
                params.append(param)
                up_dldp_l2 += T.sum(T.square(dlossdparam))
                up_dni_l2 += T.sum(T.square(dniJ))
            
            # Update the DNI (from the last step)
            # recalculate the DNI prediction since it can't be passed
            dni_out_old = self.dni.output(h_tmT)
            dni_old_h = self._slice(dni_out_old,0)
            dni_old_c = self._slice(dni_out_old,1)
            # dni target: current loss backprop'ed + new dni backprop'ed
            dni_target_h = T.grad(loss,h_tmT) \
                           +T.Lop(h_t,h_tmT,dni_out_h)
            dni_target_c = T.grad(loss,c_tmT) \
                           +T.Lop(c_t,c_tmT,dni_out_c)
            dni_error = T.sum(T.square(dni_old_h-dni_target_h)) \
                        +T.sum(T.square(dni_old_c-dni_target_c))
            for param in self.dni.params:
                grads.append(T.grad(dni_error,param))
            
            updates = adam(lr,params+self.dni.params,grads)
            
            return [c_t,h_t,loss,dni_error,
                    T.sqrt(up_dldp_l2),T.sqrt(up_dni_l2)],updates
        
        c0 = T.zeros((self.n_hidden,),dtype=theano.config.floatX)
        [c,h,
         seq_loss,seq_dni_error,
         up_dldp,up_dni],updates = theano.scan(fn=step,
                                sequences=[shufflereshape(x),
                                           shufflereshape(y)],
                                outputs_info=[T.alloc(c0,x.shape[0],self.n_hidden),
                                              T.alloc(self.h0,x.shape[0],self.n_hidden),
                                              None,None,None,None],
                                non_sequences=[self.Wx,self.Wh,self.Wy,
                                               self.b,self.by,
                                               learning_rate,dni_scale])
        return theano.function(inputs=[x,y,learning_rate,dni_scale],
                               outputs=[T.mean(seq_loss),
                                        T.mean(seq_dni_error),
                                        T.mean(up_dldp),
                                        T.mean(up_dni)],
                                updates=updates)
    
    def test(self):
        x = T.tensor3('x')
        y = T.tensor3('y')
        
        def step(x_t,y_t,c_tm1,h_tm1,
                 Wx,Wh,Wy,b,by):
            if self.norm:
                preact = layer_norm(T.dot(x_t,Wx),self.scalex) \
                         +layer_norm(T.dot(h_tm1,Wh),self.scaleh)+b
            else:
                preact = T.dot(x_t,Wx)+T.dot(h_tm1,Wh)+b
            i = sigmoid(self._slice(preact,0))
            f = sigmoid(self._slice(preact,1))
            o = sigmoid(self._slice(preact,2))
            g = tanh(self._slice(preact,3))
            c_t = c_tm1*f+g*i
            if self.norm:
                    c_t = layer_norm(c_t,self.scalec,self.shiftc)
            h_t = o*tanh(c_t)
            yo_t = softmax(T.dot(h_t,Wy)+by)
            loss_t = T.mean(categorical_crossentropy(yo_t,y_t))
            return c_t,h_t,yo_t,loss_t
        c0 = T.alloc(T.zeros((self.n_hidden,),dtype=theano.config.floatX),
                     x.shape[0],self.n_hidden)
        h0 = T.alloc(self.h0,x.shape[0],self.n_hidden)
        [c,s,yo,loss],_ = theano.scan(fn=step,
                              sequences=[x.dimshuffle([1,0,2]),
                                         y.dimshuffle([1,0,2])],
                              outputs_info=[c0,h0,None,None],
                              non_sequences=[self.Wx,self.Wh,self.Wy,
                                             self.b,self.by])
        loss = T.mean(loss)
        return theano.function(inputs=[x,y],
                               outputs=loss)


        























