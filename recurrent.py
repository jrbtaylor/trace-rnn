# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 17:31:59 2016

@author: jrbtaylor
"""

from __future__ import print_function

import numpy
from collections import OrderedDict

import theano
from theano import tensor as T
from theano.tensor import tanh
from theano.tensor.nnet import sigmoid, relu
from theano.tensor.nnet.nnet import softmax, categorical_crossentropy

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

class lstm(object):
    def __init__(self,x,n_in,n_hidden,n_out,bptt_limit):
        self.x = x
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.bptt_limit = bptt_limit
        # initialize weights
        self.Wxi = uniform_weight(n_in,n_hidden)
        self.Wsi = ortho_weight(n_hidden,rng)
        self.Wxf = uniform_weight(n_in,n_hidden)
        self.Wsf = ortho_weight(n_hidden,rng)
        self.Wxo = uniform_weight(n_in,n_hidden)
        self.Wso = ortho_weight(n_hidden,rng)
        self.Wxg = uniform_weight(n_in,n_hidden)
        self.Wsg = ortho_weight(n_hidden,rng)
        self.Wsy = uniform_weight(n_hidden,n_out)
        self.bi = const_bias(n_hidden,0)
        self.bf = const_bias(n_hidden,0)
        self.bo = const_bias(n_hidden,0)
        self.bg = const_bias(n_hidden,0)
        self.by = const_bias(n_out,0)
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
            
            # Initialize momentum
            self.gparams_mom = []
            for param in self.params:
                gparam_mom = theano.shared(
                                 numpy.zeros(param.get_value(borrow=True).shape,
                                 dtype=theano.config.floatX))
                self.gparams_mom.append(gparam_mom)
        
        def output(self,x):
            next_input = x
            for n in range(self.n_layers):
                next_input = relu(T.dot(next_input,self.W[n])+self.b[n])
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
        
        # initialize momentum
        self.gparams_mom = []
        for param in self.params:
            gparam_mom = theano.shared(
                             numpy.zeros(param.get_value(borrow=True).shape,
                             dtype=theano.config.floatX))
            self.gparams_mom.append(gparam_mom)
        
        # initialize DNI
        self.dni = dni(n_hidden,n_hidden,2)
        
    def train(self):
        x = T.tensor3('x')
        y = T.tensor3('y')
        learning_rate = T.scalar('learning_rate')
        momentum = T.scalar('momentum')
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
        def step(x_t,y_t,h_tmT,Wx,Wh,bh,Wy,by,lr,momentum,scale):
            
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
            updates = OrderedDict()
            up_dldp_l2 = 0
            up_dni_l2 = 0
            # Train the RNN: backprop (loss + DNI output)
            dni_out = self.dni.output(h_t)
            for [param,gparam_mom] in zip(self.params,self.gparams_mom):
                dlossdparam = T.grad(loss,param)
                dniJ = T.Lop(h_t,param,dni_out,disconnected_inputs='ignore')
                up_dldp_l2 += T.sum(T.square(dlossdparam))
                up_dni_l2 += T.sum(T.square(dniJ))
                updates[gparam_mom] = momentum*gparam_mom \
                                      -(1.-momentum)*lr*(dlossdparam+scale*dniJ)
                updates[param] = param+updates[gparam_mom]
                
            # Update the DNI (from the last step)
            # re-calculate the DNI prediction from the last step
            # note: can't be passed through scan or T.grad won't work
            dni_out_old = self.dni.output(h_tmT)
            # dni_target: current loss backprop'ed + new dni backprop'ed
            dni_target = T.grad(loss,h_tmT) \
                         +T.Lop(h_t,h_tmT,dni_out)
            dni_error = T.sum(T.square(dni_out_old-dni_target))
            for [param,gparam_mom] in zip(self.dni.params,self.dni.gparams_mom):
                gparam = T.grad(dni_error,param)
                updates[gparam_mom] = momentum*gparam_mom \
                                      -(1.-momentum)*lr*gparam
                updates[param] = param+updates[gparam_mom]
            
            return [h_t,loss,dni_error,T.sqrt(up_dldp_l2),T.sqrt(up_dni_l2)],updates
            
        h0 = T.zeros((self.n_hidden,),dtype=theano.config.floatX)
        [h,seq_loss,seq_dni_error,up_dldp,up_dni],updates = theano.scan(fn=step, 
                             sequences=[shufflereshape(x),
                                        shufflereshape(y)],
                             outputs_info=[T.alloc(h0,x.shape[0],self.n_hidden),
                                           None,None,None,None],
                             non_sequences=[self.Wx,self.Wh,self.bh,
                                            self.Wy,self.by,
                                            learning_rate,momentum,dni_scale])
        return theano.function(inputs=[x,y,learning_rate,momentum,dni_scale],
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


class lstm_dni(object):
    def __init__(self,n_in,n_hidden,n_out,steps,rng=rng):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.steps = steps
        
        # initialize weights
        self.Wxi = uniform_weight(n_in,n_hidden)
        self.Wsi = ortho_weight(n_hidden,rng)
        self.Wxf = uniform_weight(n_in,n_hidden)
        self.Wsf = ortho_weight(n_hidden,rng)
        self.Wxo = uniform_weight(n_in,n_hidden)
        self.Wso = ortho_weight(n_hidden,rng)
        self.Wxg = uniform_weight(n_in,n_hidden)
        self.Wsg = ortho_weight(n_hidden,rng)
        self.Wsy = uniform_weight(n_hidden,n_out)
        self.bi = const_bias(n_hidden,0)
        self.bf = const_bias(n_hidden,0)
        self.bo = const_bias(n_hidden,0)
        self.bg = const_bias(n_hidden,0)
        self.by = const_bias(n_out,0)
        
        self.params = [self.Wxi,self.Wsi,self.Wxf,self.Wsf,self.Wxo,self.Wso,self.Wxg,self.Wsg,self.Wsy,self.bi,self.bf,self.bo,self.bg,self.by]
        self.W = [self.Wxi,self.Wsi,self.Wxf,self.Wsf,self.Wxo,self.Wso,self.Wxg,self.Wsg,self.Wsy]
        self.L1 = numpy.sum([abs(w).sum() for w in self.W])
        self.L2 = numpy.sum([(w**2).sum() for w in self.W])
        
        # initialize momentum
        self.gparams_mom = []
        for param in self.params:
            gparam_mom = theano.shared(
                             numpy.zeros(param.get_value(borrow=True).shape,
                             dtype=theano.config.floatX))
            self.gparams_mom.append(gparam_mom)
        
        # initialize DNI
        self.dni = dni(n_hidden,n_hidden,2)
        
    def train(self):
        x = T.tensor3('x')
        y = T.tensor3('y')
        learning_rate = T.scalar('learning_rate')
        momentum = T.scalar('momentum')
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
        def step(x_t,y_t,c_tmT,s_tmT,
                 Wxi,Wsi,Wxf,Wsf,Wxo,Wso,Wxg,Wsg,Wsy,bi,bf,bo,bg,by,
                 lr,momentum,scale):
            
            # manually build the graph for the inner loop
            yo_t = []
            c_tm1 = c_tmT
            s_tm1 = s_tmT
            loss = 0
            for t in range(self.steps):
                i = sigmoid(T.dot(x_t[t],Wxi)+T.dot(s_tm1,Wsi)+bi)
                f = sigmoid(T.dot(x_t[t],Wxf)+T.dot(s_tm1,Wsf)+bf)
                o = sigmoid(T.dot(x_t[t],Wxo)+T.dot(s_tm1,Wso)+bo)
                g = tanh(T.dot(x_t[t],Wxg)+T.dot(s_tm1,Wsg)+bg)
                c_t = c_tm1*f+g*i
                s_t = tanh(c_t)*o
                output = softmax(T.dot(s_t,Wsy)+by)
                yo_t.append(output)
                # update for next step
                c_tm1 = c_t
                s_tm1 = s_t
                loss += T.mean(categorical_crossentropy(output,y_t[t]))
            
            loss = loss/self.steps # to take mean
            updates = OrderedDict()
            up_dldp_l2 = 0
            up_dni_l2 = 0
            # Train the LSTM: backprop (loss + DNI output)
            dni_out = self.dni.output(s_t)
            for [param,gparam_mom] in zip(self.params,self.gparams_mom):
                dlossdparam = T.grad(loss,param)
                dniJ = T.Lop(s_t,param,dni_out,disconnected_inputs='ignore')
                up_dldp_l2 += T.sum(T.square(dlossdparam))
                up_dni_l2 += T.sum(T.square(dniJ))
                updates[gparam_mom] = momentum*gparam_mom \
                                      -(1.-momentum)*lr*(dlossdparam+scale*dniJ)
                updates[param] = param+updates[gparam_mom]
            
            # Update the DNI (from the last step)
            # recalculate the DNI prediction since it can't be passed
            dni_out_old =self.dni.output(s_tmT)
            # dni target: current loss backprop'ed + new dni backprop'ed
            dni_target = T.grad(loss,s_tmT) \
                         +T.Lop(s_t,s_tmT,dni_out)
            dni_error = T.sum(T.square(dni_out_old-dni_target))
            for [param,gparam_mom] in zip(self.dni.params,self.dni.gparams_mom):
                gparam = T.grad(dni_error,param)
                updates[gparam_mom] = momentum*gparam_mom \
                                      -(1.-momentum)*lr*gparam
                updates[param] = param+updates[gparam_mom]
            
            return [c_t,s_t,loss,dni_error,
                    T.sqrt(up_dldp_l2),T.sqrt(up_dni_l2)],updates
        
        c0 = T.zeros((self.n_hidden,),dtype=theano.config.floatX)
        s0 = T.zeros((self.n_hidden,),dtype=theano.config.floatX)
        [c,s,
         seq_loss,seq_dni_error,
         up_dldp,up_dni],updates = theano.scan(fn=step,
                                sequences=[shufflereshape(x),
                                           shufflereshape(y)],
                                outputs_info=[T.alloc(c0,x.shape[0],self.n_hidden),
                                              T.alloc(s0,x.shape[0],self.n_hidden),
                                              None,None,None,None],
                                non_sequences=[self.Wxi,self.Wsi,self.Wxf,self.Wsf,
                                               self.Wxo,self.Wso,self.Wxg,self.Wsg,
                                               self.Wsy,self.bi,self.bf,self.bo,
                                               self.bg,self.by,
                                               learning_rate,momentum,dni_scale])
        return theano.function(inputs=[x,y,learning_rate,momentum,dni_scale],
                               outputs=[T.mean(seq_loss),
                                        T.mean(seq_dni_error),
                                        T.mean(up_dldp),
                                        T.mean(up_dni)],
                                updates=updates)
    
    def test(self):
        x = T.tensor3('x')
        y = T.tensor3('y')
        
        def step(x_t,y_t,c_tm1,s_tm1,Wxi,Wsi,Wxf,Wsf,Wxo,Wso,Wxg,Wsg,Wsy,bi,bf,bo,bg,by):
            i = sigmoid(T.dot(x_t,Wxi)+T.dot(s_tm1,Wsi)+bi)
            f = sigmoid(T.dot(x_t,Wxf)+T.dot(s_tm1,Wsf)+bf)
            o = sigmoid(T.dot(x_t,Wxo)+T.dot(s_tm1,Wso)+bo)
            g = tanh(T.dot(x_t,Wxg)+T.dot(s_tm1,Wsg)+bg)
            c = c_tm1*f+g*i
            s = tanh(c)*o
            yo_t = softmax(T.dot(s,Wsy)+by)
            loss_t = T.mean(categorical_crossentropy(yo_t,y_t))
            return c,s,yo_t,loss_t
        c0 = T.alloc(T.zeros((self.n_hidden,),dtype=theano.config.floatX),
                     x.shape[0],self.n_hidden)
        s0 = T.alloc(T.zeros((self.n_hidden,),dtype=theano.config.floatX),
                     x.shape[0],self.n_hidden)
        [c,s,yo,loss],_ = theano.scan(fn=step,
                              sequences=[x.dimshuffle([1,0,2]),
                                         y.dimshuffle([1,0,2])],
                              outputs_info=[c0,s0,None,None],
                              non_sequences=[self.Wxi,self.Wsi,self.Wxf,self.Wsf,
                                             self.Wxo,self.Wso,self.Wxg,self.Wsg,
                                             self.Wsy,self.bi,self.bf,self.bo,
                                             self.bg,self.by],
                              strict=True)
        loss = T.mean(loss)
        return theano.function(inputs=[x,y],
                               outputs=loss)
        
        

        



