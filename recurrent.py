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


class lstm_dni(object):
    def __init__(self,n_in,n_hidden,n_out,steps,rng=rng):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.steps = steps
        
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
        self.Ws = theano.shared(numpy.concatenate(
                    [ortho_weight(n_hidden,rng) for i in range(4)],axis=1),
                     borrow=True)
        self.Wy = theano.shared(uniform_weight(n_hidden,n_out))
        self.b = theano.shared(numpy.concatenate(
                    [const_bias(n_hidden,0) for i in range(4)],axis=0),
                     borrow=True)
        self.by = theano.shared(const_bias(n_out,0))
        
        self.params = [self.Wx,self.Ws,self.Wy,self.b,self.by]
        self.W = [self.Wx,self.Ws,self.Wy]
        self.L1 = numpy.sum([abs(w).sum() for w in self.W])
        self.L2 = numpy.sum([(w**2).sum() for w in self.W])
        
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
        def step(x_t,y_t,c_tmT,s_tmT,
                 Wx,Ws,Wy,b,by,
                 lr,scale):
            
            # manually build the graph for the inner loop
            yo_t = []
            c_tm1 = c_tmT
            s_tm1 = s_tmT
            loss = 0
            for t in range(self.steps):
                preact = T.dot(x_t[t],Wx)+T.dot(s_tm1,Ws)+b
                i = sigmoid(self._slice(preact,0))
                f = sigmoid(self._slice(preact,1))
                o = sigmoid(self._slice(preact,2))
                g = tanh(self._slice(preact,3))
                c_t = c_tm1*f+g*i
                s_t = tanh(c_t)*o
                output = softmax(T.dot(s_t,Wy)+by)
                yo_t.append(output)
                # update for next step
                c_tm1 = c_t
                s_tm1 = s_t
                loss += T.mean(categorical_crossentropy(output,y_t[t]))
            
            loss = loss/self.steps # to take mean
            up_dldp_l2 = 0
            up_dni_l2 = 0
            # Train the LSTM: backprop (loss + DNI output)
            dni_out = self.dni.output(s_t)
            grads = []
            for param in self.params:
                dlossdparam = T.grad(loss,param)
                dniJ = T.Lop(s_t,param,dni_out,disconnected_inputs='ignore')
                up_dldp_l2 += T.sum(T.square(dlossdparam))
                up_dni_l2 += T.sum(T.square(dniJ))
                grads.append(dlossdparam+scale*dniJ)
            
            # Update the DNI (from the last step)
            # recalculate the DNI prediction since it can't be passed
            dni_out_old =self.dni.output(s_tmT)
            # dni target: current loss backprop'ed + new dni backprop'ed
            dni_target = T.grad(loss,s_tmT) \
                         +T.Lop(s_t,s_tmT,dni_out)
            dni_error = T.sum(T.square(dni_out_old-dni_target))
            for param in self.dni.params:
                grads.append(T.grad(dni_error,param))
            
            updates = adam(lr,self.params+self.dni.params,grads)
            
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
                                non_sequences=[self.Wx,self.Ws,self.Wy,
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
        
        def step(x_t,y_t,c_tm1,s_tm1,
                 Wx,Ws,Wy,b,by):
            preact = T.dot(x_t,Wx)+T.dot(s_tm1,Ws)+b
            i = sigmoid(self._slice(preact,0))
            f = sigmoid(self._slice(preact,1))
            o = sigmoid(self._slice(preact,2))
            g = tanh(self._slice(preact,3))
            c_t = c_tm1*f+g*i
            s_t = tanh(c_t)*o
            yo_t = softmax(T.dot(s_t,Wy)+by)
            loss_t = T.mean(categorical_crossentropy(yo_t,y_t))
            return c_t,s_t,yo_t,loss_t
        c0 = T.alloc(T.zeros((self.n_hidden,),dtype=theano.config.floatX),
                     x.shape[0],self.n_hidden)
        s0 = T.alloc(T.zeros((self.n_hidden,),dtype=theano.config.floatX),
                     x.shape[0],self.n_hidden)
        [c,s,yo,loss],_ = theano.scan(fn=step,
                              sequences=[x.dimshuffle([1,0,2]),
                                         y.dimshuffle([1,0,2])],
                              outputs_info=[c0,s0,None,None],
                              non_sequences=[self.Wx,self.Ws,self.Wy,
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
        
        # initialize activation trace
        def floatX(data):
            return numpy.asarray(data,dtype=theano.config.floatX)
        self.trace = theano.shared(self.Wh.get_value()*floatX(0.))
        self.non_trace_params = [self.Wx,self.Wy,self.bh,self.by]
        
        # initialize DNI
        self.dni = dni(n_hidden,n_hidden,2)
        
    def train(self):
        x = T.tensor3('x')
        y = T.tensor3('y')
        learning_rate = T.scalar('learning_rate')
        dni_scale = T.scalar('dni_scale')
        
        # reset the activation traces for each new sequence
        self.trace *= 0
        eps = 1e-8 # to avoid dividing by zero
        
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
            # recurrent weights incorporate traces
            for param in [self.Ws]:
                trace_scale = (self.trace+eps)/ \
                                (T.sqrt(T.mean(T.sqr(self.trace)))+eps)
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


class lstm_trace(object):
    def __init__(self,n_in,n_hidden,n_out,steps,rng=rng):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.steps = steps
        
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
        self.Ws = theano.shared(numpy.concatenate(
                    [ortho_weight(n_hidden,rng) for i in range(4)],axis=1),
                     borrow=True)
        self.Wy = theano.shared(uniform_weight(n_hidden,n_out))
        self.bx = theano.shared(numpy.concatenate(
                    [const_bias(n_hidden,0) for i in range(4)],axis=0),
                     borrow=True)
        self.by = theano.shared(const_bias(n_out,0))
        
        self.params = [self.Wx,self.Ws,self.Wy,self.bx,self.by]
        self.W = [self.Wx,self.Ws,self.Wy]
        self.b = [self.bx,self.by]
        
        # initialize activation trace
        def floatX(data):
            return numpy.asarray(data,dtype=theano.config.floatX)
        self.trace = theano.shared(self.Ws.get_value()*floatX(0.))
        self.non_trace_params = [self.Wx,self.Wy,self.bx,self.by]
        
        self.L1 = numpy.sum([abs(w).sum() for w in self.W])
        self.L2 = numpy.sum([(w**2).sum() for w in self.W])
        
        # initialize DNI
        self.dni = dni(n_hidden,n_hidden,2)
    
    # slice for doing step calculations in parallel
    def _slice(self,x,n):
        return x[:,n*self.n_hidden:(n+1)*self.n_hidden]
        
    def train(self):
        x = T.tensor3('x')
        y = T.tensor3('y')
        learning_rate = T.scalar('learning_rate')
        trace_decay = T.scalar('trace_decay')
        dni_scale = T.scalar('dni_scale')
        
        # reset the activation traces for each new sequence
        self.trace *= 0
        eps = 1e-8 # to avoid dividing by zero
        
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
                 Wx,Ws,Wy,bx,by,
                 lr,scale):
            
            # manually build the graph for the inner loop
            yo_t = []
            c_tm1 = c_tmT
            s_tm1 = s_tmT
            loss = 0
            for t in range(self.steps):
                preact_x = T.dot(x_t[t],Wx)
                preact_s = T.dot(s_tm1,Ws)
                preact = preact_x+preact_s+bx
                i = sigmoid(self._slice(preact,0))
                f = sigmoid(self._slice(preact,1))
                o = sigmoid(self._slice(preact,2))
                g = tanh(self._slice(preact,3))
                c_t = c_tm1*f+g*i
                s_t = tanh(c_t)*o
                preact_y = T.dot(s_t,Wy)
                output = softmax(preact_y+by)
                yo_t.append(output)
                # update for next step
                c_tm1 = c_t
                s_tm1 = s_t
                loss += T.mean(categorical_crossentropy(output,y_t[t]))
                
                # update traces
                self.trace = trace_decay*self.trace+T.abs(preact_s)
            
            loss = loss/self.steps # to take mean
            up_dldp_l2 = 0
            up_dni_l2 = 0
            # Train the LSTM: backprop (loss + DNI output)
            dni_out = self.dni.output(s_t)
            grads = []
            # recurrent weights incorporate traces
            for param in [self.Ws]:
                trace_scale = (self.trace+eps)/ \
                                (T.sqrt(T.mean(T.sqr(self.trace)))+eps)
                dlossdparam = T.grad(loss,param)
                dniJ = T.Lop(s_t,param,dni_out,disconnected_inputs='ignore')
                up_dldp_l2 += T.sum(T.square(dlossdparam))
                up_dni_l2 += T.sum(T.square(dniJ))
                grads.append(trace_scale*(dlossdparam+scale*dniJ))
            # non-trace parameters get regular DNI updates
            for param in self.non_trace_params:
                dlossdparam = T.grad(loss,param)
                dniJ = T.Lop(s_t,param,dni_out,disconnected_inputs='ignore')
                up_dldp_l2 += T.sum(T.square(dlossdparam))
                up_dni_l2 += T.sum(T.square(dniJ))
                grads.append(dlossdparam+scale*dniJ)
            
            # Update the DNI (from the last step)
            # recalculate the DNI prediction since it can't be passed
            dni_out_old =self.dni.output(s_tmT)
            # dni target: current loss backprop'ed + new dni backprop'ed
            dni_target = T.grad(loss,s_tmT) \
                         +T.Lop(s_t,s_tmT,dni_out)
            dni_error = T.sum(T.square(dni_out_old-dni_target))
            for param in self.dni.params:
                grads.append(T.grad(dni_error,param))
            
            updates = adam(lr,self.params+self.dni.params,grads)
            
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
                                non_sequences=[self.Wx,self.Ws,self.Wy,
                                               self.bx,self.by,
                                               learning_rate,dni_scale])
        return theano.function(inputs=[x,y,learning_rate,dni_scale,trace_decay],
                               outputs=[T.mean(seq_loss),
                                        T.mean(seq_dni_error),
                                        T.mean(up_dldp),
                                        T.mean(up_dni)],
                                updates=updates)
    
    def test(self):
        x = T.tensor3('x')
        y = T.tensor3('y')
        
        def step(x_t,y_t,c_tm1,s_tm1,
                 Wx,Ws,Wy,b,by):
            preact = T.dot(x_t,Wx)+T.dot(s_tm1,Ws)+b
            i = sigmoid(self._slice(preact,0))
            f = sigmoid(self._slice(preact,1))
            o = sigmoid(self._slice(preact,2))
            g = tanh(self._slice(preact,3))
            c_t = c_tm1*f+g*i
            s_t = tanh(c_t)*o
            yo_t = softmax(T.dot(s_t,Wy)+by)
            loss_t = T.mean(categorical_crossentropy(yo_t,y_t))
            return c_t,s_t,yo_t,loss_t
        c0 = T.alloc(T.zeros((self.n_hidden,),dtype=theano.config.floatX),
                     x.shape[0],self.n_hidden)
        s0 = T.alloc(T.zeros((self.n_hidden,),dtype=theano.config.floatX),
                     x.shape[0],self.n_hidden)
        [c,s,yo,loss],_ = theano.scan(fn=step,
                              sequences=[x.dimshuffle([1,0,2]),
                                         y.dimshuffle([1,0,2])],
                              outputs_info=[c0,s0,None,None],
                              non_sequences=[self.Wx,self.Ws,self.Wy,
                                             self.b,self.by])
        loss = T.mean(loss)
        return theano.function(inputs=[x,y],
                               outputs=loss)

        
def adam(lr,params,grads):
    """
    Adam optimization
    
    Parameters
    ----------
    lr: Theano SharedVariable
        Initial learning rate
    params: list of Theano SharedVariable
        Model parameters
    grads: list of Theano variables
        Gradients of cost w.r.t. parameters
    
    """
    # default values
    epsilon = 1e-8
    b1 = 0.9
    b2 = 0.999
    
    def floatX(data):
        return numpy.asarray(data,dtype=theano.config.floatX)
    
    updates = []
    t = theano.shared(floatX(1.))
    t_new = t+1
    updates.append((t,t_new))
    for p,g in zip(params,grads):
        m = theano.shared(p.get_value()*floatX(0.))
        m_new = b1*m+(1-b1)*g
        updates.append((m,m_new))
        
        v = theano.shared(p.get_value()*floatX(0.))
        v_new = b2*v+(1-b2)*g**2
        updates.append((v,v_new))
        
        mhat = m_new/(1-b1**t)
        vhat = v_new/(1-b2**t)
        updates.append((p,p-lr*mhat/(T.sqrt(vhat)+epsilon)))
        
    return updates
        























