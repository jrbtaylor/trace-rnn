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

def layer_norm(h,eps=1e-5):
    return (h-T.mean(h,axis=1,keepdims=True))/(eps+T.std(h,axis=1,keepdims=True))

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
        
        # slice for doing step calculations in parallel
        def _slice(x,n):
            return x[:,n*self.n_hidden:(n+1)*self.n_hidden]        
        
        # forward function
        def forward(x_t,c_tm1,s_tm1,Wx,Ws,Wy,b,by):
            preact = T.dot(x_t,Wx)+T.dot(s_tm1,Ws)+b
            i = sigmoid(_slice(preact,0))
            f = sigmoid(_slice(preact,1))
            o = sigmoid(_slice(preact,2))
            g = tanh(_slice(preact,3))
            c = c_tm1*f+g*i
            s = tanh(c)*o
            y = softmax(T.dot(s,Wy)+by)
            return [c,s,y]
        c0 = T.alloc(T.zeros((self.n_hidden,),dtype=theano.config.floatX),x.shape[0],self.n_hidden)
        s0 = T.alloc(T.zeros((self.n_hidden,),dtype=theano.config.floatX),x.shape[0],self.n_hidden)
        ([c,s,y],updates) = theano.scan(fn=forward,
                                      sequences=x.dimshuffle([1,0,2]),
                                      outputs_info=[dict(initial=c0,taps=[-1]),
                                                    dict(initial=s0,taps=[-1]),
                                                    None],
                                      non_sequences=[self.Wx,self.Ws,self.Wy,
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
    def __init__(self,n_in,n_hidden,n_out,steps,norm=True,rng=rng):
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
                if self.norm:
                    s_t = layer_norm(s_t)
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
            if self.norm:
                    s_t = layer_norm(s_t)
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


class lstm_trace(object):
    def __init__(self,n_in,n_hidden,n_out,steps,norm=True,rng=rng):
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
        
        self.trace_params = [self.Ws]
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
        
        # re-initialize activation trace
        def floatX(data):
            return numpy.asarray(data,dtype=theano.config.floatX)
        self.trace = theano.shared(self.Ws.get_value()*floatX(0.))
        
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
        def step(x_t,y_t,c_tmT,s_tmT,
                 Wx,Ws,Wy,bx,by,
                 lr,scale,trace,decay):
            
            # manually build the graph for the inner loop
            yo_t = []
            c_tm1 = c_tmT
            s_tm1 = s_tmT
            old_trace = trace
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
                if self.norm:
                    s_t = layer_norm(s_t)
                preact_y = T.dot(s_t,Wy)
                output = softmax(preact_y+by)
                yo_t.append(output)
                # update for next step
                c_tm1 = c_t
                s_tm1 = s_t
                loss += T.mean(categorical_crossentropy(output,y_t[t]))
                
                # update traces: RMS taken over the batch dimension
                trace = decay*trace \
                         +T.sqrt(T.mean(T.sqr(preact_s),axis=0,keepdims=True))
            
            loss = loss/self.steps # to take mean
            up_dldp_l2 = 0
            up_dni_l2 = 0
            # Train the LSTM: backprop (loss + DNI output)
            dni_out = self.dni.output(s_t)
            grads = []
            # recurrent weights incorporate traces
            for param in self.trace_params:
                trace_scale = (trace+eps)/ \
                                (T.sqrt(T.mean(T.sqr(trace)))+eps)
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
            
            # params need to be passed to adam in same order as their grads
            updates = adam(lr,self.trace_params+self.non_trace_params+self.dni.params,grads)
            updates.append((old_trace,trace))
            
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
                                               learning_rate,dni_scale,
                                               self.trace,trace_decay])
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
                 Wx,Ws,Wy,bx,by):
            preact = T.dot(x_t,Wx)+T.dot(s_tm1,Ws)+bx
            i = sigmoid(self._slice(preact,0))
            f = sigmoid(self._slice(preact,1))
            o = sigmoid(self._slice(preact,2))
            g = tanh(self._slice(preact,3))
            c_t = c_tm1*f+g*i
            s_t = tanh(c_t)*o
            if self.norm:
                    s_t = layer_norm(s_t)
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
                                             self.bx,self.by])
        loss = T.mean(loss)
        return theano.function(inputs=[x,y],
                               outputs=loss)


        























