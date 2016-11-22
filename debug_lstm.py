# -*- coding: utf-8 -*-
"""
Debug the lstm implementation to figure out if the bug
is in the tensor splicing or somewhere else

Created on Mon Nov 21 14:21:12 2016

@author: jrbtaylor
"""

from __future__ import print_function

import numpy
import theano
from theano import tensor as T
from theano.tensor import tanh
from theano.tensor.nnet import sigmoid
from theano.tensor.nnet.nnet import softmax, categorical_crossentropy

import timeit

from optim import adam

rng = numpy.random.RandomState(1)

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

# the original LSTM implementation
class lstm(object):
    def __init__(self,x,n_in,n_hidden,n_out):
        self.x = x
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
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
        self.output = y
        self.pred = T.argmax(self.output,axis=1)
    
    # ----- Classification -----
    def crossentropy(self,y):
        return T.mean(categorical_crossentropy(self.output,y))
    
    def errors(self,y):
        return T.mean(T.neq(self.pred,y))
    
    # ----- Regression -----
    def mse(self,y):
        return T.mean(T.sqr(T.sub(self.output,y)))


# the tensor-slicing LSTM implementation
class lstm_slice(object):
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
        def _slice(self,x,n):
            return x[:,n*self.n_hidden:(n+1)*self.n_hidden]        
        
        # forward function
        def forward(x_t,c_tm1,s_tm1,Wx,Ws,Wy,b,by):
            preact = T.dot(x_t,Wx)+T.dot(s_tm1,Ws)+b
            i = sigmoid(self._slice(preact,0))
            f = sigmoid(self._slice(preact,1))
            o = sigmoid(self._slice(preact,2))
            g = tanh(self._slice(preact,3))
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
        return T.mean(categorical_crossentropy(self.output,y))
    
    def errors(self,y):
        return T.mean(T.neq(self.pred,y))
    
    # ----- Regression -----
    def mse(self,y):
        return T.mean(T.sqr(T.sub(self.output,y)))


# -----------------------------------------------------------------------------
# Common copy task
# n_in is the number of words + 2 (one for pause, one for copy)
# the n_in-2 words are randomly chosen and 1-hot encoded sequence_length times
# then the blank character is input pause times
# then the copy character is input (once)
# then the output repeats the original sequence (minus the pause & copy words)
# the input during the copy is the blank character again
# -----------------------------------------------------------------------------
def data(n_in,n_train,n_val,sequence_length,pause):
    rng = numpy.random.RandomState(1)
    def generate_data(examples):
        x = numpy.zeros((examples,2*sequence_length+pause+1,n_in),dtype='float32')
        y = numpy.zeros((examples,2*sequence_length+pause+1,n_in-2),dtype='float32')
        for ex in range(examples):
            # original sequence
            oneloc = rng.randint(0,n_in-2,size=(sequence_length))
            x[ex,numpy.arange(sequence_length),oneloc] = 1
            # blank characters before copy
            x[ex,sequence_length+numpy.arange(pause),n_in-2] = 1
            # copy character
            x[ex,sequence_length+pause,n_in-1] = 1
            # blank characters during copy
            x[ex,sequence_length+pause+1+numpy.arange(sequence_length),n_in-2] = 1
            # output
            y[ex,sequence_length+pause+1+numpy.arange(sequence_length),oneloc] = 1
        return x,y
    x_train,y_train = generate_data(n_train)
    x_val,y_val = generate_data(n_val)
    print(x_train.shape)
    print(y_train.shape)
    print(x_val.shape)
    print(y_val.shape)
    return [x_train,y_train,x_val,y_val]


def experiment(train_fcn,x_train,y_train,lr,lr_decay,batch_size,
               test_fcn,x_val,y_val,n_epochs,patience):
    loss = []
    val_loss = []
    
    train_idx = range(x_train.shape[0])
    
    best_val = numpy.inf
    epoch = 0
    init_patience = patience
    while epoch<n_epochs and patience>0:
        start_time = timeit.default_timer()
        
        # train
        loss_epoch = 0
        numpy.random.shuffle(train_idx)
        n_train_batches = int(numpy.floor(x_train.shape[0]/batch_size))
        for batch in range(n_train_batches):
            batch_idx = train_idx[batch*batch_size:(batch+1)*batch_size]
            x_batch = x_train[batch_idx]
            y_batch = y_train[batch_idx]
            loss_batch = train_fcn(x_batch,y_batch,lr)
            loss_epoch += loss_batch
        loss_epoch = loss_epoch/n_train_batches
        end_time = timeit.default_timer()
        print('Epoch %d  -----  time per example (msec): %f' \
             % (epoch,1000*(end_time-start_time)/x_train.shape[0]))
        print('Training loss  =  %f' % loss_epoch)
        loss.append(loss_epoch)
        
        # validate
        val_loss_epoch = 0
        n_val_batches = int(numpy.floor(x_val.shape[0]/batch_size))
        for batch in range(n_val_batches):
            x_batch = x_val[batch*batch_size:(batch+1)*batch_size]
            y_batch = y_val[batch*batch_size:(batch+1)*batch_size]
            val_loss_epoch += test_fcn(x_batch,y_batch)
        val_loss_epoch = val_loss_epoch/n_val_batches
        print('Validation loss = %f' % val_loss_epoch)
        val_loss.append(val_loss_epoch)
        
        # early stopping
        if val_loss_epoch<best_val:
            best_val = val_loss_epoch
            patience = init_patience
        else:
            patience -= 1
        
        # or stop once it gets good enough
        # DNI paper stops <0.15 bits error
        if val_loss_epoch<0.15*numpy.log(2):
            patience = 0
        
        # set up next epoch
        epoch += 1
        lr = lr*lr_decay
    
    return loss, val_loss

def log_results(filename,line,sequence_length,n_in,n_hidden,n_out,
                loss,val_loss,overwrite=False):
        import csv
        import os
        if not filename[-4:]=='.csv':
            filename = filename+'.csv'
        if line==0 and overwrite:
            # check if old log exists and delete
            if os.path.isfile(filename):
                os.remove(filename)
        file = open(filename,'a')
        writer = csv.writer(file)
        if line==0:
            writer.writerow(('sequence_length','n_in','n_hidden','n_out',
                             'Training_loss','Validation_loss'))
        writer.writerow((sequence_length,n_in,n_hidden,n_out,loss,val_loss))

def test_old(x_train,y_train,x_val,y_val,n_in,n_hidden,n_out,
             lr,lr_decay,batch_size,n_epochs,patience):
    x = T.tensor3('x')
    y = T.tensor3('y')
    learning_rate = T.scalar('learning_rate')
    model = lstm(x,n_in,n_hidden,n_out)
    train = theano.function(inputs=[x,y,learning_rate],
                            outputs=[model.crossentropy(y)],
                            updates=adam(learning_rate,model.params,
                                         T.grad(model.crossentropy(y),
                                                model.params)))
    test = theano.function(inputs=[x,y],
                           outputs=[model.crossentropy(y)])
    return experiment(train,x_train,y_train,lr,lr_decay,batch_size,
                      test,x_val,y_val,n_epochs,patience)

def test_new(x_train,y_train,x_val,y_val,n_in,n_hidden,n_out,
             lr,lr_decay,batch_size,n_epochs,patience):
    x = T.tensor3('x')
    y = T.tensor3('y')
    learning_rate = T.scalar('learning_rate')
    model = lstm_slice(x,n_in,n_hidden,n_out)
    train = theano.function(inputs=[x,y,learning_rate],
                            outputs=[model.crossentropy(y)],
                            updates=adam(learning_rate,model.params,
                                         T.grad(model.crossentropy(y),
                                                model.params)))
    test = theano.function(inputs=[x,y],
                           outputs=[model.crossentropy(y)])
    return experiment(train,x_train,y_train,lr,lr_decay,batch_size,
                      test,x_val,y_val,n_epochs,patience)


if __name__ == "__main__":
    import graph
    import argparse
    parser = argparse.ArgumentParser(description='Run LSTM experiments')
    parser.add_argument('--sequence_length',nargs='*',type=int,
                        default=[5])
    parser.add_argument('--learnrate',nargs='*',type=float,
                        default=[5e-4])
    parser.add_argument('--model',nargs='*',type=str,
                        default=['old','new'])
    sequence_length = parser.parse_args().sequence_length[0]
    lr = parser.parse_args().learnrate[0]
    model = parser.parse_args().model
    
    n_in = 4 # n_in-2 words + pause + copy
    n_out = n_in-2
    batch_size = 512
    n_train = 20*batch_size
    n_val = batch_size
    pause = 1
    x_train,y_train,x_val,y_val = data(n_in,n_train,n_val,sequence_length,pause)
    
    lr_decay = 0.95
    n_epochs = 100
    patience = 20
    n_hidden = 32
    
    for m in model:
        if m=='old':
            loss, val_loss = test_old(x_train,y_train,x_val,y_val,
                                      n_in,n_hidden,n_out,
                                      lr,lr_decay,batch_size,n_epochs,patience)
            filename = 'debug_lstm_old'
        else: # 'new'
            loss, val_loss = test_new(x_train,y_train,x_val,y_val,
                                      n_in,n_hidden,n_out,
                                      lr,lr_decay,batch_size,n_epochs,patience)
            filename = 'debug_lstm_new'
        log_results(filename,0,sequence_length,n_in,n_hidden,n_out,loss,val_loss)
        graph.make_all(filename,2)






























