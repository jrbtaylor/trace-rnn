# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 09:45:41 2016

Figure out rolling memory in scan function

@author: jrbtaylor
"""

import numpy
import theano
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

rng = RandomStreams(seed=1)

memory_size = 5
memory = theano.shared(numpy.zeros((memory_size,),dtype=theano.config.floatX))
x = T.matrix('x')
n_steps = T.lscalar('n_steps')
def step(x_t,memory):
    new_memory = T.concatenate([x_t,memory[:-1]],axis=0)
    updates = (memory,new_memory)#T.set_subtensor(memory[-1],x_t))
    return [memory],[updates]
outputs,updates = theano.scan(fn=step,
                              sequences=x,
                              non_sequences=[memory])
rollingmem = theano.function(inputs=[n_steps],
                             outputs=outputs,
                             updates=updates,
                             givens={x:rng.uniform(size=(n_steps,1),
                                                   high=1,
                                                   low=-1)})
print(rollingmem(10))







