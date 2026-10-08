import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# SGD with momentum 

# v_t = β * v_{t-1} + g_t          # velocity (exponential moving avg of gradients)
# θ_t = θ_{t-1} - lr * v_t

class SGDMomentum:

    def __init__(self,params,lr: int = 1e-4, momentum: int=0.9,weight_decay: int=0.0):
        self.params=params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay= weight_decay
        self.v = [None for _ in range(len(self.params))]

    @torch.no_grad()
    def step(self):
        for i,p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay !=0:
                g = g + p*self.weight_decay 
            if self.v[i] is None:
                self.v[i] = torch.zeros_like(p)
            self.v[i] = self.momentum*self.v[i] + g
            p=p - self.lr*self.v[i]

    def zero_grad(self):
        for p in self.params:
            if p.grad != None:
                p.grad= None


class ADAM:

    def __init__(self, params, lr, m1, m2, weight_decay, eps):
        self.params=params
        self.lr = lr
        self.m1 = m1
        self.m2 = m2
        self.weight_decay=weight_decay
        self.eps = eps
        self.m = [None for _ in range(len(self.params))]
        self.v = [None for _ in range(len(self.params))]
        self.t = 0


    @torch.no_grad()
    def step(self):

        self.t+=1

        for i,p in enumerate(self.params):

            if p.grad is None:
                continue
            g = p.grad 
            if self.weight_decay !=0 :
                g = g + self.weight_decay*p 

            if self.v[i] is None:
                            self.v[i] = torch.zeros_like(p)
            if self.m[i] is None:
                            self.m[i] = torch.zeros_like(p)

            self.m[i] = (1-self.m1)*self.m[i] + self.m1*g
            self.v[i] = (1-self.m2)*self.v[i] + self.m2*g*g

            m_hat = self.m[i]/(1 - self.m1**self.t)
            v_hat = self.v[i]/(1 - self.m2**self.t)



            p-= self.lr * m_hat / (torch.sqrt(v_hat)+ self.eps)

    def zero_grad(self):
        for p in self.params:
             if p.grad is not None:
                  p.grad = None

