'''
Author: DejaWU
Date created: 2/17/2017
Date last modified: 5/30/2017
'''

from math import log
from random import uniform

def logext(x):
    if x == 0:
        return Logp(-float("inf"))
    return Logp(log(x,10))

""" On utilise le logarithme pour améliorer la précision du type flottant """    
class Logp(object):
    def __init__(self,p):
        self.p = p
    
    def __add__(self,a):
        if self.p == -float("inf"):
            return a
        elif a.p == -float("inf"):
            return self
        if self.p <= a.p:
            return Logp(a.p + log(1+10**(self.p-a.p),10))
        return Logp(self.p + log(1+10**(a.p-self.p),10))
        
    def __mul__(self,a):
        return Logp(self.p + a.p)
    
    def __lt__(self,a):
        return self.p <= a.p
    
    def __truediv__(self,a):
        return Logp(self.p - a.p)
    
    def __repr__(self):
        return '%.6f' % (10**self.p)
    
def sumlog(l):
    c = Logp(-float("inf"))
    for r in l:
        c += r
    return c
    
class HMM(object):
    def __init__(self,N,M,a,b,init):
        self.nb_of_iterations = 0
        self.nb_of_states = N
        self.nb_of_obs_symbols = M
        self.a = a
        self.b = b
        self.init = init

    def evaluation_forward(self,o):
        N = self.nb_of_states
        T = len(o)
        init,a,b = self.init, self.a, self.b
        alpha = [[Logp(-float("inf")) for _ in range(N+1)] for _ in range(T+1)] 
        c = [Logp(1) for _ in range(T+1)]   #c est créé pour résoudre le "problème de scaling"
        
        for i in range(1,N+1):
            alpha[1][i] = init[i]*b[i][o[0]]
        c[1] = sumlog(alpha[1][i] for i in range(1,N+1))
        
        for i in range(1,N+1):
            alpha[1][i] /= c[1]
            
        for t in range(2,T+1):
            for i in range(1,N+1):
                alpha[t][i] = sumlog(alpha[t-1][j]*a[j][i] for j in range(1,N+1))*b[i][o[t-1]]
            c[t] = sumlog([alpha[t][i] for i in range(1,N+1)]) 
            for i in range(1,N+1):
                alpha[t][i] /= c[t]	
                
        return sum(item.p for item in c)
        
    def evaluation_backward(self,o):
        N = self.nb_of_states 
        T = len(o)
        init,a,b = self.init, self.a, self.b
        beta = [[Logp(-float("inf")) for _ in range(N+1)] for _ in range(T+1)]
        
        for i in range(1,N+1):
            beta[T][i] = Logp(0)
            
        for t in range(N-1,0,-1):
            for i in range(1,N+1):
                beta[t][i] = sumRat(a[i][j]*b[j][o[t]]*beta[t+1][j] for j in range(1,N+1))
        return sum(init[i]*b[i][o[0]]*beta[1][i] for i in range(1,N+1))
        
    def viterbi(self,o): #Algorithme de Viterbi
        N = self.nb_of_states
        T = len(o)
        init,a,b = self.init, self.a, self.b
        delta = [[-1 for _ in range(N+1)] for _ in range(T+1)]
        state = [[0 for _ in range(N+1)] for _ in range(T+1)]
        
        for i in range(1,N+1):
            delta[1][i] = init[i]*b[i][o[0]]
            state[1][i] = 0
            
        for t in range(2,T+1):
            for j in range(1,N+1):
                delta[t][j] = max(delta[t-1][i]*a[i][j] for i in range(1,N+1))*b[j][o[t-1]]
                state[t][j] = max(range(1,N+1),key = lambda x: delta[t-1][x]*a[x][j])
                
        q = [max(range(1,N+1),key = lambda x: delta[T][x])]
        for t in range(T-1,0,-1):
            q.insert(0,state[t+1][q[0]]) #Retour sur trace
        return q     
        
    def learning(self,o):
        self.nb_of_iterations += 1
        N = self.nb_of_states
        M = self.nb_of_obs_symbols
        T = len(o)
        init,a,b = self.init, self.a, self.b
        alpha = [[Logp(-float("inf")) for _ in range(N+1)] for _ in range(T+1)]
        beta = [[Logp(-float("inf")) for _ in range(N+1)] for _ in range(T+1)]
        c = [Logp(0) for _ in range(T+1)] 
        for i in range(1,N+1):
            alpha[1][i] = init[i]*b[i][o[0]]
        c[1] = sumlog(alpha[1][i] for i in range(1,N+1))
        
        for i in range(1,N+1):
            alpha[1][i] /= c[1]
            
        for t in range(2,T+1):
            for i in range(1,N+1):
                alpha[t][i] = sumlog(alpha[t-1][j]*a[j][i] for j in range(1,N+1))*b[i][o[t-1]]
            c[t] = sumlog([alpha[t][i] for i in range(1,N+1)]) 
            for i in range(1,N+1):
                alpha[t][i] /= c[t]
                
        for i in range(1,N+1):
            beta[T][i] = Logp(0) / c[T]
            
        for t in range(T-1,0,-1):
            for i in range(1,N+1):
                beta[t][i] = sumlog(a[i][j]*b[j][o[t]]*beta[t+1][j] for j in range(1,N+1)) / c[t]
        x = [[[Logp(-float("inf")) for _ in range(N+1)] for _ in range(N+1)] for _ in range(T+1)]
        
        for t in range(1,T):
            p0 = sumlog([alpha[t][i]*a[i][j]*b[j][o[t]]*beta[t+1][j] for j in range(1,N+1) for i in range(1,N+1)])
            for i in range(1,N+1):
                for j in range(1,N+1):
                    x[t][i][j] = alpha[t][i]*a[i][j]*b[j][o[t]]*beta[t+1][j] / p0
        gamma = [[Logp(0) for _ in range(N+1)] for _ in range(T+1)]
        
        for t in range(1,T):
            for i in range(1,N+1):
                gamma[t][i] = sumlog([x[t][i][j] for j in range(1,N+1)])
        s = sumlog([gamma[1][i] for i in range(1,N+1)])
        
        for i in range(1,N+1):
            self.init[i] = gamma[1][i] / s
            
        for i in range(1,N+1):
            for j in range(1,N+1):
                self.a[i][j] = sumlog([x[t][i][j] for t in range(1,T)]) / sumlog([gamma[t][i] for t in range(1,T)])
        
        for i in range(1,N+1):
            for j in range(1,M+1):
                self.b[i][j] = sumlog([gamma[t][i]*logext(o[t-1] == j) for t in range(1,T+1)]) / sumlog(gamma[t][i] for t in range(1,T+1)) 
                
def InitMatrix(N,M):
    mat = [[Logp(-float("inf")) for _ in range(M+1)] for _ in range(N+1)]
        for i in range(1,N+1):
        c = 1 
        for j in range(1,M+1):
            if j == M:
                mat[i][j] = Logp(log(c,10))
            else:
                t = uniform(min(c,0.9/M),min(c,1.1/M))
                while t >= c:
                    t = uniform(min(c,0.9/M),min(c,1.1/M))
                mat[i][j] = Logp(log(t,10))
                    c -= t
