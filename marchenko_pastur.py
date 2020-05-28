import numpy as np

class marchenko_pastur:
    def __init__(self, p, n):
        self._p = p; self._n = n
    def lbound(self):
        return (1 - np.sqrt(self._p/self._n))**2
                
    def ubound(self):
        return (1 + np.sqrt(self._p/self._n))**2
    
    def pdf(self, x):
        y = self._p/self._n
        lb = self.lbound()
        ub = self.ubound()
        inregion = ((lb < x) & (x < ub))
        dens = np.sqrt((ub-x)*(x-lb)) / (2*np.pi*x*y)
        dens[~inregion] = 0.0
        return dens
        
    def cdf(self, x):
        a = self.lbound()
        b = self.ubound()
        R = (b-x)*(x-a)
        num = np.sqrt(R) + (a+b)/2*np.arcsin((2*x-a-b)/(b-a)) - a*b*1/np.sqrt(a*b)*np.arcsin(((a+b)*x-2*a*b)/(x*(b-a)))
        return 0.5 + num/(2*np.pi*self._p/self._n)