"""
The main classes used by sage to encapsulate the functionality of equations of motion and states.
"""

from copy import deepcopy
from collections import Counter

from numpy import *
import sympy as sp

from propagators.utils import delta


class Equation(dict):
    '''Handles multiplication and addition of State instances.'''
        
    def __add__(self,other):
        if isinstance(other,State):
            return self + Equation({other:1})
        if isinstance(other,Equation):
            AND = self.keys() & other.keys()
            sOR = self.keys() - other.keys()
            oOR = other.keys() - self.keys()
            new = Equation({i:self[i]+other[i] for i in AND})
            new.update({i:self[i] for i in sOR if self[i]!=0})
            new.update({i:other[i] for i in oOR if other[i]!=0})
            return new
        if other == 0:
            return self
        else:
            return NotImplemented
    
    def __radd__(self,other):
        return self.__add__(other)
    
    def __sub__(self,other):
        if isinstance(other,State):
            return self + Equation({other:-1})
        if isinstance(other,Equation):
            AND = self.keys() & other.keys()
            sOR = self.keys() - other.keys()
            oOR = other.keys() - self.keys()
            new = Equation({i:self[i]-other[i] for i in AND})
            new.update({i:self[i] for i in sOR if self[i]!=0})
            new.update({i:-other[i] for i in oOR if other[i]!=0})
            return new
        if other == 0:
            return self
        else:
            return NotImplemented
    
    def __rsub__(self,other):
        if isinstance(other,State):
            return Equation({other:1}) + (-1)*self
        else:
            return NotImplemented
    
    def __mul__(self,other):
        if isinstance(other,State):
            return self[other]
        elif isinstance(other,Equation):
            AND = self.keys() & other.keys()
            return sum([self[i]*other[i] for i in AND])
        else:
            return Equation({i: self[i]*other for i in self.keys()})
    
    def __rmul__(self,other):
        return self.__mul__(other)
    
    def __missing__(self,key):
        return 0

    def __str__(self):
        return str({str(key)+":"+str(val) for key,val in self.items()})

    def __hash__(self):
        return hash(self.__str__())

    def __sympy__(self):
        return self


class State:
    '''Create a new state in the variational Hilbert space.'''
    
    def __init__(self, dim, edge=inf, period=False):
        '''size: number of free sites across from the center in all directions;
         should be set higher than the expansion order (maximal number of bosons)
        qvect: 1D array of coordinates in Fourier space'''
        self.qvect = array([0]*dim,dtype=object)
        self.config = {(0,)*dim: Counter("e")}
        self.edge = edge
        self.period = period
    
    def __eq__(self,other):
        if not isinstance(other,State):
            return False
        if (self.config==other.config and all(self.qvect==other.qvect)):
            return True
        else:
            return False
    
    def __lt__(self,other):
        sp = self.particles()
        op = other.particles()
        ss = sum(tuple(sp.values()))
        os = sum(tuple(op.values()))
        if self.dim() != other.dim():
            return self.dim() < other.dim()
        if ss != os:
            return ss < os
        if sp != op:
            return sorted(sp.items()) < sorted(op.items())
        if self.config != other.config:
            return self.format() < other.format()
        if tuple(self.qvect) != tuple(other.qvect):
            return tuple(self.qvect) < tuple(other.qvect)
    
    def __ge__(self,other):
        return not self<other
    
    def __add__(self,other):
        if self==other:
            return Equation({self:2})
        elif isinstance(other,State):
            return Equation({self:1,other:1})
        elif other == 0:
            return self
        else:
            return NotImplemented
    
    def __radd__(self,other):
        return self.__add__(other)
    
    def __sub__(self,other):
        return self + (-1)*other
    
    def __rsub__(self,other):
        return (-1)*self + other
    
    def __mul__(self,other):
        if self==other:
            return 1
        elif isinstance(other,State):
            return 0
        elif other==0:
            return 0
        elif not isinstance(other,Equation):
            return Equation({self:other})
        else:
            return NotImplemented
    
    def __rmul__(self,other):
        return self.__mul__(other)
    
    def __div__(self,other):
        return Equation({self:1/other})
    
    def __hash__(self):
        return hash((tuple(self.qvect),tuple(self.format())))
    
    def __repr__(self):
        return str(self.qvect)+" x "+str(self.format(False))
    
    def __str__(self):
        return str((tuple(self.qvect),self.format()))
    
    def __sympy__(self):
        return self

    def format(self,hashable=True):
        if hashable==True:
            form = tuple([(x,tuple(sorted(y.elements())))
                            for (x,y) in sorted(self.config.items())])
        else:
            form = [(x,list(sorted(y.elements())))
                        for (x,y) in sorted(self.config.items())]
        return form
    
    def size(self):
        return max([sum(abs(array(key))) for key in self.config])
    
    def dim(self):
        return len(self.qvect)
    
    def locate(self,particle=None):
        if particle==None:
            particle=self.particles().keys()
        return array([key for (key,val) in self.config.items()
                        if any([val[p]>0 for p in particle])])
    
    def locset(self,particle=None):
        if particle==None:
            particle=self.particles().keys()
        return set([key for (key,val) in self.config.items()
                        if any([val[p]>0 for p in particle])])
    
    def nnlist(self,particle):
        sites = self.locate(particle)
        vecs = delta(self.dim())
        return array([tuple(x+y) for x in sites for y in vecs
                          if all(sum(abs(x+y-sites),axis=-1))])
    
    def nnset(self,particle,n=1):
        sites = self.locate(particle)
        vecs = delta(self.dim())
        return set([tuple(x+y) for x in sites for y in vecs
                        if all(sum(abs(x+y-sites),axis=-1))])
    
    def particles(self,par=None):
        ls = [x for x in self.config.values()]
        if len(ls)==0:
            c = Counter()
        else:
            c = sum(ls)
        if par==None:
            return c
        else:
            return c[par]
    
    def site(self,point):
        point = tuple(point)
        try:
            return self.config[point]
        except KeyError:
            return Counter()
    
    def copy(self):
        return deepcopy(self)
    
    def fold(self,point):
        if self.period:
            return tuple(mod(array(point)+self.edge,2*self.edge+1)-self.edge)
        else:
            return tuple(point)
    
    def excite(self,point,particle):
        point = tuple(point)
        if self.period==False and any(abs(array(point)) > self.edge):
            raise ValueError(
                "Attempted excitation at {point} beyond the state edge in\n {state}.".format(point=repr(point),state=repr(self.config)))
        else:
            point = self.fold(point)
            try:
                self.config[point].update([particle])
                return self
            except KeyError:
                self.config[point] = Counter([particle])
                return self
    
    def relax(self,point,particle):
        point = tuple(point)
        if self.period==False and any(abs(array(point)) > self.edge):
            raise ValueError("Attempted relaxation at {point} beyond the state edge in\n {state}.".format(point=repr(point),state=repr(self.config)))
        else:
            point = self.fold(point)
            try:
                if self.config[point][particle]==0:
                    return 0
                self.config[point].subtract([particle])
                if not self.config[point][particle]:
                    del(self.config[point][particle])
                if not self.config[point]:
                    del(self.config[point])
                return self
            except KeyError:
                return self
    
    def kadd(self,K):
        self.qvect = mod(self.qvect + K, 2*sp.pi)
        return self
    
    def reduce(self):
        index = self.locate("e")[0]
        if any(index!=0):
            self.config = {self.fold(array(key)-index):val for
                               (key,val) in self.config.items()}
            phase = sp.exp(sp.I*dot(self.qvect,-index.astype(object)))
            return self*phase*sp.S("exp(1j*dot(k,"+str(tuple(-index))+"))")
        else:
            return self
