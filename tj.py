"""
Implementation of the propagator and interaction operators for the t-J model, including 3-site terms and full quantum fluctuations. 
Includes some specialised functions used for comparison with SCBA, which is a less flexible method than this.
"""

import sympy as sp
from scipy.sparse.linalg import spsolve

#from propagators.utils import neighbor, pairs
from sage import *
from propagators import *


## Symbols

k = sp.Symbol("k")
om = sp.Symbol("omega")
t = sp.Symbol("t")
J = sp.Symbol("J")
A = sp.Symbol("alpha")


## Expansion functions for the t-J model

def propno3s(state):
    start = state.locate("e")[0]
    C = 1
    if len(state.particles()) == 1:
        m = 2*state.dim()
    else:
        v = state.nnlist("m")
        n = len(v)    #number of NN vectors
        d = len(v[0]) #space dimension
        c = count_nonzero(sum(abs(v-start),axis=-1)) #corrected for electron prox.
        k = state.particles("m") #LSW approximation
        m = 4*d*k*(1-A)+A*2*c +2*d #multiplier for J, no prox. effect for LSW
        if start in v: #prox. effect for LSW
            m -= 2*(1-A)
        if min(sum(abs(v),axis=-1))>1:
            C = 0 #ensure that electron not far from cloud
    res = C/(om-m*J)*state
    return res


def prop3s(state):
    start = state.locate("e")[0]
    if len(state.particles()) == 1:
        green = sp.Function("G3sk")
        m = 2*state.dim()
        res = green(k,om-m*J,J)*state
        return res
    else:
        green = sp.Function("Gm")
        bosons = state.locset("m")
        sites = neighbor(bosons,2)-bosons
        res = Equation()
        for pt in sites:
            s = ElHop(state,pt)
            res += s*green(str(tuple(bosons)),str(pt),str(tuple(start)),om,J)
        return res


def intacttJ(state):
    N = state.dim()
    res = Equation()
    point = state.locate("e")[0]
    par = state.locset()
    nn = neighbor(par,2)
##### Kinetic interactions
    for d in delta(N):
        si = state.site(point+d)
        if si["m"] == 0:
            res += -t*ElAddBoson(state,d,"m")
        if si["m"] == 1:
            res += -t*ElDelBoson(state,d,"m")
##### Spin fluctuations
    for p,q,d in pairs(nn):
        si = state.site(p)
        sj = state.site(q)
        if si["e"]==0 and sj["e"]==0:
            if si["m"] == 0 and sj["m"] == 0:
                res += 2*J*AddBosonPair(state,p,d,"m")
            if si["m"] == 1 and sj["m"] == 1:
                res += 2*J*DelBosonPair(state,p,d,"m")
##### 3-site interactions
    for e in delta(state.dim()):
        sj = state.site(d+e)
        if si["e"]==0 and sj["e"]==0:
            if si["m"] == 1 and sj["m"] == 1:
                res += J*ElSwapBoson(state,d+e,"m")
          # Spin-flip terms
            if si["m"] == 0 and sj["m"] == 0:
                res += J*ElAddBosonPair(state,d,e,"m")
            if si["m"] == 1 and sj["m"] == 1:
                res += J*ElDelBosonPair(state,d,e,"m")
    return res


### Remove geometrical constraints and terms above LSW to compare with SCBA

def propBAno3s(state):
    start = state.locate("e")[0]
    C = 1
    if len(state.particles()) == 1:
        m = 2*state.dim()
    else:
        n = state.particles("m") #number of magnons
        d = state.dim() #space dimension
        v = state.locate("m")
        m = 2*n*2*d+2*d #multiplier for J
        if start in state.nnlist("m"): #prox. effect
            m -= 2
        if min(sum(abs(v),axis=-1))>1:
            C = 0 #ensure electron near cloud
    res = C/(om-m*J)*state
    return res


def propBA3s(state):
    start = state.locate("e")[0]
    if len(state.particles()) == 1:
        green = sp.S("G3sk")
        m = 2*state.dim()
        res = green(k,om-m*J,J)*state
        return res
    else:
        green = sp.S("G3s")
        res = Equation()
        n = state.particles("m") #number of magnons
        d = state.dim() #space dimension
        m = 2*n*2*d+2*d #multiplier for J
        res += s*green(om-m*J,J,(0,)*d)
        return res


def intactBAtJ(state):
    """Still includes the C1 constraint to avoid pseudo-Trugman processes"""
    N = state.dim()
    res = Equation()
    point = state.locate("e")[0]
    par = state.locset()
    nn = neighbor(par,2)
    for d in delta(N):
##### Kinetic interactions
        si = state.site(point+d)
        if si["m"]==0:
            res += -t*ElAddBoson(state,d,"m")
        if si["m"]==1:
            res += -t*ElDelBoson(state,d,"m")
##### Spin fluctuations
    for p,q,d in pairs(nn):
        si = state.site(p)
        sj = state.site(q)
        if si["e"]==0 and sj["e"]==0:
            if si["m"] == 0 and sj["m"] == 0:
                res += 2*J*AddBosonPair(state,p,d,"m")
            if si["m"] == 1 and sj["m"] == 1:
                res += 2*J*DelBosonPair(state,p,d,"m")
    return res
