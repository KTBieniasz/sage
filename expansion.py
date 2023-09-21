"""
General expansion functions used by the Equation class to expand the EOMs by applying the Hamiltonian according to a set of rules.
"""

from multiprocessing import Value

from numpy import *
import sympy as sp
from scipy.sparse import csr_matrix

from propagators import *
from sage.classes import *


def expand(state,propagate,interact):
    '''Expands the state by first applying the propagator and then the interaction. Returns two equations: the rhs of the expansion of the state and the propagators for the state.'''
    s = state.copy()
    cns = propagate(s)
    if cns:
        for (p,c) in cns.items():
            s -= interact(p)*c
        return s, cns
    else:
        return 1*s, cns


def rules(state,rls):
    '''Implements the rules used to keep or exclude states from the variational Hilbert space at expansion order n.'''
    par = state.particles()
    sites = state.locate(rls[0].keys())
    if state.period:
        pdist = array([[sum(vstack([abs(i-j),abs(abs(i-j)-2*state.edge-1)]).min(0))
                        for i in sites] for j in sites])
    else:
        pdist = array([[sum(abs(i-j)) for i in sites] for j in sites])
    pstr = array([a[1:]-a[:-1] for a in sort(pdist)])
    #
    rules = [
        sum(tuple(par.values()))<=sum(tuple(rls[0].values()))+1, #enforce expansion order
        all(pstr<=1) and sum(pstr==2)<=len(pstr), #cloud topology
        rls[1] or all([par[k]<=v for (k,v) in rls[0].items()]) #enforce boson numbers independently
        ]
    return all(rules)


fdict = {"G0k":G0k,"G0":G0,"Gj":Gj,"Gi":Gi,"G3sk":G3sk,"G3s":G3s,"Gm":Gm}


def convert(args,eqlist,const,fdict=fdict,stset=None):
    '''Function to convert the symbolic equations of motion into numpy functions which return the matrices used for solving the system.'''
    fdict = {"G0k":G0k,"G0":G0,"Gj":Gj,"Gi":Gi,"G3sk":G3sk,"G3s":G3s,"Gm":Gm}
    if stset==None:
        stset=set([st for eq in eqlist for st in eq])
    X = sorted(stset)
    D = {e:i for (i,e) in enumerate(X)}
    n = len(stset)
    idx1 = []
    idx2 = []
    col = []
    for i,line in enumerate(eqlist):
        row = []
        for (state,multip) in sorted(line.items()):
            idx1.append(i)
            idx2.append(D[state])
            row.append(multip)
        col.append(sp.lambdify(args,row,[fdict,"numpy"]))
    A = lambda *prms: csr_matrix(([x for f in col for x in f(*prms)],
                                      (idx1,idx2)),(n,n),dtype=complex128)
    const = array(const).reshape(-1,1)
    idx = nonzero(const)
    F = sp.lambdify(args,tuple(const[idx]),[fdict,"numpy"])
    B = lambda *prms: csr_matrix((F(*prms),idx),(n,1))
    return A,B,X


def eom(state,rls,propagate,interact,args,symb=False):
    '''Generates the EOMs by expanding the starting state up to order n in the number of bosons. Repetitively uses the expand function to generate the equation, verifies the resulting states with rules and inserts the conforming states into the list of states to be expanded. The process is continued until there are no new states generated. Then converts thesymbolic equations, unless symb==True.'''
    todo = set()
    done = set()
    todo.add(state)
    eom = []
    cns = []
    while todo:
        s = todo.pop()
        #print(s)
        (res, c) = expand(s,propagate,interact)
        trunk = Equation({r:v for (r,v) in res.items() if rules(r,rls)})
        eom.append(trunk)
        cns.append(state*c)
        done.add(s)
        todo = todo | set(trunk) - done
    if symb==False:
        return convert(args,eom,cns,fdict,done)
    else:
        return eom,cns,done


def conv(args,eqlist,rls=None,term=False):
    '''Function to convert the symbolic equations of motion into numpy functions which return the matrices used for solving the system.'''
    fdict = {"G0k":G0k,"G0":G0,"Gj":Gj,"Gi":Gi}
    if rls==None: #for nonterminal G0 - no rls
        stset=set([st for eq in eqlist for st in eq])
    elif term==False: #for nonterminal V - rls passed to rules
        stset=set([st for eq in eqlist for st in eq if rules(st,rls)])
    else: #for terminal matrix - rls is the list of input states
        stset=set(rls)
    A = lambda *prms: array(sp.lambdify(args, [[eq[x] for eq in eqlist]
                        for x in sorted(stset)], [fdict,"numpy"])(*prms))
    return A,stset


def Green(state,rls,propagate,interact,args):
    L = list(state)
    F = []
    for i in range(sum(rls)):
        g = [propagate(s) for s in sorted(L)]
        G0,st0 = conv(args,g,L,True)
        G1,st1 = conv(args,g)
        v = [interact(s) for s in sorted(st1)]
        V0,st2 = conv(args,v,st1,True)
        V1,st3 = conv(args,v,rls)
        F.append((G0,G1,V0,V1))
        L = st3
    n = len(L)
    def func(*prms):
        I = csr_matrix((n,n),dtype=int)
        for (G0,G1,V0,V1) in F[-1::-1]:
            G = G1(*prms)
            V = V1(*prms)
            L = (G.T).dot((V.T).conj())
            R = V.dot(G)
            I = G0(*prms) + (G.T).dot(V0(*prms).dot(G)) + L.dot(I.dot(R))
        return I
    return func
