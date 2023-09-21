"""
General state operators used to implement more complicated Hamiltonians.
These should only use the State methods, usually begin with copy() and end with reduce().
"""

from numpy import *


def Reduce(state,Q=0):
    '''This function only performs the reduction, optionally adding the ordering vector Q. Mostly useful when including Ising-type processes in the interaction.'''
    s = state.copy()
    s.kadd(Q)
    if s.locset("e"):
        return s.reduce()
    else:
        return s


def ElHop(state,site):
    '''This function hops the electron to the given site. Since usually electron should be at the origin, site is assumed to be the absolute location, not a hoping vector.'''
    point = state.locate("e")[0]
    s = state.copy()
    s.relax(point,"e")
    s.excite(site,"e")
    return s.reduce()


def ElAddBoson(state,vect,boson,Q=0):
    '''Move the electron by vect, leaving a boson behind. Optionally can add the ordering vector Q to the state.'''
    point = state.locate("e")[0]
    if state.site(point+vect)["e"] == 0:
        s = state.copy()
        vect = array(vect,int)
        s.relax(point,"e")
        s.excite(point+vect,"e")
        s.excite(point,boson)
        s.kadd(Q)
        return s.reduce()
    else: return 0


def ElDelBoson(state,vect,boson,Q=0):
    '''Delete a boson by moving a neighboring electron to its site. Optionally can add the ordering vector Q to the state.'''
    point = state.locate("e")[0]
    if state.site(point+vect)[boson] > 0:
        s = state.copy()
        vect = array(vect,int)
        s.relax(point+vect,boson)
        s.relax(point,"e")
        s.excite(point+vect,"e")
        s.kadd(Q)
        return s.reduce()
    else: return 0


def ElSwapBoson(state,vect,boson):
    '''Swap a neighbouing electron/boson pair. Vector vect points from the electron to the boson.'''
    point = state.locate("e")[0]
    if state.site(point+vect)[boson] > 0:
        s = state.copy()
        vect = array(vect,int)
        s.relax(point+vect,boson)
        s.relax(point,"e")
        s.excite(point+vect,"e")
        s.excite(point,boson)
        return s.reduce()
    else: return 0


def ElProcBosons(state,vect,exc,rel,Q=0):
    '''Generalization of the electron/boson processes. Moves an electon by vect, leaving behind the bosons in list exc and destroying the bosons from the list rel at the end site. The user has to check the boson occupancy on the end site.'''
    point = state.locate("e")[0]
    if all([state.site(point+vect)[p]>0 for p in rel]):
        s = state.copy()
        vect = array(vect,int)
        s.relax(point,"e")
        for p in rel:
            s.relax(point+vect,p)
        s.excite(point+vect,"e")
        for p in exc:
            s.excite(point,p)
        s.kadd(Q)
        return s.reduce()
    else: return 0


def ElAddBosonPair(state,vect1,vect2,boson,Q=0):
    '''Move the electron by vect, leaving a boson behind. Optionally can add the ordering vector Q to the state.'''
    point = state.locate("e")[0]
    if state.site(point+vect1)["e"]==0 and state.site(point+vect1+vect2)["e"]==0:
        s = state.copy()
        vect1,vect2 = array(vect1,int),array(vect2,int)
        s.relax(point,"e")
        s.excite(point+vect1+vect2,"e")
        s.excite(point,boson)
        s.excite(point+vect1,boson)
        s.kadd(Q)
        return s.reduce()
    else: return 0


def ElDelBosonPair(state,vect1,vect2,boson,Q=0):
    '''Delete a boson by moving a neighboring electron to its site. Optionally can add the ordering vector Q to the state.'''
    point = state.locate("e")[0]
    if state.site(point+vect1)[boson]>0 and state.site(point+vect1+vect2)[boson]>0:
        s = state.copy()
        vect1,vect2 = array(vect1,int),array(vect2,int)
        s.relax(point+vect1,boson)
        s.relax(point+vect1+vect2,boson)
        s.relax(point,"e")
        s.excite(point+vect1+vect2,"e")
        s.kadd(Q)
        return s.reduce()
    else: return 0


def HopBoson(state,point,vect,boson,Q=0):
    '''Hop the boson at point by vect, optionally adding the ordering vector Q to the state. Useful for fluctuations.'''
    if state.site(point)[boson] > 0:
        s = state.copy()
        vect = array(vect,int)
        s.relax(point,boson)
        s.excite(point+vect,boson)
        s.kadd(Q)
        return s.reduce()
    else: return 0


def AddBoson(state,point,boson,Q=0):
    '''Create a boson at point, optionally adding the ordering vector Q to the state. Useful for fluctuations.'''
    s = state.copy()
    s.excite(point,boson)
    s.kadd(Q)
    return s.reduce()


def DelBoson(state,point,boson,Q=0):
    '''Delete the boson at point, optionally adding the ordering vector Q to the state. Useful for fluctuations.'''
    if state.site(point)[boson] > 0:
        s = state.copy()
        s.relax(point,boson)
        s.kadd(Q)
        return s.reduce()
    else: return 0


def AddBosonPair(state,point,vect,boson):
    '''Create a pair of bosons on point and the site next to it by vect. Useful for fluctuations.'''
    s = state.copy()
    vect = array(vect,int)
    s.excite(point,boson)
    s.excite(point+vect,boson)
    return s.reduce()


def DelBosonPair(state,point,vect,boson):
    '''Delete a pair of bosons on point and the site next to it by vect. Useful for fluctuations.'''
    if (state.site(point)[boson]>0 and state.site(point+vect)[boson]>0):
        s = state.copy()
        vect = array(vect,int)
        s.relax(point,boson)
        s.relax(point+vect,boson)
        return s.reduce()
    else: return 0


def ProcBosons(state,point,vect,rel1,rel2,exc1,exc2,Q=0):
    '''Generalization of the boson processes. Creates/destroys the bosons in list exc1/rel1 at point and creates/destroys the bosons from the list exc2/rel2 at the end site. The user has to check the boson occupancy on the end site.'''
    vect = array(vect,int)
    if (all([state.site(point)[p]>0 for p in rel1]) and
        all([state.site(point+vect)[p]>0 for p in rel2])):
        s = state.copy()
        for p in rel1:
            s.relax(point,p)
        for p in rel2:
            s.relax(point+vect,p)
        for p in exc1:
            s.excite(point,p)
        for p in exc2:
            s.excite(point+vect,p)
        s.kadd(Q)
        return s.reduce()
    else: return 0
