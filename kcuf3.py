"""
Implementation of the propagator and interaction operators for the KCuF3 spin-orbital Kugel-Khomskii models. For details see:
Phys. Rev. B 94, 085117 (2016)
Phys. Rev. B 95, 235153 (2017)
Phys. Rev. B 100, 125109 (2019)
"""

from scipy.sparse.linalg import spsolve
from sympy import simplify

from sage import *
from propagators import *


## Symbols

J = sp.Symbol("J/4")
om = sp.Symbol("omega")
t = sp.Symbol("-t/4")
t0 = sp.Symbol("-t/4")*(1-2*sp.sin("phi"))
phi = sp.Symbol("phi")
eta = sp.Symbol("eta")


## 3D expansion functions

def ising3D(state):
    A = (1-eta)/(1+eta)/(1-3*eta)
    B = (1+3*eta)/(1+eta)/(1-3*eta)
    C = 1/(1-eta**2)
    Q = (sp.pi, sp.pi, 0)
    P = (0, sp.pi, 0)
    par = state.locset()
    nn = neighbor(par,3)
    res = 0
    for p,q,d in pairs(nn):
        si = state.site(p)
        sj = state.site(q)
        # for AB plane
        if d[-1]==0:
            if si["e"]==0 and sj["e"]==0:
                res += J*(A*sgm(si["m"])*sgm(sj["m"])+B) *(2*C/A-1-(3-4*sp.sin(phi)**2)/4 *sgm(si["o"])*sgm(sj["o"])) #H1
                res += J*C/2*(1-sgm(si["m"])*sgm(sj["m"])) *sp.sin(phi)*(sgm(si["o"])+sgm(sj["o"])) #H2
            else:# for electron
                res += J*B*(2*C/A-1) #H1
                res += J*C/2*sp.sin(phi)*(sj["e"]*sgm(si["o"])+si["e"]*sgm(sj["o"])) #H2
            # GS correction
            res -= J*(A+B)*(2*C/A-1-(3-4*sp.sin(phi)**2)/4) #<H1>, <H2>=0
        # for C direction
        if d[-1]!=0:
            if si["e"]==0 and sj["e"]==0:
                res += J*(-A*sgm(si["m"])*sgm(sj["m"])+B) *(2*C/A-1+sp.sin(phi)**2*sgm(si["o"])*sgm(sj["o"])) #H1
                res += J*C*(1-sgm(si["m"])*sgm(sj["m"])-(A+3*B)/C*sp.sin(phi)) *sp.sin(phi)*(sgm(si["o"])+sgm(sj["o"])) #H2
            else:# for electron
                res += J*B*(2*C/A-1) #H1
                res += J*C*(1-(A+3*B)/C*sp.sin(phi))*sp.sin(phi) *(sj["e"]*sgm(si["o"])+si["e"]*sgm(sj["o"])) #H2
            # GS correction
            res -=  J*(B-A)*(2*C/A-1+sp.sin(phi)**2) #<H1>
            res -= -J*2*(A+3*B)*sp.sin(phi)**2 #<H2>
    return Equation({state:simplify(res)})


def prop3D(state):
    start = state.locate("e")[0]
    if len(state.particles()) == 1:
        green = sp.Function("G0k")
        E0 = ising3D(state)[state]
        res = green("k+array("+str(tuple(state.qvect))+")",
                        om-E0,t0,2)*state
        return res
    else:
        green = sp.Function("Gi")
        bosons = state.locset(["o","m"])
        nn = neighbor(bosons)-bosons
        plane = [p[:-1] for p in bosons if p[-1]==start[-1]]
        sites = [p for p in nn if p[-1]==start[-1]] #consider only sites in the same z-plane as electron
        res = Equation()
        #if len(sites)==0: sites.append(tuple(start))
        for pt in sites:
            el = State(3)
            st = state.copy().relax(start,"e")
            E0 = ising3D(st)[st] + ising3D(el)[el]# energy of bosons + indep el.
            # E0 = ising3D(state)[state]
            s = ElHop(state,pt)
            res += s*green(str(tuple(plane)),str(pt[:-1]),str(tuple(start[:-1])),
                               om-E0,t0)
        return res


def intact3D(state):
    ## For now only kinetic interaction defined for general phi!
    A = (1-eta)/(1+eta)/(1-3*eta)
    B = (1+3*eta)/(1+eta)/(1-3*eta)
    C = 1/(1-eta**2)
    Q = (sp.pi, sp.pi, 0)
    P = (0, sp.pi, 0)
    t01 = t*sp.cos(phi)
    t11 = -t*(1+2*sp.sin(phi))
    s00 = t*(1+sp.sin(phi))
    s11 = -t*(1-sp.sin(phi))
    point = state.locate("e")[0]
    #par = state.locset()
    nn = neighbor([(0,0,0)],2)
    #nnn = neighbor(par,2)
    el = State(3)
    st = state.copy().relax(point,"e")
    res = ising3D(state)-(ising3D(st)[st]+ising3D(el)[el])*Reduce(state) #proximity effect, not included in Gi!
    for d in delta(3):
        end = state.site(point+d)
        # for AB plane
        if d[-1] == 0:
            if end["o"] == 0 and end["m"] == 0:
                res += (-2*t01*ElAddBoson(state,d,"o",Q)
                        -sp.sqrt(3)*t*sp.exp(sp.I*dot(P,d)) *ElAddBoson(state,d,"o")
                            )
            if end["o"] == 1 and end["m"] == 0:
                res += (2*t01*ElDelBoson(state,d,"o",Q)
                        -sp.sqrt(3)*t*sp.exp(sp.I*dot(P,d)) *ElDelBoson(state,d,"o")
                        +t11*ElSwapBoson(state,d,"o")
                            )
            if end["o"] == 0 and end["m"] == 1:
                res += (-2*t01*ElProcBosons(state,d,["o","m"],["m"],Q)
                        -sp.sqrt(3)*t*sp.exp(sp.I*dot(P,d)) *ElProcBosons(state,d,["o","m"],["m"])
                        +t11*ElSwapBoson(state,d,"m")
                            )
            if end["o"] == 1 and end["m"] == 1:
                res += (2*t01*ElProcBosons(state,d,["m"],["o","m"],Q)
                        -sp.sqrt(3)*t*sp.exp(sp.I*dot(P,d)) *ElProcBosons(state,d,["m"],["o","m"])
                        +t11*ElProcBosons(state,d,["o","m"],["o","m"])
                            )
        #For C direction
        if d[-1] != 0:
            if end["o"] == 0 and end["m"] == 0:
                res += 2*(s00*ElAddBoson(state,d,"m")
                          -t01*ElProcBosons(state,d,["o","m"],[],Q))
            if end["o"] == 1 and end["m"] == 0:
                res += 2*(s11*ElProcBosons(state,d,["o","m"],["o"])
                          -t01*ElProcBosons(state,d,["m"],["o"],Q))
            if end["o"] == 0 and end["m"] == 1:
                res += 2*(s00*ElDelBoson(state,d,"m")
                          -t01*ElProcBosons(state,d,["o"],["m"],Q))
            if end["o"] == 1 and end["m"] == 1:
                res += 2*(s11*ElProcBosons(state,d,["o"],["o","m"])
                          -t01*ElProcBosons(state,d,[],["o","m"],Q))
## Pseudo-Ising and spin-fluctuations
    for p,q,d in pairs(nn):
        si = state.site(p)
        sj = state.site(q)
        # for AB plane
        if d[-1]==0:
        # Pseudo-Ising with Q
            if si["e"]==0 and sj["e"]==0:
                res += -J*C/2*(1-sgm(si["m"])*sgm(sj["m"])) *sp.exp(sp.I*(dot(P,d)+dot(Q,p)))*sp.sqrt(3) *sp.cos(phi)*(sgm(si["o"])-sgm(sj["o"])) *Reduce(state,Q) #H2
            else:# for electron
                res += -J*C/2*sp.exp(sp.I*(dot(P,d)+dot(Q,p)))*sp.sqrt(3) *sp.cos(phi)*(sj["e"]*sgm(si["o"])-si["e"]*sgm(sj["o"])) *Reduce(state,Q)
        # FM fluctuations
            if si["e"]==0 and sj["e"]==0:# magnon hopping in plane
                res += J*(2*A*(2*C/A-1+(4*sp.sin(phi)**2-3)/4*sgm(si["o"]) *sgm(sj["o"]))*(HopBoson(state,p,d,"m")+HopBoson(state,q,-d,"m"))
                        -C*sp.sin(phi)*(sgm(si["o"])+sgm(sj["o"])) *(HopBoson(state,p,d,"m")+HopBoson(state,q,-d,"m"))
                        +C*sp.sqrt(3)*(sgm(si["o"])-sgm(sj["o"])) *sp.cos(phi)*sp.exp(sp.I*(dot(P,d)+dot(Q,p))) *(HopBoson(state,p,d,"m",Q)+HopBoson(state,q,-d,"m",Q)))
        # for C direction, AF fluctuations
        if d[-1]!=0:
            if si["e"]==0 and sj["e"]==0:
                if si["m"]==0 and sj["m"]==0:
                    res += 2*J*(A*(2*C/A-1+sp.sin(phi)**2*sgm(si["o"])*sgm(sj["o"]))
                                    +C*sp.sin(phi)*(sgm(si["o"])+sgm(sj["o"]))) *AddBosonPair(state,p,d,"m")
                if si["m"]==1 and sj["m"]==1:
                    res += 2*J*(A*(2*C/A-1+sp.sin(phi)**2*sgm(si["o"])*sgm(sj["o"]))
                                    +C*sp.sin(phi)*(sgm(si["o"])+sgm(sj["o"]))) *DelBosonPair(state,p,d,"m")
    return res


### 2D ab-plane expansion functions; restrictions included in H0

def propAB(state):
    J = sp.Symbol("3*J/8")
    start = state.locate("e")[0]
    om = sp.Symbol("omega")
    if len(state.particles()) == 1:
        green = sp.Function("G0k")
        m = 2*state.dim()
        s = green("k+array("+str(tuple(state.qvect))+")",
                      str(om-m*J),"t")*state
        return s
    else:
        green = sp.Function("Gj")
        bosons = state.locset("o")
        nnn = neighbor(bosons,2)-bosons
        res = Equation()
        for pt in nnn:
            s = ElHop(state,pt)
            res += s*green(str(tuple(bosons)),str(pt),str(tuple(start)),
                               str(om),"t",str(J))
        return res


def intactAB(state):
    Q = (sp.pi,sp.pi)
    P = (0, sp.pi)
    t = sp.Symbol("t")
    J = sp.Symbol("J/8")
    res = Equation()
    point = state.locate("e")[0]
    nn = neighbor([(0,0)],2)
    for d in delta(state.dim()):
        phase = sp.exp(sp.I*dot(P,d))
        if state.site(point+d)["o"] == 0:
            res += (2*t*ElAddBoson(state,d,"o") +
                    phase*sp.sqrt(3)*t*ElAddBoson(state,d,"o",Q))
        if state.site(point+d)["o"] == 1:
            res += (2*t*ElDelBoson(state,d,"o") +
                    -phase*sp.sqrt(3)*t*ElDelBoson(state,d,"o",Q) +
                    t*ElSwapBoson(state,d,"o"))
## Fluctuations
    for p,q,d in pairs(nn):
        si = state.site(p)
        sj = state.site(q)
        if si["e"]==0 and sj["e"]==0:
            #Tx*Tz
            M = -sp.sqrt(3)*J*sp.exp(sp.I*dot(P,d))
            if si["o"] == 0 and sj["o"] == 0:
                res += M*(AddBoson(state,q,"o")+AddBoson(state,p,"o"))
            if si["o"] == 0 and sj["o"] == 1:
                res += M*(DelBoson(state,q,"o")-AddBoson(state,p,"o"))
            if si["o"] == 1 and sj["o"] == 0:
                res += -M*(AddBoson(state,q,"o")-DelBoson(state,p,"o"))
            if si["o"] == 1 and sj["o"] == 1:
                res += -M*(DelBoson(state,q,"o")+DelBoson(state,p,"o"))
            #Tx*Tx
            if si["o"] == 0 and sj["o"] == 0:
                res += -J*AddBosonPair(state,p,d,"o")
            if si["o"] == 0 and sj["o"] == 1:
                res += -J*HopBoson(state,q,-d,"o")
            if si["o"] == 1 and sj["o"] == 0:
                res += -J*HopBoson(state,p,d,"o")
            if si["o"] == 1 and sj["o"] == 1:
                res += -J*DelBosonPair(state,p,d,"o")
    return res


### 2D ab-plane expansion functions; restrictions included in V

def propAB2(state):
    start = state.locate("e")[0]
    if len(state.particles()) == 1:
        green = sp.Function("G0k")
        m = 2*state.dim()
        s = green("k+array("+str(tuple(state.qvect))+")",
                      "omega-"+str(m)+"*J","t")*state
        return s
    else:
        green = sp.Function("Gj")
        bosons = state.locset("o")
        sites = state.nnset("o")
        res = Equation()
        for pt in sites:
            s = ElHop(state,pt)
            res += s*green(str(tuple(bosons)),str(pt),str(tuple(start)),
                               "omega","t","J",False)
        return res


def intactAB2(state):
    Q = sp.pi
    res = Equation()
    point = state.locate("e")[0]
    for d in delta(state.dim()):
        phase = sp.exp(sp.I*dot(array((0,sp.pi)),d.astype(object)))
        if state.site(point)["o"] == 0 and state.site(point+d)["o"] == 0:
            res += (sp.Symbol("2*t")*ElAddBoson(state,d,"o") +
                    phase*sp.Symbol("sqrt(3)*t")*ElAddBoson(state,d,"o",Q))
        if state.site(point)["o"] == 0 and state.site(point+d)["o"] == 1:
            res += (sp.Symbol("2*t")*ElDelBoson(state,d,"o") +
                    phase*sp.Symbol("-sqrt(3)*t")*ElDelBoson(state,d,"o",Q) +
                    sp.Symbol("t")*ElSwapBoson(state,d,"o") +
                    sp.Symbol("-t")*ElHop(state,d) + sp.Symbol("-2*J")*state)
        if state.site(point)["o"] == 1:
            res += sp.Symbol("-t")*ElHop(state,d) + sp.Symbol("-2*J")*state
    return res


## Some helper functions

def sgm(i, n=1):
    return 1-2*sp.Integer(i)/n
