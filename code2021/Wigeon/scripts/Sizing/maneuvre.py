import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

avg_alt = 1110
design_dive_speed = lambda Vc: 1.2 * Vc
rho = 1.099650576194018

def maneuvrenv(V, Vs, WoS, CLmax, pos=True):
    n = lambda CL, V, WoS: 0.5 * rho * V ** 2 * CL / WoS
    nmin, nmax = -1, 2.5 # UAM reader
    Vc, VD = Vs
    interpolate = lambda V, V1, V2, n1, n2: n1 + (V - V1) * (n2 - n1) / (V2 - V1)
    return min(n(CLmax, V, WoS), nmax) if pos else \
    ( max(-n(CLmax, V, WoS), nmin) if V <= Vc else interpolate(V, Vc, VD, nmin, 0))

def plotmaneuvrenv(WoS, Vc, CLmax):
    VD = design_dive_speed(Vc)
    Vs = Vc, VD
    x = np.linspace(0, VD, 100)
    ax = sns.lineplot(x=x, y=[maneuvrenv(V, Vs, WoS, CLmax, True) for V in x], color='red')
    sns.lineplot(x=x, y=[maneuvrenv(V, Vs, WoS, CLmax, False) for V in x], color='red', label='Maneuvre Envelope')
    ax.set(xlabel="V [m/s]", ylabel="n [-]")
    plt.plot([VD, VD], [maneuvrenv(VD, Vs, WoS, CLmax, False), maneuvrenv(VD, Vs, WoS, CLmax, True)], color='red')
    return np.max([maneuvrenv(V, Vs, WoS, CLmax, True) for V in x])
    # plt.savefig('maneuvre.png')

def posgustload(V, Vs, us, ns, CLalpha, WoS):
    n = lambda V, u: 1 + rho * V * CLalpha * u / (2 * WoS)
    (ub, uc, ud), (Vb, Vc, VD), (nb, nc, nd)  = us, Vs, ns
    interpolate = lambda V, V1, V2, n1, n2: n1 + (V - V1) * (n2 - n1) / (V2 - V1)
    return n(V, ub) if 0 <= V <= Vb else \
    ( interpolate(V, Vb, Vc, nb, nc) if Vb < V <= Vc else interpolate(V, Vc, VD, nc, nd) )

neggustload = lambda V, Vs, us, ns, CLalpha, WoS: 2 - posgustload(V, Vs, us, ns, CLalpha, WoS)

def plotgustenv(Vb, Vc, CLalpha, WoS):
    n = lambda V, u: 1 + rho * V * CLalpha * u / (2 * WoS)

    Vb, Vc, VD = Vs = (Vb, Vc, design_dive_speed(Vc)) # Change if VD Changes
    ub, uc, ud = us = 20.12, 15.24, 7.62 # Obtained from CS
    nb, nc, nd = ns = n(Vb, ub), n(Vc, uc), n(VD, ud)

    x = np.linspace(0, VD, 100)
    ax = sns.lineplot(x=x, y=[posgustload(V, Vs, us, ns, CLalpha, WoS) for V in x], color='blue')
    ax.set(xlabel="V [m/s]", ylabel="n [-]")
    sns.lineplot(x=x, y=[neggustload(V, Vs, us, ns, CLalpha, WoS) for V in x], color='blue', label='Gust Load Envelope')
    plt.plot([VD, VD], [neggustload(VD, Vs, us, ns, CLalpha, WoS), posgustload(VD, Vs, us, ns, CLalpha, WoS)], color='blue')
    return np.max([posgustload(V, Vs, us, ns, CLalpha, WoS) for V in x])
    # plt.savefig('gust.png')