import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from input.GeneralConstants import *


def maneuvrenv(V, Vs, WoS, CLmax, nmin, nmax, pos=True):
    n = lambda CL, V, WoS: 0.5 * rho_cruise * V ** 2 * CL / WoS
    Vc, VD = Vs
    interpolate = lambda V, V1, V2, n1, n2: n1 + (V - V1) * (n2 - n1) / (V2 - V1)
    return min(n(CLmax, V, WoS), nmax) if pos else \
    ( max(-n(CLmax, V, WoS), nmin) if V <= Vc else interpolate(V, Vc, VD, nmin, 0))

def plotmaneuvrenv(WoS, Vc, CLmax, nmin, nmax):
    VD = 1.2*Vc
    Vs = Vc, VD
    x = np.linspace(0, VD, 100)
    ax = sns.lineplot(x=x, y=[maneuvrenv(V, Vs, WoS, CLmax, nmin, nmax, True) for V in x], color='blue', zorder=3)
    sns.lineplot(x=x, y=[maneuvrenv(V, Vs, WoS, CLmax, nmin, nmax, False) for V in x], color='blue', label='Manoeuvre Envelope',zorder=3)
    ax.set(xlabel="V [m/s]", ylabel="n [-]")
    plt.plot([VD, VD], [maneuvrenv(VD, Vs, WoS, CLmax, nmin, nmax, True), maneuvrenv(VD, Vs, WoS, CLmax, nmin, nmax, True)], color='blue',zorder=3)
    plt.plot([VD, VD],[0, nmax], color='blue',zorder=3)
    plt.grid(True)
    plt.xlim(0,VD+7)
    plt.plot([-5,VD+7],[0,0], color='black', lw=1)


    #plt.savefig('manoeuvre_env.png')
    return np.max([maneuvrenv(V, Vs, WoS, CLmax, nmin, nmax, True) for V in x])

def posgustload(V, Vs, us, ns, CLalpha, WoS):
    n = lambda V, u: 1 + rho_cruise * V * CLalpha * u / (2 * WoS)
    (ub, uc, ud), (Vb, Vc, VD), (nb, nc, nd)  = us, Vs, ns
    interpolate = lambda V, V1, V2, n1, n2: n1 + (V - V1) * (n2 - n1) / (V2 - V1)
    return n(V, ub) if 0 <= V <= Vb else \
    ( interpolate(V, Vb, Vc, nb, nc) if Vb < V <= Vc else interpolate(V, Vc, VD, nc, nd) )

neggustload = lambda V, Vs, us, ns, CLalpha, WoS: 2 - posgustload(V, Vs, us, ns, CLalpha, WoS)


def plot_dash(V, n):
    plt.plot([0, V],[1, n], linestyle='dashed', color='black', zorder=1, alpha=0.5)


def plotgustenv(V_s, Vc, CLalpha, WoS, TEXT=False):
    n = lambda V, u: 1 + rho_cruise * V * CLalpha * u / (2 * WoS)
    Vb = np.sqrt(n(Vc, uc))*V_s
    Vb, Vc, VD = Vs = (Vb, Vc, 1.2*Vc) # Change if VD Changes
    us = ub, uc, ud  # Obtained from CS
    nb, nc, nd = ns = n(Vb, ub), n(Vc, uc), n(VD, ud)
    x = np.linspace(0, VD, 100)
    ax = sns.lineplot(x=x, y=[posgustload(V, Vs, us, ns, CLalpha, WoS) for V in x], color='black', zorder=2)
    ax.set(xlabel="V [m/s]", ylabel="n [-]")
    sns.lineplot(x=x, y=[neggustload(V, Vs, us, ns, CLalpha, WoS) for V in x], color='black', label='Gust Load Envelope',zorder=2)
    plt.plot([VD, VD], [neggustload(VD, Vs, us, ns, CLalpha, WoS), posgustload(VD, Vs, us, ns, CLalpha, WoS)], color='black',zorder=2)
    plot_dash(Vc, nc)
    plot_dash(Vc, 2 - nc)
    plot_dash(VD, nd)
    plot_dash(VD, 2 - nd)
    plt.plot([Vb, Vb], [2-nb, nb], linestyle='dashed', color='black', zorder=1, alpha=0.5)
    plt.plot([Vc, Vc], [2-nc, nc], linestyle='dashed', color='black', zorder=1, alpha=0.5)
    if TEXT:
        plt.text(Vb + 1, 0.1, 'Vb', fontsize = 11, weight='bold')
        plt.text(V_s + 1, 0.1, 'Vs', fontsize=11, weight='bold')
        plt.text(Vc + 1, 0.1, 'Vc', fontsize=11, weight='bold')
        plt.text(VD + 1, 0.1, 'Vd', fontsize=11, weight='bold')
        plt.plot([V_s,V_s],[0, 0.05], color='black')
        plt.plot([Vb, Vb], [0, 0.05], color='black')
        plt.plot([Vc, Vc], [0, 0.05], color='black')
        plt.plot([Vc, Vc], [0, 0.05], color='black')



    return np.max([posgustload(V, Vs, us, ns, CLalpha, WoS) for V in x])
    # plt.savefig('gust.png')






"""def posgustload(V, Vs, us, ns, CLalpha, WoS):
    n = lambda V, u: 1 + rho_cruise * V * CLalpha * u / (2 * WoS)
    (ub, uc, ud), (Vb, Vc, VD), (nb, nc, nd)  = us, Vs, ns
    interpolate = lambda V, V1, V2, n1, n2: n1 + (V - V1) * (n2 - n1) / (V2 - V1)
    return n(V, ub) if 0 <= V <= Vb else \
    ( interpolate(V, Vb, Vc, nb, nc) if Vb < V <= Vc else interpolate(V, Vc, VD, nc, nd) )

def n_gust(V, CLalpha, u, WoS): return  1 + rho_cruise * V * CLalpha * u / (2 * WoS)

def pos_gust_load(V, Vb, Vc, ub, uc, ud, WS, CLalpha):
    if V <= Vb: n = n_gust(V, CLalpha, ub, WS)
    elif V <= Vc: n = n_gust(V, CLalpha, uc, WS)
    else: n = n_gust(V, CLalpha, ud, WS)
    return n

def neg_gust_load(n): return 2 - n

def plot_gust_load(V, Vb, Vc, ub, uc, ud, WS, CLalpha):
    VD = 1.2*Vc
    V = np.linspace(0, VD, 100)
    plt.plot(V, pos_gust_load(V, Vb, Vc, ub, uc, ud, WS, CLalpha))"""

