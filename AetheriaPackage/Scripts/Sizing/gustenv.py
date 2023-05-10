def posgustload(V, Vs, us, ns, CLalpha, WoS):
    n = lambda V, u: 1 + rho(avg_alt) * V * CLalpha * u / (2 * WoS)
    (ub, uc, ud), (Vb, Vc, VD), (nb, nc, nd)  = us, Vs, ns
    interpolate = lambda V, V1, V2, n1, n2: n1 + (V - V1) * (n2 - n1) / (V2 - V1)
    return n(V, ub) if 0 <= V <= Vb else \
    ( interpolate(V, Vb, Vc, nb, nc) if Vb < V <= Vc else interpolate(V, Vc, VD, nc, nd) )

neggustload = lambda V, Vs, us, ns, CLalpha, WoS: 2 - posgustload(V, Vs, us, ns, CLalpha, WoS)

def plotgustenv(Vb, Vc, CLalpha, WoS):
    n = lambda V, u: 1 + rho(avg_alt) * V * CLalpha * u / (2 * WoS)

    Vb, Vc, VD = Vs = (Vb, Vc, design_dive_speed(Vc)) # Change if VD Changes
    ub, uc, ud = us = 20.12, 15.24, 7.62 # Obtained from CS
    nb, nc, nd = ns = n(Vb, ub), n(Vc, uc), n(VD, ud)

    x = np.linspace(0, VD, 100)
    ax = sns.lineplot(x=x, y=[posgustload(V, Vs, us, ns, CLalpha, WoS) for V in x], color='blue')
    ax.set(xlabel="V [m/s]", ylabel="n [-]")
    sns.lineplot(x=x, y=[neggustload(V, Vs, us, ns, CLalpha, WoS) for V in x], color='blue', label='Gust Load Envelope')
    plt.plot([VD, VD], [neggustload(VD, Vs, us, ns, CLalpha, WoS), posgustload(VD, Vs, us, ns, CLalpha, WoS)], color='blue')
    print(f'Maximum load factor is: {np.max([nb,nc,nd])}')
    # plt.savefig('gust.png')
