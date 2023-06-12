# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

"""CALCULATE TAIL LENGTH BASED ON BETA AND ASPECT RATIO"""
def find_tail_length(h0, b0, Beta, V, l, AR, n):
    roots = np.roots([np.pi / 3, np.pi * l, 0, -V/n]) # Find positive roots of cubic function of Tank Volume (two tanks)
    positive_roots = [root.real for root in roots if np.isreal(root) and root > 0]
    r = positive_roots[0] # radius of tank
    bc = 2 * n * r # width of crashed fuselage at end of tank
    hc = bc / AR # height of crashed fuselage at end of tank
    A_f = bc ** 2 / (AR * Beta ** 2) # area of fuselage at end of tank
    hf = np.sqrt(A_f / AR) # height of fuselage at end of tank
    bf = A_f/hf # width of fuselage at end of tank
    l_t = h0 * l / (h0 - hf) # length of tail
    upsweep = np.arctan2((h0 - hf), l) # upsweep angle
    return l_t, upsweep, bc, hc, hf, bf

"""CONVERGE TAIL LENGTH BY CONVERGING ASPECT RATIO"""
def converge_tail_length(h0, b0, Beta, V, l, ARe, n):
    AR0 = b0/h0
    AR = AR0
    error, i = 1, 0 # iteration error and number
    ARarr = [] # aspect ratio array
    while error > 0.005: # stop when error is smaller than 0.5%
        ARarr.append(AR)
        tail_data = list(find_tail_length(h0, b0, Beta, V, l, AR, n))
        AR = l / tail_data[0] * (ARe - AR0) + AR0
        error = np.abs((ARarr[-1] - AR)/AR)
        i += 1
        if i > 200: # stop if iteration number if more than 200 (no convergence)
            error = 0
    #print("Converged after: ", i, "iterations to AR: ", AR)
    tail_data.append(AR)
    return tail_data # returns tail length, upsweep, bc, hc, hf, bf

"""MAKE 2D SENSITIVY PLOT FOR BETA AND ARe"""
def plot_variable(h0, b0, V, l_tank, n,  parameter, parameter_values, fixed_parameter, fixed_value):
    l_tail = []

    for i in range(len(l_tank)):
        l_tail_row = []
        for j in range(len(parameter_values)):
            if parameter == 'ARe':
                ARe = parameter_values[j]
                Beta = fixed_value
            elif parameter == 'Beta':
                ARe = fixed_value
                Beta = parameter_values[j]

            l_t, upsweep, bc, hc, hf, bf, AR = converge_tail_length(h0, b0, Beta, V, l_tank[i], ARe, n)

            if 1 <= l_t <= 7.5 and hf < h0 and l_t > l_tank[i] and bf < b0  and AR > 0 and bc > 0 and hf > 0 and bf > 0 and hc > 0 and hc>bc/n:
                #and hc > bc / n
                l_tail_row.append(l_t)
            else:
                l_tail_row.append(np.nan)

        l_tail.append(l_tail_row)

    l_tank, parameter_values = np.meshgrid(l_tank, parameter_values)
    l_tail = np.array(l_tail)

    # Plot the colored plot
    fig, ax = plt.subplots()

    if parameter == 'ARe':
        c = ax.pcolormesh(l_tank, parameter_values, l_tail.T, cmap='YlOrRd')
    elif parameter == 'Beta':
        c = ax.pcolormesh(l_tank, parameter_values, l_tail.T, cmap='viridis_r')

    ax.set_xlabel('Tank Length [m]')
    ax.set_ylabel(parameter)
    ax.set_title(f'Fixed {fixed_parameter}: {fixed_value}')

    # Add colorbar
    cbar = plt.colorbar(c, label='Tail Length [m]')

    plt.show()

def minimum_tail_length(h0, b0, Beta, V, l_tank, ARe, n):
    l_tail = []
    l_tank = list(l_tank)
    indices_to_remove = []  # Track indices to be removed

    for i in range(len(l_tank)):
        l_t, upsweep, bc, hc, hf, bf, AR = converge_tail_length(h0, b0, Beta, V, l_tank[i], ARe, n)
        l_tail.append(l_t)

        if l_t < l_tank[i] or l_t > 8 or AR < 0 or bf < 0 or hf < 0 or hc < 0 or bc < 0 or hc<bc/n or bf>b0 or hf>h0:
            indices_to_remove.append(i)

    # Remove values from l_tail based on indices to remove
    l_tail = [l for i, l in enumerate(l_tail) if i not in indices_to_remove]

    # Remove values from l_tank based on indices to remove
    l_tank = [l for i, l in enumerate(l_tank) if i not in indices_to_remove]

    plt.plot(l_tail, l_tank)
    plt.xlabel("Tail length [m]")
    plt.ylabel("Tank length [m]")
    plt.show()

    l_tail = np.array(l_tail)
    if len(l_tail) > 0:
        min_index = np.argmin(l_tail)
        tail_data = converge_tail_length(h0, b0, Beta, V, l_tank[min_index], ARe, n)
        tail_data.append(l_tank[min_index])
    else:
        print("No possible tail length")


    return tail_data

def stress_strain_curve(stress_p, E, plain_stress):
    e = np.arange(0,1,0.001)
    densification = 0.95
    strain_p = stress_p / E
    crush_strength = np.zeros(len(e))

    for i in range(len(e)):
        if e[i] <= strain_p:
            crush_strength[i] = e[i]*E
        if e[i] > strain_p and e[i] <= densification:
            crush_strength[i] = plain_stress
        if e[i] > densification:
            crush_strength[i] = plain_stress + (e[i]-densification)*E

    plt.plot(e, crush_strength/(10**6))
    plt.show()
    return crush_strength, e


def decel_calculation():
    a = [0]
    v = [9.1]
    s = [0]
    t = [0]
    m = 3000
    Ek = [0.5*m*v[-1]**2]
    s_tot = 0.4
    dt = 0.00001
    g = 9.81
    A = 1
    sigma_cr, e = stress_strain_curve(1*10**6, 5*10**6, 0.35*10**6)
    """
    while v[-1] > 0:
        strain = s[-1]/s_tot
        index = np.abs(e - strain).argmin()
        Fcrush = sigma_cr[index]*A
        F = - Fcrush
        print("Fcrush: ", Fcrush)
        a.append(F/m)
        v.append(v[-1]+a[-1]*dt)
        print(v[-1])
        s.append(s[-1]+v[-1]*dt)
        print(s[-1])
        t.append(t[-1]+dt)
    """
    while v[-1] > 0:
        strain = s[-1]/s_tot
        index = np.abs(e - strain).argmin()
        Fcrush = sigma_cr[index]*A
        #Fcrush = 0.315*10**6
        ds = v[-1]*dt
        work_done = Fcrush*ds
        Ek.append(Ek[-1]- work_done)
        if Ek[-1] > 0:
            v.append(np.sqrt(2*Ek[-1]/m))
            a.append((v[-1]-v[-2])/dt)
            t.append(t[-1]+dt)
            s.append(s[-1] + ds)
        else:
            Ek.pop(-1)
            break

    plt.plot(t[1:-1], np.array(a[1:-1])/g)
    plt.xlabel("Time")
    plt.ylabel("Acceleration")
    plt.show()
    plt.plot(t, s)
    plt.xlabel("Time")
    plt.ylabel("Distance")
    plt.show()
    plt.plot(t, v)
    plt.xlabel("Time")
    plt.ylabel("Velocity")
    plt.show()


def simple_crash_box(m, a, sigma_cr, v):
    s = v**2/(2*a)
    A = m*a/sigma_cr
    return s, A

