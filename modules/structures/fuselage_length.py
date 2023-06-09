# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

"""CALCULATE TAIL LENGTH BASED ON BETA AND ASPECT RATIO"""
def find_tail_length(h0, b0, Beta, V, l, AR):
    roots = np.roots([np.pi / 3, np.pi * l, 0, -V/2]) # Find positive roots of cubic function of Tank Volume (two tanks)
    positive_roots = [root.real for root in roots if np.isreal(root) and root > 0]
    r = positive_roots[0] # radius of tank
    bc = 4 * r # width of crashed fuselage at end of tank
    hc = bc / AR # height of crashed fuselage at end of tank
    A_f = bc ** 2 / (AR * Beta ** 2) # area of fuselage at end of tank
    hf = np.sqrt(A_f / AR) # height of fuselage at end of tank
    bf = A_f/hf # width of fuselage at end of tank
    l_t = h0 * l / (h0 - hf) # length of tail
    upsweep = np.arctan2((h0 - hf), l) # upsweep angle
    return l_t, upsweep, bc, hc, hf, bf

"""CONVERGE TAIL LENGTH BY CONVERGING ASPECT RATIO"""
def converge_tail_length(h0, b0, Beta, V, l, AR, ARe, AR0):
    error, i = 1, 0 # iteration error and number
    ARarr = [] # aspect ratio array
    while error > 0.005: # stop when error is smaller than 0.5%
        ARarr.append(AR)
        tail_data = list(find_tail_length(h0, b0, Beta, V, l, AR))
        AR = l / tail_data[0] * (ARe - AR0) + AR0
        error = np.abs((ARarr[-1] - AR)/AR)
        i += 1
        if i > 200: # stop if iteration number if more than 200 (no convergence)
            error = 0
    #print("Converged after: ", i, "iterations to AR: ", AR)
    tail_data.append(AR)
    return tail_data # returns tail length, upsweep, bc, hc, hf, bf

"""MAKE 2D SENSITIVY PLOT FOR BETA AND ARe"""
def plot_variable(h0, b0, V, l_tank, parameter, parameter_values, fixed_parameter, fixed_value):
    AR0 = b0 / h0 # define inputs

    # initialise
    AR = AR0
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

            l_t, upsweep, bc, hc, hf, bf, AR = converge_tail_length(h0, b0, Beta, V, l_tank[i], AR, ARe, AR0)

            if 1 <= l_t <= 7.5 and hf < h0 and l_t > l_tank[i] and bf < b0 and hc > bc / 2 and AR > 0 and bc > 0 and hf > 0:
                l_tail_row.append(l_t)
            else:
                l_tail_row.append(np.nan)

        l_tail.append(l_tail_row)

    l_tank, parameter_values = np.meshgrid(l_tank, parameter_values)
    l_tail = np.array(l_tail)

def tank_radius(l_tank, V_tank):
    """WIDTH OF FUSELAGE AT THE END OF HYDROGEN TANK, BASED ON TANK RADIUS"

    :param l_tank: length of the tank in m
    :type l_tank: float
    :param V_tank: total volume of the tank in m^3
    :type V_tank:  float
    :return: tank radius in m
    :rtype: float
    """    
    roots = np.roots(np.pi/3, np.pi*l_tank, 0, V_tank)
    return [root.real for root in roots if root.real > 0 and root.imag == 0]

    if parameter == 'ARe':
        c = ax.pcolormesh(l_tank, parameter_values, l_tail.T, cmap='YlOrRd')
    elif parameter == 'Beta':
        c = ax.pcolormesh(l_tank, parameter_values, l_tail.T, cmap='viridis_r')
def width_fuselage_crushed(r): 
    """ WIDTH OF FUSELAGE AT THE END OF HYDROGEN TANK, BASED ON TANK RADIUS"

    :param r: tank radius in m
    :type r: float
    :return: width fuselage crushed in m
    :rtype: float
    """    
    return 4*r

    ax.set_xlabel('Tank Length [m]')
    ax.set_ylabel(parameter)
    ax.set_title(f'Fixed {fixed_parameter}: {fixed_value}')
def height_fuselage_crushed(r, Beta, h0, b0):
    """ HEIGHT OF FUSELAGE AT THE END OF HYDROGEN TANK, CONSTRAINED BY BETA

    :param r: radius of tank in meters
    :type r: float
    :param Beta: Crash diameter coefficient, defines the safe area after a crash
    :type Beta: float
    :param h0: Height of the cabin, thus the height at the start of the tailcone
    :type h0: float
    :param b0:Height of the cabin, thus the height at the start of the tailcone
    :type b0: _type_
    :return: _description_
    :rtype: _type_
    """    
    b_c = 4*r
    A = b0/h0
    A_f = b_c**2/(A*Beta**2)
    return np.sqrt(A_f/A)

    # Add colorbar
    cbar = plt.colorbar(c, label='Tail Length [m]')
def find_length_fuselage(h0, h_f, l_tank):
    """HEIGHT OF FUSELAGE AT THE END OF HYDROGEN TANK, CONSTRAINED BY BETA

    :param h0: _description_
    :type h0: _type_
    :param h_f: _description_
    :type h_f: _type_
    :param l_tank: _description_
    :type l_tank: _type_
    :return: _description_
    :rtype: _type_
    """    
    return (h0-h_f)/h_f*l_tank

def upsweep_angle(h0, h_f, l_tank):
    """ FIND UPSWEEP ANGLE

    :param h0: _description_
    :type h0: _type_
    :param h_f: _description_
    :type h_f: _type_
    :param l_tank: _description_
    :type l_tank: _type_
    :return: _description_
    :rtype: _type_
    """    
    return np.arctan2((h0-h_f)/l_tank)