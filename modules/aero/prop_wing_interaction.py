import numpy as np
import matplotlib.pyplot as plt


def prop_lift_slipstream(T, S_W, n_e, D, b_W, V_0, angle_of_attack, CL_wing, CL_alpha_s_eff, i_cs, rho, delta_alpha_zero_f, alpha_0):
    """_summary_
    :param T: thrust -> weight of the aircraft
    :type T: _type_
    :param S_W: surface area wing
    :type S_W: _type_
    :param n_e: number of engines on the wing
    :type n_e: _type_
    :param D: diameter rotors
    :type D: _type_
    :param V_0: free stream velocity
    :type V_0: _type_
    :param CL_alpha_s_eff: effective lift-curve slope of the slipstream section of the wing
    :type CL_alpha_s_eff: _type_
    :param rho: density 
    :type rho: _type_
    :param i_cs: angle of incidence with respect to fuselage 
    :type i_cs: _type_
    :param b_W: wing span 
    :type rho: _type_
    :param delta_alpha_zero_f: change in zero-life angle of atack due to flap deflection
    :type delta_alpha_zero_f: _type_
    :param alpha_0: zero-life of the airfoil section
    :type alpha_0: _type_
    :return: _description_
    :rtype: _type_
    """    
    # https://arc.aiaa.org/doi/pdf/10.2514/6.2017-0236

    # -------- lift from wing including slipstream effect -----------
    C_T = 2*T/(rho*V_0*V_0*S_W)
    V_delta_over_V_0 = np.sqrt(1+(C_T*S_W*4/(n_e*D*D)))-1
    V_delta = V_delta_over_V_0 * V_0

    D_star = D*np.sqrt((V_0+(V_delta/2))/(V_0+V_delta))

    #effective aspect ratio of the wing imersed in the slipstream
    A_w = b_W*b_W/S_W
    A_s = n_e*D*D/b_W   # from wigeon, needs verification, couldnt find explanation for this...
    A_s_eff = A_s + (A_w - A_s)*((V_0/(V_0+V_delta))**(A_w-A_s))

    #angle-of-attack of this section
    alpha_star = np.arctan((V_0*np.sin(angle_of_attack))/(V_0*np.cos(angle_of_attack) + (V_delta/2)))
    alpha_s = alpha_star + i_cs - alpha_0 - delta_alpha_zero_f
    
    #downwash due to slipstream
    sin_epsilon_s = (2*CL_alpha_s_eff * np.sin(alpha_s))/(np.pi*A_s_eff)
    sin_epsilon = (2*CL_wing)/(np.pi*A_w)

    CL_w = 2*((np.pi/4)*b_W*b_W - n_e*(np.pi/4)*D_star*D_star)*sin_epsilon/S_W
    CL_s = (n_e*np.pi*D_star*D_star*(V_0+V_delta)**2 * sin_epsilon_s)/(2*S_W*V_0*V_0)
    CL_ws = CL_w + CL_s
    CL_slipstream = CL_ws - CL_wing

    #total CL
    return CL_slipstream, CL_ws



def prop_lift_thrust(c_d, angle_of_attack):
    CL_T = c_d * np.sin(angle_of_attack)
    return CL_T

a = prop_lift_slipstream(T=650, S_W=6, n_e=2, D=1.9, b_W=5, V_0=83.33, angle_of_attack=0.054, CL_wing=0.434, CL_alpha_s_eff=0.07, i_cs=0, rho=1.225, delta_alpha_zero_f=0, alpha_0=-0.03)

print(a)