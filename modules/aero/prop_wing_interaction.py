import numpy as np



def prop_interaction(C_T, S_W, n_c, D, V_0, CL_alpha_s_eff, alpha_s, rho, S_i, delta_alpha_zero_f, alpha_0):
    
    # https://arc.aiaa.org/doi/pdf/10.2514/6.2017-0236

    #lift from wing including slipstream effect
    V_delta_over_V_0 = np.sqrt(1+(C_T*S_W*4/(n_c*D*D)))-1
    V_delta = V_delta_over_V * V_0

    D_star = D*np.sqrt((V_0+V_delta/2)/V+V_delta)

    #effective aspect ratio of the wing imersed in the slipstream
    A_s_eff = A_s + (A_w - A_s)*((V_0/(V_0/V_delta))**(A_w-A_s))

    #angle-of-attack of this section
    alpha_star = np.arctan((V_0*np.sin(angle_of_attack))/(V_0*np.cos(angle_of_attack) + (V_delta/2)))
    alpha_s = alpha_star + i_cs - alpha_0 - delta_alpha_zero_f
    
    #downwash due to slipstream
    sin_epsilon_s = (2*CL_alpha_s_eff * np.sin(alpha_s))/(np.pi*A_s_eff)
    sin_epsilon = (2*CL_w)/(np.pi*A_w)

    CL_ws = 2*((np.pi/4)*b_W*b_W - n_e*(np.pi/4)*D_star*D_star)*sin_epsilon/S_i + (n_e*np.pi*D_star*D_star*(V+V_delta)**2 * sin_epsilon_s)/(2*S_i*V*V)

    #lift from thrust perpendicular to free stream velocity
    C_T = 2*T/(rho*V*V*S)
    CL_T = C_T * np.sin(alpha)

    #total CL
    CL_tot = CL_ws + CL_T

    return CL_tot