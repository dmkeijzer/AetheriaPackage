import numpy as np

# https://arc.aiaa.org/doi/pdf/10.2514/6.2017-0236

    # -------- lift from wing including slipstream effect -----------
def C_T(T, rho, V_0, S_W):
    C_T = 2*T/(rho*V_0*V_0*S_W)
    return C_T

def V_delta(C_T, S_W, n_e, D, V_0):
    V_delta_over_V_0 = np.sqrt(1+(C_T*S_W*4/(n_e*np.pi*D*D)))-1
    V_delta = V_delta_over_V_0 * V_0
    return V_delta

def D_star(D, V_0, V_delta):
    D_star = D*np.sqrt((V_0+(V_delta/2))/(V_0+V_delta))
    return D_star

    #effective aspect ratio of the wing imersed in the slipstream
def A_s_eff(b_W, S_W, n_e, D, V_0, V_delta):
    A_w = b_W*b_W/S_W
    A_s = n_e*D*D/S_W   # from wigeon, needs verification, couldnt find explanation for this...
    A_s_eff = A_s + (A_w - A_s)*((V_0/(V_0+V_delta))**(A_w-A_s))
    return A_s_eff, A_w, A_s

    # datcom
def CL_effective_alpha(mach, A_s_eff, sweep_half):
    beta = np.sqrt(1 - mach**2)
    CL_alpha_s_eff =  (2*np.pi*A_s_eff)/(2 + np.sqrt(4 + ((A_s_eff*beta)/0.95)**2*(1 + (np.tan(sweep_half)**2)/(beta**2))))
    return CL_alpha_s_eff

    #angle-of-attack of this section
def alpha_s(CL_wing, CL_alpha_s_eff, i_cs, angle_of_attack, alpha_0, V_0, V_delta, delta_alpha_zero_f):
    alpha_star = np.arctan2((V_0*np.sin(angle_of_attack)),(V_0*np.cos(angle_of_attack) + (V_delta/2)))
    alpha_s = alpha_star + i_cs - alpha_0 - delta_alpha_zero_f
    return alpha_s, i_cs
    
    #downwash due to slipstream
def sin_epsilon_angles(CL_alpha_s_eff, alpha_s, A_s_eff, CL_wing, A_w):
    sin_epsilon_s = (2*CL_alpha_s_eff * np.sin(alpha_s))/(np.pi*A_s_eff)
    sin_epsilon = (2*CL_wing)/(np.pi*A_w)
    return sin_epsilon, sin_epsilon_s

    # CL of the wing excluding propellors calculated through slipstream equation
def CL_ws(S_W, b_W, n_e, D_star, sin_epsilon, V_0, V_delta, sin_epsilon_s, CL_wing):

    CL_w = (2/S_W)*((np.pi/4)*b_W*b_W - (n_e*(np.pi/4)*D_star*D_star)) * sin_epsilon
    
    # CL of the propellors calculated through slipstream equation
    CL_s = n_e*(np.pi*D_star*D_star/(2*S_W)) * ((V_0+V_delta)**2/(V_0*V_0))*sin_epsilon_s
    CL_ws = CL_w + CL_s
    CL_slipstream = CL_ws - CL_wing

    #total CL slipstream
    return CL_slipstream, CL_ws, CL_w, CL_s



def prop_lift_thrust(T, rho, V_0, S_W, angle_of_attack):
    C_T = 2*T/(rho*V_0*V_0*S_W)
    CL_T = C_T * np.sin(angle_of_attack)
    return CL_T


