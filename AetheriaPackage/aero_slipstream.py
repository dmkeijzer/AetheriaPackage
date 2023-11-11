
import numpy as np
from AetheriaPackage.ISA_tool import ISA
import AetheriaPackage.GeneralConstants as const

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



def slipstream_cruise(WingClass,EngineClass, AeroClass, mission):
        # with open(r"input/data_structures/aetheria_constants.json") as jsonFile:
        #         data = json.load(jsonFile)
        atm = ISA(const.h_cruise)
        t_cr = atm.temperature()
        rho_cr = atm.density()
        mhu = atm.viscosity_dyn()
        # print(rho_cr)
        EngineClass.total_disk_area = mission.MTOM / const.diskloading
        diameter_propellers = 2*np.sqrt(EngineClass.total_disk_area/ np.pi )

        # Angles
        i_cs_var = 0.0733 # calculated from lift at cruise
        angle_of_attack_fuse = 0
        angle_of_attack_prop = angle_of_attack_fuse + i_cs_var

        # ---------- CRUISE ---------
        drag_cruise = 0.5*rho_cr*WingClass.surface*const.v_cr*const.v_cr*AeroClass.cd_cruise
        # drag_cruise = 0
        # Thrust coefficient
        C_T_var = 0.15

        #change in V
        V_delta_var = V_delta(C_T=C_T_var, S_W=WingClass.surface, n_e=3, D=diameter_propellers, V_0=const.v_cr)

        # effective Diameter
        D_star_var = D_star(D=diameter_propellers, V_0=const.v_cr, V_delta=V_delta_var)

        A_eff_var = A_s_eff(b_W=WingClass.span, S_W=WingClass.surface, n_e=3, D=diameter_propellers, V_0=const.v_cr, V_delta=V_delta_var)[0]

        # DATCOM
        CL_eff_alpha_var = CL_effective_alpha(mach=AeroClass.mach_cruise, A_s_eff= A_eff_var, sweep_half=-WingClass.sweep_LE)

        # angles
        angles = alpha_s(CL_wing=AeroClass.cL_cruise, CL_alpha_s_eff=CL_eff_alpha_var, i_cs = i_cs_var, angle_of_attack= angle_of_attack_prop, alpha_0=const.alpha_zero_l, V_0=const.v_cr, V_delta=V_delta_var, delta_alpha_zero_f=0)
        alpha_s_var = angles[0]

        sin_epsilon = sin_epsilon_angles(CL_alpha_s_eff=CL_eff_alpha_var, alpha_s=alpha_s_var, A_s_eff=A_eff_var, CL_wing=AeroClass.cL_cruise, A_w=WingClass.aspect_ratio)[0]
        sin_epsilon_s = sin_epsilon_angles(CL_alpha_s_eff=CL_eff_alpha_var, alpha_s=alpha_s_var, A_s_eff=A_eff_var, CL_wing=AeroClass.cL_cruise, A_w=WingClass.aspect_ratio)[1]

        CL_slipstream_final = CL_ws(S_W=WingClass.surface, b_W=WingClass.span, n_e=3, D_star=D_star_var, sin_epsilon=sin_epsilon, V_0=const.v_cr, V_delta=V_delta_var, sin_epsilon_s=sin_epsilon_s, CL_wing=AeroClass.cL_cruise)[0]
        CL_old = AeroClass.cL_cruise

        CL_ws_var = CL_ws(S_W=WingClass.surface, b_W=WingClass.span, n_e=3, D_star=D_star_var, sin_epsilon=sin_epsilon, V_0=const.v_cr, V_delta=V_delta_var, sin_epsilon_s=sin_epsilon_s, CL_wing=AeroClass.cL_cruise)[1]
        CL_wing_section = CL_ws(S_W=WingClass.surface, b_W=WingClass.span, n_e=3, D_star=D_star_var, sin_epsilon=sin_epsilon, V_0=const.v_cr, V_delta=V_delta_var, sin_epsilon_s=sin_epsilon_s, CL_wing=AeroClass.cL_cruise)[2]
        CL_s_section = CL_ws(S_W=WingClass.surface, b_W=WingClass.span, n_e=3, D_star=D_star_var, sin_epsilon=sin_epsilon, V_0=const.v_cr, V_delta=V_delta_var, sin_epsilon_s=sin_epsilon_s, CL_wing=AeroClass.cL_cruise)[3]

        prop_lift_var = prop_lift_thrust(T=drag_cruise, rho=rho_cr, V_0=const.v_cr, S_W=WingClass.surface, angle_of_attack=angle_of_attack_prop)

        CL_total_cruise = CL_ws_var + prop_lift_var

        downwash_angle_wing = np.sin(sin_epsilon)
        downwash_angle_prop = np.sin(sin_epsilon_s)
        average_downwash_angle = (downwash_angle_prop*diameter_propellers*3 + downwash_angle_wing*(WingClass.span-3*diameter_propellers))/WingClass.span

        AeroClass.downwash_angle = average_downwash_angle
        AeroClass.downwash_angle_wing = downwash_angle_wing
        AeroClass.downwash_angle_prop = downwash_angle_prop
        AeroClass.cL_plus_slipstream = CL_total_cruise
        #     print(CL_wing_section, CL_s_section, CL_old, prop_lift_var)
        return WingClass, EngineClass, AeroClass, mission

