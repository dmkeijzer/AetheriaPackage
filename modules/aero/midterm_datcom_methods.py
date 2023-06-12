import numpy as np


def datcom_cl_alpha(A, mach, sweep_half):
    beta = np.sqrt(1 - mach**2)
    return (2*np.pi*A)/(2 + np.sqrt(4 + ((A*beta)/0.95)**2*(1 + (np.tan(sweep_half)**2)/(beta**2))))


def flap_planform(cL_max, cL_max_flaps60, sweep_hingeline):

    delta_cl_max_base = 0.9     # from Raymer for plain flaps
    k1 = 1.0                    # from Raymer for 25% cf/c
    k2 = 1.0                    # from Raymer for 60 deg flap deflection
    k3 = 1.0                    # from Raymer for flap kinematics (1.0 for plain flaps)

    delta_cl_max_airfoil = k1*k2*k3*delta_cl_max_base
    delta_cL_max = cL_max_flaps60 - cL_max

    S_wf_over_S = delta_cL_max/(0.92 * delta_cl_max_airfoil*np.cos(sweep_hingeline))    # ratio of flap surface area over wing surface area
    return S_wf_over_S


def cL_max_flaps60(cl_max_airfoil, v_stall, v_stall_flaps60):

    cL_max = 0.9*cl_max_airfoil
    return v_stall**2/v_stall_flaps60**2 * cL_max


def cL_alpha_wingfus(cL_alpha_wing, diameter_fus, wing_span):

    x = diameter_fus/wing_span
    K_wf = 1 + 0.025*x - 0.25*x**2
    return K_wf*cL_alpha_wing
