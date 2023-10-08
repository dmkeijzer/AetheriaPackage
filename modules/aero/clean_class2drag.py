import numpy as np
import os
import json
import input.GeneralConstants  as const
import sys
import pathlib as pl
sys.path.append(str(list(pl.Path(__file__).parents)[2]))

os.chdir(str(list(pl.Path(__file__).parents)[2]))


# Define functions for all components of Class 2 drag estimations
def Reynolds(rho_cruise, V_cruise, mac, mu, k):
    """Returns the Reynold number

    :param rho_cruise: Density at cruise altitude
    :type rho_cruise: _type_
    :param V_cruise: Cruise velocity
    :type V_cruise: _type_
    :param mac: Mean aerodynamic chord
    :type mac: float
    :param mu: dynamic viscosity
    :type mu: _type_
    :param k: surface factor in the order of 1e-5 and 1e-7
    :type k: float
    :return: Reyolds number
    :rtype: _type_
    """
    return min((rho_cruise * V_cruise * mac / mu), 38.21 * (mac / k) ** 1.053)

def deps_da(cL_alpha, AR):
    return (2*cL_alpha)/(np.pi*AR)

def Mach_cruise(V_cruise, gamma, R, T_cruise):
    """_summary_

    :param V_cruise: Cruise speed [m/s]
    :type V_cruise: _type_
    :param gamma: _description_
    :type gamma: _type_
    :param R: _description_
    :type R: _type_
    :param T_cruise: _description_
    :type T_cruise: _type_
    :return: _description_
    :rtype: _type_
    """
    a = np.sqrt(gamma * R * T_cruise)
    return V_cruise / a


def FF_fus(l, d):
    """_summary_

    :param l: length fuselage
    :type l: _type_
    :param d: diameter fuselage
    :type d: _type_
    :return: _description_
    :rtype: _type_
    """
    f = l / d
    return 1 + 60 / (f ** 3) + f / 400

def sweep_m(sweep_le, loc_max_t, c_root, span, taper):
    """_summary_

        :param sweep_le: leading edge sweep angle
        :type sweep_le: _type_
        :param loc_max_t: location of maximum thickness of airfoil (x/c)
        :type loc_max_t: _type_
        :param c_root: root chord length
        :type c_root: _type_
        :param span: _description_
        :type span: _type_
        :param taper: _description_
        :type taper: _type_
        :return: _description_
        :rtype: _type_
        """
    return np.arctan2(np.tan(sweep_le) - (1-taper)*loc_max_t*(2*c_root) / span, 1)


def FF_wing(toc, xcm, M, sweep_m):
    """_summary_

    :param toc: thickness over chord ratio
    :type toc: _type_
    :param xcm: (x/c)m, position of maximum thickness
    :type xcm: _type_
    :param M: Mach number
    :type M: _type_
    :return: _description_
    :rtype: _type_
    """

    return (1 + 0.6 * toc / xcm + 100 * toc * 4) * (1.34 * (M * 0.18) * (np.cos(sweep_m)) * 0.28)

def FF_tail(toc_tail, xcm_tail, M, sweep_m):
    """_summary_

    :param toc_tail: thickness over chord ratio
    :type toc_tail: _type_
    :param xcm_tail: (x/c)m, position of maximum thickness
    :type xcm_tail: _type_
    :param M: Mach number
    :type M: _type_
    :return: _description_
    :rtype: _type_
    """

    return (1 + 0.6 * toc_tail / xcm_tail + 100 * toc_tail * 4) * (1.34 * (M * 0.18) * (np.cos(sweep_m)) * 0.28)

def S_wet_fus(d, l1, l2, l3):
    """_summary_

    :param d: diameter fuselage
    :type d: _type_
    :param l1: length cockpit/parabolic section
    :type l1: _type_
    :param l2: length cylindrical section
    :type l2: _type_
    :param l3: length conical section
    :type l3: 
    :return: _description_
    :rtype: _type_
    """
    return (np.pi * d / 4) * (((1 / (3 * l1 ** 2)) * ((4 * l1 ** 2 + ((d ** 2) / 4)) ** 1.5 - ((d ** 3) / 8))) - d + 4 * l2 + 2 * np.sqrt(l3 ** 2 + (d ** 2) / 4))


def CD_upsweep(u, d, S_wet_fus):
    """_summary_

    :param u: upsweep angle (rad)
    :type u: _type_
    :param d: diameter fuselage
    :type d: _type_
    :param S_wet_fus: _description_
    :type S_wet_fus: _type_
    :return: _description_
    :rtype: _type_
    """
    return 3.83 * (u ** 2.5) * np.pi * d ** 2 / (4 * S_wet_fus)


def CD_base(M, A_base, S_wet_fus):
    """_summary_

    :param M: _description_
    :type M: _type_
    :param A_base: base area fuselage
    :type A_base: _type_
    :param S: _description_
    :type S: _type_
    :return: _description_
    :rtype: _type_
    """
    return (0.139 + 0.419 * (M - 0.161) ** 2) * A_base / S_wet_fus


def C_fe_fus(frac_lam_fus, Reynolds, M):
    """_summary_

    :param frac_lam_fus: fraction laminar flow fuselage
    :type frac_lam_fus: _type_
    :param Reynolds: _description_
    :type Reynolds: _type_
    :return: _description_
    :rtype: _type_
    """
    C_f_lam = 1.328 / np.sqrt(Reynolds)
    C_f_turb = 0.455 / (((np.log10(Reynolds)) ** 2.58)
                        * (1 + 0.144 * M ** 2) ** 0.65)
    return frac_lam_fus * C_f_lam + (1 - frac_lam_fus) * C_f_turb


def C_fe_wing(frac_lam_wing, Reynolds, M):
    """_summary_

    :param frac_lam_wing: fraction laminar flow wing
    :type frac_lam_wing: _type_
    :param Reynolds: _description_
    :type Reynolds: _type_
    :return: _description_
    :rtype: _type_
    """
    C_f_lam = 1.328 / np.sqrt(Reynolds)
    C_f_turb = 0.455 / (((np.log10(Reynolds)) ** 2.58)
                        * (1 + 0.144 * M ** 2) ** 0.65)

    return frac_lam_wing * C_f_lam + (1 - frac_lam_wing) * C_f_turb


def CD_fus(C_fe_fus, FF_fus, S_wet_fus):
    """_summary_

    :param C_fe_fus: skin friction coefficient fuselage
    :type C_fe_fus: _type_
    :param FF_fus: form factor fuselage
    :type FF_fus: _type_
    :param S_wet_fus: _description_
    :type S_wet_fus: _type_
    :return: _description_
    :rtype: _type_
    """
    IF_fus = 1.0        # From WIGEON script
    return C_fe_fus * FF_fus * IF_fus * S_wet_fus


def CD_wing(C_fe_wing, FF_wing, S_wet_wing, S):
    """_summary_

    :param C_fe_wing: skin friction coefficient wing
    :type C_fe_wing: _type_
    :param FF_wing: form factor wing
    :type FF_wing: _type_
    :param S_wet_wing: _description_
    :type S_wet_wing: _type_
    :return: _description_
    :rtype: _type_
    """
    IF_wing = 1.1      # From WIGEON script

    CD_wing = max(float(C_fe_wing * FF_wing * IF_wing * S_wet_wing), 0.007)
    return CD_wing


def CD_tail(C_fe_wing, FF_tail, S_wet_tail):

    IF_tail = 1.0
    CD_tail = max(float(C_fe_wing * FF_tail * IF_tail * S_wet_tail), 0.005)
    return CD_tail


def CD0(S, S_tail, S_fus, CD_fus, CD_wing, CD_upsweep, CD_base, CD_tail, CD_flaps):
    """_summary_

    :param S: wing area
    :type S: _type_
    :param CD_fus: _description_
    :type CD_fus: _type_
    :param CD_wing: _description_
    :type CD_wing: _type_
    :param CD_upsweep: _description_
    :type CD_upsweep: _type_
    :param CD_base: _description_
    :type CD_base: _type_
    :return: _description_
    :rtype: _type_
    """

    leakage_factor = 1.075  # accounts for leakage from propellers etc.

    return ((CD_wing / S) + (CD_tail / S_tail) + (CD_fus / S_fus)) * leakage_factor + CD_upsweep + CD_base + CD_flaps


def CDi(CL, A, e):
    """_summary_

    :param CL: _description_
    :type CL: _type_
    :param A: _description_
    :type A: _type_
    :param e: _description_
    :type e: _type_
    :return: _description_
    :rtype: _type_
    """

    CDi = max(float(CL**2 / (np.pi * A * e)), 0.007)
    return CDi


def CD_flaps(angle_flap_deg):
    F_flap = 0.0144         # for plain flaps
    cf_c = 0.25             # standard value
    S_flap_S_ref = 0.501    # from Raymer's methods

    return F_flap*cf_c*S_flap_S_ref*(angle_flap_deg-10)

def CD(CD0, CDi):
    """_summary_

    :param CD0: _description_
    :type CD0: _type_
    :param CDi: _description_
    :type CDi: _type_
    :return: _description_
    :rtype: _type_
    """
    return CD0 + CDi


def lift_over_drag(CL_output, CD_output):
    """_summary_

    :param CL_output: _description_
    :type CL_output: _type_
    :param CD_output: _description_
    :type CD_output: _type_
    :return: _description_
    :rtype: _type_
    """
    return CL_output / CD_output


def Oswald_eff(A):
    """_summary_

    :param A: aspect ratio
    :type A: _type_
    :return: _description_
    :rtype: _type_
    """
    return 1.78 * (1 - 0.045 * A**0.68) - 0.64


def Oswald_eff_tandem(b1, b2, h):
    """_summary_

    :param b1: span front wing
    :type b1: _type_
    :param b2: span aft wing
    :type b2: _type_
    :param h: height difference between wingss
    :type h: _type_
    :param b: _description_
    :type b: _type_
    :param Oswald_eff: _description_
    :type Oswald_eff: _type_
    :return: _description_
    :rtype: _type_
    """
    b_avg = (b1 + b2) / 2
    factor = 0.5 + (1 - 0.66 * (h / b_avg)) / (2.1 + 7.4 * (h / b_avg))
    return factor * 0.8



def integrated_drag_estimation(WingClass, FuselageClass, VTailClass, AeroClass):

    # General flight variables
    re_var = Reynolds(const.rho_cr, const.v_cr, WingClass.chord_mac, const.mhu, const.k)
    M_var = Mach_cruise(const.v_cr, const.gamma, const.R, const.t_cr)
    AeroClass.e = Oswald_eff(WingClass.aspect_ratio)


    # Writing to Aeroclass
    AeroClass.deps_da = deps_da(AeroClass.cL_alpha, WingClass.aspect_ratio)

    # Form factors
    FF_fus_var = FF_fus(FuselageClass.length_fuselage, FuselageClass.diameter_fuselage)
    FF_wing_var = FF_wing(const.toc, const.xcm, M_var, sweep_m(WingClass.sweep_LE, const.xcm, WingClass.chord_root, WingClass.span, WingClass.taper))
    FF_tail_var = FF_tail(const.toc_tail, const.xcm_tail, M_var, VTailClass.quarterchord_sweep)

    # Wetted area
    S_wet_fus_var = S_wet_fus(FuselageClass.diameter_fuselage, FuselageClass.length_cockpit, FuselageClass.length_cabin, FuselageClass.length_tail)
    S_wet_wing_var = 2 * WingClass.surface  # from ADSEE slides
    S_wet_tail_var = 2 * VTailClass.surface

    # Miscellaneous drag
    AeroClass.cd_upsweep = CD_upsweep(FuselageClass.upsweep, FuselageClass.diameter_fuselage, S_wet_fus_var)
    AeroClass.cd_base = CD_base(M_var, const.A_base, S_wet_fus_var)

    # Skin friction coefficienct
    C_fe_fus_var = C_fe_fus(const.frac_lam_fus, re_var, M_var)
    C_fe_wing_var = C_fe_wing(const.frac_lam_wing, re_var, M_var)
    C_fe_tail_var = C_fe_wing(const.frac_lam_wing, re_var, M_var)

    # Total cd
    CD_fus_var = CD_fus(C_fe_fus_var, FF_fus_var, S_wet_fus_var)
    CD_wing_var = CD_wing(C_fe_wing_var, FF_wing_var, S_wet_wing_var, WingClass.surface)
    CD_tail_var = CD_tail(C_fe_tail_var, FF_tail_var, S_wet_tail_var)
    AeroClass.cd0_cruise = CD0(WingClass.surface, VTailClass.surface, FuselageClass.length_fuselage*FuselageClass.width_fuselage_outer, CD_fus_var, CD_wing_var,AeroClass.cd_upsweep, AeroClass.cd_base, CD_tail_var, CD_flaps=0)

    # Lift times S
    # cL_tail_times_Sh = VTailClass.cL_cruise * HortailClass.surface #TODO Idk if this is correct or even necessary
    # cL_wing_times_S = AeroClass.cL_plus_slipstream*WingClass.surface

    # total_cL = (cL_wing_times_S + cL_tail_times_Sh) / (WingClass.surface + HortailClass.surface)


    # Summation and L/D calculation
    CDi_var = CDi(AeroClass.cL_cruise, WingClass.aspect_ratio, AeroClass.e)
    AeroClass.cd_cruise = CD(AeroClass.cd0_cruise, CDi_var)
    AeroClass.ld_cruise = lift_over_drag(AeroClass.cL_cruise,AeroClass.cd_cruise)




    # ------------------------ DRAG DURING STALL -------------- 
    # General flight variables
    re_var = Reynolds(const.rho_sl, const.v_stall, WingClass.chord_mac, const.mhu, const.k)
    M_var = Mach_cruise(const.v_stall, const.gamma, const.R, const.t_stall)



    # Form factor
    FF_fus_var = FF_fus(FuselageClass.length_fuselage, FuselageClass.diameter_fuselage)
    FF_wing_var = FF_wing(const.toc, const.xcm, M_var, sweep_m(WingClass.sweep_LE, const.xcm, WingClass.chord_root, WingClass.span, WingClass.taper))
    FF_tail_var = FF_tail(const.toc_tail, const.xcm_tail, M_var, VTailClass.quarterchord_sweep)


    # Wetted area
    S_wet_fus_var = S_wet_fus(FuselageClass.diameter_fuselage, FuselageClass.length_cockpit, FuselageClass.length_cabin, FuselageClass.length_tail)
    S_wet_wing_var = 2 * WingClass.surface  # from ADSEE slides
    S_wet_tail_var = 2 * VTailClass.surface

    # Miscellaneous drag
    CD_upsweep_var = CD_upsweep(FuselageClass.upsweep, FuselageClass.diameter_fuselage, S_wet_fus_var)
    CD_base_var = CD_base(M_var, const.A_base, S_wet_fus_var)

    # Skin friction coefficienct
    C_fe_fus_var = C_fe_fus(const.frac_lam_fus, re_var, M_var)
    C_fe_wing_var = C_fe_wing(const.frac_lam_wing, re_var, M_var)
    C_fe_tail_var = C_fe_wing(const.frac_lam_wing, re_var, M_var)

    # Total cd
    CD_fus_var = CD_fus(C_fe_fus_var, FF_fus_var, S_wet_fus_var)
    CD_wing_var = CD_wing(C_fe_wing_var, FF_wing_var, S_wet_wing_var, WingClass.surface)
    CD_tail_var = CD_tail(C_fe_tail_var, FF_tail_var, S_wet_tail_var)
    CD_flaps_var = CD_flaps(60)
    AeroClass.cd0_stall = CD0(WingClass.surface, VTailClass.surface, FuselageClass.length_fuselage*FuselageClass.width_fuselage_outer, CD_fus_var, CD_wing_var, CD_upsweep_var, CD_base_var, CD_tail_var, CD_flaps_var)

    # Total cl
    # total_cL_stall = AeroClass.cL_plus_slipstream_stall

    # Summation and L/D calculation
    CDi_var = CDi(AeroClass.cL_max_flaps60, WingClass.aspect_ratio, AeroClass.e)
    AeroClass.cd_stall = CD(AeroClass.cd0_stall, CDi_var)
    AeroClass.ld_stall = lift_over_drag(AeroClass.cL_max_flaps60, AeroClass.cd_stall)

    return WingClass, FuselageClass, VTailClass, AeroClass


