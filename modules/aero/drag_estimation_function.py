import sys
import pathlib as pl
import os
import json
import numpy as np

sys.path.append(str(list(pl.Path(__file__).parents)[3]))
os.chdir(str(list(pl.Path(__file__).parents)[3]))

# Import from modules and input folder
import input.data_structures.GeneralConstants  as const
from modules.aero.clean_class2drag import *
from modules.aero.midterm_datcom_methods import *
from input.data_structures import *
from input.data_structures.ISA_tool import ISA
from input.data_structures.vee_tail import VeeTail
# AeroClass = Aero()
# FuselageClass = Fuselage()
# VTailClass = VeeTail()
# HorTailClass = HorTail()
# WingClass = Wing()
# AeroClass.load()
# FuselageClass.load()
# VTailClass.load()
# HorTailClass.load()
# WingClass.load()


os.chdir(str(list(pl.Path(__file__).parents)[2]))
# import CL_cruise from json files


# atm = ISA(const.h_cruise)
# t_cr = atm.temperature()
# rho_cr = atm.density()
# mhu = atm.viscosity_dyn()

def final_drag_estimation(WingClass, FuselageClass, VTailClass, AeroClass, HortailClass):
    mac = WingClass.chord_mac

    # General flight variables
    re_var = Reynolds(const.rho_cr, const.v_cr, mac, const.mhu, const.k)
    M_var = Mach_cruise(const.v_cr, const.gamma, const.R, const.t_cr)
    Oswald_eff_var = Oswald_eff(WingClass.aspectratio)


    # Writing to Aeroclass
    AeroClass.e = Oswald_eff_var
    AeroClass.deps_da = deps_da(AeroClass.cL_alpha, WingClass.aspectratio)

    # Form factor
    FF_fus_var = FF_fus(FuselageClass.length_fuselage, FuselageClass.diameter_fuselage)
    FF_wing_var = FF_wing(const.toc, const.xcm, M_var, sweep_m(WingClass.sweep_LE, const.xcm, WingClass.chord_root, WingClass.span, WingClass.taper))
    FF_tail_var = FF_tail(const.toc_tail, const.xcm_tail, M_var, HortailClass.sweep_halfchord_h)

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
    CD0_var = CD0(WingClass.surface, VTailClass.surface, FuselageClass.length_fuselage*FuselageClass.width_fuselage_outer, CD_fus_var, CD_wing_var, CD_upsweep_var, CD_base_var, CD_tail_var, CD_flaps=0)

    # Lift times S
    cL_tail_times_Sh = VTailClass.CL_cruise * HortailClass.surface
    cL_wing_times_S = AeroClass.cL_plus_slipstream*WingClass.surface

    total_cL = (cL_wing_times_S + cL_tail_times_Sh) / (WingClass.surface + HortailClass.surface)


    # Summation and L/D calculation
    CDi_var = CDi(AeroClass.cL_cruise, WingClass.aspectratio, AeroClass.e)
    CD_var = CD(CD0_var, CDi_var)
    lift_over_drag_var = lift_over_drag(total_cL, CD_var)

    print("CD0_wing", CD_wing_var / WingClass.surface)
    print("CD cruise", CD_var)
    print("CL cruise", total_cL)
    print("L/D cruise", lift_over_drag_var)

    # Writing to JSON file
    AeroClass.ld_cruise = lift_over_drag_var
    AeroClass.cd_cruise = CD_var
    AeroClass.cd0_cruise = CD0_var
    AeroClass.cd_upsweep = CD_upsweep_var
    AeroClass.cd_base = CD_base_var


    # ------------------------ DRAG DURING STALL -------------- 
    # General flight variables
    re_var = Reynolds(rho_sl, const.v_stall, mac, const.mhu, const.k)
    M_var = Mach_cruise(const.v_stall, const.gamma, const.R, t_stall)
    Oswald_eff_var = Oswald_eff(WingClass.aspectratio)


    # Writing to Class
    AeroClass.e = Oswald_eff_var
    AeroClass.deps_da = 0.1

    # Form factor
    FF_fus_var = FF_fus(FuselageClass.length_fuselage, FuselageClass.diameter_fuselage)
    FF_wing_var = FF_wing(const.toc, const.xcm, M_var, sweep_m(WingClass.sweep_LE, const.xcm, WingClass.chord_root, WingClass.span, WingClass.taper))
    FF_tail_var = FF_tail(const.toc_tail, const.xcm_tail, M_var, HorTailClass.sweep_halfchord_h)


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
    CD0_var = CD0(WingClass.surface, VTailClass.surface, FuselageClass.length_fuselage*FuselageClass.width_fuselage_outer, CD_fus_var, CD_wing_var, CD_upsweep_var, CD_base_var, CD_tail_var, CD_flaps_var)

    # Total cl
    total_cL_stall = AeroClass.cL_plus_slipstream_stall

    # Summation and L/D calculation
    CDi_var = CDi(AeroClass.cL_max_flaps60, WingClass.aspectratio, AeroClass.e)
    CD_var = CD(CD0_var, CDi_var)
    lift_over_drag_var = lift_over_drag(total_cL_stall, CD_var)


    # Writing to classes file
    AeroClass.ld_stall = lift_over_drag_var
    AeroClass.cd_stall = CD_var
    AeroClass.cd0_stall = CD0_var
    AeroClass.mach_stall = M_var

    AeroClass.dump()
    return WingClass, FuselageClass, VTailClass, AeroClass, HortailClass


