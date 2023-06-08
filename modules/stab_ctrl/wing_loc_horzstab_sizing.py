import json
import sys
import pathlib as pl
import os
import numpy as np
import pandas as pd

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))
import matplotlib.pyplot as plt
# from input.data_structures import *
from scripts.stab_ctrl.potato_plot import J1loading



# WingClass = Wing()
# FuseClass = Fuselage()
# HorTailClass = HorTail()
#
#
# WingClass.load()
# FuseClass.load()
# HorTailClass.load()
#
#
#
# depsda = HorTailClass.downwash
# MAC = WingClass.chord_mac
# Vh_V_2 = 0.95 #From SEAD, V-tail somewhat similar to fin-mounted stabiliser
# S = WingClass.surface
# b_f = FuseClass.width_fuselage
# h_f = FuseClass.height_fuselage
# b = WingClass.span
# Lambdac4 = WingClass.quarterchord_sweep
# taper = WingClass.taper
# A_h = HorTailClass.aspect_ratio
# eta = 0.95
# Mach = WingClass.mach_cruise
# Lambdah2 = 0 #Assumed
# CLaw = WingClass.cL_alpha
# c_root = WingClass.chord_root
# l_f = FuseClass.length_fuselage
# CL0_approach = WingClass.cL_alpha0_approach
# Cm_ac_wing = WingClass.cm
# CLAh_approach= WingClass.cL_approach #Assumes fuselage contribution negligible
# x_lemac_x_rootc = WingClass.X_lemac
# SM = 0.05 #Stability margin, standard

# CLaAh = CLaw*(1+2.15*b_f/b)*(S-b_f*c_root)/S + np.pi * b_f ** 2 / (2 * S)
#
# x_ac_stab_nacelles_bar = 0 # Missing nacelles data/counteracting effect for our design
# x_ac_stab_wing_bar = 0.24 # From graph from Torenbeek
# x_ac_stab_fus1_bar = -(1.8 * b_f * h_f * l_fn)/(CLaAh * S * MAC)
# x_ac_stab_fus2_bar = (0.273 * b_f * S * (b-b_f) * np.tan(Lambdac4))/((1+taper)*MAC**2*(b+2.15 * b_f))
#
# x_ac_stab_bar = x_ac_stab_wing_bar + x_ac_stab_fus1_bar + x_ac_stab_fus2_bar + x_ac_stab_nacelles_bar
#
# beta = np.sqrt(1-Mach**2)
#
# CLah = 2 * np.pi * A_h/(2+np.sqrt(4+(A_h*beta/eta)**2)*(1+(np.tan(Lambdah2))**2/beta**2))
#
# m_stab = 1 / ((CLah / CLaAh)*(1-depsda)*(l_h/MAC)*Vh_V_2)
# q_stab = (x_ac_stab_bar - SM)/((CLah / CLaAh)*(1-depsda)*(l_h/MAC)*Vh_V_2)
# #ShS_stab = m_stab * x_aft_stab_bar + q_stab
#
#
#
# CLh_approach = -0.35 * A_h ** (1/3)
# Cm_ac_flaps = None
# Cm_ac_fuselage = -1.8 * (1 - 2.5*b_f /l_f) * np.pi * b_f * h_f * l_f * CL0 / (4*S*MAC * CLaAh) #CLaAh for ctrl is different than for stab if cruise in compressible flow
# Cm_ac_nacelles = 0 #Assumed/missing data on nacelles
# Cm_ac = Cm_ac_wing + Cm_ac_flaps + Cm_ac_fuselage + Cm_ac_nacelles
#
# m_ctrl = 1 / ((CLh_approach/CLAh_approach)* (l_h/MAC)*Vh_V_2)
# q_ctrl = ((Cm_ac/CLAh_approach) - x_ac_stab_bar) / ((CLh_approach/CLAh_approach)* (l_h/MAC)*Vh_V_2) #x_ac_bar for ctrl is different than for stab if cruise in compressible flow
# #ShS_ctrl = m_ctrl * x_front_ctrl_bar + q_ctrl

def stabcg(ShS, x_ac, CLah, CLaAh, depsda, lh, c, VhV2, SM):
    x_cg = x_ac + (CLah/CLaAh)*(1-depsda)*ShS*(lh/c)*VhV2 - SM
    return x_cg
def ctrlcg(ShS, x_ac, Cmac, CLAh, CLh, lh, c, VhV2):
    x_cg = x_ac - Cmac/CLAh + CLh*lh*ShS * VhV2 / (c * CLAh)
    return x_cg

def CLaAhcalc(CLaw, b_f, b, S, c_root):
    CLaAh = CLaw * (1 + 2.15 * b_f / b) * (S - b_f * c_root) / S + np.pi * b_f ** 2 / (2 * S)
    return CLaAh

def x_ac_fus_1calc(b_f, h_f, l_fn, CLaAh, S, MAC):
    x_ac_stab_fus1_bar = -(1.8 * b_f * h_f * l_fn) / (CLaAh * S * MAC)
    return x_ac_stab_fus1_bar

def x_ac_fus_2calc(b_f, S, b, Lambdac4, taper, MAC):
    x_ac_stab_fus2_bar = (0.273 * b_f * S * (b - b_f) * np.tan(Lambdac4)) / ((1 + taper) * MAC ** 2 * b*(b + 2.15 * b_f))
    return x_ac_stab_fus2_bar

def betacalc(M):
    return np.sqrt(1-M**2)

def CLahcalc(A_h, beta, eta, Lambdah2):
    CLah = 2 * np.pi * A_h / (2 + np.sqrt(4 + (A_h * beta / eta) ** 2 * (1 + (np.tan(Lambdah2)) ** 2 / beta ** 2)))
    return CLah

def stab_formula_coefs(CLah, CLaAh, depsda, l_h, MAC, Vh_V_2, x_ac_stab_bar, SM):
    m_stab = 1 / ((CLah / CLaAh) * (1 - depsda) * (l_h / MAC) * Vh_V_2)
    q_stab = (x_ac_stab_bar - SM) / ((CLah / CLaAh) * (1 - depsda) * (l_h / MAC) * Vh_V_2)
    return m_stab, q_stab

def CLh_approach_estimate(A_h):
    CLh_approach = -0.35 * A_h ** (1 / 3)
    return CLh_approach

def cmac_fuselage_contr(b_f, l_f, h_f, CL0_approach, S, MAC, CLaAh):
    Cm_ac_fuselage = -1.8 * (1 - 2.5 * b_f / l_f) * np.pi * b_f * h_f * l_f * CL0_approach / (4 * S * MAC * CLaAh)
    return Cm_ac_fuselage

def ctrl_formula_coefs(CLh_approach, CLAh_approach, l_h, MAC, Vh_V_2, Cm_ac, x_ac_stab_bar):
    m_ctrl = 1 / ((CLh_approach / CLAh_approach) * (l_h / MAC) * Vh_V_2)
    q_ctrl = ((Cm_ac / CLAh_approach) - x_ac_stab_bar) / ((CLh_approach / CLAh_approach) * (l_h / MAC) * Vh_V_2)
    return m_ctrl, q_ctrl

def wing_location_horizontalstab_size(WingClass, FuseClass, HorTailClass, plot=False, CLh_approach = None):
    log_final = np.zeros((0,6))
    depsda = HorTailClass.downwash
    MAC = WingClass.chord_mac
    Vh_V_2 = 0.95  # From SEAD, V-tail somewhat similar to fin-mounted stabiliser
    S = WingClass.surface
    b_f = FuseClass.width_fuselage
    h_f = FuseClass.height_fuselage
    b = WingClass.span
    Lambdac4 = WingClass.quarterchord_sweep
    taper = WingClass.taper
    A_h = HorTailClass.aspect_ratio
    eta = 0.95
    Mach = WingClass.mach_cruise
    Lambdah2 = 0  # Assumed
    CLaw = WingClass.cL_alpha
    c_root = WingClass.chord_root
    l_f = FuseClass.length_fuselage
    CL0_approach = WingClass.cL_alpha0_approach
    Cm_ac_wing = WingClass.cm_ac
    CLAh_approach = WingClass.cL_approach  # Assumes fuselage contribution negligible
    x_lemac_x_rootc = WingClass.X_lemac
    SM = 0.05  # Stability margin, standard

    if CLh_approach == None:
        CLh_approach = CLh_approach_estimate(A_h)

    for wing_loc in np.linspace(0.3, 0.65, np.size(np.arange(-1,2,0.002))):
        log_stab = np.zeros((0, 2))
        log_ctrl = np.zeros((0, 2))
        x_ac_stab_wing_bar = 0.24  # From graph from Torenbeek
        l_h = l_f * (1-wing_loc)
        l_fn = wing_loc * l_f - x_ac_stab_wing_bar * MAC - x_lemac_x_rootc
        cglims = J1loading(wing_loc * l_f, l_f)[0]
        frontcgexc = (cglims["frontcg"] - wing_loc * l_f + x_ac_stab_wing_bar * MAC)/ MAC
        rearcgexc = (cglims["rearcg"] - wing_loc * l_f + x_ac_stab_wing_bar * MAC)/ MAC

        CLaAh = CLaAhcalc(CLaw, b_f, b, S, c_root)

        x_ac_stab_nacelles_bar = 0  # Missing nacelles data/counteracting effect for our design
        x_ac_stab_fus1_bar = x_ac_fus_1calc(b_f, h_f, l_fn, CLaAh, S, MAC)
        x_ac_stab_fus2_bar = x_ac_fus_2calc(b_f, S, b, Lambdac4, taper, MAC)

        x_ac_stab_bar = x_ac_stab_wing_bar + x_ac_stab_fus1_bar + x_ac_stab_fus2_bar + x_ac_stab_nacelles_bar

        beta = betacalc(Mach)

        CLah = CLahcalc(A_h, beta, eta, Lambdah2)

        m_stab, q_stab = stab_formula_coefs(CLah, CLaAh, depsda, l_h, MAC, Vh_V_2, x_ac_stab_bar, SM)

        Cm_ac_flaps = -0.1825#From delta CL0
        Cm_ac_fuselage = cmac_fuselage_contr(b_f, l_f, h_f, CL0_approach, S, MAC, CLaAh)  # CLaAh for ctrl is different than for stab if cruise in compressible flow
        Cm_ac_nacelles = 0  # Assumed/missing data on nacelles
        Cm_ac = Cm_ac_wing + Cm_ac_flaps + Cm_ac_fuselage + Cm_ac_nacelles

        m_ctrl, q_ctrl = ctrl_formula_coefs(CLh_approach, CLAh_approach, l_h, MAC, Vh_V_2, Cm_ac, x_ac_stab_bar) # x_ac_bar for ctrl is different than for stab if cruise in compressible flow

        #log_cgexc = np.vstack((log_cgexc, np.array([cglims["frontcg"], cglims["rearcg"], wing_loc])))
        for x_aft_stab_bar in np.arange(-1,2,0.002):
            ShS_stab = m_stab * x_aft_stab_bar + q_stab
            log_stab = np.vstack((log_stab, np.array([x_aft_stab_bar, ShS_stab])))
        for x_front_ctrl_bar in np.arange(-1,2,0.002):
            ShS_ctrl = m_ctrl * x_front_ctrl_bar + q_ctrl
            log_ctrl = np.vstack((log_ctrl, np.array([x_front_ctrl_bar, ShS_ctrl])))

        log_stab = log_stab[np.where(log_stab[:,0] > rearcgexc)[0],:]
        log_ctrl = log_ctrl[np.where(log_ctrl[:,0] < frontcgexc)[0],:]
        if np.size(log_stab) == 0 or np.size(log_ctrl) == 0:
            continue
        ShSmin = max(np.min(log_stab[:,1]), np.min(log_ctrl[:,1]))
        log_final = np.vstack((log_final, np.array([wing_loc, ShSmin, frontcgexc, rearcgexc, ctrlcg(ShSmin, x_ac_stab_bar, Cm_ac,CLAh_approach, CLh_approach, l_h, MAC, Vh_V_2), stabcg(ShSmin, x_ac_stab_bar, CLah, CLaAh, depsda, l_h,MAC, Vh_V_2, SM)])))
    if plot:
        plt.plot(log_final[:,0], log_final[:,1])
        plt.show()


    WingClass.x_lewing = log_final[np.where(log_final[:,1] == np.min(log_final[:,1]))[0], 0:2][0][0] *FuseClass.length_fuselage - 0.24 * WingClass.chord_mac - WingClass.X_lemac
    HorTailClass.hortailsurf_wingsurf =  log_final[np.where(log_final[:,1] == np.min(log_final[:,1]))[0], 0:2][0][1]
    HorTailClass.surface = log_final[np.where(log_final[:,1] == np.min(log_final[:,1]))[0], 0:2][0][1] * WingClass.surface
    #Update all values
    WingClass.dump()
    #FuseClass.dump()
    HorTailClass.dump()

    return log_final[np.where(log_final[:,1] == np.min(log_final[:,1]))[0], 0:2], log_final[np.where(log_final[:,1] == np.min(log_final[:,1]))[0], -1] - x_ac_stab_bar

if __name__ == "__main__":
    from input.data_structures import *
    WingClass = Wing()
    FuseClass = Fuselage()
    HorTailClass = HorTail()
    WingClass.load()
    FuseClass.load()
    HorTailClass.load()
    wing_location_horizontalstab_size(WingClass, FuseClass, HorTailClass, plot = True)