import json
import sys
import pathlib as pl
import os
import numpy as np
import pandas as pd
from warnings import warn
import pdb

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

import matplotlib.pyplot as plt
from modules.stab_ctrl.loading_diagram import loading_diagram
import input.GeneralConstants as const
from modules.stab_ctrl.aetheria_stability_derivatives_edited import downwash, downwash_k


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

def stab_formula_coefs(CLah, CLaAh, depsda, l_h, MAC,Vh_V_2, x_ac_stab_bar, SM):
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


def wing_location_horizontalstab_size(WingClass, FuseClass, Aeroclass, VtailClass, AircraftClass, PowerClass, EngineClass, StabClass, A_h,  plot=False, CLh_approach = None, stepsize = 0.002, cg_shift = 0):
    CLAh_approach = Aeroclass.cL_max * 0.9 # Assumes fuselage contribution negligible

    if CLh_approach == None:
        CLh_approach = CLh_approach_estimate(A_h)

    # Initalise wing placement optimisaton
    dict_log = {
        "wing_loc_lst": [],
        "cg_front_lst": [],
        "cg_rear_lst": [],
        "Shs_min_lst": [],
        "cg_dict_marg_lst": [],
        "stab_lst": [],
        "ctrl_lst": [],
    }

    for wing_loc in np.linspace(0.3, 0.65, np.size(np.arange(-1,2,stepsize))):
        l_h = FuseClass.length_fuselage * (1-wing_loc)
        l_fn = wing_loc * FuseClass.length_fuselage - const.x_ac_stab_wing_bar * WingClass.chord_mac - WingClass.x_lemac
        depsda = downwash(downwash_k(l_fn, WingClass.span), Aeroclass.cL_alpha, WingClass.aspect_ratio) # TODO compute downwash from functions
        cg_dict, cg_dict_margin = loading_diagram(wing_loc * FuseClass.length_fuselage, FuseClass.length_fuselage, FuseClass, WingClass, VtailClass, AircraftClass, PowerClass, EngineClass )
        cg_front_bar = (cg_dict_margin["frontcg"] - wing_loc * FuseClass.length_fuselage + const.x_ac_stab_wing_bar * WingClass.chord_mac)/ WingClass.chord_mac
        cg_rear_bar = (cg_dict_margin["rearcg"] - wing_loc * FuseClass.length_fuselage + const.x_ac_stab_wing_bar * WingClass.chord_mac)/ WingClass.chord_mac
        CLaAh = CLaAhcalc(Aeroclass.cL_alpha, FuseClass.width_fuselage_outer, WingClass.span, WingClass.surface, WingClass.chord_root)

        # Computing aerodynamic centre
        x_ac_stab_fus1_bar = x_ac_fus_1calc(FuseClass.width_fuselage_outer, FuseClass.height_fuselage_outer, l_fn, CLaAh, WingClass.surface, WingClass.chord_mac)
        x_ac_stab_fus2_bar = x_ac_fus_2calc(FuseClass.width_fuselage_outer, WingClass.surface, WingClass.span, WingClass.quarterchord_sweep, WingClass.taper, WingClass.chord_mac)
        x_ac_stab_bar = const.x_ac_stab_wing_bar + x_ac_stab_fus1_bar + x_ac_stab_fus2_bar + const.x_ac_stab_nacelles_bar

        # Computing moment about the aerodynamic centre
        Cm_ac_fuselage = cmac_fuselage_contr(FuseClass.width_fuselage_outer, FuseClass.length_fuselage, FuseClass.height_fuselage_outer, Aeroclass.cL_alpha0_approach, WingClass.surface, WingClass.chord_mac, CLaAh)  # CLaAh for ctrl is different than for stab if cruise in compressible flow
        Cm_ac = Aeroclass.cm_ac + const.Cm_ac_flaps + Cm_ac_fuselage + const.Cm_ac_nacelles
        
        # computing misc variables required
        beta = betacalc(const.mach_cruise)
        CLah = CLahcalc(A_h, beta, const.eta_a_f, const.sweep_half_chord_tail)

        # Creating actually scissor plot
        cg_bar  = np.linspace(-1,2,2000)
        m_ctrl, q_ctrl = ctrl_formula_coefs(CLh_approach, CLAh_approach, l_h, WingClass.chord_mac, const.Vh_V_2, Cm_ac, x_ac_stab_bar) # x_ac_bar for ctrl is different than for stab if cruise in compressible flow
        m_stab, q_stab = stab_formula_coefs(CLah, CLaAh, depsda, l_h, WingClass.chord_mac, const.Vh_V_2, x_ac_stab_bar, const.stab_margin)
        ShS_stab = m_stab * cg_bar - q_stab
        ShS_ctrl = m_ctrl * cg_bar + q_ctrl

        # retrieving minimum tail sizing
        idx_ctrl = cg_bar == min(cg_bar, key=lambda x:abs(x - cg_front_bar))
        idx_stab = cg_bar == min(cg_bar, key=lambda x:abs(x - cg_rear_bar))
        ShSmin = max(ShS_ctrl[idx_ctrl], ShS_stab[idx_stab])[0]


        if False:

            plt.plot(cg_bar, ShS_stab, label= "stability")
            plt.plot(cg_bar, ShS_ctrl, label= "Control")
            plt.hlines(ShSmin, cg_front_bar, cg_rear_bar)
            plt.annotate(f"{wing_loc=}", (1,0))
            plt.annotate(f"{cg_front_bar=}", (1,-.1))
            plt.annotate(f"{cg_rear_bar=}", (1,-.17))
            plt.annotate(f"{ShSmin=}", (1,-.24))
            plt.legend()
            plt.grid()
            plt.show()

        # Storing results
        dict_log["wing_loc_lst"].append(wing_loc)
        dict_log["cg_front_lst"].append(cg_front_bar)
        dict_log["cg_rear_lst"].append(cg_rear_bar)
        dict_log["Shs_min_lst"].append(ShSmin)
        dict_log["cg_dict_marg_lst"].append(cg_dict_margin)
        dict_log["stab_lst"].append(ShS_stab)
        dict_log["ctrl_lst"].append(ShS_ctrl)


    # Selecting optimum design
    design_idx = np.argmin(dict_log["Shs_min_lst"])
    design_shs = dict_log["Shs_min_lst"][design_idx]
    design_wing_loc = dict_log["wing_loc_lst"][design_idx]
    design_cg_front_bar = dict_log["cg_front_lst"][design_idx]
    design_cg_rear_bar = dict_log["cg_rear_lst"][design_idx]
    design_cg_dict_margin = dict_log["cg_dict_marg_lst"][design_idx]
    design_stab_lst = dict_log["stab_lst"][design_idx]
    design_ctrl_lst = dict_log["ctrl_lst"][design_idx]

    if plot:
        fig, axs = plt.subplots(2,1)

        axs[0].plot(cg_bar, design_stab_lst, label= "stability")
        axs[0].plot(cg_bar, design_ctrl_lst, label= "Control")
        axs[0].hlines(design_shs, design_cg_front_bar, design_cg_rear_bar)
        axs[0].annotate(f"{design_wing_loc=}", (1,0))
        axs[0].annotate(f"{design_cg_front_bar=}", (1,-.3))
        axs[0].annotate(f"{design_cg_rear_bar=}", (1,-.6))
        axs[0].annotate(f"{design_shs=}", (1,-.9))
        axs[0].legend()
        axs[0].grid()


        axs[1].plot(dict_log["wing_loc_lst"], dict_log["Shs_min_lst"])
        axs[1].set_xlabel("Wing Location")
        axs[1].set_ylabel("Shs")
        axs[1].grid()
        plt.show()

    WingClass.x_lewing = design_wing_loc*FuseClass.length_fuselage - 0.24 * WingClass.chord_mac - WingClass.x_lemac
    VtailClass.virtual_hor_surface = design_shs*WingClass.surface

    return design_shs, design_wing_loc, design_cg_front_bar, design_cg_rear_bar, design_cg_dict_margin


