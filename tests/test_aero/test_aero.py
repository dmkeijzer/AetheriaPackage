
import pytest
import os
import sys
import pathlib as pl
import numpy as np

sys.path.append(str(list(pl.Path(__file__).parents)[1]))
os.chdir(str(list(pl.Path(__file__).parents)[1]))


from modules.aero.avl_access import get_lift_distr, get_strip_array, get_tail_lift_distr
from modules.aero.clean_class2drag  import Reynolds, Mach_cruise, FF_fus, FF_wing, S_wet_fus, CD_upsweep, CD_base, C_fe_fus, C_fe_wing, CD_fus, CD_wing, CD0, CDi, CD, lift_over_drag,Oswald_eff, Oswald_eff_tandem, integrated_drag_estimation
from modules.aero.prop_wing_interaction import *
from modules.misc_tools.ISA_tool import ISA 



def test_get_lift_distr(wing, aero):
    lift_func, results = get_lift_distr(wing, aero, plot=False, test= True)
    span_points = np.linspace(0, wing.span/2, 300)

    assert np.isclose(results["Cruise"]["Totals"]["CLtot"], aero.cL_cruise)
    assert results["Cruise"]["Totals"]["Alpha"] > 0.3  # Assert that angle of attach has a reasonable value
    assert (np.diff(np.vectorize(lift_func)(span_points)) < 0).all() # Assert that the lift only decreases towards the tip

def test_get_tail_lift_distr(wing, veetail, aero):
    lift_func, results = get_tail_lift_distr(wing, veetail, aero, plot=False, test=True)
    span_points = np.linspace(0, wing.span/2, 300)


def test_get_strip_forces(wing, aero):
    y_le_arr, cl_strip_arr= get_strip_array( wing, aero, plot= False)

    assert np.max(y_le_arr) < wing.span/2 and np.min(y_le_arr) > 0  # Make sure all coordinates are within bounds
    assert np.where(cl_strip_arr == np.max(cl_strip_arr))[0][0] > 1  # Assert maximum lift coefficient is not at the root
    assert (cl_strip_arr < aero.cL_cruise + 0.1).all() # Assert reasonalbe values for lift coefficients


def test_drag2_calculation(values_drag):
    reynolds = Reynolds(**values_drag["reynolds"])
    mach_cruise = Mach_cruise(**values_drag["mach_cruise"])
    ff_fus = FF_fus(**values_drag["ff_fus"])
    ff_wing = FF_wing(**values_drag["ff_wing"])
    s_wet_fus = S_wet_fus(**values_drag["s_wet_fus"])
    cd_upsweep = CD_upsweep(**values_drag["cd_upsweep"])
    cd_base = CD_base(**values_drag["cd_base"])
    c_fe_fus = C_fe_fus(**values_drag["c_fe_fus"])
    c_fe_wing = c_fe_wing(**values_drag["c_fe_wing"])
    cd_fus = CD_fus(**values_drag["cd_fus"])
    cd_wing = CD_wing(**values_drag["cd_wing"])
    cd0 = CD0(**values_drag["cd0"])
    cdi = CDi(**values_drag["cdi"])
    cd = CD(**values_drag["cd"])
    lift_over_drag = lift_over_drag(**values_drag["lift_over_drag"])
    oswald_eff = Oswald_eff(**values_drag["oswald_eff"])
    oswald_eff_tandem = Oswald_eff_tandem(**values_drag["oswald_eff_tandem"])


    assert np.isclose(reynolds.mass, 573.014)
    assert np.isclose(mach_cruise, 0.3)
    assert np.isclose(ff_fus.ff, 0.2)
    assert np.isclose(ff_wing.ff, 0.3)
    assert np.isclose(s_wet_fus.s_wet, 150)
    assert np.isclose(cd_upsweep.cd, 0.2)
    assert np.isclose(cd_base.cd, 0.3)
    assert np.isclose(c_fe_fus.c_fe, 0.1)
    assert np.isclose(c_fe_wing.c_fe, 0.2)
    assert np.isclose(cd_fus.cd, 0.05)
    assert np.isclose(cd_wing.cd, 0.04)
    assert np.isclose(cd0.cd0, 0.1)
    assert np.isclose(cdi.cdi, 0.16)
    assert np.isclose(cd.cd, 0.22)
    assert np.isclose(lift_over_drag.L_over_D, 15)
    assert np.isclose(oswald_eff.e, 0.84)
    assert np.isclose(oswald_eff_tandem.e, 0.83)


def test_prop_wing_interaction(values_slipstream):
    thrust_coef = C_T(**values_slipstream["C_T"])
    V_delta_test = V_delta(**values_slipstream["V_delta"])
    D_star_test = D_star(**values_slipstream["D_star"])
    effective_aspect = A_s_eff(**values_slipstream["A_s_eff"])[0]
    CL_effective = CL_effective_alpha(**values_slipstream["CL_eff"])
    angle_of_attack = alpha_s(**values_slipstream["angle_of_attack"])[1]
    i_cs = alpha_s(**values_slipstream["i_cs"])[2]
    alpha_s_test = alpha_s(**values_slipstream["alpha_s"])[0]
    sin_eps = sin_epsilon_angles(**values_slipstream["sin_epsilon"])[0]
    sin_eps_s = sin_epsilon_angles(**values_slipstream["sin_epsilon_s"])[1]
    CL_ws_test = CL_ws(**values_slipstream["CL_ws"])[1]
    prop_l_thrust = prop_lift_thrust(**values_slipstream["prop_lift_thrust"])

    assert np.isclose(thrust_coef, 0.038422)
    assert np.isclose(V_delta_test, 1.21311)
    assert np.isclose(D_star_test, 1.94365)
    assert np.isclose(effective_aspect, 4.7232)
    assert np.isclose(CL_effective, 4.1715)
    assert np.isclose(angle_of_attack, 0.0174533)
    assert np.isclose(i_cs, 2.3333333)
    assert np.isclose(alpha_s_test, 1.34976)
    assert np.isclose(sin_eps, 0.02546479)
    assert np.isclose(sin_eps_s, 0.0031777796)
    assert np.isclose(CL_ws_test, 0.804174088)
    assert np.isclose(prop_l_thrust, 0.0019203)

def test_integrated_drag(wing, fuselage, veetail,aero):
    wing_res, fuse_res, veetail_res, aero_res = integrated_drag_estimation(wing, fuselage, veetail, aero)
    assert aero_res.cd0_cruise

