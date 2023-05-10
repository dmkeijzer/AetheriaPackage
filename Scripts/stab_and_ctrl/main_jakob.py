import numpy as np
from stab_and_ctrl.Scissor_Plots import Wing_placement_sizing
from stab_and_ctrl.Vertical_tail_sizing import VT_sizing
from stab_and_ctrl.loading_diagram import CgCalculator
from stab_and_ctrl.landing_gear_placement import LandingGearCalc
from stab_and_ctrl.hover_controllabilty import HoverControlCalcTandem
import constants as const
from matplotlib import pyplot as plt

# example values based on inputs_config_1.json
W = 3652.352770706565*9.80665
h = 305
lfus = 7.2
hfus = 1.705
wfus = 1.38
xcg = 1.680
V0 = 52
Vstall = 40
M0 = V0 / np.sqrt(const.gamma * const.R * 288.15)
CD0 = 0.03254
theta0 = 0
CLfwd = 1.781
CLrear = 1.737
CLdesfwd = 0.82
CLdesrear = 0.82
Clafwd = 6.1879
Clarear = Clafwd
Cmacfwd = -0.0645
Cmacrear = -0.0645
Sfwd = 5.25
Srear = 5.25
Afwd = 10
Arear = 10
Gamma = 0
Lambda_c4_fwd = 0
Lambda_c4_rear = 0
cfwd = 0.65
crear = 0.65
bfwd = np.sqrt(Sfwd * Afwd)
brear = np.sqrt(Srear * Arear)
efwd = 0.958
erear = 0.958
taper = 0.45
n_rot_f = 6
n_rot_r = 6
rot_y_range_f = [0.5 * bfwd * 0.1, 0.5 * bfwd * 0.9]
rot_y_range_r = [0.5 * brear * 0.1, 0.5 * brear * 0.9]
K = 4959.86
ku = 0.1
n_failures = 1

mbat = 400
mwing = 400
mengine = 25
mtom = W / 9.80665
mcargo = 4 * 7
mpax = 88
mpil = 88
mfus = mtom - mbat - 2 * mwing - (n_rot_f + n_rot_r) * mengine - mcargo - 4 * mpax - mpil

x_pil = 1.9
x_pax = [2.9, 4.3]
x_cargo = 5.15
x_bat = 2.9
x_fus = 2.6

y_pax = 0.3

z_fus = 0.7
z_bat = 0.4
z_cargo = 0.9
z_pax = 0.9
z_pil = 0.9
z_wf = 0.5
z_wr = 1.4

max_tw = 5
x_ng_min = 0.7
y_max_rotor = bfwd * 0.9
z_rotor_line_root = z_wf + 0.1
rotor_rad = 0.5
fus_back_bottom = [x_cargo, 0]
fus_back_top = [lfus, hfus]
x_cg_margin = 0.05
theta = np.deg2rad(15)
phi = np.deg2rad(7)
psi = np.deg2rad(55)
min_ng_load_frac = 0.08

Pbr_cruise_pe = 110024/1.2 * 0.9 /16
elevator_effect = 1.4

# position parameters for the wing
d = 0
dy = 0
Sfwd_Srear = Sfwd/Srear

done = False

while not done:
    x_wf = cfwd / 4 + d
    x_wr = lfus - 3 / 4 * crear

    cgcalc = CgCalculator(mwing + mengine * n_rot_f, mwing + mengine * n_rot_r,
                          mfus, mbat, mcargo, mpax, mpil, [x_fus, 0, 0.5],
                          [x_bat, 0, z_bat], [x_cargo, 0, z_cargo],
                          [[x_pax[0], -y_pax, z_pax], [x_pax[0], y_pax, z_pax],
                           [x_pax[1], -y_pax, z_pax], [x_pax[1], y_pax, z_pax]],
                          [x_pil, 0, z_pil])
    x_range, y_range, z_range = cgcalc.calc_cg_range([x_wf, z_wf], [x_wr, z_wr])

    lgcalc = LandingGearCalc(max_tw, x_ng_min, y_max_rotor, Gamma,
                             z_rotor_line_root, rotor_rad, fus_back_bottom,
                             fus_back_top)
    x_ng, x_mlg, tw, h_mlg = lgcalc.optimum_placement(x_range, x_cg_margin,
                                                      z_range[1], theta, phi,
                                                      psi, min_ng_load_frac)

    if tw > max_tw:
        # TODO: do something to fix it and iterate
        print("Warning: track width too large!")

    print("nose gear x:", x_ng, "main landing gear x:", x_mlg)
    lgcalc.plot_lg(x_range, x_cg_margin, z_range[1], x_ng, x_mlg, tw, h_mlg)
    plt.show()

    x_min_test, x_max_test = 0, lfus
    dx = 0.1

    hcct = HoverControlCalcTandem(mtom, n_rot_f, n_rot_r, x_wf, x_wr,
                                  rot_y_range_f, rot_y_range_r, K, ku)
    x_min, x_max, _, _ = hcct.calc_crit_x_cg_range(x_max_test, x_max_test, 0.1,
                                                   [x_range[1], y_range[1]],
                                                   [n_failures])[0]

    if (x_min is None or x_range[0] < x_min or x_range[1] < x_min
            or x_range[0] > x_max or x_range[1] > x_max):
        # TODO: do something to fix it and iterate
        print("Warning: Uncontrollable in hover!")

    wps = Wing_placement_sizing(W,h, lfus, hfus, wfus, V0, CD0, CLfwd,
                                CLrear, CLdesfwd, CLdesrear, Clafwd, Clarear,
                                Cmacfwd, Cmacrear, Sfwd, Srear, Afwd, Arear,
                                Gamma, Lambda_c4_fwd, Lambda_c4_rear, cfwd,
                                crear, bfwd, brear, efwd, erear, taper,
                                n_rot_f, n_rot_r, rot_y_range_f, rot_y_range_r,
                                K, ku, z_range[1], d, dy, Pbr_cruise_pe)

    x_test_range = np.arange(x_min_test, x_max_test, dx)
    Sfwd_Srear_stab, Sfwd_Srear_ctrl = wps.Sr_Sfwd(np.array(x_range),
                                                   elevator_effect, d)

    # FIXME: this assumes that the CG shifts forward with increasing Swfd/Srear
    if Sfwd_Srear_stab[0] < Sfwd_Srear or Sfwd_Srear_stab[1] < Sfwd_Srear:
        # TODO: do something to fix it and iterate
        print("Warning: Statically unstable in cruise!")

    if Sfwd_Srear_ctrl[0] > Sfwd_Srear or Sfwd_Srear_ctrl[1] > Sfwd_Srear:
        # TODO: do something to fix it and iterate
        print("Warning: Uncontrollable in cruise!")


#
# vt_sizing = VT_sizing(W,h,xcg,lfus,hfus,wfus,V0,Vstall,M0,CD0,theta0,
#                       CLfwd,CLrear,CLafwd,CLarear,
#                       Cmacfwd,Cmacrear,Sfwd,Srear,Afwd,Arear,0,0,cfwd,crear,bfwd,brear,taper)
#
# elevator_effect = 1.4
# d = 0
# dx = 0.1
#
# nE = 16
# Tt0 = 4000
# yE = bfwd/2
# lv = lfus-xcg
# vt_sizing.plotting(nE,Tt0,yE,lv,br_bv=0.87,cr_cv=0.4)
# # xcg_middle = (0.2187 + 3.3439) / 2
# # wps.hover_calc.fail_rotors([0, 3, 5, 6])
# # xcgs = np.linspace(xcg_middle - 2, xcg_middle + 2, 100)
# # acais = []
# # for xcg in xcgs:
# #     acais.append((wps.hover_calc.acai([xcg, 0])))
# # plt.plot(xcgs, acais)
# # plt.axvline(xcg_middle)
# # plt.axhline(0)
# # plt.title("[0, 3, 5, 6]")
# # plt.figure()
# #
# # wps.hover_calc.fail_rotors([1, 2, 4, 7])
# # acais = []
# # for xcg in xcgs:
# #     acais.append((wps.hover_calc.acai([xcg, 0])))
# # plt.plot(xcgs, acais)
# # plt.axvline(xcg_middle)
# # plt.axhline(0)
# # plt.title("[1, 2, 4, 7]")
# # plt.show()
#
# wps.plotting(0, lfus, dx, elevator_effect, d)
