import numpy as np

import constants as const
import Aero_tools as at

import Drag_midterm2_2 as drag_comp
import Wing_design_midterm2 as wing_des
import Airfoil_analysis_midterm2 as airfoil

import engine_sizing_positioning_midterm2 as eng_siz
import battery_midterm2 as bat
import PropandPower.BEM as BEM
import PropandPower.Blade_plotter as bp

import performance_analysis_midterm2 as perf
import Flight_performance_final_midterm2 as energy_calc

import Vertical_tail_sizing_midterm2 as vert_tail
import Weight_midterm2 as weight
# import structures.Weight as weight

"""
Here we will run the midterm code (and part of the new code when available) to have converged values
Create copies of the midterm tools in this folder if necessary and make sure they comply with the new coding rules
Among other things, the idea is to start with a higher weight estimate, closer to the current estimated weight

Start with performance to get wing loading and such, needed for aero

The go to aero to get wing planform

From there we can size the engines

Here we can update the weight and iterate

Lastly, we can size the vertical tail

Vertical tail does not have outputs needed for other departments and it required to manually read a graph, hence it can
be left out of the optimisation and only run at the eng

"""
# --------------------- Fixed parameters and constants ------------------------
# Constants from constants.py
g0 = const.g
rho0 = const.rho_0
gamma = const.gamma
R = const.R

# Fuselage shape is fixed, so fixed variables (update in constants.py if needed)
w_fus = const.w_fuselage
l1_fus = const.l_nosecone
l2_fus = const.l_cylinder
l3_fus = const.l_tailcone
l_fus = l1_fus + l2_fus + l3_fus
h_fus = const.h_fuselage
fus_upsweep = const.upsweep


# --------------------- Initial estimates ---------------------
# Aero
CLmax = 1.46916
s1, s2 = const.s1, const.s2   # Ratio of front and back wing areas to total area
S1, S2 = 8.25, 8.25           # surface areas of wing one and two
S_tot = S1+S2                 # Total wing surface area
AR_wing = 7.5                 # Aspect ratio of a wing, not aircraft
AR_tot = AR_wing/2            # Aspect ratio of aircraft

# Wingtips
# S_wt = 0    # Surface of the winglets
h_wt_1 = 0.5  # Height of front wingtips
h_wt_2 = 0.5  # Height of back wingtips


# Performance
h_cr = 1000
V_cr = 52.87
C_L_cr = 0.8
V_stall = 40
V_max = 100
n_turn = 2
ROC = 10
ROC_hover = 2


# Propulsion
n_prop = 12                 # Number of engines [-]
disk_load = 250             # [kg/m^2]
clearance_fus_prop = 0.3    # Horizontal separation between the fuselage and the first propeller [m]
clearance_prop_prop = 0.3   # Horizontal separation between propellers [m]
xi_0 = 0.1                  # r/R ratio of hub diameter to out diameters [-]
m_bat = 800                 # Initial estimate for battery mass [kg]


# Structures
# TODO: Revise initial mass
MTOM = 3000         # maximum take-off mass from statistical data - Class I estimation

n_ult = 3.2 * 1.5   # 3.2 is the max we found, 1.5 is the safety factor


# Stability
S_v = 1.558     # Area of the vertical tail [m^2]
h_tail = 1.396  # Height of vertical tail [m]

# ------------------ Constants for weight estimation ----------------
# TODO: revise Pmax
Pmax = 15.25  # this is defined as maximum perimeter in Roskam, so i took top down view of the fuselage perimeter

# PAX
# From project guide: 95 kg per pax (including luggage)
n_pax = 5  # number of passengers (pilot included)
m_pax = 88  # assume average mass of a passenger according to Google
cargo_m = (95-m_pax)*n_pax  # Use difference of pax+luggage - pax to get cargo mass

# Fuselage and CGs
pos_fus = l_fus/2                       # fuselage centre of mass away from the nose
pos_lgear = pos_fus + 0.4               # Main landing gear position away from the nose
pos_frontwing, pos_backwing = 0.5, 7    # positions of the wings away from the nose

mass_per_prop = 480 / n_prop
m_prop = [mass_per_prop] * n_prop       # list of mass of engines (so 30 kg per engine with nacelle and propeller)
# pos_prop = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0,
#             7.0]  # 8 on front wing and 8 on back wing
pos_prop = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0]  # 6 on front wing and 6 on back wing
# pos_prop = [0.2, 0.2, 0.2, 0.2, 7.0, 7.0, 7.0, 7.0]  # 4 on front wing and 4 on back wing


# ------------- Initial mass estimate -------------
wing = weight.Wing(MTOM, S1, S2, n_ult, AR_wing, [pos_frontwing, pos_backwing])
# wing = weight.Wing(MTOM, S1, S2, n_ult, AR_wing, AR_wing, [pos_frontwing, pos_backwing])
fuselage = weight.Fuselage(MTOM, Pmax, l_fus, n_pax, pos_fus)
lgear = weight.LandingGear(MTOM, pos_lgear)
props = weight.Propulsion(n_prop, m_prop, pos_prop)
Mass = weight.Weight(m_pax, wing, fuselage, lgear, props, cargo_m=cargo_m, cargo_pos=6, battery_m=m_bat,
                     battery_pos=3.6, p_pax=[1.5, 3, 3, 4.2, 4.2])

# Initial estimate for the mass
MTOM = Mass.mtom

# TODO: revise approach of reiterating
# Reiterate once because wing uses value for MTOM
wing = weight.Wing(MTOM, S1, S2, n_ult, AR_wing, [pos_frontwing, pos_backwing])
# wing = weight.Wing(MTOM, S1, S2, n_ult, AR_wing, AR_wing, [pos_frontwing, pos_backwing])
fuselage = weight.Fuselage(MTOM, Pmax, l_fus, n_pax, pos_fus)
lgear = weight.LandingGear(MTOM, pos_lgear)
props = weight.Propulsion(n_prop, m_prop, pos_prop)
Mass = weight.Weight(m_pax, wing, fuselage, lgear, props, cargo_m=cargo_m, cargo_pos=6, battery_m=m_bat,
                     battery_pos=3.6, p_pax=[1.5, 3, 3, 4.2, 4.2])

# Initial estimate for the mass
MTOM = Mass.mtom
print("Initial MTOM:", MTOM, "[kg]")
print(" ")

iterate = True
count = 0
while iterate or (count < 10):

    # Get atmospheric values at cruise
    ISA = at.ISA(h_cr)
    rho = ISA.density()             # Density
    a = ISA.soundspeed()            # Speed of sound
    dyn_vis = ISA.viscosity_dyn()   # Dynamic viscosity

    M = V_cr/a                      # Cruise Mach number

    # Aero
    wing_design = wing_des.wing_design(AR_tot, s1, 0, s2, 0, M, S_tot, l_fus-1, h_fus-0.3, w_fus, h_wt_1, h_wt_2)

    # [b2, c_r2, c_t2, c_MAC2, y_MAC2, X_LEMAC2]
    wing_plan_1, wing_plan_2 = wing_design.wing_planform_double()

    taper = wing_plan_1[2]/wing_plan_1[1]

    # CL_max
    CLmax = wing_design.CLmax_s()[0]

    # # Lift slope
    # CL_alpha_1 = wing_design.liftslope()
    # CL_alpha_2 = wing_design.liftslope()

    # ------ Drag ------

    # Oswald efficiency factor
    # e = drag_comp.e_OS(AR_tot) * drag_comp.e_factor('tandem', h_fus-0.3, wing_plan_1[0], drag_comp.e_OS(AR_tot))

    # Airfoil characteristics
    airfoil_stats = airfoil.airfoil_stats()

    drag = drag_comp.componentdrag('tandem', S_tot, l1_fus, l2_fus, l3_fus, np.sqrt(w_fus*h_fus), V_cr, rho,
                                   wing_plan_1[3], AR_tot, M, const.k, const.flamf, const.flamw, dyn_vis, const.tc,
                                   const.xcm, 0, wing_design.sweep_atx(0)[0], fus_upsweep, wing_plan_1[2], h_fus-0.3,
                                   const.IF_f, const.IF_w, const.IF_v, airfoil_stats[2], const.Abase, S_v,
                                   s1, s2, h_wt_1, h_wt_2)

    # TODO: get CL for cruise
    CD0 = drag.CD0()
    CD_cr = drag.CD(C_L=C_L_cr)

    # ----------------- Vertical drag -------------------
    Afus = np.pi * np.sqrt(w_fus * h_fus)**2 / 4

    CDs = drag.CD(CLmax)
    CDs_f = drag.CD0_f
    CDs_w = CDs - CDs_f

    CD_a_w_v = wing_design.CDa_poststall(const.tc, CDs_w, CDs_f, Afus, 0, "wing", drag.CD)
    CD_a_f_v = wing_design.CDa_poststall(const.tc, CDs_w, CDs_f, Afus, 90, "fus", drag.CD)

    CD_vertical = CD_a_w_v + CD_a_f_v

    # Propulsion
    # Get drag at cruise
    D_cr = CD_cr * 0.5 * rho * V_cr**2 * S_tot

    # Size the propellers
    prop_sizing = eng_siz.PropSizing(wing_plan_1[0], w_fus, n_prop, clearance_fus_prop, clearance_prop_prop, MTOM, xi_0)

    prop_radius = prop_sizing.radius()
    prop_area = prop_sizing.area_prop()
    disk_load = prop_sizing.disk_loading()

    # act_disk = ADT.ActDisk(TW_ratio, MTOM, v_e, V_cr, D, D)
    # With fuselage shape and span we can have size of the engines
    # From that we can use BEM to design the blades (here number of blades is an input,
    # so we can optimise it if necessary, or just assume one

    # ----------------------- Performance ------------------------
    # Cl_alpha_curve, CD_a_w, CD_a_f, alpha_lst, Drag

    # post_stall = Wing_params.post_stall_lift_drag(tc, CDs, CDs_f, Afus)

    init_sizing = perf.initial_sizing(h_cr, None, drag, V_stall, V_max, n_turn, ROC, V_cr, ROC_hover, MTOM*g0,
                                      CD_vertical, const.eff_prop, const.eff_hover, disk_load)

    # Get wing loading and from that the area
    WS = init_sizing.sizing()[0]

    S_tot = MTOM*g0/WS

    S1, S2 = S_tot*s1, S_tot*s2

    V = at.speeds(h_cr, MTOM, CLmax, S_tot, drag)

    # Cruise speed
    V_cr, CL_cr_check = V.cruise()

    # # Stall speed
    # V_stall = V.stall()

    # print("CL comparison:", CL_cr_check, C_L_cr, V_cr)

    # Cruise CL of the wings
    L_cr = MTOM*g0
    L_cr_1 = L_cr/2
    L_cr_2 = L_cr/2

    CL_cr_1 = 2*L_cr_1/(rho * V_cr**2 * S1)
    CL_cr_2 = 2*L_cr_2/(rho * V_cr**2 * S2)
    C_L_cr = CL_cr_2

    # Aero to pass to mission
    alpha_lst = np.arange(0, 89, 0.1)
    Cl_alpha_curve = wing_design.CLa(const.tc, CDs, CDs_f, Afus, alpha_lst)
    CD_a_w = wing_design.CDa_poststall(const.tc, CDs, CDs_f, Afus, alpha_lst, "wing", drag.CD)
    CD_a_f = wing_design.CDa_poststall(const.tc, CDs, CDs_f, Afus, alpha_lst, "fus", drag.CD)

    # Energy sizing
    mission = energy_calc.mission(MTOM, h_cr, V_cr, CLmax, S_tot, n_prop*prop_area, Cl_alpha_curve, CD_a_w, CD_a_f,
                                  alpha_lst, drag, m_bat)

    # Get approximate overall efficiency
    eff_overall = 0.91 * 0.57 + 0.699 * 0.43
    energy = 1.1 * mission.total_energy()[0] * 2.77778e-7 * 1000 / eff_overall  # From [J] to [Wh]

    # Battery sizing
    battery = bat.Battery(500, 1000, energy, 1)

    m_bat = battery.mass()

    # -------------------- Update weight ------------------------
    # TODO update battery weight
    wing = weight.Wing(MTOM, S1, S2, n_ult, AR_wing, [pos_frontwing, pos_backwing])
    # wing = weight.Wing(MTOM, S1, S2, n_ult, AR_wing, AR_wing, [pos_frontwing, pos_backwing])
    fuselage = weight.Fuselage(MTOM, Pmax, l_fus, n_pax, pos_fus)
    lgear = weight.LandingGear(MTOM, pos_lgear)
    props = weight.Propulsion(n_prop, m_prop, pos_prop)
    Mass = weight.Weight(m_pax, wing, fuselage, lgear, props, cargo_m=cargo_m, cargo_pos=6, battery_m=m_bat,
                         battery_pos=3.6, p_pax=[1.5, 3, 3, 4.2, 4.2])

    # Update mass and get CG
    MTOM_new = Mass.mtom
    x_CG_MTOM = Mass.mtom_cg

    if np.abs((MTOM_new-MTOM)/MTOM) < 0.001:
        iterate = False
        MTOM = MTOM_new

    else:
        MTOM = MTOM_new
    # print("New MTOM:", MTOM)
    # print(" ")
    count += 1


# Stability
vertical_tail = vert_tail.VT_sizing(MTOM*g0, h_cr, x_CG_MTOM, l_fus, h_fus, w_fus, V_cr, V_stall, CD0,
                                    CL_cr_1, CL_cr_2, 0, 0, S1, S2, AR_wing, AR_wing, 0, 0,
                                    wing_plan_1[3], wing_plan_2[3], wing_plan_1[0], wing_plan_2[0], taper, ARv=1.25)

v_tail = vertical_tail.final_VT_rudder(n_prop, D_cr, wing_plan_1[0]/2, l_fus-x_CG_MTOM, 0.9, 0.35)

print("Converged MTOM:", MTOM, "[kg]")
# print("CG position:", x_CG_MTOM)
# print("")
#
# print("Energy:", energy, "[kWh]")
# print("Battery mass:", m_bat, "[kg]")
# print("Wing surface:", S_tot, "[m^2]")
# print("")
print("Propeller radius:", prop_radius, "[m]")
# print("Disk loading:", disk_load, "[kg/m^2]")
# print("Cruise drag:", D_cr, "[N]")
print("Thrust per engine at cruise:", D_cr/n_prop, "[N]")
# print("")
# print("Span:", wing_plan_1[0])
# print("MAC:", wing_plan_1[3])
# print("")
print("Cruise speed:", V_cr, "[m/s]")
# print("Cruise height:", h_cr, "[m]")
# print("")
# print("CL_cr:", C_L_cr, "CD_cr:", CD_cr, "L/D at cr:", C_L_cr/CD_cr)
# print("W/D:", MTOM*g0/D_cr)
# print("")

# Sv,C_vr,C_vt,bv,Sweep_v_c2,c_r,c_r_root,c_r_tip,b_r,ARv

# print("Params for Miguel")
# print("CG", x_CG_MTOM)
# print("V_cr", V_cr)
# print("V_stall", V_stall)
# print("CLmax", CLmax)
# print("CLalpha", CL_alpha_1, CL_alpha_2)
# print("S1, S2", S1, S2)
# print("AR:", AR_wing)
# print("MAC", wing_plan_1[3], wing_plan_2[3])
# print("Spans:", wing_plan_1[0], wing_plan_2[0])
# print("Taper:", taper)
# # [b2, c_r2, c_t2, c_MAC2, y_MAC2, X_LEMAC2]
# print("C_r", wing_plan_1[1], wing_plan_2[1])
# print("")
# print("New S_v:", v_tail[0])
# print("C_vr:", v_tail[1])
# print("C_vt:", v_tail[2])
# print("b_v:", v_tail[3])

# print("Vertical tail surface", vertical_tail.final_VT_rudder(n_prop, ))
print(" ")


# Propeller blade design
B = 5
M_tip = 0.3
omega = M_tip*a/prop_radius
rpm = omega/0.10472
print("Propeller rpm:", rpm)

# rpm = 1500
# omega = rpm * 0.10472

# V_tip = omega*prop_radius

print("Propeller blade")
print("")
print("Design thrust:", 3*D_cr/n_prop)
print("")
print(B, prop_radius, rpm, xi_0, rho, dyn_vis, V_cr, 20, a, 100000, 3*D_cr/n_prop)
blade = BEM.BEM(B, prop_radius, rpm, xi_0, rho, dyn_vis, V_cr, 20, a, 100000, T=3*D_cr/n_prop)
# blade = BEM.BEM(B, prop_radius, rpm, xi_0, rho, dyn_vis, V_cr, 20, a, 100000, MTOM*g0)

design = blade.optimise_blade(0)[1]
coefs = blade.optimise_blade(0)[3]

# print("Chord per station:", design[0])
# print("")
# print("Pitch per station in deg:", np.rad2deg(design[1]))
# print("")
# print("AoA per station in [deg]:", np.rad2deg(design[2]))
# print("")
# print("Radial coordinates [m]:", design[3])
# print("")
# print("D/L ratio per station:", design[4])
# print("")
# print("Propeller efficiency:", design[5])
# print("")
# print("Thrust coefficient:", design[6])
# print("")
# print("Power coefficient:", design[7])
# print("")
# print("Exit speed per station:", V_e)
# print("")
# print("Average exit speed per station:", np.average(V_e))
# print("")
# print("Propulsive efficiency:", 2/(1 + np.average(V_e)/V_cruise))


plotter = bp.PlotBlade(design[0], design[1], design[3], prop_radius, xi_0)

plotter.plot_blade()
plotter.plot_3D_blade()

# Off-design analysis
Omega = rpm * 2 * np.pi / 60

RN = (Omega * design[3]) * design[0] * rho / dyn_vis

print(V_cr, B, prop_radius, design[0], design[1], design[3], coefs[0], coefs[1], rpm, rho, dyn_vis, a, RN)
blade_hover = BEM.OffDesignAnalysisBEM(V_cr, B, prop_radius, design[0], design[1], design[3],
                                       coefs[0], coefs[1], rpm, rho, dyn_vis, a, RN)
blade_hover_analysis = blade_hover.analyse_propeller()

print("Thrust from analysis", blade_hover_analysis[0])

# Polinomial regression for smooth distribution
coef_chords = np.polynomial.polynomial.polyfit(design[3], design[0], 5)
coef_pitchs = np.polynomial.polynomial.polyfit(design[3], design[1], 5)

radial_stations_Koen = np.array([1/10, (1/10 + 1/11*9/10), (1/10 + 2/11*9/10), (1/10 + 3/11*9/10), (1/10 + 4/11*9/10),
                                 (1/10 + 5/11*9/10), (1/10 + 6/11*9/10), (1/10 + 7/11*9/10), (1/10 + 8/11*9/10),
                                 (1/10 + 9/11*9/10), (1/10 + 10/11*9/10), (1/10 + 11/11*9/10)-0.001])*prop_radius


chord_fun = np.polynomial.polynomial.Polynomial(coef_chords)
pitch_fun = np.polynomial.polynomial.Polynomial(coef_pitchs)

koen_chords = chord_fun(radial_stations_Koen) * 1000
koen_pitch = np.rad2deg(pitch_fun(radial_stations_Koen))

# print("Koen chords:", koen_chords)
# print("Koen pitchs:", koen_pitch)
# print("Hub radius:", xi_0)

coef_cl = np.polynomial.polynomial.polyfit(design[3], coefs[0], 5)
coef_cd = np.polynomial.polynomial.polyfit(design[3], coefs[1], 5)

cl_fun = np.polynomial.polynomial.Polynomial(coef_cl)
cd_fun = np.polynomial.polynomial.Polynomial(coef_cd)

koen_cls = cl_fun(radial_stations_Koen)
koen_cds = cd_fun(radial_stations_Koen)


# Rotational speed
Omega = rpm * 2 * np.pi / 60
# This is an initial estimate for the Reynolds number per blade
RN = (Omega * radial_stations_Koen) * koen_chords * rho / dyn_vis


blade_hover = BEM.OffDesignAnalysisBEM(V_cr, B, prop_radius, koen_chords/1000, np.deg2rad(koen_pitch), radial_stations_Koen,
                                       koen_cls, koen_cds, rpm, rho, dyn_vis, a, RN)

blade_hover_analysis = blade_hover.analyse_propeller()

print("Thrust from analysis CATIA version", blade_hover_analysis[0])


plotter = bp.PlotBlade(koen_chords/1000, np.deg2rad(koen_pitch), radial_stations_Koen, prop_radius, xi_0)

plotter.plot_blade()
plotter.plot_3D_blade()

# # Plot
# axs[1].axis('equal')
#
# # Plot actual points
# axs[1].scatter(self.radial_coords, y_maxs)
# axs[1].scatter(self.radial_coords, y_mins)
#
# # Plot smooth distribution  TODO: revise
# radius = np.linspace(self.xi_0, self.R, 200)
# axs[1].plot(radius, y_min_fun(radius))
# axs[1].plot(radius, y_max_fun(radius))

