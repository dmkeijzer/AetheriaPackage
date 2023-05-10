import numpy as np
import Aero_tools as at
import constants_final as const

# Aero
import Preliminary_Lift.Drag as drag_comp
import Preliminary_Lift.Wing_design as wingdes
import Preliminary_Lift.Airfoil_analysis as airfoil

# Performance
import Flight_performance.Flight_performance_final as FP

# Propulsion
import PropandPower.engine_sizing_positioning as eng_siz
import PropandPower.battery as batt

# Stability and Control
import stab_and_ctrl.Vertical_tail_sizing as vert_tail
from stab_and_ctrl.hover_controllabilty import HoverControlCalcTandem
from stab_and_ctrl.landing_gear_placement import LandingGearCalc
from stab_and_ctrl.loading_diagram import CgCalculator
from stab_and_ctrl.xcg_limits import xcg_limits, optimise_wings, Cma, deps_da_empirical

# Structures
import structures.Weight as wei

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

# Wingtips
h_wt_1 = const.h_wt_1  # Height of front wingtips
h_wt_2 = const.h_wt_2  # Height of back wingtips


# Propulsion
n_prop_1 = const.n_prop_1
n_prop_2 = const.n_prop_2
n_prop = const.n_prop

# Structures
n_ult = const.n_ult


# ------------------ Constants for weight estimation ----------------
# TODO: revise Pmax
Pmax_weight = 15.25  # this is defined as maximum perimeter in Roskam, so i took top down view of the fuselage perimeter

# ------------- Initial mass estimate -------------
def mass(MTOM, S1, S2, n_ult, AR_wing1, AR_wing2, pos_frontwing, pos_backwing, Pmax, l_fus, n_pax, pos_fus,
         pos_lgear, n_prop, m_prop, pos_prop, m_pax, cargo_m, m_bat):
    wing = wei.Wing(MTOM, S1, S2, n_ult, AR_wing1, AR_wing2, [pos_frontwing, pos_backwing])
    m_wf = wing.wweight1
    m_wr = wing.wweight2
    fuselage = wei.Fuselage(MTOM, Pmax, l_fus, n_pax, pos_fus)
    m_fus = fuselage.mass
    cg_fus = fuselage.pos
    lgear = wei.LandingGear(MTOM, pos_lgear)
    cg_gear = lgear.pos
    props = wei.Propulsion(n_prop, m_prop, pos_prop)
    cg_props = props.pos_prop
    m_prop = props.mass
    Mass = wei.Weight(m_pax, wing, fuselage, lgear, props, cargo_m=cargo_m, cargo_pos=6, battery_m=m_bat,
                      battery_pos=3.6, p_pax=[1.5, 3, 3, 4.2, 4.2])

    return Mass.mtom, m_wf, m_wr, m_fus, m_prop, cg_fus, cg_gear, cg_props, Mass.mtom_cg

def find_mac(S, b, taper):
    """
    Calculate mean aerodynamic chord of a wing
    :param S: Wing surface area
    :param b: Wingspan
    :param taper: Wing taper
    :return: Mean aerodynamic chord
    """
    cavg = S / b
    cr = 2 / (1 + taper) * cavg
    mac = 2/3 * cr * (1 + taper + taper ** 2) / (1 + taper)
    return mac


def xmac_to_xle(sweep_25, A, taper, b, dihedral):

    # y position of the mac
    y_mac = b*(1 + 2*taper)/(6*(1 + taper))

    # Get the sweep angle of the leading edge
    sweep_le = np.arctan(np.tan(np.radians(sweep_25)) + (1-taper)/((1+taper)*A))

    # Calculate xlemac wrt to the root
    x_mac   = np.tan(sweep_le)*y_mac + 0.25*find_mac(b**2/A, b, taper)

    # Calculate height of the mac wrt the root
    z_mac   = np.tan(dihedral)*y_mac

    return x_mac, z_mac


class RunDSE:
    def __init__(self, initial_estimates: np.array):
        """
        This class integrates all the code and runs the optimisation

                Fixed:
        Range
        Aerofoil (Clmax, Cmac, …)
        Cabin design
        Taper ratios
        Quarter-chord sweep
        Weight of propulsion system
        Constraints on wing placement and dimensions
        Change in CL on front and rear wing due to elevator
        Clearance requirements

                            Internal (first estimates):
        Loading diagram (CG position and excursion)
        Rotor diameter
        Stall speed
        Energy consumption
        Peak power
        Maximum take-off mass
        Wing mass
        Landing gear placement
        Battery mass
        MTOM

                                    Optimisation variables:
        Tail-cone length
        Wing surface area
        Aspect ratios
        Relative wing sizes
        # Battery placement
        Wing placement

        :param fixed_inputs:        These are the fixed inputs passed into the integrated code (from constants.py)
        :param initial_estimates:   These are the initial estimates for the changing internal variables, to initialise
                                    the code
        """
        # self.fixed_inps = fixed_inputs
        self.initial_est = initial_estimates

    def run(self, optim_inputs, internal_inputs):
        """
        Here goes the main integration code
        It takes as inputs the optimisation variables and also estimates of its internal variables (e.g. battery mass)
        So for one set of optim inputs it results the optimisation outputs (e.g. mass and energy consumption)
        and also the internal parameters needed for iteration


        """
        # Initial estimates/
        MTOM = internal_inputs[0]
        V_cr = internal_inputs[2]
        h_cr = internal_inputs[3]
        C_L_cr = internal_inputs[4]
        CLmax = internal_inputs[5]
        prop_radius = internal_inputs[6]
        de_da = internal_inputs[7]
        Sv = internal_inputs[8]
        V_stall = internal_inputs[9]
        max_power = internal_inputs[10]
        AR_wing1 = internal_inputs[11]
        AR_wing2 = internal_inputs[12]

        Sr_Sf = internal_inputs[13]
        s1 = internal_inputs[14]
        # Ratio is input
        s2 = s1 * Sr_Sf

        # Positions of the wings [horizontally, vertically]
        xf = internal_inputs[15]
        zf = internal_inputs[16]
        xr = internal_inputs[17]
        zr = internal_inputs[18]

        pos_front_wing = [xf, zf]
        pos_back_wing = [xr, zr]

        b1 = internal_inputs[19]
        b2 = internal_inputs[20]

        # Distances (positive if back wing is further aft and higher)
        wing_distance_hor = xr - xf
        wing_distance_ver = zr - zf

        # ----------- Get atmospheric values at cruise --------------
        ISA = at.ISA(h_cr)
        rho = ISA.density()             # Density
        a = ISA.soundspeed()            # Speed of sound
        dyn_vis = ISA.viscosity_dyn()   # Dynamic viscosity

        M = V_cr / a                    # Cruise Mach number

        # TODO: check these two lines
        # Wing loading and wing area
        WS_stall = 0.5 * rho * V_stall * V_stall * CLmax
        S_tot = MTOM * g0 / WS_stall

        # Wing areas
        S2 = S_tot*s2
        S1 = S2/Sr_Sf
        s1 = S1/S_tot


        # Aero
        wing_design = wingdes.wing_design(AR_wing1, AR_wing2, s1, const.sweepc41, s2, const.sweepc42, M, S_tot,
                                          wing_distance_hor, wing_distance_ver, w_fus, h_wt_1, h_wt_2, const.k_wl,
                                          const.i1)

        # [b2, c_r2, c_t2, c_MAC2, y_MAC2, X_LEMAC2]
        wing_plan_1, wing_plan_2 = wing_design.wing_planform_double()

        # ------ Drag ------
        Afus = np.pi * np.sqrt(w_fus * h_fus)**2 / 4

        # Airfoil characteristics
        airfoil_stats = airfoil.airfoil_stats()

        drag = drag_comp.componentdrag('tandem', S_tot, l1_fus, l2_fus, l3_fus, np.sqrt(w_fus * h_fus), V_cr, rho,
                                       wing_plan_1[3], AR_wing1, AR_wing2, M, const.k, const.flamf, const.flamw, dyn_vis, const.tc,
                                       const.xcm, 0, wing_design.sweep_atx(0)[0], fus_upsweep, wing_plan_1[2],
                                       wing_distance_ver, const.IF_f, const.IF_w, const.IF_v, airfoil_stats[2],
                                       const.Abase, Sv, s1, s2, h_wt_1, h_wt_2, const.k_wl)


        taper = wing_plan_1[2] / wing_plan_1[1]

        CDs = drag.CD(CLmax)
        CDs_f = drag.CD0_f
        CDs_w = CDs - CDs_f

        # CL_max
        alpha_wp = 1    # If we only want CLmax (and not slope) this does not matter

        # Drag at stall
        error = 1
        while error > 0.05:  # FIXME this goes crazy on the second iteration
            D_stall = drag.CD(CLmax) * 0.5 * rho * V_stall**2 * S_tot
            T_per_eng_during_stall = D_stall/n_prop

            CLmaxnew = wing_design.CLa_wprop(T_per_eng_during_stall, V_stall, rho, 2*prop_radius, n_prop_1, n_prop_2,
                                             const.tc, CDs_w, CDs_f, Afus, alpha_wp, de_da)[1]
            print("1", CLmaxnew, "end\n")
            error = np.abs(CLmaxnew-CLmax)/CLmax
            CLmax = CLmaxnew

        # FIXME no iter
        # D_stall = drag.CD(CLmax) * 0.5 * rho * V_stall**2 * S_tot
        # T_per_eng_during_stall = D_stall/n_prop
        #
        # CLmax = wing_design.CLa_wprop(T_per_eng_during_stall, V_stall, rho, 2*prop_radius, n_prop_1, n_prop_2,
        #                               const.tc, CDs_w, CDs_f, Afus, alpha_wp, de_da)[1]

        # print("1", CLmax, "2", T_per_eng_during_stall, V_stall, rho, 2*prop_radius, n_prop_1, n_prop_2,
        #       const.tc, CDs_w, CDs_f, Afus, alpha_wp, de_da, "end\n")

        CD0 = drag.CD0()
        CD_cr = drag.CD(C_L=C_L_cr)

        # ----------------- Vertical drag -------------------

        # CD_a_w_v = wing_design.CDa_poststall(const.tc, CDs_w, CDs_f, Afus, 0, "wing", drag.CD, de_da)
        # CD_a_f_v = wing_design.CDa_poststall(const.tc, CDs_w, CDs_f, Afus, 90, "fus", drag.CD, de_da)

        # Propulsion
        # Get drag at cruise
        D_cr = CD_cr * 0.5 * rho * V_cr ** 2 * S_tot

        # Size the propellers
        prop_sizing = eng_siz.PropSizing(wing_plan_1[0], w_fus, n_prop, const.c_fp, const.c_pp, MTOM, const.xi_0)

        prop_radius = prop_sizing.radius()
        prop_area = prop_sizing.area_prop()

        # ----------------------- Performance ------------------------

        V = at.speeds(h_cr, MTOM, CLmax, S_tot, drag)

        # Cruise speed
        V_cr, CL_cr_check = V.cruise()

        print("V_cr", V_cr, "MTOM", MTOM, "CLmax", CLmax, "S_tot", S_tot)
        print("")

        # Update the stall speed
        V_stall = V.stall()

        # print("CL comparison:", CL_cr_check, C_L_cr, V_cr)

        # Cruise CL of the wings
        L_cr = MTOM * g0
        L_cr_1 = L_cr * s1
        L_cr_2 = L_cr * s2

        # Lift coefficients at cruise
        CL_cr_1 = 2 * L_cr_1 / (rho * V_cr ** 2 * S1)
        CL_cr_2 = 2 * L_cr_2 / (rho * V_cr ** 2 * S2)
        C_L_cr = 2 * L_cr / (rho * V_cr ** 2 * S_tot)

        # Aero to pass to mission
        alpha_lst = np.arange(-3, 89, 0.1)
        Cl_alpha_curve = wing_design.CLa(const.tc, CDs, CDs_f, Afus, alpha_lst, de_da)[0]
        CD_a_w = wing_design.CDa_poststall(const.tc, CDs, CDs_f, Afus, alpha_lst, "wing", drag.CD, de_da)
        CD_a_f = wing_design.CDa_poststall(const.tc, CDs, CDs_f, Afus, alpha_lst, "fus", drag.CD, de_da)

        # Energy sizing
        mission = FP.mission(MTOM, h_cr, V_cr, CLmax, S_tot, n_prop * prop_area, P_max=max_power,
                             Cl_alpha_curve=Cl_alpha_curve, CD_a_w=CD_a_w, CD_a_f=CD_a_f, alpha_lst=alpha_lst,
                             Drag=drag, t_loiter=30*60, rotational_rate=5, mission_dist=const.mission_range)

        # Get approximate overall efficiency
        energy, t_tot, max_power, max_thrust, t_hor = mission.total_energy()

        # Overall efficiency from battery to engine
        eff_overall = const.eff_bat_eng_cr * (t_hor/t_tot) + const.eff_bat_eng_h * (1-(t_hor/t_tot))
        energy = energy * 2.77778e-7 * 1000 / eff_overall  # From [J] to [Wh]
        # TODO: check safety factor (1.02 *)

        # Cruise power
        P_cr = mission.power_cruise_config(h_cr, V_cr, MTOM)

        # Engine sizing

        # Maximum power [W] and thrust [N] of the engines (total)
        time, P_max_eng_tot, T_max_tot = mission.total_energy()[1:4]
        P_max_eng_ind = P_max_eng_tot/n_prop                    # Maximum power [W] of the engines (per engine)
        P_max_bat = P_max_eng_tot/const.eff_bat_eng_h           # Maximum power [W] to be delivered by the battery

        T_max_ind = T_max_tot/n_prop                # Maximum thrust to be delivered by the engines (per engine)

        # Battery sizing
        sp_en_den = const.sp_en_den
        vol_en_den = const.vol_en_den
        batt_cost = const.bat_cost
        DoD = const.DoD
        P_den = const.P_den
        safety_factor = 1  # TODO: discuss
        EOL_C = const.EOL_C

        # sp_en_den, vol_en_den, tot_energy, cost, DoD, P_den, P_max, safety, EOL_C
        battery = batt.Battery(sp_en_den, vol_en_den, energy, batt_cost, DoD, P_den, P_max_bat, safety_factor, EOL_C)

        m_bat = battery.mass()

        # The mass of one engine is the specific mass of the engines (kg/W) x Total power of ONE ENGINE
        m_prop = const.sp_mass_en * P_max_eng_ind

        # -------------------- Update weight ------------------------
        pos_fus = l_fus*0.4
        MAC1 = find_mac(S1, b1, taper)
        MAC2 = find_mac(S2, b2, taper)
        pos_prop_front = [(xf - 0.25*MAC1) - 0.2] * n_prop_1
        pos_prop_back = [(xr - 0.25*MAC2) - 0.2] * n_prop_2

        pos_prop = np.hstack((np.array(pos_prop_front), np.array(pos_prop_back)))

        pos_lgear = (1.75+6)/2  # TODO revise if needed
        MTOM, m_wf, m_wr, m_fus, m_prop, cg_fus0, cg_gear, cg_props, x_CG_MTOM = mass(MTOM, S1, S2, n_ult, AR_wing1,
                                                                                     AR_wing2, pos_front_wing,
                                                                                     pos_back_wing, Pmax_weight, l_fus,
                                                                                     const.n_pax, pos_fus, pos_lgear,
                                                                                     n_prop, m_prop, pos_prop,
                                                                                     const.m_pax,  const.m_cargo_tot,
                                                                                     m_bat)

        # ----------------- Stability and control -------------------
        # W = 2950 * 9.80665
        # h = 1000
        # lfus = 7.2
        # hfus = 1.705
        # wfus = 1.38
        # xcg = 3.0
        # V0 = 64.72389428906716
        # Vstall = 40
        # Pbr = 110024 / 1.2 * 0.9 / 12
        # # M0 = V0 / np.sqrt(const.gamma * const.R * 288.15)
        # CD0 = 0.03254
        # CLfwd = 1.44333 # Maximum lift coefficient of forward wing
        # CLrear = 1.44333 # Maximum lift coefficient of rear wing
        # CLdesfwd = 0.7382799 # Design lift coefficient of forward wing for cruise
        # CLdesrear = 0.7382799 # Design lift coefficient of rear wing for cruise
        # CLafwd = 5.1685 # Lift slope of forward wing
        # Clafwd = 6.1879 # Airfoil lift slope (fwd wing)
        # Clarear = Clafwd # Airfoil lift slope (rear wing)
        # Cd0fwd = 0.00347  # Airfoil drag coefficient [-]
        # Cd0rear = Cd0fwd
        # CD0fwd = 0.00822  # Wing drag coefficient [-]
        # CD0rear = CD0fwd
        # Cmacfwd = -0.0645 # Pitching moment coefficient at ac [-] (fwd wing)
        # Cmacrear = -0.0645 # Pitching moment coefficient at ac [-] (rear wing)
        # Sfwd = 8.417113787320769 # Forward wing surface area [m^2]
        # Srear = 8.417113787320769 # Rear wing surface area [m^2]
        # taperfwd = 0.45
        # taperrear = 0.45
        # S = Srear + Sfwd
        # Afwd = 9 * 1
        # Arear = 9
        # Gamma = 0
        # Lambda_c4_fwd = 0.0 * np.pi / 180 # Sweep at c/4 [rad]
        # Lambda_c4_rear = 0.0 * np.pi / 180
        # cfwd = 1.014129367767935
        # crear = 1.014129367767935
        # c = Srear / S * crear + Sfwd / S * cfwd
        # bfwd = np.sqrt(Sfwd * Afwd)
        # brear = np.sqrt(Srear * Arear)
        # b = max(bfwd, brear)
        # print(b)
        # e = 1.1302
        # efwd = 0.958 # Span efficiency factor of fwd wing
        # erear = 0.958 # Span efficiency factor of rear wing
        # taper = 0.45
        # n_rot_f = 6
        # n_rot_r = 6
        # rot_y_range_f = [0.5 * bfwd * 0.1, 0.5 * bfwd * 0.9]
        # rot_y_range_r = [0.5 * brear * 0.1, 0.5 * brear * 0.9]
        # K = 4959.86
        # ku = 0.1
        # Zcg = 0.70 # TALK ABOUT STRUCTURES FOR BETTER ESTIMATE
        # elev_fac = 1.4
        # Vr_Vf_2 = 0.90 # TO BE CHANGED -> AERODYNAMICS
        # # crmaxf, crmaxr MAXIMUM root chord lengths of both wings [m]
        # # bmaxf, bmaxr MAXIMUM span lengths of both wings [m]
        # # Arangef, Aranger max and min values of Aspect ratio [-]
        # # xcg_range most front based on loading diagram
        # # max_thrust: total maximum that can be achieved in hover
        # #  x_wf = x_wf, x_wr = x_wr: x-location of rotors approx aerodynamic centers
        # # cg_pax, cg_pil, cg_wf, cg_wr: cg locations of passengers, pilot, front wing and rear wing


        # Hover controllability
        x_f_rotated = xf - xmac_to_xle(const.sweepc41, AR_wing1, taper, b1, const.dihedral1)[0] + 0.45*wing_plan_1[1]
        x_r_rotated = xr - xmac_to_xle(const.sweepc42, AR_wing2, taper, b2, const.dihedral2)[0] + 0.45*wing_plan_2[1]
        HoverControlCalcTandem(MTOM, n_rot_f=n_prop_1, n_rot_r=n_prop_2, x_wf=x_f_rotated, x_wr=x_r_rotated,
                               rot_y_range_f=[w_fus/2 + const.c_fp + prop_radius, wing_plan_1[0]],
                               rot_y_range_r=[w_fus/2 + const.c_fp + prop_radius, wing_plan_2[0]],
                               K=max_thrust/n_prop, ku=0.1)

        # Loading
        cg_pax = [[3.75, 0.5, h_fus*0.4], [3.75, -0.5, h_fus*0.4], [6, 0.5, h_fus*0.4], [6, -0.5, h_fus*0.4]]
        # Approximated with new layout
        cg_pil = [1.75, 0, h_fus/2]  # Pilot is higher than pax
        cg_fus = [l_fus*0.5, 0, h_fus*0.5]
        cg_calc = CgCalculator(m_wf, m_wr, m_fus, m_bat, const.m_cargo_tot, const.m_pax, const.m_pax,
                               cg_fus=cg_fus, cg_bat=const.cg_bat, cg_cargo=const.cargo_pos, cg_pax=cg_pax, cg_pil=cg_pil)

        # Get the cg range, based on wing placement, the loading order can be changed if needed
        cg_wf = [xf + 0.25*MAC1, zf]
        cg_wr = [xr + 0.25*MAC2, zr]
        [x_front, x_aft], _, [_, z_top] = cg_calc.calc_cg_range(cg_wf, cg_wr)

        # x_cg limit

        # For cmac: use airfoil analysis Cm_ac
        # For CLmax: Wing_design CLa_wprop (for entire aircraft)
        # CLdes: use CL_des of entire aircraft
        # CD0
        # CLa fwd and rear, second and third output ASK STABILITY IF THEY INCLUDE DOWNWASH THEMSELVES
        # Vr_Vf = 1

        # Wing characteristics
        Cmac_airfoil = airfoil.Cm_ac(const.sweepc41, AR_wing1)[1]
        Cmacfwd = airfoil.Cm_ac(const.sweepc41, AR_wing1)[0]  # TODO: changes with AR
        Cmacrear = airfoil.Cm_ac(const.sweepc42, AR_wing2)[0]
        CLfwd, CLrear = wing_design.CLa_wprop(T_per_eng_during_stall, V_stall, rho, 2*prop_radius, n_prop_1, n_prop_2,
                                              const.tc, CDs_w, CDs_f, Afus, alpha_wp, de_da=0)[4:6]
        CLdesfwd = drag.CL_des()[0]
        CLdesrear = drag.CL_des()[0]
        CD0fwd = drag.Cd_w(0)
        CD0rear = CD0fwd
        CLafwd = wing_design.liftslope(0)[1][0]  # TODO: unit check
        CLarear = wing_design.liftslope(0)[2][0]  # TODO: unit check
        Clafwd = airfoil.airfoil_stats()[4] * 180/np.pi  # TODO: unit check
        Clarear = airfoil.airfoil_stats()[4] * 180/np.pi  # TODO: unit check

        # Optimize the wing size and aspect ratios for stability and control, ignoring the stability constraint for now
        y_mac_1 = b1 * (1 + 2 * taper) / (6 * (1 + taper))
        y_mac_2 = b2 * (1 + 2 * taper) / (6 * (1 + taper))

        # Leading edge position
        xrangef_LE = [0, 2.1 - wing_plan_1[1]]
        xranger_LE = [6, l_fus-wing_plan_2[1]]

        # MAC position
        xrangef = [0 + xmac_to_xle(const.sweepc41, AR_wing1, taper, b1, const.dihedral1)[0],
                   2.1 - wing_plan_1[1] + xmac_to_xle(const.sweepc41, AR_wing1, taper, b1, const.dihedral1)[0]]
        xranger = [6 + xmac_to_xle(const.sweepc42, AR_wing2, taper, b2, const.dihedral2)[0],
                   l_fus - wing_plan_2[1] + xmac_to_xle(const.sweepc42, AR_wing2, taper, b2, const.dihedral2)[0]]

        zrangef = [0 + y_mac_1 * const.dihedral1, 0.25*h_fus + y_mac_1 * const.dihedral1]
        zranger = [0.7*h_fus + y_mac_2 * const.dihedral2, h_fus + y_mac_2 * const.dihedral2]

        Zcg = 0.4 * const.h_fuselage  # Estimate

        # Check inputs
        # print(Cmacfwd, Cmacrear, CLfwd, CLrear, CLdesfwd, CLdesrear, CD0fwd, CD0rear, taper, taper,
        #                                                              const.sweepc41, const.sweepc42, const.e_f,
        #                                                              const.e_r, Clafwd, Clarear, Zcg, const.Vr_Vf_2,
        #                                                              const.elev_fac, rho, P_cr/n_prop, S_tot, MTOM*g0,
        #                                                              xrangef,  xranger, zrangef, zranger, const.crmaxf,
        #                                                              const.crmaxr, const.b_max, const.b_max,
        #                                                              const.A_range_f, const.A_range_r, [x_front, x_aft])

        # [AR_wing1, AR_wing2, xf, xr, zf, zr, Sr_Sf] = optimise_wings(Cmac_airfoil, CLfwd, CLrear, CLdesfwd,
        #                                                              CLdesrear, CD0fwd, CD0rear, taper, taper,
        #                                                              const.sweepc41, const.sweepc42, const.e_f,
        #                                                              const.e_r, Clafwd, Clarear, Zcg, const.Vr_Vf_2,
        #                                                              const.elev_fac, rho, P_cr/n_prop, S_tot, MTOM*g0,
        #                                                              xrangef,  xranger, zrangef, zranger, const.crmaxf,
        #                                                              const.crmaxr, const.b_max, const.b_max,
        #                                                              const.A_range_f, const.A_range_r,
        #                                                              xcg_range=[x_front, x_aft],    # xcg_range=[x_front, x_aft]
        #                                                              impose_stability=False)

        [AR_wing1, AR_wing2, xf, xr, zf, zr, Sr_Sf] = optimise_wings(CLfwd, CLrear, CLdesfwd,
                                                                     CLdesrear, CD0fwd, CD0rear, taper, taper,
                                                                     const.sweepc41, const.sweepc42, const.e_f,
                                                                     const.e_r, Clafwd, Clarear, Zcg, const.Vr_Vf_2,
                                                                     const.elev_fac, rho, P_cr/n_prop, S_tot, MTOM*g0,
                                                                     xrangef,  xranger, zrangef, zranger, const.crmaxf,
                                                                     const.crmaxr, const.b_max, const.b_max,
                                                                     const.A_range_f, const.A_range_r,
                                                                     xcg_range=[x_front, x_aft-0.07],    # xcg_range=[x_front, x_aft]
                                                                     impose_stability=True)
                                                                     # TODO: revise stability margin

        print("AR1", AR_wing1, "AR2", AR_wing2, xf, xr, zf, zr, "Sr_Sf", Sr_Sf)
        print("")

        # Calculate new S1 with new ratio
        S1 = S_tot/(1 + Sr_Sf)
        S2 = S1 * Sr_Sf

        # New span of wing 1
        b1 = np.sqrt(AR_wing1 * S1)
        b2 = np.sqrt(AR_wing2 * S2)

        # Transform wing positions to leading edge of the root
        x_le_f = xf - xmac_to_xle(const.sweepc41, AR_wing1, taper, b1, const.dihedral1)[0]
        x_le_r = xr - xmac_to_xle(const.sweepc42, AR_wing2, taper, b2, const.dihedral2)[0]

        # Transform z-position
        z_le_f = zf - xmac_to_xle(const.sweepc41, AR_wing1, taper, b1, const.dihedral1)[1]
        z_le_r = zr - xmac_to_xle(const.sweepc42, AR_wing2, taper, b2, const.dihedral2)[1]

        # lambda_c4f, bf, lh, h_ht, A, CLaf, rho, Pbr, Sf, CLf, W
        de_da = deps_da_empirical(const.sweepc41, b1, xr - xf, zr - zf, AR_wing1, CLafwd, rho, P_cr / n_prop, S1, CL_cr_1, MTOM * g0)


        """
        :param h_ht: Distance between wings normal to their chord planes 
        """  # FIXME: I used vertical distance

        # Landing gear placement
        h_bottom = 0
        gears = LandingGearCalc(1.5*w_fus, x_ng_min=0.3, y_max_rotor=wing_plan_1[0],
                                gamma=float(np.radians(5)), z_rotor_line_root=pos_front_wing[1] + h_bottom,
                                rotor_rad=prop_radius,
                                fus_back_bottom=const.fus_back_bottom, fus_back_top=const.fus_back_top)

        x_ng, x_mlg, track_width, z_mlg = gears.optimum_placement([x_front, x_aft], x_cg_margin=0,
                                                                  z_cg_max=z_top, theta=const.pitch_lim,
                                                                  phi=const.lat_lim, psi=const.turn_over,
                                                                  min_ng_load_frac=const.min_ng_load)

        CM_a = Cma(Clafwd, Clarear, const.sweepc41, const.sweepc42, taper, taper, CL_cr_1, CL_cr_2, AR_wing1, AR_wing2,
                   const.e_f, const.e_r, xf, xr, zf, zr, Zcg, const.Vr_Vf_2, Sr_Sf, x_CG_MTOM, S_tot, rho, P_cr/n_prop,
                   MTOM*g0)

        # Load vertical tail
        vertical_tail = vert_tail.VT_sizing(MTOM * g0, h_cr, x_CG_MTOM, l_fus, h_fus, w_fus, V_cr, V_stall, CD0,
                                            CL_cr_1, CL_cr_2, Clafwd, Clarear, S1, S2, AR_wing1, AR_wing2,
                                            const.sweepc41, const.sweepc42, wing_plan_1[3], wing_plan_2[3],
                                            b1, b2, taper, ARv=const.ARv, sweepTE=const.sweep_vtail)

        # nE, Tt0, yE, br_bv, cr_cv, ARv, sweepTE
        v_tail = vertical_tail.final_VT_rudder(n_prop, D_cr, max(b1/2, b2/2), const.br_bv, const.cr_cv, const.ARv,
                                               const.sweep_vtail)

        Sv = v_tail[0]

        # Variables that are updated (the 0 is a placeholder, not used)
        internal_inputs = [MTOM, 0, V_cr, h_cr, C_L_cr, CLmax, prop_radius, de_da, Sv, V_stall, max_power, AR_wing1,
                           AR_wing2, Sr_Sf, s1, xf, zf, xr, zr, b1, b2]


        # Outputs for optimisation cost function
        optim_outputs = [MTOM, energy, time, CM_a]

        print("MTOM:", MTOM, ", energy:", energy, ", battery mass:", m_bat)
        print("")

        # Other necessary outputs
        other_outputs = [track_width, z_mlg]

        return optim_outputs, internal_inputs, other_outputs

    def multirun(self, N_iters, optim_inputs):
        """
        With this you can run the integrated code as many times as you want per optimisation, so that you get internal
        convergence of the internal parameters

        :param N_iters: Number of iterations of the code for each optimisation iteration
        """
        internal_inputs = self.initial_est

        for i in range(N_iters):
            print("Iteration #", i)
            print("")
            optim_outputs, internal_inputs, other_outputs = self.run(optim_inputs, internal_inputs)

        """
        Function to iterate until converged
        """
        # N_iter = 0
        # convergence = 1
        # m_old = 3000        # Initial mass
        # while (convergence > 0.01) or (N_iter < 10):
        #     optim_outputs, internal_inputs, other_outputs = self.run(optim_inputs, internal_inputs)
        #
        #     N_iter += 1
        #
        #     # Check convergence of mass
        #     m_new = internal_inputs[0]
        #     convergence = np.abs(m_new - m_old)/m_old
        #     m_old = m_new

        return optim_outputs, internal_inputs, other_outputs
