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
from stab_and_ctrl.xcg_limits import Cma, deps_da_empirical, xcg_ctrl

# Structures
import structures.Weight as wei

# --------------------- Fixed parameters and constants ------------------------
# TODO:
#   Revise list of parameters to change and double check with departments final values
#   Agree with Koen on final value for engine sizing, battery sizing
#   Double check with stability that TW of 1.5 is enough for OEI
#   Fix array error which will come
#   Print weight fractions for Nikita
#   Print disk radius and loading
#   Print different propeller radii and ask Miguel about them, add to Vertical_tail_sizing
#   Change inputs for VT_controllability
#   Find vertical tail placement
#   Find clearance for aft wing
#   The return value you want is [[[0], [1], [2], [3], [4], [5], [6], ..., [11]]]

# Constants from constants.py
g0 = const.g
rho0 = const.rho_0
gamma = const.gamma
R = const.R

# Fuselage shape is fixed, so fixed variables (update in constants.py if needed)
w_fus = const.w_fuselage

l1_fus = const.l_nosecone
l2_fus = const.l_cylinder

h_fus = const.h_fuselage
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
Pmax_weight = 17  # this is defined as maximum perimeter in Roskam, so I took top down view of the fuselage perimeter

# ------------- Initial mass estimate -------------
def mass(MTOM, S1, S2, n_ult, AR_wing1, AR_wing2, pos_frontwing, pos_backwing, Pmax, l_fus, n_pax, pos_fus,
         pos_lgear, n_prop, m_prop, pos_prop, m_pax, cargo_m, m_bat, Sv, v_tail_rchord, contingency = False,
         cg_bat=[None, None, None]):

    print(cg_bat, "bat cg")
    wing = wei.Wing(MTOM, S1, S2, n_ult, AR_wing1, AR_wing2, [pos_frontwing, pos_backwing])
    m_wf = wing.wweight1
    m_wr = wing.wweight2
    fuselage = wei.Fuselage(MTOM, Pmax, l_fus, n_pax, pos_fus)
    m_fus = fuselage.mass
    cg_fus = fuselage.pos
    lgear = wei.LandingGear(MTOM, pos_lgear)

    cg_gear = lgear.pos
    # print(m_prop)
    props = wei.Propulsion(n_prop, m_prop, pos_prop)
    cg_props = props.pos_prop
    m_prop = props.mass

    # Vertical tail sizing
    vtail = wei.Vtail(MTOM, Sv, const.ARv, v_tail_rchord, const.tc, const.sweep_vtail_c4)

    # class Vtail:
    #     def __init__(self, mtom, Sv, Av, rchord, toc, sweep_deg):

    Mass = wei.Weight(m_pax, wing, fuselage, lgear, props, cargo_m=cargo_m, cargo_pos=const.cargo_pos[0],
                      battery_m=m_bat, battery_pos=cg_bat[0], p_pax=[const.x_pil, const.x_f_pax, const.x_f_pax,
                      const.x_r_pax, const.x_r_pax], contingency = contingency, Vtail_mass=vtail.mass, vtail_pos=pos_backwing[0])

    return Mass.mtom, m_wf + n_prop_1*np.sum(m_prop)/n_prop, m_wr + n_prop_2*np.sum(m_prop)/n_prop, m_fus, m_prop, \
           cg_fus, cg_gear, cg_props, \
           Mass.mtom_cg, lgear.mass, vtail.mass


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

        # Ratio between front and rear wing
        Sr_Sf = internal_inputs[13]

        # Fractions of total wing area
        s1 = (1 + Sr_Sf)**-1
        s2 = s1 * Sr_Sf

        # Positions of the wings [horizontally, vertically]
        xf = internal_inputs[15]
        zf = internal_inputs[16]
        xr = internal_inputs[17]
        zr = internal_inputs[18]
        max_thrust_stall = internal_inputs[19]

        root_chord_vtail = internal_inputs[20]
        TW_ratio_control = internal_inputs[21]

        x_front = internal_inputs[22]
        x_aft   = internal_inputs[23]

        bat_pos = internal_inputs[25]

        cg_bat = [bat_pos, 0, 0.4*const.h_fuselage]

        # Distances (positive if back wing is further aft and higher)
        wing_distance_hor = xr - xf
        wing_distance_ver = zr - zf

        # ----------- Get atmospheric values at cruise --------------
        ISA = at.ISA(h_cr)
        rho = ISA.density()             # Density
        a = ISA.soundspeed()            # Speed of sound
        dyn_vis = ISA.viscosity_dyn()   # Dynamic viscosity

        M = V_cr / a                    # Cruise Mach number

        # Wing loading and wing area
        WS_stall = 0.5 * rho * V_stall * V_stall * CLmax
        S_tot = MTOM * g0 / WS_stall

        # Wing areas
        # print(AR_wing2, S_tot, s2)
        S2 = S_tot*s2
        S1 = S2/Sr_Sf
        s1 = S1/S_tot

        # Spans
        b1 = np.sqrt(AR_wing1*S1)
        b2 = np.sqrt(AR_wing2*S2)

        # Aero
        wing_design = wingdes.wing_design(AR_wing1, AR_wing2, s1, const.sweepc41, s2, const.sweepc42, M, S_tot,
                                          wing_distance_hor, wing_distance_ver, w_fus, h_wt_1, h_wt_2, const.k_wl,
                                          const.i1)

        # Wing planform shape parameters
        wing_plan_1, wing_plan_2 = wing_design.wing_planform_double()

        # Assuming taper is the same for front and rear wing
        taper = wing_plan_1[2] / wing_plan_1[1]

        # Calculate the tailcone length, based on aft wing placement
        l_tc = xr - xmac_to_xle(const.sweepc41, AR_wing1, taper, b1, const.dihedral1)[0] + \
               (2 * b2 / (AR_wing2 * (1 + taper))) - (const.l_nosecone + const.l_cylinder) + const.wing_clearance_aft

        # Fuselage upsweep, based on tailcone length
        fus_upsweep = np.arctan(0.4 / l_tc)

        # Length of the fuselage
        l_fus = l_tc + const.l_nosecone + const.l_cylinder

        # print(l_fus)

        # ------ Drag ------
        Afus = np.pi * np.sqrt(w_fus * h_fus)**2 / 4

        # Airfoil characteristics
        airfoil_stats = airfoil.airfoil_stats()

        drag = drag_comp.componentdrag('tandem', S_tot, const.l_nosecone, const.l_cylinder, l_tc, np.sqrt(w_fus * h_fus), V_cr, rho,
                                       wing_plan_1[3], AR_wing1, AR_wing2, M, const.k, const.flamf, const.flamw, dyn_vis, const.tc,
                                       const.xcm, 0, wing_design.sweep_atx(0)[0], fus_upsweep, wing_plan_1[2],
                                       wing_distance_ver, const.IF_f, const.IF_w, const.IF_v, airfoil_stats[2],
                                       const.Abase, Sv, s1, s2, h_wt_1, h_wt_2, const.k_wl)

        CDs = float(drag.CD(CLmax))
        CDs_f = float(drag.CD0_f)
        CDs_w = CDs - CDs_f

        alpha_wp = 1    # If we only want CLmax (and not slope) this does not matter

        # This part iteratively calculates the CLmax, based on the thrust during stall
        error = 1
        while error > 0.05:

            # Drag at stall
            D_stall = float(drag.CD(CLmax)) * 0.5 * rho * V_stall**2 * S_tot

            # Thrust per engine at stall, based on the drag and a maximum thrust
            T_per_eng_during_stall = np.minimum(D_stall/n_prop, max_thrust_stall/n_prop)

            # Get the new CLmax, taking into account the airflow from the propellers
            CLmaxnew = wing_design.CLa_wprop(T_per_eng_during_stall, V_stall, rho, 2*prop_radius, n_prop_1, n_prop_2,
                                             const.tc, CDs_w, CDs_f, Afus, alpha_wp, de_da)[1]

            error = np.abs(CLmaxnew-CLmax)/CLmax
            CLmax = CLmaxnew

        CLmax = np.minimum(CLmax, 3)
        CD0 = float(drag.CD0())
        CD_cr = float(drag.CD(C_L=C_L_cr))

        # ----------------- Vertical drag -------------------

        # Propulsion
        # Get drag at cruise
        D_cr = CD_cr * 0.5 * rho * V_cr ** 2 * S_tot

        # print("Before prop sizing")
        # Size the propellers
        prop_sizing1 = eng_siz.PropSizing(wing_plan_1[0], w_fus, n_prop, const.c_fp, const.c_pp, MTOM, const.xi_0)

        prop_radius1 = prop_sizing1.radius()
        prop_area1 = prop_sizing1.area_prop()

        prop_sizing2 = eng_siz.PropSizing(wing_plan_2[0], w_fus, n_prop, const.c_fp, const.c_pp, MTOM, const.xi_0)

        prop_radius2 = prop_sizing2.radius()
        prop_area2 = prop_sizing2.area_prop()

        tot_prop_area = prop_area1*n_prop_1 + prop_area2*n_prop_2

        # ----------------------- Performance ------------------------

        V = at.speeds(h_cr, MTOM, CLmax, S_tot, drag)

        # Cruise speed
        V_cr, CL_cr_check = V.cruise()

        # Update the stall speed
        V_stall = V.stall()

        # Cruise CL of the wings
        L_cr = MTOM * g0
        L_cr_1 = L_cr * s1
        L_cr_2 = L_cr * s2

        # Lift coefficients at cruise
        CL_cr_1 = 2 * L_cr_1 / (rho * V_cr ** 2 * S1)
        CL_cr_2 = 2 * L_cr_2 / (rho * V_cr ** 2 * S2)
        C_L_cr = 2 * L_cr / (rho * V_cr ** 2 * S_tot)

        # Aero to pass to mission
        # print("Before aero thingies to pass into mission")
        alpha_lst = np.arange(-3, 89, 0.1)
        Cl_alpha_curve = wing_design.CLa(const.tc, CDs, CDs_f, Afus, alpha_lst, de_da)[0]
        CD_a_w = wing_design.CDa_poststall(const.tc, CDs, CDs_f, Afus, alpha_lst, "wing", drag.CD, de_da)
        CD_a_f = wing_design.CDa_poststall(const.tc, CDs, CDs_f, Afus, alpha_lst, "fus", drag.CD, de_da)

        # Energy sizing
        mission = FP.mission(MTOM, h_cr, V_cr, CLmax, S_tot, tot_prop_area, P_max=max_power,
                             Cl_alpha_curve=Cl_alpha_curve, CD_a_w=CD_a_w, CD_a_f=CD_a_f, alpha_lst=alpha_lst,
                             Drag=drag, t_loiter=15*60, rotational_rate=5, mission_dist=const.mission_range)

        max_thrust_stall = mission.max_thrust(rho, V_stall)

        # Get approximate overall efficiency
        energy, t_tot, P_max_eng_mission, max_thrust, t_hor = mission.total_energy(simplified=False)

        energy_wc = energy

        # Overall efficiency from battery to engine
        eff_overall = const.eff_bat_eng_cr * (t_hor/t_tot) + const.eff_bat_eng_h * (1-(t_hor/t_tot))
        energy = energy * 2.77778e-7 * 1000*const.energy_cont / eff_overall  # From [J] to [Wh]

        # Cruise power
        P_cr, D_cruise = mission.power_cruise_config(h_cr, V_cr, MTOM)

        # Engine sizing

        # Maximum power [W] and thrust [N] of the engines (total)
        # time, P_max_eng_mission, T_max_tot = mission.total_energy()[1:4]
        time = t_tot
        # Maximum TW needed for controllability
        P_max_control = mission.thrust_to_power(TW_ratio_control * MTOM * const.g, 0, rho)[1]


        P_max_eng_tot = max(P_max_eng_mission, P_max_control)

        P_max_eng_ind = P_max_eng_tot/n_prop                    # Maximum power [W] of the engines (per engine)
        P_max_bat = P_max_eng_tot/const.eff_bat_eng_h           # Maximum power [W] to be delivered by the battery

        # Battery sizing
        sp_en_den = const.sp_en_den
        vol_en_den = const.vol_en_den
        batt_cost = const.bat_cost
        DoD = const.DoD
        P_den = const.P_den
        safety_factor = 1  # TODO: discuss
        EOL_C = const.EOL_C

        battery = batt.Battery(sp_en_den, vol_en_den, energy, batt_cost, DoD, P_den, P_max_bat, safety_factor, EOL_C)

        m_bat = battery.mass()

        # -------------------- Update weight ------------------------
        pos_fus = l_fus*0.4
        MAC1 = find_mac(S1, b1, taper)
        MAC2 = find_mac(S2, b2, taper)
        pos_prop_front = [(xf - 0.25*MAC1) - 0.2] * n_prop_1
        pos_prop_back = [(xr - 0.25*MAC2) - 0.2] * n_prop_2

        pos_prop = np.hstack((np.array(pos_prop_front), np.array(pos_prop_back)))

        pos_eng_front = [(xf - 0.25 * MAC1)] * n_prop_1
        pos_eng_back = [(xr - 0.25 * MAC2)] * n_prop_2

        pos_eng = np.hstack((np.array(pos_eng_front), np.array(pos_eng_back)))

        # The mass of one engine is the specific mass of the engines (kg/W) x Total power of ONE ENGINE
        m_prop = const.sp_mass_en * P_max_eng_ind * np.ones(np.shape(pos_prop))

        # Assuming the center of gravity of the landing gears is in their centre.
        # Should not have a lot of affect, as their weights are relatively low
        pos_lgear = (const.x_ng + const.x_tg)/2

        pos_front_wing = [xf + 0.25 * MAC1, zf]
        pos_back_wing = [xr + 0.25 * MAC2, zr]

        # Calculate some mass and balance related things
        MTOM, m_wf, m_wr, m_fus, m_prop_ct, cg_fus0, cg_gear, cg_props, x_CG_MTOM, m_gear, vtail_mass = mass(MTOM, S1, S2, n_ult, AR_wing1,
                                                                                     AR_wing2, pos_front_wing,
                                                                                     pos_back_wing, Pmax_weight, l_fus,
                                                                                     const.n_pax, pos_fus, pos_lgear,
                                                                                     n_prop, m_prop, pos_prop,
                                                                                     const.m_pax,  const.m_cargo_tot,
                                                                                     m_bat, Sv, root_chord_vtail,
                                                                                     contingency = True, cg_bat=cg_bat)

        # ----------------- Stability and control -------------------

        # Hover controllability
        x_f_rotated = xf - xmac_to_xle(const.sweepc41, AR_wing1, taper, b1, const.dihedral1)[0] + 0.45*wing_plan_1[1]
        x_r_rotated = xr - xmac_to_xle(const.sweepc42, AR_wing2, taper, b2, const.dihedral2)[0] + 0.45*wing_plan_2[1]

        TW_ratio_control = 1.3
        check_hover_boolean = False
        while not check_hover_boolean:
            control = HoverControlCalcTandem(MTOM, n_rot_f=n_prop_1, n_rot_r=n_prop_2, x_wf=x_f_rotated, x_wr=x_r_rotated,
                                   rot_y_range_f=[w_fus / 2 + const.c_fp + prop_radius1, wing_plan_1[0]],
                                   rot_y_range_r=[w_fus / 2 + const.c_fp + prop_radius2, wing_plan_2[0]],
                                   K=TW_ratio_control * MTOM * g0 / n_prop, ku=0.1)

            # Find engines that are allowed to fail
            check_hover = control.find_max_allowable_rotor_failures(cg=[x_aft, 0], max_n_failures=1)
            if len(check_hover) > 0:
                check_hover = check_hover[0]

            if check_hover == [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]]:
                check_hover_boolean = True
                print(check_hover, "hover check", TW_ratio_control)

            else:
                TW_ratio_control += 0.1

        # Loading
        cg_pax = [[const.x_f_pax, 0.5, h_fus*0.4], [const.x_f_pax, -0.5, h_fus*0.4], [const.x_r_pax, 0.5, h_fus*0.4],
                  [const.x_r_pax, -0.5, h_fus*0.4]]

        # Approximated with new layout
        cg_pil = [const.x_pil, 0, h_fus/2]  # Pilot is higher than pax
        cg_fus = [l_fus*0.4, 0, h_fus*0.5]

        # print("Before CG from stability")
        cg_calc = CgCalculator(m_wf, m_wr, m_fus, m_bat, const.m_cargo_tot, const.m_pax, const.m_pax,
                               cg_fus=cg_fus, cg_bat=cg_bat, cg_cargo=const.cargo_pos, cg_pax=cg_pax,
                               cg_pil=cg_pil, m_vt = vtail_mass, cg_vt = pos_back_wing[0])

        # Get the cg range, based on wing placement, the loading order can be changed if needed
        cg_wf = [xf + 0.25*MAC1, zf]
        cg_wr = [xr + 0.25*MAC2, zr]

        [x_front, x_aft], _, [_, z_top] = cg_calc.calc_cg_range(cg_wf, cg_wr)
        x_front = float(x_front)
        x_aft = float(x_aft)
        z_top = float(z_top)
        x_ld = cg_calc.calc_cg(cg_wf, cg_wr, True, [0,1,2,3], True)[0]
        print('loading diagram cg', x_front, x_aft, x_ld)

        # Some aerodynamic constants
        CD0fwd = drag.Cd_w(0)
        CD0rear = CD0fwd
        CLafwd = wing_design.liftslope(0)[1][0]  # TODO: unit check
        Clafwd = airfoil.airfoil_stats()[4] * 180/np.pi  # TODO: unit check
        Clarear = airfoil.airfoil_stats()[4] * 180/np.pi  # TODO: unit check

        Zcg = 0.4 * const.h_fuselage  # Estimate

        # Calculate new S1 with new ratio
        S1 = S_tot/(1 + Sr_Sf)
        S2 = S1 * Sr_Sf

        # New span of wing 1
        b1 = np.sqrt(AR_wing1 * S1)
        b2 = np.sqrt(AR_wing2 * S2)

        # Downwash
        de_da = deps_da_empirical(const.sweepc41, b1, xr - xf, zr - zf, AR_wing1, CLafwd, rho, P_cr / n_prop, S1,
                                  CL_cr_1, MTOM * g0)

        # Some wing parameters needed for gear placement
        Cr1 = wing_plan_1[1]                                        # [m] Front root chord
        Cr_rotating_1 = 2*Cr1*(taper - 1)*const.y_tilt_1/b1 + Cr1    # [m] Root chord of the rotating part of the front

        gears = LandingGearCalc(const.x_ng, const.x_tg, const.tw_ng, Cr_rotating_1*(1-const.c_rot_1), const.y_tilt_1,
                                taper*Cr1*(1-const.c_rot_1), b1, taper,
                                zf - xmac_to_xle(const.sweepc41, AR_wing1, taper, b1, const.dihedral1)[1],
                                const.dihedral1, b1/2, 0, prop_radius1, l_fus, const.h_fuselage)

        # Calculate the track width of the tail gear, and the height of the landing gear
        tw_tg, h_lg, reason = gears.optimum_placement([x_front, x_aft],
                                                      z_cg_max=Zcg,
                                                      theta= const.pitch_lim,
                                                      phi=const.lat_lim,
                                                      psi=const.turn_over,
                                                      min_lf = 0.08)  # TODO: Check if this is reasonable

        print(tw_tg, w_fus, reason)
        if tw_tg > 2*w_fus:
            print('Check tail gear track width: ', tw_tg, 'Reason:', reason)

        max_coeffs = wing_design.CLa_wprop(T_per_eng_during_stall, V_stall, rho, prop_radius1 * 2, n_prop_1,
                                           n_prop_2, const.tc_wing, CDs_w, CDs_f, Afus, alpha_wp, de_da)
        CLmf, CLmr = max_coeffs[4], max_coeffs[5]

        CM_a = Cma(Clafwd, Clarear, const.sweepc41, const.sweepc42, taper, taper, CL_cr_1, CL_cr_2, AR_wing1, AR_wing2,
                   const.e_f, const.e_r, xf, xr, zf, zr, Zcg, const.Vr_Vf_2, Sr_Sf, x_aft, S_tot, rho, P_cr/n_prop,
                   MTOM*g0)

        # Load vertical tail
        vertical_tail = vert_tail.VT_sizing(MTOM * g0, h_cr, x_CG_MTOM, l_fus, h_fus, w_fus, V_cr, V_stall, CD0,
                                            CL_cr_1, CL_cr_2, Clafwd, Clarear, S1, S2, AR_wing1, AR_wing2,
                                            const.sweepc41, const.sweepc42, wing_plan_1[3], wing_plan_2[3],
                                            b1, b2, taper, ARv=const.ARv, sweepTE=const.sweep_vtail)

        # nE, Tt0, yE, br_bv, cr_cv, ARv, sweepTE
        v_tail = vertical_tail.final_VT_rudder(prop_radius1, prop_radius2, int(n_prop/4), n_prop, D_cr, const.br_bv,
                                               const.cr_cv, const.ARv,  const.sweep_vtail)

        cg_fwd_lim = xcg_ctrl(const.sweepc41, const.sweepc42, CLmf*const.elev_fac, CLmr, CD0fwd, CD0rear, AR_wing1,
                              AR_wing2, const.e_f, const.e_r, find_mac(S1, b1, taper), find_mac(S2, b2, taper), xf, xr,
                              zf, zr, Zcg, const.Vr_Vf_2, Sr_Sf)

        Sv = v_tail[0]
        root_chord_vtail = v_tail[1]

        # Variables that are updated (the 0 is a placeholder, not used)
        internal_inputs = [MTOM, S_tot, V_cr, h_cr, C_L_cr, CLmax, prop_radius1, de_da, Sv, V_stall, P_max_eng_tot, AR_wing1,
                           AR_wing2, Sr_Sf, s1, xf, zf, xr, zr, max_thrust_stall, root_chord_vtail, TW_ratio_control,
                           x_front, x_aft, l_fus, bat_pos]

        # Aerodynamic moments
        Cmac1 = airfoil.Cm_ac(const.sweepc41, AR_wing1)[0]
        Cmac2 = airfoil.Cm_ac(const.sweepc42, AR_wing2)[0]

        # Redo the mass calculations, but without contingencies. This is done only to store it
        MTOM_nc, m_wf_nc, m_wr_nc, m_fus_nc, m_prop_nc, cg_fus0_nc, \
        cg_gear_nc, cg_props_nc, x_CG_MTOM_nc, m_gear_nc, vtail_mass_nc      = mass(MTOM, S1, S2, n_ult, AR_wing1,
                                                                                    AR_wing2, pos_front_wing,
                                                                                    pos_back_wing, Pmax_weight, l_fus,
                                                                                    const.n_pax, pos_fus, pos_lgear,
                                                                                    n_prop, m_prop, pos_prop,
                                                                                    const.m_pax,  const.m_cargo_tot,
                                                                                    m_bat, Sv, root_chord_vtail,
                                                                                    contingency = False, cg_bat=cg_bat)

        print("xcg and xcg nc", x_CG_MTOM, x_CG_MTOM_nc)

        # Outputs for optimisation cost function
        optim_outputs = [MTOM, energy, time, CM_a, cg_fwd_lim - x_front, MTOM_nc]

        mission_nc = FP.mission(float(MTOM_nc), float(h_cr), float(V_cr), float(CLmax), float(S_tot),
                                                        float(tot_prop_area), P_max=float(max_power),
                                                        Cl_alpha_curve=Cl_alpha_curve, CD_a_w=CD_a_w, CD_a_f=CD_a_f, alpha_lst=alpha_lst,
                                                        Drag=drag, t_loiter=15 * 60, rotational_rate=5, mission_dist=const.mission_range)

        energy_nc, t_tot_nc, P_max_nc, T_max_nc, t_hov_nc = mission_nc.total_energy(simplified=False)

        lines       = [["MAC1", find_mac(S1, b1, taper)],  # Mean Aerodynamic Chord [m]
                       ["MAC2", find_mac(S2, b2, taper)],
                       ["taper", taper],  # [-]
                       ["rootchord1", 2*b1/(AR_wing1*(1+taper))],  # [m]
                       ["rootchord2", 2*b2/(AR_wing2*(1+taper))],
                       ["thicknessChordRatio", const.tc_wing],  # [-]
                       ["xAC", 0.25],  # [-] position of ac with respect to the chord
                       ["MTOM_nc", MTOM_nc],
                       ["MTOM", MTOM],
                       ["AR1", AR_wing1],
                       ["AR2", AR_wing2],
                       ["S1", S1],
                       ["S2", S2],  # surface areas of wing one and two
                       ["span1", b1],
                       ["span2", b2],
                       ["nmax", 3.2],  # maximum load factor
                       ["Pmax", Pmax_weight],
                       # this is defined as maximum perimeter in Roskam, so i took top down view of the fuselage perimeter
                       ["lf", l_fus],  # length of fuselage
                       ["m_pax", const.m_pax],  # average mass of a passenger according to Google
                       ["n_prop", const.n_prop],  # number of engines
                       ["n_pax", const.n_pax],  # number of passengers (pilot included)
                       ["pos_fus", cg_fus0],  # fuselage centre of mass away from the nose
                       ["pos_lgear", cg_gear],  # landing gear position away from the nose
                       ["pos_frontwing", xf],   # Position of the aerodynamic centre of the wing
                       ["pos_backwing", xr],
                       ["zpos_frontwing", zf],  # Position of the aerodynamic centre of the wing
                       ["zpos_backwing", zr],
                       ["m_prop", m_prop],  # list of mass of engines (so 30 kg per engine with nacelle and propeller)
                       ["pos_prop", pos_prop],
                       # 8 on front wing and 8 on back wing
                       ["Mac1", Cmac1],  # aerodynamic moment around AC
                       ["Mac2", Cmac2],
                       ["flighttime", t_tot/3600],  # [hr]
                       ["takeofftime", (t_tot-t_hor)/(2*3600)],
                       ["enginePlacement", pos_eng],  # list(np.linspace(0.1 * b / 2, 0.8 * b / 2, 4)),
                       # engineMass,400 * 9.81 / 8, # See m_prop
                       ["T_max", max_thrust],  # [s] Time in vertical config
                       ["T_max_ctrl", TW_ratio_control],
                       ["battery_pos", cg_bat[0]],  # [m] Battery x-position
                       ["cargo_m", const.m_cargo_tot],
                       ["cargo_pos", const.cargo_pos[0]],
                       ["battery_m", m_bat],
                       ["P_max", max_power],  # [W] Maximum disk power needed
                       ["vol_bat", energy/ const.vol_en_den],
                       ["price_bat", const.bat_cost*energy/1000],
                       ["h_winglet_1", h_wt_1],
                       ["h_winglet_2", h_wt_2],
                       ["V_cruise", V_cr],
                       ["h_cruise", h_cr],
                       ["CLmax", CLmax],
                       ["CD0fwd", CD0fwd],
                       ["CD0fwd", CD0rear],
                       ["CD0", CD0],
                       ["Clafwd", Clafwd],
                       ["Clarear", Clarear],
                       ["CL_cr_fwd", CL_cr_1],
                       ["CL_cr_rear", CL_cr_2],
                       ["CL_cr", C_L_cr],
                       ["P_br_cruise_per_engine", P_cr/n_prop],
                       ["T_cr_per_engine", D_cruise/n_prop],
                       ["x_cg_MTOM_nc", x_CG_MTOM_nc],
                       ["Prop_radius_front", prop_radius1],
                       ["Prop_radius_back", prop_radius2],
                       ["Disk_load_front", 0.5*MTOM/(prop_area1*n_prop/2)],
                       ["Disk_load_back", 0.5*MTOM/(prop_area2*n_prop/2)],
                       ["Root chord vertical tail", root_chord_vtail],
                       ["Front_wing_mass", m_wf_nc],
                       ["Rear_wing_mass", m_wr_nc],
                       ["Fuselage_mass", m_fus_nc],
                       ["Landing_gear_mass", m_gear_nc],
                       ["Vertical_tail_mass", vtail_mass],
                       ["Propulsion_mass", m_prop_nc],
                       ["Energy_nc", energy_nc],
                       ["Energy", energy_wc],
                       ["Cr_vert", root_chord_vtail],
                       ["m_v_tail", vtail_mass],
                       ["S_vtail", Sv],
                       ["b_vtail", v_tail[3]]]

        txt = open("final_values.txt", 'w')
        txt.truncate(0)
        for element in lines:
            txt.write(element[0] + " = " + str(element[1]) + "\n")
        txt.close()

        # mission_nc = FP.mission(float(MTOM_nc), float(h_cr), float(V_cr), float(CLmax), float(S_tot),
        #                         float(tot_prop_area), P_max=float(max_power),
        #                         Cl_alpha_curve=Cl_alpha_curve, CD_a_w=CD_a_w, CD_a_f=CD_a_f, alpha_lst=alpha_lst,
        #                         Drag=drag, t_loiter=15 * 60, rotational_rate=5, mission_dist=const.mission_range)
        #
        # T_cr_nc, P_cr_nc = mission_nc.power_cruise_config(h_cr, V_cr, MTOM)
        #
        # E_tot_nc, t_tot_nc, P_max_nc, T_max_nc, t_hov_nc = mission_nc.total_energy(simplified=True)

        # print(E_tot_nc, t_tot_nc, P_max_nc, T_max_nc, t_hov_nc, T_cr_nc, P_cr_nc)
        #
        # print("MTOM:            ", MTOM)
        # print("     - Battery:  ", m_bat)
        print('MTOM', MTOM_nc)
        print("     - Wing fore:", m_wf, AR_wing1)
        print("     - Wing aft: ", m_wr, AR_wing2)
        print("Mass battery:", m_bat)
        # print("     - Fuselage: ", m_fus)
        # print("     - pax:      ", const.m_pax * const.n_pax)
        # print("Total based on components:",
        #       m_bat + m_wf + m_wr + m_fus + const.m_pax * const.n_pax + const.m_cargo_tot + m_gear)
        # print("MTOW w/o contingency:", MTOM_nc)
        # print("Cruise speed:    ", V_cr)
        # print("Max power:       ", max_power)
        # print("Energy used:     ", energy * 3.6e-3)
        # print("Wing surface:    ", S_tot)
        # print("Wing spans:      ", b1, b2)
        # print("CM alpha:        ", CM_a)
        # print("Controllability: ", cg_fwd_lim - x_front)

        # Other necessary outputs
        other_outputs = [tw_tg, None]
        # print("end")
        return optim_outputs, internal_inputs, other_outputs

    def multirun(self, N_iters, optim_inputs):
        """
        With this you can run the integrated code as many times as you want per optimisation, so that you get internal
        convergence of the internal parameters

        :param N_iters: Number of iterations of the code for each optimisation iteration
        """
        internal_inputs = self.initial_est

        for i in range(N_iters):
            # print("Iteration #", i)
            # print("")
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
