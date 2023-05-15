from dataclasses import dataclass
import numpy as np

import constants as const

import stab_and_ctrl.hover_controllabilty as hc
import stab_and_ctrl.landing_gear_placement as lgp
import stab_and_ctrl.loading_diagram as ld

# TODO: account for nose gear, battery, front wing, and pilot
#  interfering with each other in the front


@dataclass
class Fuselage:
    """
    Class containing all input fields from the fuselage to stability and
    controllability
    @author Jakob Schoser
    """
    m_empty: float  # empty fuselage mass (excl. battery, incl. vertical
    #                      tail and landing gear)
    m_cargo: float  # cargo mass
    m_pil: float  # pilot mass
    m_ppax: float  # mass per passenger

    cg_empty: list  # empty fuselage CG
    cg_cargo: list  # cargo CG
    cg_pil: list  # pilot CG
    cg_pax: list  # list of passenger CGs

    max_tw: float  # maximum track width
    min_x_ng: float  # most forward allowable nose gear position
    us_bp: list  # fuselage up-sweep bottom point [x, z]
    us_tp: list  # fuselage up-sweep top point [x, z]

    l: float  # fuselage length
    h: float  # fuselage height
    w: float  # fuselage width


@dataclass
class Wing:
    """
    Class containing all input fields from a wing to stability and
    controllability
    @author Jakob Schoser
    """
    xcg: float
    zcg: float

    m: float  # wing mass (excl. engines)
    Cd0_af: float  # zero-lift drag coefficient (aerofoil)
    Cd0: float  # zero-lift drag coefficient (wing)
    e: float  # span efficiency factor
    CLmax: float  # maximum lift coefficient of the wing
    CLdes: float  # design CL of the wing (cruise)
    Clalpha_af: float  # lift curve slope of the aerofoil
    Cmac: float  # pitching moment coefficient
    S: float  # surface area
    mac: float  # mean aerodynamic chord length
    AR: float  # aspect ratio
    taper: float  # taper ratio
    y_b_eng: list  # placement range of rotors
    h_eng: float  # height of the engines with respect to the wing
    Gamma: float  # dihedral angle in radians
    lambda_c4: float  # quarter-chord sweep


@dataclass
class PropAndPower:
    """
    Class containing all input fields from propulsion and power to stability
    and controllability
    @author Jakob Schoser
    """
    n_engf: int  # number of engines on front wing
    n_engr: int  # number of engines on rear wing
    m_peng: float  # mass per engine (incl. propeller)
    max_T_hover_pe: float  # maximum thrust per engine in hover
    P_shaft_cruise_pe_f: float  # shaft power per front engine during cruise
    T_cruise_f: float  # thrust of front engines combined during cruise
    T_cruise_r: float  # thrust of rear engines combined during cruise
    prop_diam: float  # propeller diameter
    m_bat: float  # battery mass


@dataclass
class Performance:
    """
    Class containing all input fields from general performance to stability and
    controllability
    @author Jakob Schoser
    """
    v_stall: float  # stall speed
    v_cruise: float  # cruise speed
    v_max: float  # maximum speed
    alt: float  # altitude


@dataclass
class StabAndControlOutputs:
    """
    Class containing all output fields from stability and controllability
    @author Jakob Schoser
    """
    xcg_range: list
    ycg_range: list
    zcg_range: list
    cruise_stable: bool  # static longitudinal stability in cruise?
    stall_controllable: bool  # can be trimmed at stall?
    hover_controllable: bool  # controllable in hover?
    hover_controllable_oei: bool  # controllable in hover with OEI?
    x_ng: float  # x-coordinate of the nose gear
    x_mlg: float  # x-coordinate of the main landing gear
    tw: float  # track width of the main landing gear
    h_mlg: float  # height of the main landing gear


def stab_and_ctrl_main(fus: Fuselage, wf: Wing, wr: Wing, pnp: PropAndPower,
                       perf: Performance):
    """
    Function that takes input values to calculate relevant outputs based on
    stability and control considerations.
    @author Jakob Schoser
    :param fus: Fuselage parameters
    :param wf: Forward wing parameters
    :param wr: Rear wing parameters
    :param pnp: Propulsion and power parameters
    :param perf: General performance parameters
    :return:
    """
    x_cg_margin = 0.1
    theta = np.deg2rad(15)
    phi = np.deg2rad(7)
    psi = np.deg2rad(55)
    min_ng_load_frac = 0.08
    ku = 0.1
    n_failures = 1
    dx = 0.05
    wf_offset_x = 0
    wf_offset_z = 0
    elev_eff = 1.4
    Sfwd_Srear = wf.S / wr.S

    out = StabAndControlOutputs(
        [3, 0, 0.2],
        1,
        [1/4 * wf.mac + wf_offset_x, 0.2 + wf_offset_z],
        [fus.l - 3/4 * wr.mac, 1.5],
        1,
        4,
        1.5,
        0.5,
        [i < pnp.n_engf // 2 or 0 <= i - pnp.n_engf < pnp.n_engr // 2
         for i in range(pnp.n_engf + pnp.n_engr)],
        TailAndCtrlSurf()
    )

    done = False
    while not done:
        m_wf = wf.m + pnp.n_engf * pnp.m_peng
        m_wr = wr.m + pnp.n_engr * pnp.m_peng

        mtom = (fus.m_empty + fus.m_pil + fus.m_cargo +
                fus.m_ppax * len(fus.cg_pax) + pnp.m_bat + m_wf + m_wr)

        bf = np.sqrt(wf.S * wf.AR)
        br = np.sqrt(wr.S * wr.AR)

        x_min_test, x_max_test = 0, fus.l

        cgcalc = ld.CgCalculator(m_wf, m_wr, fus.m_empty, pnp.m_bat,
                                 fus.m_cargo, fus.m_ppax, fus.m_pil,
                                 fus.cg_empty, out.cg_bat, fus.cg_cargo,
                                 fus.cg_pax, fus.cg_pil)
        x_range, y_range, z_range = cgcalc.calc_cg_range(out.cg_wf, out.cg_wr)

        lgcalc = lgp.LandingGearCalc(fus.max_tw, fus.min_x_ng,
                                     wf.y_b_eng[1] * np.sqrt(wf.AR * wf.S),
                                     wf.Gamma, wf.h_eng + out.cg_wf[1],
                                     pnp.prop_diam/2, fus.us_bp, fus.us_tp)
        out.x_ng, out.x_mlg, out.tw, out.h_mlg = lgcalc.optimum_placement(
            x_range, x_cg_margin, z_range[1], theta, phi, psi, min_ng_load_frac
        )
        if out.tw is None:
            # TODO: do something to fix it and iterate
            print("Warning: track width too large!")

        hcct = hc.HoverControlCalcTandem(mtom, pnp.n_engf, pnp.n_engf,
                                         out.cg_wf[0], out.cg_wr[0],
                                         wf.y_b_eng * bf, wr.y_b_eng * br,
                                         pnp.max_T_hover_pe, ku)
        x_min, x_max, _, _ = hcct.calc_crit_x_cg_range(
            x_max_test, x_max_test, dx, [x_range[1], y_range[1]], [n_failures]
        )[0]

        if (x_min is None or x_range[0] < x_min or x_range[1] < x_min
                or x_range[0] > x_max or x_range[1] > x_max):
            # TODO: do something to fix it and iterate
            print("Warning: Uncontrollable in hover!")

        # FIXME: assumes that both wings have the same Cd0, Gamma and taper
        wps = sp.Wing_placement_sizing(mtom * const.g, perf.alt, fus.l, fus.h,
                                       fus.w, perf.v_cruise, wf.Cd0, wf.CLmax,
                                       wr.CLmax, wf.CLdes, wf.CLdes,
                                       wf.Clalpha_af, wf.Clalpha_af, wf.Cmac,
                                       wr.Cmac, wf.S, wr.S, wf.AR, wr.AR,
                                       wf.Gamma, wf.lambda_c4, wr.lambda_c4,
                                       wf.mac, wr.mac, bf, br, wf.e, wr.e,
                                       wf.taper, pnp.n_engf, pnp.n_engr,
                                       wf.y_b_eng * bf, wr.y_b_eng * br,
                                       pnp.max_T_hover_pe, ku, z_range[1],
                                       wf_offset_x, wf_offset_z,
                                       pnp.P_shaft_cruise_pe_f)

        Sfwd_Srear_stab, Sfwd_Srear_ctrl = wps.Sr_Sfwd(np.array(x_range),
                                                       elev_eff, wf_offset_x)

        # FIXME: assumes that the CG shifts forward with increasing Swfd/Srear
        if Sfwd_Srear_stab[0] < Sfwd_Srear or Sfwd_Srear_stab[1] < Sfwd_Srear:
            # TODO: do something to fix it and iterate
            print("Warning: Statically unstable in cruise!")

        if Sfwd_Srear_ctrl[0] > Sfwd_Srear or Sfwd_Srear_ctrl[1] > Sfwd_Srear:
            # TODO: do something to fix it and iterate
            print("Warning: Uncontrollable in cruise!")
