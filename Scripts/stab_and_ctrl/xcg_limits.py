import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt

from Preliminary_Lift.LLTtest2 import downwash
from Preliminary_Lift.Airfoil_analysis import Cm_ac


def xcg_stab(CLaf, CLar, CLf, CLr, Af, Ar, ef, er, xf, xr, zf, zr, zcg,
             Vr_Vf_2, Sr_Sf, de_da):
    """
    Calculate the maximum CG x-position for static longitudinal stability
    at cruise conditions.
    :param CLaf: Lift slope of the front wing
    :param CLar: Lift slope of the rear wing
    :param CLf: Lift coefficient of the front wing (cruise)
    :param CLr: Lift coefficient of the rear wing (cruise)
    :param Af: Aspect ratio of the front wing
    :param Ar: Aspect ratio of the rear wing
    :param ef: Span efficiency factor of the front wing
    :param er: Span efficiency factor of the front wing
    :param xf: x-position of the aerodynamic centre of the front wing
    :param xr: x-position of the aerodynamic centre of the rear wing
    :param zf: z-position of the aerodynamic centre of the front wing
    :param zr: z-position of the aerodynamic centre of the rear wing
    :param zcg: z-position of the CG
    :param Vr_Vf_2: Velocity ratio between front and rear wing, squared
    :param Sr_Sf: Rear wing surface area / front wing surface area
    :param de_da: Downwash gradient of front wing on rear wing
    :return: maximum CG x-position for static longitudinal stability
    at cruise conditions
    """
    num = (2 * CLf / (np.pi * Af * ef) * CLaf * (zcg - zf)
           - 2 * CLr / (np.pi * Ar * er) * CLar * (zr - zcg) * (1 - de_da)
           + CLaf * xf
           + CLar * xr * Vr_Vf_2 * Sr_Sf * (1 - de_da))

    den = CLaf + CLar * Vr_Vf_2 * Sr_Sf * (1 - de_da)

    return num / den


def Cma(Claf, Clar, lambda_c4f, lambda_c4r, taperf, taperr, CLf, CLr, Af, Ar,
        ef, er, xf, xr, zf, zr, zcg, Vr_Vf_2, Sr_Sf, xcg, S, rho, Pbr, W):
    lambda_c2f = lambda_c4_to_lambda_c2(Af, taperf, lambda_c4f)
    lambda_c2r = lambda_c4_to_lambda_c2(Ar, taperr, lambda_c4r)
    CLaf = CLa(Claf, Af, lambda_c2f)
    CLar = CLa(Clar, Ar, lambda_c2r)
    bf,  _ = bf_br(S, Sr_Sf, Af, Ar)
    Sf = S / (1 + Sr_Sf)
    Sr = S - Sf
    bf = np.sqrt(Sf * Af)
    br = np.sqrt(Sr * Ar)
    macf = find_mac(Sf, bf, taperf)
    macr = find_mac(Sr, br, taperr)
    mac = (macf * Sf + macr * Sr) / S

    de_da = deps_da_empirical(lambda_c4f, bf, xr - xf, zr - zf, Af, CLaf,
                              rho, Pbr, Sf, CLf, W)
    return (CLaf * (xcg - xf)
            - CLar * (xr - xcg) * (1 - de_da) * Vr_Vf_2 * Sr_Sf
            - 2 * CLf / (np.pi * Af * ef) * CLaf * (zcg - zf)
            + 2 * CLr / (np.pi * Ar * er) * CLar * (zr - zcg)
            * (1 - de_da) * Vr_Vf_2 * Sr_Sf) * Sf / (S * mac)


# TODO: improve downwash estimation
def deps_da_not_use(bf, br, crf, crr, ctf, ctr, lambda_c4f, lambda_c4r, Sf, Af,
                    CLf, alphaf, lh, h_ht, rho, Pbr, W, V):
    """
    Estimate the downwash gradient of the front wing on the rear wing based
    on lifting-line theory (accounting for additional downwash due to
    propellers).
    :param bf:
    :param br:
    :param crf:
    :param crr:
    :param ctf:
    :param ctr:
    :param lambda_c4f:
    :param lambda_c4r:
    :param Sf:
    :param Af:
    :param CLf:
    :param alphaf:
    :param lh:
    :param h_ht:
    :param rho:
    :param Pbr:
    :param W:
    :param V:
    :return:
    """
    r = lh * 2 / bf
    mtv = h_ht * 2 / bf  # Approximation
    de_da = downwash(bf, Af, crf, ctf, lambda_c4f, alphaf, h_ht, lh, br, crr, ctr, lambda_c4r, V)
    phi = np.arcsin(mtv/r)
    # print("Check", rho, Pbr, Sf, CLf, lh, W, phi)
    dsde_da = np.where(
        np.logical_and(np.rad2deg(phi) < 30, np.rad2deg(phi) > 0),
        6.5 * (rho * Pbr ** 2 * Sf ** 3 * CLf ** 3 / (lh ** 4 * W ** 3)) ** (1 / 4) * (np.sin(phi * 6)) ** 2.5,
        0
    )

    return de_da+dsde_da


def deps_da_empirical(lambda_c4f, bf, lh, h_ht, A, CLaf, rho, Pbr, Sf, CLf, W):
    """
    Estimate the downwash gradient of the front wing on the rear wing
    (accounting for additional downwash due to propellers).
    :param lambda_c4f: Quarter-chord sweep of the front wing
    :param bf: Span of the front wing
    :param lh: x-distance between aerodynamic centres of front and rear wing
    :param h_ht: Distance between wings normal to their chord planes
    :param A:  Aspect ratio of the front wing
    :param CLaf: Lift slope of the front wing
    :param rho: Air density
    :param Pbr: Brake power of one engine (half of the engines?) on front wing
    :param Sf: Surface area of the front wing
    :param CLf: Lift coefficient of the front wing
    :param W: Aircraft weight
    :return: Downwash gradient of the front wing on the rear wing
    """
    r = lh * 2 / bf
    mtv = h_ht * 2 / bf  # Approximation
    Keps = (0.1124 + 0.1265 * lambda_c4f + 0.1766 * lambda_c4f ** 2) / r ** 2 + 0.1024 / r + 2
    Keps0 = 0.1124 / r ** 2 + 0.1024 / r + 2
    v = 1 + (r ** 2 / (r ** 2 + 0.7915 + 5.0734 * mtv ** 2)) ** (0.3113)
    de_da = Keps / Keps0 * CLaf / (np.pi * A) * (
            r / (r ** 2 + mtv ** 2) * 0.4876 / (np.sqrt(r ** 2 + 0.6319 + mtv ** 2)) + v * (
            1 - np.sqrt(mtv ** 2 / (1 + mtv ** 2))))
    phi = np.arcsin(mtv/r)
    dsde_da = np.where(
        np.logical_and(np.rad2deg(phi) < 30, np.rad2deg(phi) > 0),
        6.5 * (rho * Pbr ** 2 * Sf ** 3 * CLf ** 3 / (lh ** 4 * W ** 3)) ** (1 / 4) * (np.sin(phi * 6)) ** 2.5,
        0
    )

    return de_da/2+dsde_da



# TODO: implement effect of aspect ratio on Cmac
def xcg_ctrl(lambda_c4f, lambda_c4r, CLf, CLr, CD0f, CD0r, Af, Ar, ef,
             er, macf, macr, xf, xr, zf, zr, zcg, Vr_Vf_2, Sr_Sf):
    """
    Calculate the minimum CG x-position for pitch controllability at stall.
    :param lambda_c4f: Quarter-chord sweep of the front wing
    :param lambda_c4r: Quarter-chord sweep of the rear wing
    :param CLf: Lift coefficient of the front wing (maximum incl. elevator)
    :param CLr: Lift coefficient of the rear wing (maximum incl. elevator)
    :param CD0f: Zero-lift drag coefficient of front wing
    :param CD0r: Zero-lift drag coefficient of rear wing
    :param Af: Aspect ratio of the front wing
    :param Ar: Aspect ratio of the rear wing
    :param ef: Span efficiency factor of the front wing
    :param er: Span efficiency factor of the front wing
    :param macf: Mean aerodynamic chord of front wing
    :param macr: Mean aerodynamic chord of rear wing
    :param xf: x-position of the aerodynamic centre of the front wing
    :param xr: x-position of the aerodynamic centre of the rear wing
    :param zf: z-position of the aerodynamic centre of the front wing
    :param zr: z-position of the aerodynamic centre of the rear wing
    :param zcg: z-position of the CG
    :param Vr_Vf_2: Velocity ratio between front and rear wing, squared
    :param Sr_Sf: Rear wing surface area / front wing surface area
    :return: Minimum CG x-position for pitch controllability at stall
    """
    Cmacf = Cm_ac(lambda_c4f, Af)[0]
    Cmacr = Cm_ac(lambda_c4r, Ar)[0]

    num = (-Cmacf
           - Cmacr * Vr_Vf_2 * Sr_Sf * macr / macf
           + CLf * xf / macf
           + CLr * xr / macf * Vr_Vf_2 * Sr_Sf
           + (CD0f + CLf**2 / (np.pi * Af * ef)) * (zcg - zf) / macf
           - (CD0r + CLr**2 / (np.pi * Ar * er)) * (zr - zcg) / macf * Vr_Vf_2 * Sr_Sf)

    den = CLf / macf + CLr / macf * Vr_Vf_2 * Sr_Sf

    return num / den


def get_Sr_Sf_for_controllability(lambda_c4f, lambda_c4r, CLf, CLr, elev_fac,
                                  CD0f, CD0r, Af, Ar, ef, er, macf, macr, xf,
                                  xr, zf, zr, zcg, xcg_front, Vr_Vf_2):
    """
    Calculate the maximum Sr_Sf to place the controllability point in front
    of the front CG position.
    :param lambda_c4f:
    :param lambda_c4r:
    :param CLf:
    :param CLr:
    :param elev_fac:
    :param CD0f:
    :param CD0r:
    :param Af:
    :param Ar:
    :param ef:
    :param er:
    :param macf:
    :param macr:
    :param xf:
    :param xr:
    :param zf:
    :param zr:
    :param zcg:
    :param xcg_front:
    :param Vr_Vf_2:
    :return:
    """
    CLf = CLf * elev_fac

    Cmacf = Cm_ac(lambda_c4f, Af)[0]
    Cmacr = Cm_ac(lambda_c4r, Ar)[0]

    num = (-Cmacf
           - CLf * (xcg_front - xf) / macf
           + (CD0f + CLf ** 2 / (np.pi * Af * ef)) * (zcg - zf) / macf)

    den = (Cmacr * Vr_Vf_2 * macr / macf
           - CLr * Vr_Vf_2 * (xr - xcg_front) / macf
           + (CD0r + CLr**2 / (np.pi * Ar * er)) * Vr_Vf_2 * (zr - zcg) / macf)

    return num / den


def CLa(Cla, A, lambda_c2, M=0, eta=0.95):
    """
    Calculate wing lift slope based on aerofoil lift slope
    :param Cla: Aerofoil lift slope
    :param A: Wing aspect ratio
    :param lambda_c2: Wing half-chord sweep angle [rad]
    :param M: Mach number
    :param eta: Ratio of Cla to 2pi, but should be kept constant at 0.95
     according to source
    :return: Wing lift slope
    """
    beta = np.sqrt(1 - M ** 2)
    return Cla * A / (2 + np.sqrt(4 + ((A * beta / eta) ** 2)
                                  * (1 + (np.tan(lambda_c2) / beta) ** 2)))


def lambda_c4_to_lambda_c2(A, taper, lambda_c4):
    """
    Calculate half-chord sweep based on quarter-chord sweep
    :param A: Wing aspect ratio
    :param taper: Wing taper
    :param lambda_c4: Wing quarter-chord sweep
    :return: Wing half-chord sweep
    """
    tanSweep_c4 = np.tan(lambda_c4)
    tanSweep_c2 = tanSweep_c4 - 4/A*(50-25)/100*(1-taper)/(1+taper)
    return np.arctan(tanSweep_c2)


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


def crf_crr(S, Sr_Sf, Af, Ar, taperf, taperr):
    """
    Calculate front and rear root chord for a given tandem wing configuration
    :param S: Total wing surface area
    :param Sr_Sf: Front wing surface area / rear wing surface area
    :param Af: Front wing aspect ratio
    :param Ar: Rear wing aspect ratio
    :param taperf: Front wing taper
    :param taperr: Rear wing taper
    :return: front wing root chord, rear wing root chord
    """
    Sf = S / (1 + Sr_Sf)
    Sr = S - Sf
    bf = np.sqrt(Sf * Af)
    br = np.sqrt(Sr * Ar)
    cavgf = Sf / bf
    cavgr = Sr / br
    return cavgf * 2 / (1 + taperf), cavgr * 2 / (1 + taperr)


def bf_br(S, Sr_Sf, Af, Ar):
    """
    Calculate front and rear wingspan for a given tandem wing configuration
    :param S: Total wing surface area
    :param Sr_Sf: Front wing surface area / rear wing surface area
    :param Af: Front wing aspect ratio
    :param Ar: Rear wing aspect ratio
    :return: front wingspan, rear wingspan
    """
    Sf = S / (1 + Sr_Sf)
    Sr = S - Sf
    return np.sqrt(Sf * Af), np.sqrt(Sr * Ar)


def xcg_limits(CLmaxf, CLmaxr, CLdesf, CLdesr, CD0f, CD0r,
               taperf, taperr, lambda_c4f, lambda_c4r, ef, er, Claf, Clar,
               zcg, Vr_Vf_2, elev_fac, rho, Pbr, S, W, Af, Ar, xf, xr, zf,
               zr, Sr_Sf):
    """
    Find the front and aft limits on the CG for controllability and stability.
    This combines the functions xcg_ctrl and xcg_stab for use in the
    optimisation.
    :param CLmaxf:
    :param CLmaxr:
    :param CLdesf:
    :param CLdesr:
    :param CD0f:
    :param CD0r:
    :param taperf:
    :param taperr:
    :param lambda_c4f:
    :param lambda_c4r:
    :param ef:
    :param er:
    :param Claf:
    :param Clar:
    :param zcg:
    :param Vr_Vf_2:
    :param elev_fac:
    :param rho:
    :param Pbr:
    :param S:
    :param W:
    :param Af:
    :param Ar:
    :param xf:
    :param xr:
    :param zf:
    :param zr:
    :param Sr_Sf:
    :return:
    """
    Sf = S / (1 + Sr_Sf)
    Sr = S - Sf
    bf = np.sqrt(Sf * Af)
    br = np.sqrt(Sr * Ar)
    macf = find_mac(Sf, bf, taperf)
    macr = find_mac(Sr, br, taperr)
    lambda_c2f = lambda_c4_to_lambda_c2(Af, taperf, lambda_c4f)
    lambda_c2r = lambda_c4_to_lambda_c2(Ar, taperr, lambda_c4r)
    CLaf = CLa(Claf, Af, lambda_c2f)
    CLar = CLa(Clar, Ar, lambda_c2r)
    de_da = deps_da_empirical(lambda_c4f, bf, xr - xf, zr - zf, Af, CLaf, rho, Pbr, Sf, CLdesf, W)
    xstab = xcg_stab(CLaf, CLar, CLdesf, CLdesr, Af, Ar, ef, er, xf, xr, zf,
                     zr, zcg, Vr_Vf_2, Sr_Sf, de_da)
    xctrl = xcg_ctrl(lambda_c4f, lambda_c4r, CLmaxf * elev_fac, CLmaxr, CD0f, CD0r, Af,
                     Ar, ef, er, macf, macr, xf, xr, zf, zr, zcg, Vr_Vf_2,
                     Sr_Sf)
    return np.array([xctrl, xstab])


def optimise_wings(CLmaxf, CLmaxr, CLdesf, CLdesr, CD0f, CD0r,
                   taperf, taperr, lambda_c4f, lambda_c4r, ef, er, Claf, Clar,
                   zcg, Vr_Vf_2, elev_fac, rho, Pbr, S, W, xrangef,
                   xranger, zrangef, zranger, crmaxf, crmaxr, bmaxf, bmaxr,
                   Arangef, Aranger, xcg_range, impose_stability=True,
                   init_Af=7, init_Ar=7, init_xf=0.5, init_xr=6.5, init_zf=0.5,
                   init_zr=1.5, init_Sr_Sf=1.):
    # FIXME: span efficiency factor is assumed to be constant
    """
    Optimise for maximum wingspan on the front wing while meeting stability
    and controllability requirements and structural constraints.

    Parameters to be optimised:
        - Aspect ratios of both wings
        - x-positions of both wings
        - z-positions of both wings
        - Sr/Sf (relative size of the wings)

    Constraints:
        - Allowable aspect ratio range for both wings
        - Allowable x-range for both wings
        - Allowable z-range for both wings
        - Maximum root chord for both wings
        - Maximum wingspan for both wings
        - CG such that aircraft is always controllable (and stable)

    Optimisation goal:
        - Make wingspans of front and rear wing as similar as possible


    :param CLmaxf:
    :param CLmaxr:
    :param CLdesf:
    :param CLdesr:
    :param CD0f:
    :param CD0r:
    :param taperf:
    :param taperr:
    :param lambda_c4f:
    :param lambda_c4r:
    :param ef:
    :param er:
    :param Claf:
    :param Clar:
    :param zcg:
    :param Vr_Vf_2:
    :param elev_fac:
    :param rho:
    :param Pbr:
    :param S:
    :param W:
    :param xrangef:
    :param xranger:
    :param zrangef:
    :param zranger:
    :param crmaxf:
    :param crmaxr:
    :param bmaxf:
    :param bmaxr:
    :param xcg_range:
    :return:
    """
    # initialise vector of initial conditions
    x0 = np.array([init_Af, init_Ar, init_xf, init_xr,
                   init_zf, init_zr, init_Sr_Sf])

    # TODO: minimise the instability instead
    # cost function to choose between solutions
    def cost(x):
        return (bf_br(S, x[6], x[0], x[1])[0]
                - bf_br(S, x[6], x[0], x[1])[1]) ** 2

    # wrapper function for the CG limits for the optimisation of x
    def find_cg_limits(x):
        return xcg_limits(CLmaxf, CLmaxr, CLdesf, CLdesr, CD0f,
                          CD0r, taperf, taperr, lambda_c4f, lambda_c4r, ef,
                          er, Claf, Clar, zcg, Vr_Vf_2, elev_fac, rho, Pbr,
                          S, W, x[0], x[1], x[2], x[3], x[4], x[5], x[6])

    # apply direct constraints to input parameters
    input_constr = optimize.LinearConstraint(
        np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0]
        ]),
        np.array([Arangef[0], Aranger[0], xrangef[0],
                  xranger[0], zrangef[0], zranger[0]]),
        np.array([Arangef[1], Aranger[1], xrangef[1],
                  xranger[1], zrangef[1], zranger[1]])
    )

    # apply constraints indirectly related to input parameters
    crmaxf_constr = optimize.NonlinearConstraint(
        lambda x: crf_crr(S, x[6], x[0], x[1], taperf, taperr)[0],
        0, crmaxf
    )
    crmaxr_constr = optimize.NonlinearConstraint(
        lambda x: crf_crr(S, x[6], x[0], x[1], taperf, taperr)[1],
        0, crmaxr
    )
    bmaxf_constr = optimize.NonlinearConstraint(
        lambda x: bf_br(S, x[6], x[0], x[1])[0],
        0, bmaxf
    )
    bmaxr_constr = optimize.NonlinearConstraint(
        lambda x: bf_br(S, x[6], x[0], x[1])[1],
        0, bmaxr
    )

    # depending whether stability is required or not,
    # apply the appropriate constraints
    if impose_stability:
        xcgrange_constr = optimize.NonlinearConstraint(
            find_cg_limits,
            np.array([-np.inf, xcg_range[1]]), np.array([xcg_range[0], np.inf])
        )
    else:
        xcgrange_constr = optimize.NonlinearConstraint(
            lambda x: find_cg_limits(x)[0],
            0, xcg_range[0]  # TODO: come back to this
        )

    # optimise
    result = optimize.minimize(cost, x0, method='trust-constr',
                               constraints=[
                                   input_constr,
                                   crmaxf_constr,
                                   crmaxr_constr,
                                   bmaxf_constr,
                                   bmaxr_constr,
                                   xcgrange_constr
                               ])

    if not result.success:
        return None

    # for some reason, the optimisation sometimes terminates successfully
    # even if stability and control constraints are not satisfied. Here,
    # we filter out those cases
    cg_limits = find_cg_limits(result.x)
    if xcg_range[0] < cg_limits[0] or xcg_range[1] < cg_limits[0]:
        # return None
        print("First wingop if statement executed")
    if (impose_stability and xcg_range[0] > cg_limits[1]
            or xcg_range[1] > cg_limits[1]):
        print("Second wingop if statement executed")

    return result.x


def plot_Sf_Sr_Af_plane(CLmaxf, CLmaxr, CLdesf, CLdesr, CD0f,
                        CD0r, taperf, taperr, lambda_c4f, lambda_c4r, ef, er,
                        Claf, Clar, zcg, Vr_Vf_2, elev_fac, rho, Pbr, S, W,
                        Ar, xf, xr, zf, zr, xcg_range, plot_Sr_Sf_range,
                        plot_Af_range, Sr_Sf=None, Af=None, res=100):
    """
    Create a plot to visualise the effect of the relative wing sizes and
    front wing aspect ratio (the most powerful parameters) on the centre of
    gravity limits.
    :param CLmaxf:
    :param CLmaxr:
    :param CLdesf:
    :param CLdesr:
    :param CD0f:
    :param CD0r:
    :param taperf:
    :param taperr:
    :param lambda_c4f:
    :param lambda_c4r:
    :param ef:
    :param er:
    :param Claf:
    :param Clar:
    :param zcg:
    :param Vr_Vf_2:
    :param elev_fac:
    :param rho:
    :param Pbr:
    :param S:
    :param W:
    :param Ar:
    :param xf:
    :param xr:
    :param zf:
    :param zr:
    :param xcg_range:
    :param plot_Sr_Sf_range:
    :param plot_Af_range:
    :param Sr_Sf:
    :param Af:
    :param res:
    :return:
    """
    Sr_Sf_range = np.linspace(plot_Sr_Sf_range[0], plot_Sr_Sf_range[1], res)
    Af_range = np.linspace(plot_Af_range[0], plot_Af_range[1], res)
    Sr_Sf_grid, Af_grid = np.meshgrid(Sr_Sf_range, Af_range)

    xcg_ctrl_grid, xcg_stab_grid = xcg_limits(CLmaxf, CLmaxr,
                                              CLdesf, CLdesr, CD0f, CD0r,
                                              taperf, taperr, lambda_c4f,
                                              lambda_c4r, ef, er, Claf, Clar,
                                              zcg, Vr_Vf_2, elev_fac, rho,
                                              Pbr, S, W, Af_grid, Ar, xf, xr,
                                              zf, zr, Sr_Sf_grid)

    plt.xlabel(r"$S_r/S_f$")
    plt.ylabel(r"$A_f$")
    margin = Cma(Claf, Clar, lambda_c4f, lambda_c4r, taperf, taperr, CLdesf,
                 CLdesr, Af_grid, Ar, ef, er, xf, xr, zf, zr, zcg, Vr_Vf_2,
                 Sr_Sf_grid, xcg_range[1], S, rho, Pbr, W)
    # margin = xcg_stab_grid
    # margin = xcg_stab_grid - xcg_ctrl_grid
    max_val = np.abs(margin).max()
    plt.pcolormesh(Sr_Sf_grid, Af_grid, margin, vmin=-max_val, vmax=max_val,
                   cmap="coolwarm")
    plt.colorbar()
    plt.contour(Sr_Sf_grid, Af_grid, xcg_ctrl_grid,
                [xcg_range[0]], colors=["tab:blue"])
    plt.contour(Sr_Sf_grid, Af_grid, xcg_stab_grid,
                [xcg_range[1]], colors=["tab:orange"])
    plt.legend()

    if Sr_Sf is not None:
        plt.scatter([Sr_Sf], [Af])


if __name__ == "__main__":
    # Cmacf = -0.0645
    # Cmacr = -0.0645
    CLmaxf = 1.7135/1.4
    CLmaxr = 0.67448
    # CLdesf = 0.7382799
    # CLdesr = 0.7382799
    CD0f = 0.00856
    CD0r = 0.00856
    taperf = 0.45
    taperr = 0.45
    lambda_c4f = 0
    lambda_c4r = 0
    ef = 0.65
    er = 0.65
    Claf = 6.028232202020173
    Clar = 6.028232202020173
    zcg = 0.4 * 1.7
    Vr_Vf_2 = 1
    elev_fac = 1.4
    rho = 1.111617926993772
    Pbr = 10561.929511285156
    S = 18.379085418840855
    # W = 2939.949692 * 9.80665
    # Initial estimates
    MTOM = 34460.871830379496  #2800
    W = MTOM * 9.80665
    V_cr = 66
    h_cr = 1000
    C_L_cr = 0.5960658181816159
    prop_radius = 0.55
    de_da = 0.25
    Sv = 1.1
    V_stall = 40
    max_power = 1.5e6
    AR_wing1_init = 8
    AR_wing2_init = 9
    Sr_Sf_init = 1.7

    # Positions of the wings [horizontally, vertically]
    xf_init = 0.5
    zf_init = 0.3
    xr_init = 6
    zr_init = 1.7

    xf = [0.45, 0.65]
    xr = [6, 7]
    zf = [0.25, 0.35]
    zr = [1.65, 1.7]
    crmaxf = 2.1
    crmaxr = 3
    bmax = 11
    xcg_range = [3.8870312891046765, 3.8870312891046765+0.2]
    Arange = [7, 12]

    impose_stability = True

    Af, Ar, xf, xr, zf, zr, Sr_Sf = optimise_wings(
        CLmaxf, CLmaxr, C_L_cr, C_L_cr, CD0f,
        CD0r, taperf, taperr, lambda_c4f, lambda_c4r, ef, er,
        Claf, Clar, zcg, Vr_Vf_2, elev_fac, rho, Pbr, S, W,
        xf, xr, zf, zr, crmaxf, crmaxf, bmax, bmax, Arange, Arange,
        xcg_range, impose_stability=impose_stability,
        init_Af=6, init_Ar=9, init_xf=xf_init, init_xr=xr_init, init_zf=zf_init,
        init_zr=zr_init, init_Sr_Sf=Sr_Sf_init
    )

    print(Af, Ar, xf, xr, zf, zr, Sr_Sf)

    plot_Sr_Sf_range = [0.5, 3]
    plot_Af_range = [3, 13]

    plot_Sf_Sr_Af_plane(CLmaxf, CLmaxr, C_L_cr, C_L_cr, CD0f,
                        CD0r, taperf, taperr, lambda_c4f, lambda_c4r, ef, er,
                        Claf, Clar, zcg, Vr_Vf_2, elev_fac, rho, Pbr, S, W,
                        Ar, xf, xr, zf, zr, xcg_range, plot_Sr_Sf_range,
                        plot_Af_range, Sr_Sf=Sr_Sf, Af=Af)
    plt.show()

