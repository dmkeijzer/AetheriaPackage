import numpy as np


def airfoil_to_wing_CLa(cla, A):
    """
    cla: airfoil cl_alpha [rad^-1]
    A: wing aspect ratio [-]

    returns
    cLa: wing cL_alpha [rad^-1]

    works for horizontal and vertical tails as well
    """
    cLa = cla / (1 + cla / (np.pi * A))
    return cLa


def downwash_k(lh, b):
    """
    lh: distance from wing ac to horizontal tail [m]
    b: wingspan [m]

    returns
    k: factor in downwash formula
    """
    k = 1 + (1 / (np.sqrt(1 + (lh / b) ** 2))) * (1 / (np.pi * lh / b) + 1)
    return k


def downwash(k, CLa, A):
    """
    k: factor calculated with downwash_k()
    CLa: wing CL-alpha [rad^-1]
    A: wing aspect ratio [-]
    """
    depsda = k * CLa / (np.pi * A)
    return depsda


def Cma_fuse(Vfuse, S, c):
    """
    Vfuse: fuselage volume [m^3]
    S: wing surface area [m^2]
    c: mean aerodynamic chord [m]

    returns
    Cma_fuse: Cma component from fuselage [rad^-1]
    """
    Cma_fuse = 2 * Vfuse / (S * c)
    return Cma_fuse


def Cnb_fuse(Vfuse, S, b):
    """
    Vfuse: fuselage volume [-]
    S: wing surface area [m^2]
    b: wingspan [m]

    returns
    Cnb_fuse: Cnb component from fuselage [rad^-1]
    """
    Cnb_fuse = -2 * Vfuse / (S * b)
    return Cnb_fuse


def CDacalc(CL0, CLa, A):
    """
    CL0: wing lift at 0 angle of attack [-]
    CLa: wing CL_alpha [rad^-1]
    A: wing aspect ratio [-]

    returns
    CDa: wing CD_alpha [rad^-1]
    """
    CDa = 2 * CL0 * CLa / (np.pi * A)
    return CDa


def Cxa(CL0, CDa):
    """
    CL0: wing lift at 0 angle of attack [-]
    CDa: wing CD_alpha [rad^-1]

    returns
    Cxa: X-force coefficient derivative wrt alpha [rad^-1]
    """
    Cxa = CL0 - CDa
    return Cxa


def Cxq():
    """
    returns
    Cxq: X-force coefficient derivative wrt q
    ALWAYS 0
    """
    return 0


def Cza(CLa, CD0):
    """
    CLa: wing CL_alpha [rad^-1]
    CD0: CD_0 [-]

    returns
    Cza: Z-force coefficient derivative wrt alpha [rad^-1]
    """
    Cza = -CLa - CD0
    return Cza


def Vhcalc(Sh, lh, S, c):
    """
    Sh: surface area of horizontal stabiliser [m^2]
    lh: distance from wing ac to horizontal tail [m]
    S: surface area of wing [m^2]
    c: mean aerodynamic chord [m]

    returns
    Vh: horizontal tail volume coefficient [-]
    """
    Vh = Sh * lh / (S * c)
    return Vh


def Czq(CLah, Vh):
    """
    CLah: Horizontal stabiliser CL_alpha [rad^-1]
    Vh: horizontal tail volume coefficient [-]

    returns
    Czq: Z-force coefficient derivative wrt q [rad^-1]
    """
    Czq = -2 * CLah * Vh
    return Czq


def Cma(CLa, lcg, c, CLah, Vh, depsda, Cmafuse):
    """
    CLa: wing CL_alpha [rad^-1]
    lcg: distance from wing ac to cg [m]
    c: mean aerodynamic chord [m]
    CLah: horizontal stabiliser CL_alpha [rad^-1]
    Vh: horizontal tail volume coefficient [-]
    depsda: downwash gradient [-]
    Cmafuse: fuselage contribution to Cma [rad^-1]

    returns
    Cma: Aircraft Cm coefficient derivative wrt alpha [rad^-1]
    """
    Cma = CLa * lcg / c - CLah * Vh * (1 - depsda) + Cmafuse
    return Cma


def Cmq(CLah, Vh, lh, c, Cmqfuse):
    """
    CLah: horizontal stabiliser CL_alpha [rad^-1]
    Vh: horizontal tail volume coefficient [-]
    lh: distance from wing ac to horizontal tail [m]
    c: mean aerodynamic chord [m]
    Cmqfuse: fuselage contribution to Cmq [rad^-1]

    returns
    Cmq: Aircraft Cm coefficient derivative wrt q [rad^-1]
    """
    Cmq = -2 * CLah * Vh * lh / c + Cmqfuse
    return Cmq


def Vvcalc(Sv, lv, S, b):
    """
    Sv: surface area of vertical tail [m^2]
    lv: distance from wing ac to vertical tail [m]
    S: surface area of wing [m^2]
    b: wingspan [m]

    returns
    Vv: vertical tail volume coefficient [-]
    """
    Vv = Sv * lv / (S * b)
    return Vv


def Cyb(Sv, S, CLav):
    """
    Sv: surface area of vertical tail [m^2]
    S: surface area of wing [m^2]
    CLav: vertical tail CL_alpha [rad^-1]

    returns
    Cyb: Y-force coefficient derivative wrt sideslip angle [rad^-1]
    """
    Cyb = -Sv * CLav / S
    return Cyb


def Cyr(Vv, CLav):
    """
    Vv: vertical tail volume coefficient [-]
    CLav: vertical tail CL_alpha [rad^-1]

    returns
    Cyr: Y-force coefficient derivative wrt yaw rate [rad^-1]
    """
    Cyr = -2 * Vv * CLav
    return Cyr


def Cyp():
    """
    returns
    Cyp: Y-force coefficient derivative wrt roll rate
    ALWAYS 0
    """
    return 0


def Clb(CLa, dihedral, taper):
    """
    CLa: wing CL_alpha [rad^-1]
    dihedral: wing dihedral [rad]
    taper: wing taper ratio [-]

    returns
    Clb: roll-moment coefficient derivative wrt sideslip angle [rad^-1]
    """
    Clb = -CLa * dihedral * (1 + 2 * taper) / (6 * (1 + taper))
    return Clb


def Clp(CLa, taper):
    """
    CLa: wing CL_alpha [rad^-1]
    taper: wing taper ratio [-]

    returns
    Clp: roll-moment coefficient derivative wrt roll rate [rad^-1]
    """
    Clp = -CLa * (1 + 3 * taper) / (12 * (1 + taper))
    return Clp


def Clr(CL0):
    """
    CL0: wing CL at 0 angle of attack [-]

    returns
    Clr: roll-moment coefficient derivative wrt yaw rate [rad^-1]
    """
    return CL0 / 4


def Cnb(CLav, Vv, Cnbfuse):
    """
    CLav: vertical tail CL_alpha [rad^-1]
    Vv: vertical tail volume coefficient [-]
    Cnbfuse: fuselage contribution to Cnb [rad^-1]

    returns
    Cnb: yaw-moment coefficient derivative wrt sideslip angle [rad^-1]
    """
    Cnb = CLav * Vv + Cnbfuse
    return Cnb


def Cnp(CL0):
    """
    CL0: wing CL at 0 angle of attack [-]

    returns
    Cnp: yaw-moment coefficient derivative wrt roll rate [rad^-1]
    """
    return -CL0 / 8


def Cnr(CLav, Vv, lv, b):
    """
    CLav: vertical tail CL_alpha [rad^-1]
    Vv: vertical tail volume coefficient [-]
    lv: distance from wing ac to vertical tail [m]
    b: wingspan [m]

    returns
    Cnr: yaw-moment coefficient derivative wrt yaw rate
    """
    Cnr = -2 * CLav * Vv * lv / b
    return (Cnr)


def longitudinal_derivatives(c, lh, CL0, CD0, lcg, Cmafuse=None, Cmqfuse=0, CLa=None, CLah=None, depsda=None,
                             CDa=None, Vh=None, Vfuse=None, S=None, cla=None, A=None, clah=None,
                             Ah=None, b=None, k=None, Sh=None):
    """
    c: mean aerodynamic chord [m]
    lh: distance from wing ac to horizontal tail [m]
    CL0: Wing CL at angle of attack 0 [-]
    CD0: CD_0 [-]
    lcg: distance from wing ac to cg [m]

    Cmafuse: fuselage contribution to Cma [rad^-1]
    Cmqfuse: fuselage contribution to Cmq [rad^-1]
    CLa: Wing CL_alpha [rad^-1]
    CLah: Horizontal tail CL_alpha [rad^-1]
    depsda: downwash gradient [-]
    CDa: CD derivative wrt angle of attack [rad^-1]
    Vh: Horizontal tail volume coefficient [-]
    Vfuse: Volume of fusealge [m^3]
    S: wing surface area [m^2]
    cla: wing airfoil lift coefficient [rad^-1]
    A: wing aspect ratio [-]
    clah: horizontal tail airfoil lift coefficient [rad^-1]
    Ah: Horizontal tail aspect ratio [-]
    b: wingspan [m]
    k: factor for downwash gradient
    Sh: horizontal tail surface area [m^2]

    returns
    dict: dictionary containing longitudinal stability derivatives
    """
    if Cmafuse == None:
        assert Vfuse != None, "Missing input: Vfuse"
        assert S != None, "Missing input: S"
        Cmafuse = Cma_fuse(Vfuse, S, c)
    if Cmqfuse == None:
        Cmqfuse = 0
    if CLa == None:
        assert cla != None, "Missing input: cla"
        assert A != None, "Missing input: A"
        CLa = airfoil_to_wing_CLa(cla, A)
    if CLah == None:
        assert clah != None, "Missing input: clah"
        assert Ah != None, "Missing input: Ah"
        CLah = airfoil_to_wing_CLa(clah, Ah)
    if depsda == None:
        if k == None:
            assert b != None, "Missing input:b"
            downwash_k(lh, b)
        assert A != None, "Missing input: A"
        depsda = downwash(k, CLa, A)
    if CDa == None:
        assert A != None, "Missing input: A"
        CDa = CDacalc(CL0, CLa, A)
    if Vh == None:
        assert Sh != None, "Missing input: Sh"
        assert S != None, "Missing input: S"
        Vh = Vhcalc(Sh, lh, S, c)

    dict = {}
    dict["Cxa"] = Cxa(CL0, CDa)
    dict["Cxq"] = Cxq()
    dict["Cza"] = Cza(CLa, CD0)
    dict["Czq"] = Czq(CLah, Vh)
    dict["Cma"] = Cma(CLa, lcg, c, CLah, Vh, depsda, Cmafuse)
    dict["Cmq"] = Cmq(CLah, Vh, lh, c, Cmqfuse)
    return dict


def lateral_derivatives(Sv, lv, S, b, dihedral, taper, CL0, CLav=None, Vv=None, CLa=None, Cnbfuse=None, clav=None,
                        Av=None, cla=None, A=None, Vfuse=None):
    """
    Sv: vertical tail surface area [m^2]
    lv: distance from wing ac to vertical tail [m]
    S: wing surface area [m^2]
    b: wingspan [m]
    dihedral: wing dihedral angle [rad]
    taper: wing taper ratio [-]
    CL0: wing CL at 0 angle of attack [-]

    CLav: vertical tail CL_alpha [rad^-1]
    Vv: vertical tail volume coefficient [-]
    CLa: wing CL_alpha [rad^-1]
    Cnbfuse: fuselage contribution to Cnb [rad^-1]
    clav: vertical tail airfoil cl_alpha [rad^-1]
    Av: vertical tail aspect ratio [-]
    cla:wing airfoil cl_alpha [rad^-1]
    A: wing aspect ratio [-]
    Vfuse: fuselage volume [m^3]

    returns
    dict: dictionary containing lateral stability derivatives
    """

    if CLav == None:
        assert clav != None, "Missing input: clav"
        assert Av != None, "Missing input: Av"
        CLav = airfoil_to_wing_CLa(clav, Av)
    if Vv == None:
        Vv = Vvcalc(Sv, lv, S, b)
    if CLa == None:
        assert cla != None, "Missing input: cla"
        assert A != None, "Missing input: A"
        CLa = airfoil_to_wing_CLa(cla, A)
    if Cnbfuse == None:
        assert Vfuse != None, "Missing input: Vfuse"
        Cnbfuse = Cnb_fuse(Vfuse, S, b)

    dict = {}
    dict["Cyb"] = Cyb(Sv, S, CLav)
    dict["Cyp"] = Cyp()
    dict["Cyr"] = Cyr(Vv, CLav)
    dict["Clb"] = Clb(CLa, dihedral, taper)
    dict["Clp"] = Clp(CLa, taper)
    dict["Clr"] = Clr(CL0)
    dict["Cnb"] = Cnb(CLav, Vv, Cnbfuse)
    dict["Cnp"] = Cnp(CL0)
    dict["Cnr"] = Cnr(CLav, Vv, lv, b)
    return dict