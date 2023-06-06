import numpy as np


def CZ_adot(CLah,Sh,S,Vh_V2,depsda,lh,c):

    CZ_adot=-CLah*Sh/S*Vh_V2*depsda*lh/c

    return CZ_adot


def Cm_adot(CLah,Sh,S,Vh_V2,depsda,lh,c):

    Cm_adot=-CLah*Sh/S*Vh_V2*depsda*(lh/c)**2

    return Cm_adot




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

def muc(m,rho,S,c):
        """
        Computes dimensionless mass for symmetric motion mu_c
        :return: mu_c
        """
        return m/(rho*S*c)

def mub(m,rho,S,b):
        """
        Computes dimensionless mass for asymmetric motion mu_b
        :return: mu_b
        """
        return m/(rho*S*b)

def Cz0(W,theta_0,rho,V,S):
    Cz0= -W*np.cos(theta_0)/(0.5*rho*V**2*S)
    return Cz0

def Cx0(W,theta_0,rho,V,S):
    Cx0= W*np.sin(theta_0)/(0.5*rho*V**2*S)
    return Cx0


def longitudinal_derivatives(CD, CL, W,rho,S, m, c, lh, CL0, CD0, lcg, Vh_V2, theta_0, V, Cmafuse=None, Cmqfuse=None, CLa=None, CLah=None, depsda=None,
                             CDa=None, Vh=None, Vfuse=None, cla=None, A=None, clah=None,
                             Ah=None, b=None, k=None, Sh=None):
    """
    CD: aircraft drag coefficient[-]
    CL: aircraft lift coefficient [-]
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
    Vh_V_ratio: Horizontal tail speed to freestream speed ratio [-]
    W: aircraft weight [N]
    rho= air density [kg/m^3]
    g = gravitational acceleration [m/s^2]
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
    dict["Cz_adot"]=CZ_adot(CLah,Sh,S,Vh_V2,depsda,lh,c)
    dict["Cm_adot"]=Cm_adot(CLah,Sh,S,Vh_V2,depsda,lh,c)
    dict["muc"]=muc(m, rho,S,c)
    dict["Cxu"]=-2*CD
    dict["Czu"]=-2*CL
    #dict["Cx0"]=-CL
    dict["Cx0"]=Cx0(W,theta_0,rho,V,S)
    dict["Cz0"]=Cz0(W,theta_0,rho,V,S)
    dict["Cmu"]=0  #Because the derivative of CL and Ct with respect to the Mach number is essentially 0. 

    return dict



def lateral_derivatives(Cnb,m,rho, Sv, lv, S, b, dihedral, taper, CL0, CLav=None, Vv=None, CLa=None, clav=None,
                        Av=None, cla=None, A=None, Cn_beta_dot=None,CY_beta_dot=None): #Cnbfuse=None, Vfuse=None
    """
    Cnb: this is the derivative the yaw moment coefficient with respect to sideslip angle beta- [-]
    theta_0: initial pitch angle [rad]
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
    W: aircraft weight [N]
    rho= air density [kg/m^3]
    g = gravitational acceleration [m/s^2]

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
    # if Cnbfuse == None:
    #     assert Vfuse != None, "Missing input: Vfuse"
    #     Cnbfuse = Cnb_fuse(Vfuse, S, b)
    if Cn_beta_dot == None:
        Cn_beta_dot=0
    if CY_beta_dot == None:
        CY_beta_dot=0

    dict = {}
    dict["Cyb"] = Cyb(Sv, S, CLav)
    dict["Cyp"] = Cyp()
    dict["Cyr"] = Cyr(Vv, CLav)
    dict["Clb"] = Clb(CLa, dihedral, taper)
    dict["Clp"] = Clp(CLa, taper)
    dict["Clr"] = Clr(CL0)
    #dict["Cnb"] = Cnb(CLav, Vv, Cnbfuse)
    dict["Cnp"] = Cnp(CL0)
    dict["Cnr"] = Cnr(CLav, Vv, lv, b)

    dict["Cy_beta_dot"] = CY_beta_dot
    dict["Cn_beta_dot"] = Cn_beta_dot
    dict["mub"]=mub(m,rho,S,b)
    dict["Cnb"]=Cnb
    return dict



def eigval_finder_sym(Iyy, m, c, long_stab_dervs):      #Iyy = 12081.83972
    """
    Iyy: moment of inertia around Y-axis
    m: MTOM
    c: MAC
    long_stab_dervs: dictionary containing all longitudinal stability derivatives + muc

    returns
    array with eigenvalues NON-DIMENSIONALISED
    """
    CX0 = long_stab_dervs["Cx0"]
    CXa = long_stab_dervs["Cxa"]
    CXu = long_stab_dervs["Cxu"]
    CZ0 = long_stab_dervs["Cz0"]
    CZa = long_stab_dervs["Cza"]
    CZu = long_stab_dervs["Czu"]
    CZq = long_stab_dervs["Czq"]
    CZadot = long_stab_dervs["Cz_adot"]
    Cma = long_stab_dervs["Cma"]
    Cmq = long_stab_dervs["Cmq"]
    Cmu = long_stab_dervs["Cmu"]
    Cmadot = long_stab_dervs["Cm_adot"]
    muc = long_stab_dervs["muc"]

    KY2 = Iyy / (m*c**2)
    Aeigval = 4 * muc **2 * KY2 * (CZadot - 2 * muc)
    Beigval = Cmadot * 2 * muc * (CZq + 2 * muc) - Cmq * 2 * muc * (CZadot - 2 * muc) - 2 * muc * KY2 * (CXu * (CZadot - 2*muc) - 2 * muc * CZa)
    Ceigval = Cma * 2 * muc * (CZq + 2*muc) - Cmadot * (2 * muc * CX0 + CXu * (CZq + 2*muc)) + Cmq * (CXu * (CZadot - 2*muc) - 2*muc*CZa) + 2 * muc*KY2*(CXa*CZu - CZa * CXu)
    Deigval = Cmu * (CXa*(CZq + 2*muc) - CZ0 * (CZadot - 2 * muc)) - Cma * (2*muc*CX0 + CXu * (CZq + 2*muc)) + Cmadot * (CX0*CXu - CZ0*CZu) + Cmq*(CXu * CZa - CZu * CXa)
    Eeigval = -Cmu * (CX0 * CXa + CZ0 * CZa) + Cma * (CX0 * CXu + CZ0 * CZu)
    return np.roots(np.array([Aeigval, Beigval, Ceigval, Deigval, Eeigval]))


def eigval_finder_asymm(Ixx, Izz, Ixz, m, b, CL, lat_stab_dervs):   #Ixx = 10437.12494 Izz = 21722.48912

    """
    Ixx: moment of inertia around X-axis
    Izz: moment of inertia around Z-axis
    Ixz: moment of gyration around X-Z
    m: MTOM
    b: wingspan
    CL: cruise CL
    lat_stab_dervs: dictionary containing all lateral stability derivatives + mub

    returns
    array with eigenvalues NON-DIMENSIONALISED
    """
    CYb = lat_stab_dervs["Cyb"]
    CYp = lat_stab_dervs["Cyp"]
    CYr = lat_stab_dervs["Cyr"]
    Clb = lat_stab_dervs["Clb"]
    Clp = lat_stab_dervs["Clp"]
    Clr = lat_stab_dervs["Clr"]
    Cnb = lat_stab_dervs["Cnb"]
    Cnp = lat_stab_dervs["Cnp"]
    Cnr = lat_stab_dervs["Cnr"]
    mub = lat_stab_dervs["mub"]


    KX2 = Ixx / (m*b**2)
    KZ2 = Izz / (m*b**2)
    KXZ = Ixz / (m*b**2)
    Aeigval = 16 * mub ** 3 * (KX2 * KZ2 - KXZ ** 2)
    Beigval = -4 * mub ** 2 * (
                2 * CYb * (KX2 * KZ2 - KXZ ** 2) + Cnr * KX2 + Clp * KZ2 + (
                    Clr + Cnp) * KXZ)
    Ceigval = 2 * mub * ((CYb * Cnr - CYr * Cnb) * KX2 + (
                CYb * Clp - Clb * CYp) * KZ2 + ((CYb * Cnp - Cnb * CYp) + (
                CYb * Clr - Clb * CYr)) * KXZ + 4 * mub * Cnb * KX2 + 4 * mub * Clb * KXZ + 0.5 * (
                                     Clp * Cnr - Cnp * Clr))
    Deigval = -4 * mub * CL * (Clb * KZ2 + Cnb * KXZ) + 2 * mub * (
                Clb * Cnp - Cnb * Clp) + 0.5 * CYb * (
                          Clr * Cnp - Cnr * Clp) + 0.5 * CYp * (
                          Clb * Cnr - Cnb * Clr) + 0.5 * CYr * (
                          Clp * Cnb - Cnp * Clb)
    Eeigval = CL * (Clb * Cnr - Cnb * Clr)
    return np.roots(np.array([Aeigval, Beigval, Ceigval, Deigval, Eeigval]))


if __name__ == "__main__":
    a = longitudinal_derivatives(0.04, 0.6, 2500 * 9.8, 1.2, 12, 2500, 1.2, 3, 0.1, 0.02, 0.1, 0.95, np.radians(3), 80,
                             Cmafuse=None, Cmqfuse=0, CLa=3.7, CLah=1.6, depsda=0.11,
                             CDa=None, Vh=None, Vfuse=7, cla=None, A=7, clah=None,
                             Ah=4, b=10, k=None, Sh=3)

    c = {"symm": {"Iyy": 12080, "m": 2500, "c": 1.2,
                 "long_stab_dervs": longitudinal_derivatives(0.04, 0.6, 2500 * 9.8, 1.2, 12, 2500, 1.2, 3, 0.1, 0.02,
                                                             0.1, 0.95, np.radians(3), 80, Cmafuse=None, Cmqfuse=0,
                                                             CLa=3.7, CLah=1.6, depsda=0.11,
                                                             CDa=None, Vh=None, Vfuse=7, cla=None, A=7, clah=None,
                                                             Ah=4, b=10, k=None, Sh=3)},
        "asymm": {"Ixx": 10440, "Izz": 21720, "Ixz": 500, "m": 2500, "b": 10, "CL": 0.6,
                  "lat_stab_dervs": lateral_derivatives(0.08, 2500, 1.2, 2, 3, 12, 10, np.radians(2), 0.4, 0.1,
                                                        CLav=1.8, Vv=None, CLa=3.7, clav=None,
                                                        Av=3, cla=None, A=7, Cn_beta_dot=None, CY_beta_dot=None)}}

    b = eigval_finder_sym(**c['symm'])
    print(c)
