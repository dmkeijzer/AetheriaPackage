from math import sqrt, pi, cos, sin, tan
import numpy as np
from scipy.constants import g
import matplotlib.pyplot as plt
import numpy as np
from AetheriaPackage.avlwrapper import Geometry, Surface, Section, NacaAirfoil, Control, Point, Spacing, Session, Case, Parameter
import AetheriaPackage.GeneralConstants as const

def get_lift_distr(wing, aero, plot= False, test= False):

    """ Returns the lift distribution as a function of the span using an AVLWrapper, see refs below.

    ref: https://github.com/jbussemaker/AVLWrapper.git

    :param wing: wing class from data structurs
    :type wing: wing class
    :param aero: aero class from data structures
    :type aero: wing structure
    :param plot: Plot the lift distribution and the avl model which is used, defaults to False
    :type plot: bool, optional
    :param test: bool used for the test, returns parameters required for the assert statements, defaults to False
    :type test: bool, optional
    :return: Lift distribution as a function of the span
    :rtype: function
    """    

    # wing root section with a flap control and NACA airfoil
    root_section = Section(leading_edge_point=Point(0, 0, 0),
                        chord= wing.chord_root,
                        airfoil=NacaAirfoil(naca='2412'))

    # wing tip
    tip_section = Section(leading_edge_point=Point(np.sin(wing.sweep_LE)*wing.span/2, wing.span/2, 0),
                        chord= wing.chord_tip,
                        airfoil=NacaAirfoil(naca='2412'))

    # wing surface defined by root and tip sections
    wing_surface = Surface(name="Wing",
                        n_chordwise=13,
                        chord_spacing=Spacing.cosine,
                        n_spanwise=20,
                        span_spacing=Spacing.equal,
                        y_duplicate=0.0,
                        sections=[root_section, tip_section])

    # geometry object (which corresponds to an AVL input-file)
    geometry = Geometry(name="wing_joby",
                        mach= const.v_cr/const.t_cr,
                        reference_area= wing.surface,
                        reference_chord= wing.chord_mac,
                        reference_span=wing.span,
                        reference_point=Point(0, 0, 0),
                        surfaces=[wing_surface])

    cruise_case = Case(name='Cruise', 
                       trimmed=Parameter(name= "alpha", constraint="CL", value=aero.cL_cruise))  # Case defined by one angle-of-attack

    # create session with the geometry object and the cases
    session = Session(geometry=geometry, cases=[cruise_case])

    # get results and write the resulting dict to a JSON-file
    results = session.get_results()
    if plot:
        session.show_geometry()

    # Extract strip data from AVL and build aer0 function
    strip_data = results["Cruise"]["StripForces"]["Wing"]

    y_le_per_strip = np.array(strip_data["Yle"]) 
    area_per_strip = np.array(strip_data["Area"])
    cl_per_strip = np.array(strip_data["cl"])
    lift_force_per_strip = 1/2*const.rho_cr*const.v_cr**2*area_per_strip*cl_per_strip
    lift_from_tip = np.flip(np.cumsum(lift_force_per_strip))

    coeff = np.polyfit(y_le_per_strip, lift_from_tip, 2)

    def lift_func(y):
        return coeff[0]*y**2 + coeff[1]*y + coeff[2]


    if plot:
        # sns.set_style("whitegrid")
        span_points = np.linspace(0, wing.span/2, 300)
        lift_values = np.vectorize(lift_func)(span_points)
        plt.plot(span_points, lift_values, label= "Wing Loading")
        plt.xlabel("Half span [m]")
        plt.ylabel("Lift force [N]")
        plt.legend()
        plt.grid(alpha=0.8, lw= 0.7)
        plt.show()

    if test:
        return lift_func, results

    return lift_func
        
def get_strip_array(wing, aero, plot= False):
    """ Returns the lift distribution as a function of the span using an AVLWrapper, see refs below.

    ref: https://github.com/jbussemaker/AVLWrapper.git

    :param wing: wing class from data structurs
    :type wing: wing class
    :param aero: aero class from data structures
    :type aero: wing structure
    :param plot: Plot the lift distribution and the avl model which is used, defaults to False
    :type plot: bool, optional
    :param test: bool used for the test, returns parameters required for the assert statements, defaults to False
    :type test: bool, optional
    :return: Lift distribution as a function of the span
    :rtype: function
    """    

    # wing root section with a flap control and NACA airfoil
    root_section = Section(leading_edge_point=Point(0, 0, 0),
                        chord= wing.chord_root,
                        airfoil=NacaAirfoil(naca='2412'))

    # wing tip
    tip_section = Section(leading_edge_point=Point(np.sin(wing.sweep_LE)*wing.span, wing.span/2, 0),
                        chord= wing.chord_tip,
                        airfoil=NacaAirfoil(naca='2412'))

    # wing surface defined by root and tip sections
    wing_surface = Surface(name="Wing",
                        n_chordwise=13,
                        chord_spacing=Spacing.cosine,
                        n_spanwise=25,
                        span_spacing=Spacing.equal,
                        y_duplicate=0.0,
                        sections=[root_section, tip_section])

    # geometry object (which corresponds to an AVL input-file)
    geometry = Geometry(name="wing_joby",
                        mach= const.v_cr/const.t_cr,
                        reference_area= wing.surface,
                        reference_chord= wing.chord_mac,
                        reference_span=wing.span,
                        reference_point=Point(0, 0, 0),
                        surfaces=[wing_surface])

    cruise_case = Case(name='Cruise', 
                       trimmed=Parameter(name= "alpha", constraint="CL", value=aero.cL_cruise))  # Case defined by one angle-of-attack

    # create session with the geometry object and the cases
    session = Session(geometry=geometry, cases=[cruise_case])

    # get results and write the resulting dict to a JSON-file
    results = session.get_results()
    if plot:
        session.show_geometry()

    # Extract strip data from AVL and build aer0 function
    strip_data = results["Cruise"]["StripForces"]["Wing"]

    y_le_per_strip = np.array(strip_data["Yle"]) 
    cl_per_strip = np.array(strip_data["cl"])


    return y_le_per_strip, cl_per_strip


def get_tail_lift_distr(wing, veetail, aero, plot= False, test= False):

    """ Returns the lift distribution as a function of the span using an AVLWrapper, see refs below.

    ref: https://github.com/jbussemaker/AVLWrapper.git

    :param wing: wing class from data structurs
    :type wing: wing class
    :param aero: aero class from data structures
    :type aero: wing structure
    :param plot: Plot the lift distribution and the avl model which is used, defaults to False
    :type plot: bool, optional
    :param test: bool used for the test, returns parameters required for the assert statements, defaults to False
    :type test: bool, optional
    :return: Lift distribution as a function of the span
    :rtype: function
    """    

    # wing root section with a flap control and NACA airfoil
    root_section = Section(leading_edge_point=Point(0, 0, 0),
                        chord= wing.chord_root,
                        airfoil=NacaAirfoil(naca='2412'))

    # wing tip
    tip_section = Section(leading_edge_point=Point(np.sin(wing.sweep_LE)*wing.span/2, wing.span/2, 0),
                        chord= wing.chord_tip,
                        airfoil=NacaAirfoil(naca='2412'))

    # wing surface defined by root and tip sections
    wing_surface = Surface(name="Wing",
                        n_chordwise=13,
                        chord_spacing=Spacing.cosine,
                        n_spanwise=20,
                        span_spacing=Spacing.equal,
                        y_duplicate=0.0,
                        sections=[root_section, tip_section])

   # elevator control for the tail surface
    elevator = Control(name="elevator",
                       gain=1.0,
                       x_hinge=0.75,
                       duplicate_sign=1.0)

    # wing root section with a flap control and NACA airfoil
    root_section_tail = Section(leading_edge_point=Point(veetail.length_wing2vtail, 0, 0),
                        chord= veetail.chord_root,
                        airfoil=NacaAirfoil(naca='0012'),
                        controls= [elevator]
                        )

    # wing tip
    tip_section_tail = Section(leading_edge_point=Point(np.sin( veetail.sweep_LE)* veetail.span/2 + veetail.length_wing2vtail , veetail.span/2*np.cos(veetail.dihedral), veetail.span/2*np.sin(veetail.dihedral)),
                        chord= veetail.chord_tip,
                        airfoil=NacaAirfoil(naca='0012'),
                       controls= [elevator] )

    # wing surface defined by root and tip sections
    tail_surface = Surface(name="v_tail",
                        n_chordwise=5,
                        chord_spacing=Spacing.cosine,
                        n_spanwise=7,
                        span_spacing=Spacing.equal,
                        y_duplicate=0.0,
                        sections=[root_section_tail, tip_section_tail])

    # geometry object (which corresponds to an AVL input-file)
    geometry = Geometry(name="wing_joby",
                        mach= const.v_cr/const.t_cr,
                        reference_area= wing.surface,
                        reference_chord= wing.chord_mac,
                        reference_span=wing.span,
                        reference_point=Point(0, 0, 0),
                        surfaces=[wing_surface, tail_surface])

    cruise_case = Case(name='Cruise', 
                       trimmed_cruise=Parameter(name= "alpha", constraint="CL", value=aero.cL_cruise),
                       trimmed_elevator =Parameter(name= "elevator", constraint="Cm", value=0)
                       )  # Case defined by one angle-of-attack

    # create session with the geometry object and the cases
    session = Session(geometry=geometry, cases=[cruise_case])

    if plot:
        session.show_geometry()

    # get results and write the resulting dict to a JSON-file
    results = session.get_results()

    # Extract strip data from AVL and build aer0 function
    strip_data = results["Cruise"]["StripForces"]["v_tail"]

    y_le_per_strip = np.array(strip_data["Yle"]) 
    area_per_strip = np.array(strip_data["Area"])
    cl_per_strip = np.array(strip_data["cl"])
    lift_force_per_strip = 1/2*const.rho_cr*const.v_cr**2*area_per_strip*cl_per_strip*np.cos(veetail.dihedral)
    lift_from_tip = np.flip(np.cumsum(lift_force_per_strip))

    coeff = np.polyfit(y_le_per_strip, lift_from_tip, 2)

    def lift_func(y):
        return coeff[0]*y**2 + coeff[1]*y + coeff[2]


    if plot:
        # sns.set_style("whitegrid")
        span_points = np.linspace(0, wing.span/2, 300)
        lift_values = np.vectorize(lift_func)(span_points)
        plt.plot(span_points, lift_values, label= "Wing Loading")
        plt.xlabel("Half span [m]")
        plt.ylabel("Lift force [N]")
        plt.legend()
        plt.grid(alpha=0.8, lw= 0.7)
        plt.show()

    if test:
        return lift_func, results

    return lift_func


def Reynolds(rho_cruise, V_cruise, mac, mu, k):
    """Returns the Reynold number

    :param rho_cruise: Density at cruise altitude
    :type rho_cruise: _type_
    :param V_cruise: Cruise velocity
    :type V_cruise: _type_
    :param mac: Mean aerodynamic chord
    :type mac: float
    :param mu: dynamic viscosity
    :type mu: _type_
    :param k: surface factor in the order of 1e-5 and 1e-7
    :type k: float
    :return: Reyolds number
    :rtype: _type_
    """
    return min((rho_cruise * V_cruise * mac / mu), 38.21 * (mac / k) ** 1.053)

def deps_da(cL_alpha, AR):
    return (2*cL_alpha)/(np.pi*AR)

def Mach_cruise(V_cruise, gamma, R, T_cruise):
    """_summary_

    :param V_cruise: Cruise speed [m/s]
    :type V_cruise: _type_
    :param gamma: _description_
    :type gamma: _type_
    :param R: _description_
    :type R: _type_
    :param T_cruise: _description_
    :type T_cruise: _type_
    :return: _description_
    :rtype: _type_
    """
    a = np.sqrt(gamma * R * T_cruise)
    return V_cruise / a


def FF_fus(l, d):
    """_summary_

    :param l: length fuselage
    :type l: _type_
    :param d: diameter fuselage
    :type d: _type_
    :return: _description_
    :rtype: _type_
    """
    f = l / d
    return 1 + 60 / (f ** 3) + f / 400

def sweep_m(sweep_le, loc_max_t, c_root, span, taper):
    """_summary_

        :param sweep_le: leading edge sweep angle
        :type sweep_le: _type_
        :param loc_max_t: location of maximum thickness of airfoil (x/c)
        :type loc_max_t: _type_
        :param c_root: root chord length
        :type c_root: _type_
        :param span: _description_
        :type span: _type_
        :param taper: _description_
        :type taper: _type_
        :return: _description_
        :rtype: _type_
        """
    return np.arctan2(np.tan(sweep_le) - (1-taper)*loc_max_t*(2*c_root) / span, 1)


def FF_wing(toc, xcm, M, sweep_m):
    """_summary_

    :param toc: thickness over chord ratio
    :type toc: _type_
    :param xcm: (x/c)m, position of maximum thickness
    :type xcm: _type_
    :param M: Mach number
    :type M: _type_
    :return: _description_
    :rtype: _type_
    """

    return (1 + 0.6 * toc / xcm + 100 * toc * 4) * (1.34 * (M * 0.18) * (np.cos(sweep_m)) * 0.28)

def FF_tail(toc_tail, xcm_tail, M, sweep_m):
    """_summary_

    :param toc_tail: thickness over chord ratio
    :type toc_tail: _type_
    :param xcm_tail: (x/c)m, position of maximum thickness
    :type xcm_tail: _type_
    :param M: Mach number
    :type M: _type_
    :return: _description_
    :rtype: _type_
    """

    return (1 + 0.6 * toc_tail / xcm_tail + 100 * toc_tail * 4) * (1.34 * (M * 0.18) * (np.cos(sweep_m)) * 0.28)

def S_wet_fus(d, l1, l2, l3):
    """_summary_

    :param d: diameter fuselage
    :type d: _type_
    :param l1: length cockpit/parabolic section
    :type l1: _type_
    :param l2: length cylindrical section
    :type l2: _type_
    :param l3: length conical section
    :type l3: 
    :return: _description_
    :rtype: _type_
    """
    return (np.pi * d / 4) * (((1 / (3 * l1 ** 2)) * ((4 * l1 ** 2 + ((d ** 2) / 4)) ** 1.5 - ((d ** 3) / 8))) - d + 4 * l2 + 2 * np.sqrt(l3 ** 2 + (d ** 2) / 4))


def CD_upsweep(u, d, S_wet_fus):
    """_summary_

    :param u: upsweep angle (rad)
    :type u: _type_
    :param d: diameter fuselage
    :type d: _type_
    :param S_wet_fus: _description_
    :type S_wet_fus: _type_
    :return: _description_
    :rtype: _type_
    """
    return 3.83 * (u ** 2.5) * np.pi * d ** 2 / (4 * S_wet_fus)


def CD_base(M, A_base, S_wet_fus):
    """_summary_

    :param M: _description_
    :type M: _type_
    :param A_base: base area fuselage
    :type A_base: _type_
    :param S: _description_
    :type S: _type_
    :return: _description_
    :rtype: _type_
    """
    return (0.139 + 0.419 * (M - 0.161) ** 2) * A_base / S_wet_fus


def C_fe_fus(frac_lam_fus, Reynolds, M):
    """_summary_

    :param frac_lam_fus: fraction laminar flow fuselage
    :type frac_lam_fus: _type_
    :param Reynolds: _description_
    :type Reynolds: _type_
    :return: _description_
    :rtype: _type_
    """
    C_f_lam = 1.328 / np.sqrt(Reynolds)
    C_f_turb = 0.455 / (((np.log10(Reynolds)) ** 2.58)
                        * (1 + 0.144 * M ** 2) ** 0.65)
    return frac_lam_fus * C_f_lam + (1 - frac_lam_fus) * C_f_turb


def C_fe_wing(frac_lam_wing, Reynolds, M):
    """_summary_

    :param frac_lam_wing: fraction laminar flow wing
    :type frac_lam_wing: _type_
    :param Reynolds: _description_
    :type Reynolds: _type_
    :return: _description_
    :rtype: _type_
    """
    C_f_lam = 1.328 / np.sqrt(Reynolds)
    C_f_turb = 0.455 / (((np.log10(Reynolds)) ** 2.58)
                        * (1 + 0.144 * M ** 2) ** 0.65)

    return frac_lam_wing * C_f_lam + (1 - frac_lam_wing) * C_f_turb


def CD_fus(C_fe_fus, FF_fus, S_wet_fus):
    """_summary_

    :param C_fe_fus: skin friction coefficient fuselage
    :type C_fe_fus: _type_
    :param FF_fus: form factor fuselage
    :type FF_fus: _type_
    :param S_wet_fus: _description_
    :type S_wet_fus: _type_
    :return: _description_
    :rtype: _type_
    """
    IF_fus = 1.0        # From WIGEON script
    return C_fe_fus * FF_fus * IF_fus * S_wet_fus


def CD_wing(C_fe_wing, FF_wing, S_wet_wing, S):
    """_summary_

    :param C_fe_wing: skin friction coefficient wing
    :type C_fe_wing: _type_
    :param FF_wing: form factor wing
    :type FF_wing: _type_
    :param S_wet_wing: _description_
    :type S_wet_wing: _type_
    :return: _description_
    :rtype: _type_
    """
    IF_wing = 1.1      # From WIGEON script

    CD_wing = max(float(C_fe_wing * FF_wing * IF_wing * S_wet_wing), 0.007)
    return CD_wing


def CD_tail(C_fe_wing, FF_tail, S_wet_tail):

    IF_tail = 1.0
    CD_tail = max(float(C_fe_wing * FF_tail * IF_tail * S_wet_tail), 0.005)
    return CD_tail


def CD0(S, S_tail, S_fus, CD_fus, CD_wing, CD_upsweep, CD_base, CD_tail, CD_flaps):
    """_summary_

    :param S: wing area
    :type S: _type_
    :param CD_fus: _description_
    :type CD_fus: _type_
    :param CD_wing: _description_
    :type CD_wing: _type_
    :param CD_upsweep: _description_
    :type CD_upsweep: _type_
    :param CD_base: _description_
    :type CD_base: _type_
    :return: _description_
    :rtype: _type_
    """

    leakage_factor = 1.075  # accounts for leakage from propellers etc.

    return ((CD_wing / S) + (CD_tail / S_tail) + (CD_fus / S_fus)) * leakage_factor + CD_upsweep + CD_base + CD_flaps


def CDi(CL, A, e):
    """_summary_

    :param CL: _description_
    :type CL: _type_
    :param A: _description_
    :type A: _type_
    :param e: _description_
    :type e: _type_
    :return: _description_
    :rtype: _type_
    """

    CDi = max(float(CL**2 / (np.pi * A * e)), 0.007)
    return CDi


def CD_flaps(angle_flap_deg):
    F_flap = 0.0144         # for plain flaps
    cf_c = 0.25             # standard value
    S_flap_S_ref = 0.501    # from Raymer's methods

    return F_flap*cf_c*S_flap_S_ref*(angle_flap_deg-10)

def lift_over_drag(CL_output, CD_output):
    """_summary_

    :param CL_output: _description_
    :type CL_output: _type_
    :param CD_output: _description_
    :type CD_output: _type_
    :return: _description_
    :rtype: _type_
    """
    return CL_output / CD_output


def Oswald_eff(A):
    """_summary_

    :param A: aspect ratio
    :type A: _type_
    :return: _description_
    :rtype: _type_
    """
    return 1.78 * (1 - 0.045 * A**0.68) - 0.64

def component_drag_estimation(WingClass, FuselageClass, VTailClass, AeroClass):

    # General flight variables
    re_var = Reynolds(const.rho_cr, const.v_cr, WingClass.chord_mac, const.mhu, const.k)
    M_var = Mach_cruise(const.v_cr, const.gamma, const.R, const.t_cr)
    AeroClass.e = Oswald_eff(WingClass.aspect_ratio)


    # Writing to Aeroclass
    AeroClass.deps_da = deps_da(AeroClass.cL_alpha, WingClass.aspect_ratio)

    # Form factors
    FF_fus_var = FF_fus(FuselageClass.length_fuselage, FuselageClass.diameter_fuselage)
    FF_wing_var = FF_wing(const.toc, const.xcm, M_var, sweep_m(WingClass.sweep_LE, const.xcm, WingClass.chord_root, WingClass.span, WingClass.taper))
    FF_tail_var = FF_tail(const.toc_tail, const.xcm_tail, M_var, VTailClass.quarterchord_sweep)

    # Wetted area
    S_wet_fus_var = S_wet_fus(FuselageClass.diameter_fuselage, FuselageClass.length_cockpit, FuselageClass.length_cabin, FuselageClass.length_tail)
    S_wet_wing_var = 2 * WingClass.surface  # from ADSEE slides
    S_wet_tail_var = 2 * VTailClass.surface

    # Miscellaneous drag
    AeroClass.cd_upsweep = CD_upsweep(FuselageClass.upsweep, FuselageClass.diameter_fuselage, S_wet_fus_var)
    AeroClass.cd_base = CD_base(M_var, const.A_base, S_wet_fus_var)

    # Skin friction coefficienct
    C_fe_fus_var = C_fe_fus(const.frac_lam_fus, re_var, M_var)
    C_fe_wing_var = C_fe_wing(const.frac_lam_wing, re_var, M_var)
    C_fe_tail_var = C_fe_wing(const.frac_lam_wing, re_var, M_var)

    # Total cd
    CD_fus_var = CD_fus(C_fe_fus_var, FF_fus_var, S_wet_fus_var)
    CD_wing_var = CD_wing(C_fe_wing_var, FF_wing_var, S_wet_wing_var, WingClass.surface)
    CD_tail_var = CD_tail(C_fe_tail_var, FF_tail_var, S_wet_tail_var)
    AeroClass.cd0_cruise = CD0(WingClass.surface, VTailClass.surface, FuselageClass.length_fuselage*FuselageClass.width_fuselage_outer, CD_fus_var, CD_wing_var,AeroClass.cd_upsweep, AeroClass.cd_base, CD_tail_var, CD_flaps=0)

    # Summation and L/D calculation
    CDi_var = CDi(AeroClass.cL_cruise, WingClass.aspect_ratio, AeroClass.e)
    AeroClass.cd_cruise = AeroClass.cd0_cruise + CDi_var
    AeroClass.ld_cruise = lift_over_drag(AeroClass.cL_cruise,AeroClass.cd_cruise)

    return WingClass, FuselageClass, VTailClass, AeroClass



def l_function(lam, spc, y, n, eps= 1e-10):
  """ Weissinger-L function, formulation by De Young and Harper.
      lam: sweep angle of quarter-chord (radians)
      spc: local span/chord
      y: y/l = y*
      n: eta/l = eta* """
 
  if abs(y-n) < eps:
    weissl = tan(lam)

  else:
    yp = abs(y)
    if n < 0.:
      weissl = sqrt((1.+spc*(yp+n)*tan(lam))**2. + spc**2.*(y-n)**2.) / \
                    (spc*(y-n) * (1.+spc*(yp+y)*tan(lam))) - 1./(spc*(y-n)) + \
                    2.*tan(lam) * sqrt((1.+spc*yp*tan(lam))**2. \
                                        + spc**2.*y**2.) / \
                    ((1.+spc*(yp-y)*tan(lam)) * (1.+spc*(yp+y)*tan(lam))) 
    else:
      weissl = -1./(spc*(y-n)) + sqrt((1.+spc*(yp-n)*tan(lam))**2. + \
                                       spc**2.*(y-n)**2.) / \
               (spc*(y-n) * (1.+spc*(yp-y)*tan(lam)))

  return weissl 

def weissinger_l( wing, alpha_root, m, plot= False): 
  """ Weissinger-L method for a swept, tapered, twisted wing.
      wing.span: span
      wing.root: chord at the root
      wing.tip: chord at the tip
      wing.sweep: quarter-chord sweep (degrees)
      wing.washout: twist of tip relative to root, +ve down (degrees)
      al: angle of attack (degrees) at the root
      m: number of points along the span (an odd number). 

      Returns:
      y: vector of points along span
      cl: local 2D lift coefficient cl
      ccl: cl * local chord (proportional to sectional lift)
      al_i: local induced angle of attack
      CL: lift coefficient for entire wing
      CDi: induced drag coefficient for entire wing """

  # Convert angles to radians
  lam = wing.quarterchord_sweep
  tw = -wing.washout

  # Initialize solution arrays
  O = m+2
  phi   = np.zeros((m))
  y     = np.zeros((m))
  c     = np.zeros((m))
  spc   = np.zeros((m))
  twist = np.zeros((m))
  theta = np.zeros((O))
  n     = np.zeros((O))
  rhs   = np.zeros((m,1))
  b     = np.zeros((m,m))
  g     = np.zeros((m,m))
  A     = np.zeros((m,m))

  # Compute phi, y, chord, span/chord, and twist on full span
  for i in range(m):
    phi[i]   = (i+1)*pi/float(m+1)                   #b[v,v] goes to infinity at phi=0
    y[i]     = cos(phi[i])                           #y* = y/l
    c[i]     = wing.chord_root + (wing.chord_tip-wing.chord_root)*y[i] #local chord
    spc[i]   = wing.span/c[i]                        #span/(local chord)
    twist[i] = abs(y[i])*tw                          #local twist

  # Compute theta and n
  for i in range(O):
    theta[i] = (i+1)*pi/float(O+1)
    n[i]     = cos(theta[i])
  n0 = 1.
  phi0 = 0.
  nO1 = -1.
  phiO1 = pi

  # Construct the A matrix, which is the analog to the 2D lift slope
  # print("Calculating aerodynamics ...")
  for j in range(m):
    # print("Point " + str(j+1) + " of " + str(m))
    rhs[j,0] = alpha_root + twist[j]

    for i in range(m):
      if i == j: b[j,i] = float(m+1)/(4.*sin(phi[j]))
      else: b[j,i] = sin(phi[i]) / (cos(phi[i])-cos(phi[j]))**2. * \
            (1. - (-1.)**float(i-j))/float(2*(m+1))

      g[j,i] = 0.
      Lj0 = l_function(lam, spc[j], y[j], n0)
      LjO1 = l_function(lam, spc[j], y[j], nO1)
      fi0 = 0.
      fiO1 = 0.
      for mu in range(m):
        fi0 += 2./float(m+1) * (mu+1)*sin((mu+1)*phi[i])*cos((mu+1)*phi0)
        fiO1 += 2./float(m+1) * (mu+1)*sin((mu+1)*phi[i])*cos((mu+1)*phiO1)

      for r in range(O):
        Ljr = l_function(lam, spc[j], y[j], n[r])
        fir = 0.
        for mu in range(m):
          fir += 2./float(m+1) * (mu+1)*sin((mu+1)*phi[i])*cos((mu+1)*theta[r])
        g[j,i] += Ljr*fir
      g[j,i] = -1./float(2*(O+1)) * ((Lj0*fi0 + LjO1*fiO1)/2. + g[j,i])

      if i == j: A[j,i] = b[j,i] + wing.span/(2.*c[j])*g[j,i]
      else: A[j,i] = wing.span/(2.*c[j])*g[j,i] - b[j,i]

  # Scale the A matrix
  A *= 1./wing.span 

  # Calculate ccl
  ccl = np.linalg.solve(A, rhs)

  # Add a point at the tip where the solution is known
  y = np.hstack((np.array([1.]), y))
  ccl = np.hstack((np.array([0.]), ccl[:,0]))
  c = np.hstack((np.array([wing.chord_tip]), c))
  twist = np.hstack((np.array([tw]), twist))

  # Return only the right-hand side (symmetry)
  nrhs = int((m+1)/2)+1    # Explicit int conversion needed for Python3
  y = y[0:nrhs]
  ccl = ccl[0:nrhs]

  # Sectional cl and induced angle of attack
  cl = np.zeros(nrhs)
  al_i = np.zeros(nrhs)
  for i in range(nrhs):
    cl[i] = ccl[i]/c[i]
    al_e = cl[i]/(2*np.pi)
    al_i[i] = alpha_root + twist[i] - al_e

  # Integrate to get CL and CDi
  CL = 0.
  CDi = 0.
  area = 0.
  for i in range(1,nrhs):
    dA = 0.5*(c[i]+c[i-1]) * (y[i-1]-y[i])
    dCL = 0.5*(cl[i-1]+cl[i]) * dA
    dCDi = sin(0.5*(al_i[i-1]+al_i[i])) * dCL
    CL += dCL
    CDi += dCDi
    area += dA
  CL /= area 
  CDi /= area

  if plot:
    # Mirror to left side for plotting
    npt = y.shape[0]
    y = np.hstack((y, np.flipud(-y[0:npt-1])))
    cl = np.hstack((cl, np.flipud(cl[0:npt-1])))
    ccl = np.hstack((ccl, np.flipud(ccl[0:npt-1])))

    fig, axarr = plt.subplots(2, sharex=True)

    axarr[0].plot(y*wing.span/2, cl, 'r', y*wing.span/2, ccl/wing.chord_mac, 'b' )
    axarr[0].set_xlabel('y')
    axarr[0].set_ylabel('Sectional lift coefficient')
    axarr[0].legend(['Cl', 'cCl / MAC'], numpoints=1)
    axarr[0].grid()
    axarr[0].annotate("CL: {:.4f}\nCDi: {:.5f}".format(CL,CDi), xy=(0.02,0.95), 
                      xycoords='axes fraction', verticalalignment='top', 
                      bbox=dict(boxstyle='square', fc='w', ec='m'), color='m')

    yroot = 0.
    ytip =wing.span/2.
    xroot = [0.,wing.chord_root]
    xrootc4 =wing.chord_root/4.
    xtipc4 = xrootc4 +wing.span/2.*tan(wing.quarterchord_sweep*pi/180.)
    xtip = [xtipc4 - 0.25*wing.chord_tip, xtipc4 + 0.75*wing.chord_tip]

    ax = axarr[1]


    x = [xroot[0],xtip[0],xtip[1],xroot[1], \
         xtip[1],xtip[0],xroot[0]]
    y_plot = [yroot,ytip,ytip,yroot, \
        -ytip,   -ytip,yroot]
    xrng = max(x) - min(x)
    yrng =wing.span

    ax.plot(y_plot, x, 'k')
    ax.set_xlabel('y')
    ax.set_ylabel('x')
    ax.set_xlim(-ytip-yrng/7.,ytip+yrng/7.)
    ax.set_ylim(min(x)-xrng/7., max(x)+xrng/7.)
    ax.set_aspect('equal', 'datalim')
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.grid()
    ax.annotate("Area: {:.4f}\nAR: {:.4f}\nMAC: {:.4f}".format(wing.surface, 
                wing.aspect_ratio,wing.chord_mac), xy=(0.02,0.95),
                xycoords='axes fraction', verticalalignment='top',
                bbox=dict(boxstyle='square', fc='w', ec='m'), color='m')

    plt.show()


  return y*wing.span/2., cl, ccl, al_i*180./pi, CL, CDi



def get_aero_planform( aero, wing, m, plot= False ):
  """ Returns 

  :param flight_perf: FlightPerfomance data structurer
  :type flight_perf: FlightPerformance
  :param vtol: VTOL data struct
  :type vtol: VTOL
  :param wing: SingleWing data struct
  :type wing: SingleWing
  :param m:  number of points along the span (an odd number). 
  :type m: int

  """  

  alpha_lst = list()
  cL_lst =  list()
  induced_drag_lst = list()

  for alpha in np.arange(0, np.radians(15), np.pi/180):
    span_points, cl_vector, ccl_vector, local_aoa, cL, CDi = weissinger_l( wing, alpha, m)
    alpha_lst.append(alpha)
    induced_drag_lst.append(CDi)
    cL_lst.append(cL)

  aero.cL_alpha = (cL_lst[-1] - cL_lst[0])/(alpha_lst[-1] - alpha_lst[0]) 
  lift_over_drag_arr = cL_lst

  if plot:
    # Create a figure with 3 subplots arranged in 3 rows
    fig, axs = plt.subplots(2, 1, figsize=(6, 8))  # 3 rows, 1 column

    # You can now customize each subplot as needed, for example:
    axs[0].plot(np.degrees(alpha_lst), cL_lst,  "k->")
    axs[0].set_xlabel("Angle of Attack")
    axs[0].set_ylabel("Lift coefficient")
    axs[0].grid()


    axs[1].plot(cL_lst, induced_drag_lst, "k->")
    axs[1].set_xlabel("Lift coefficient")
    axs[1].set_ylabel("Induced Drag coefficient")
    axs[1].grid()

    plt.tight_layout()
    plt.show()
  return np.array(alpha_lst),np.array(cL_lst), np.array(induced_drag_lst)

def wing_planform(wing, MTOM: float, WS_cruise: float):
    """ Sizes the wing based on the wingloading, mtom and wingloading in 
    cruise configuration
    """    

    wing.surface = MTOM / WS_cruise * 9.81
    wing.span  = np.sqrt( wing.aspect_ratio * wing.surface)
    wing.chord_root = 2 * wing.surface / ((1 + wing.taper) * wing.span)
    wing.chord_tip = wing.taper * wing.chord_root
    wing.chord_mac = (2 / 3) * wing.chord_root  * ((1 + wing.taper + wing.taper ** 2) / (1 + wing.taper))
    wing.y_mac = (wing.span / 6) * ((1 + 2 * wing.taper) / (1 + wing.taper))
    wing.sweep_LE = 0.25 * (2 * wing.chord_root / wing.span) * (1 - wing.taper) + np.tan(np.radians(wing.quarterchord_sweep))

    wing.x_lemac = wing.y_mac * wing.sweep_LE
    return wing

def vtail_planform(vtail):

    vtail.span  = np.sqrt( vtail.aspect_ratio * vtail.surface)
    vtail.chord_root = 2 * vtail.surface / ((1 + vtail.taper) * vtail.span)
    vtail.chord_tip = vtail.taper * vtail.chord_root
    vtail.chord_mac = (2 / 3) * vtail.chord_root  * ((1 + vtail.taper + vtail.taper ** 2) / (1 + vtail.taper))

    return vtail

def winglet_correction(wing, winglet_correction: float):
    wing.effective_aspect_ratio = wing.aspect_ratio * winglet_correction
    wing.effective_span = wing.span*np.sqrt(wing.effective_aspectratio/wing.aspect_ratio)
    return wing
    
def winglet_factor(h_wl, b, k_wl):  #https://www.fzt.haw-hamburg.de/pers/Scholz/Aero/AERO_PUB_Winglets_IntrinsicEfficiency_CEAS2017.pdf

    return (1+(2/k_wl)*(h_wl/b))**2
