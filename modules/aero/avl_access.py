
import os
import json
import sys
import numpy as np
import sys
import pathlib as pl
import matplotlib.pyplot as plt

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from modules.avlwrapper import Geometry, Surface, Section, NacaAirfoil, Control, Point, Spacing, Session, Case, Parameter

from input.data_structures.wing import Wing
from input.data_structures.aero import Aero
import input.data_structures.GeneralConstants as const

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
    tip_section = Section(leading_edge_point=Point(np.sin(wing.sweep_LE)*wing.span, wing.span/2, 0),
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

    coeff = np.polyfit(y_le_per_strip, lift_force_per_strip, 2)

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


