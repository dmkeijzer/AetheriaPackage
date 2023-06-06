from input.data_structures.aero import Aero
from input.data_structures.wing import Wing
from modules.avlwrapper import Geometry, Surface, Section, NacaAirfoil, Control, Point, Spacing, Session, Case, Parameter
import json
import sys
import numpy as np
import pathlib as pl
sys.path.append(str(list(pl.Path(__file__).parents)[2]))

Wingclass = Wing()
Wingclass.load()
Aeroclass = Aero()
Aeroclass.load()
if __name__ == '__main__':

    # wing root section with a flap control and NACA airfoil
    root_section = Section(leading_edge_point=Point(0, 0, 0),
                           chord=Wingclass.chord_root,
                           airfoil=NacaAirfoil(naca='2412'))

    # wing tip
    tip_section = Section(leading_edge_point=Point(np.sin(Wingclass.sweep_LE)*Wingclass.span, Wingclass.span/2, 0),
                          chord=Wingclass.chord_tip,
                          airfoil=NacaAirfoil(naca='2410'))

    # wing surface defined by root and tip sections
    wing_surface = Surface(name="Wing",
                           n_chordwise=8,
                           chord_spacing=Spacing.cosine,
                           n_spanwise=12,
                           span_spacing=Spacing.cosine,
                           y_duplicate=0.0,
                           sections=[root_section, tip_section])

    # geometry object (which corresponds to an AVL input-file)
    geometry = Geometry(name="Test wing",
                        reference_area=Wingclass.surface,
                        reference_chord=Wingclass.chord_mac,
                        reference_span=Wingclass.span,
                        reference_point=Point(0, 0, 0),
                        surfaces=[wing_surface])

    # Cases (multiple cases can be defined)
    # Case defined by one angle-of-attack which is i_cs, calculated before convergence
    cruise_case = Case(name='Cruise', alpha=3.1493,
                       CL=Aeroclass.cL_cruise, CDo=Aeroclass.cd0)

    # More elaborate case, angle-of-attack of 4deg, elevator parameter which sets Cm (pitching moment) to 0.0
    cruise_trim_case = Case(name='Trimmed',
                            alpha=4.0,
                            elevator=Parameter(name='elevator', constraint='Cm', value=0.0))

    # create session with the geometry object and the cases
    # session = Session(geometry=geometry, cases=[cruise_case, landing_case])
    session = Session(geometry=geometry, cases=[cruise_case])

    # get results and write the resulting dict to a JSON-file
    session.show_geometry()
    results = session.get_results()
    with open('out.json', 'w') as f:
        f.write(json.dumps(results))
