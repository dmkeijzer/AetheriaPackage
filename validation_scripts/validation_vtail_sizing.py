import numpy as np
import sys
import pathlib as pl

import AetheriaPackage.data_structs as data
from AetheriaPackage.sim_contr import span_vtail, size_vtail_opt


path = r"C:\Users\damie\OneDrive\Desktop\Damien\DSE\AetheriaPackage\output\run_optimizaton_Oct_30_22.32\design_state_Oct_30_22.32.json"

fuselage = data.Fuselage.load(path)
aircraft = data.AircraftParameters.load(path)
WingClass = data.Wing.load(path)
Aeroclass = data.Aero.load(path)
VtailClass = data.VeeTail.load(path)
PowerClass = data.Power.load(path)
EngineClass = data.Engine.load(path)
StabClass = data.Stab.load(path)


min_span = span_vtail(1,fuselage.diameter_fuselage,30*np.pi/180)
size_vtail_opt(WingClass, fuselage, VtailClass, StabClass, Aeroclass, aircraft, PowerClass, EngineClass, min_span, plot= True )


