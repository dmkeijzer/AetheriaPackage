import numpy as np
import sys
import pathlib as pl

sys.path.append(str(list(pl.Path(__file__).parents)[1]))

from scripts.power.finalPowersizing import power_system_convergences
from modules.stab_ctrl.loading_diagram import loading_diagram
import input.data_structures  as data
import input.GeneralConstants as const
from modules.stab_ctrl.aetheria_stability_derivatives_edited import downwash, downwash_k
from modules.stab_ctrl.vtail_sizing_optimal import size_vtail_opt
from scripts.structures.vtail_span import span_vtail


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

print()

