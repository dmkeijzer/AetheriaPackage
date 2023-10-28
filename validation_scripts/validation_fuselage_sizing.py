import numpy as np
import sys
import pathlib as pl

sys.path.append(str(list(pl.Path(__file__).parents)[1]))

import input.data_structures  as data
import input.GeneralConstants as const
from modules.structures.fuselage_length import get_fuselage_sizing, plot_variable


path = r"C:\Users\damie\OneDrive\Desktop\Damien\DSE\AetheriaPackage\output\run_optimizaton_Oct_28_13.51\design_state_Oct_28_13.51.json"

fuselage = data.Fuselage.load(path)
aircraft = data.AircraftParameters.load(path)
tank = data.HydrogenTank()
pstack = data.FuelCell()
WingClass = data.Wing.load(path)
Aeroclass = data.Aero.load(path)
VtailClass = data.VeeTail.load(path)
PowerClass = data.Power.load(path)
EngineClass = data.Engine.load(path)
StabClass = data.Stab.load(path)


get_fuselage_sizing(tank, pstack, aircraft, fuselage, validate=True)
plot_variable(1.81, 1.6, fuselage.volume_powersys,  np.linspace(1,8, 50), 2, "ARe", np.linspace(1.5,3,30), "Beta", 0.5  )
