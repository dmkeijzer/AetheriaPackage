import AetheriaPackage.propulsion as prop
from AetheriaPackage.data_structs import *
from AetheriaPackage.ISA_tool import ISA
import numpy as np

mission = AircraftParameters.load(r"output\run_optimizaton_Nov_23_15.25\design_state_Nov_23_15.25.json")
engine = Engine.load(r"output\run_optimizaton_Nov_23_15.25\design_state_Nov_23_15.25.json")
aero = Aero.load(r"output\run_optimizaton_Nov_23_15.25\design_state_Nov_23_15.25.json")

prop.propcalc(aero, mission, engine, 1000)


print(engine.t_factor)
