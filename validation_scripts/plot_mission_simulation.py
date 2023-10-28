import numpy as np
import sys
import pathlib as pl

sys.path.append(str(list(pl.Path(__file__).parents)[1]))


from scripts.power.finalPowersizing import power_system_convergences
from modules.flight_perf.performance import get_performance_updated
from input.data_structures import Power, AircraftParameters, Aero, Wing, Engine

json_path = r"C:\Users\damie\OneDrive\Desktop\Damien\DSE\AetheriaPackage\output\run_optimizaton_Oct_17_22.24\design_state_Oct_17_22.24.json"

aero = Aero.load(json_path)
mission = AircraftParameters.load(json_path)
wing = Wing.load(json_path)
engine = Engine.load(json_path)
power = Power.load(json_path)

get_performance_updated(aero, mission, wing, engine, power, True)

