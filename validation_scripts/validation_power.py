import numpy as np
import sys
import pathlib as pl

sys.path.append(str(list(pl.Path(__file__).parents)[1]))


from scripts.power.finalPowersizing import power_system_convergences
from input.data_structures import Power, AircraftParameters


json_path = r"C:\Users\damie\OneDrive\Desktop\Damien\DSE\AetheriaPackage\output\run_optimizaton_Oct_17_22.24\design_state_Oct_17_22.24.json"

power = Power.load(json_path)
aircraft = AircraftParameters.load(json_path)

power_system_convergences(power, aircraft)

print(power)
