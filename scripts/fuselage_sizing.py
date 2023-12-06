import AetheriaPackage.structures as struct
from AetheriaPackage.data_structs import *
from AetheriaPackage.ISA_tool import ISA
import numpy as np

mission = AircraftParameters.load(r"output\run_optimizaton_Nov_25_15.04\design_state_Nov_25_15.04.json")
engine = Engine.load(r"output\run_optimizaton_Nov_25_15.04\design_state_Nov_25_15.04.json")
fuselage= Fuselage.load(r"output\run_optimizaton_Nov_25_15.04\design_state_Nov_25_15.04.json")
Pstack = FuelCell()
Tank = HydrogenTank()

struct.get_fuselage_sizing(Tank, Pstack, mission, fuselage)
V = 
fl.plot_variable(h0, b0, , l_tank, number_tanks,'Beta', np.linspace(0.1, 0.6, 60), 'ARe', 2.25)

