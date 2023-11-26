import AetheriaPackage.sim_contr as sc
from AetheriaPackage.data_structs import *
from AetheriaPackage.ISA_tool import ISA
import numpy as np

aircraft= AircraftParameters.load(r"output\Final_design_Nov_25_15.48\design_state_Nov_25_15.48.json")
wing = Wing.load(r"output\Final_design_Nov_25_15.48\design_state_Nov_25_15.48.json")
engine = Engine.load(r"output\Final_design_Nov_25_15.48\design_state_Nov_25_15.48.json")
power = Power.load(r"output\Final_design_Nov_25_15.48\design_state_Nov_25_15.48.json")
fuselage = Fuselage.load(r"output\Final_design_Nov_25_15.48\design_state_Nov_25_15.48.json")
vtail = VeeTail.load(r"output\Final_design_Nov_25_15.48\design_state_Nov_25_15.48.json")
Stability = Stab.load(r"output\Final_design_Nov_25_15.48\design_state_Nov_25_15.48.json")
aero = Aero.load(r"output\Final_design_Nov_25_15.48\design_state_Nov_25_15.48.json")

min_span = sc.span_vtail(1,fuselage.diameter_fuselage,30*np.pi/180)
sc.size_vtail_opt( wing, fuselage, vtail, Stability, aero, aircraft, power, engine, min_span, CLh_initguess= -0.32)


print(vtail.shs)

