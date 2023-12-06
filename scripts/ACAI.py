
import sys
sys.path.append(".")
from AetheriaPackage.sim_contr import acai, pylon_calc, create_rotor_loc
import numpy as np
from scipy.constants import g
from AetheriaPackage.data_structs import *
import numpy as np

aircraft= AircraftParameters.load(r"output\Final_design_Nov_25_15.48\design_state_Nov_25_15.48.json")
wing = Wing.load(r"output\Final_design_Nov_25_15.48\design_state_Nov_25_15.48.json")
engine = Engine.load(r"output\Final_design_Nov_25_15.48\design_state_Nov_25_15.48.json")
power = Power.load(r"output\Final_design_Nov_25_15.48\design_state_Nov_25_15.48.json")
fuselage = Fuselage.load(r"output\Final_design_Nov_25_15.48\design_state_Nov_25_15.48.json")
vtail = VeeTail.load(r"output\Final_design_Nov_25_15.48\design_state_Nov_25_15.48.json")
stability = Stab.load(r"output\Final_design_Nov_25_15.48\design_state_Nov_25_15.48.json")
aero = Aero.load(r"output\Final_design_Nov_25_15.48\design_state_Nov_25_15.48.json")

r = 0.275
r1  = np.sqrt(3)*r/2
Bf = np.array([[1,1,1,1,1,1],
                [0, -r1,-r1,0,r1,r1],
                [r, r/2, -r/2, -r, -r/2, r/2],
                [-0.1,0.1,-0.1,0.1,-0.1,0.1]])
fcmin = 0*np.ones((6,1))
fcmax=6.125*np.ones((6,1))
Tg = np.array([[1.535*9.8],
                [0],
                [0],
                [0]])
res = acai(Bf, fcmin, fcmax, Tg)
print(res)
assert np.isclose(res, 1.486052554907109)
## From OG ACAI paper

# loc =  create_rotor_loc(wing.span, 1, vtail.span, vtail.dihedral, 
#                         aircraft.wing_loc*fuselage.length_fuselage - wing.chord_root/4,
#                         wing.chord_root, wing.chord_tip, wing.sweep_LE,
#                         fuselage.length_fuselage, vtail.chord_root)
# res = pylon_calc(wing, vtail, fuselage, stability, aircraft, loc, pylon_step=2, Tfac_step= 0.4)
# print(res)
