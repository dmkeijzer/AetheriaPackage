import numpy as np
import sys
import os
import pathlib as pl
import json
# Path handling
sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))


# Importing modules and input data
from modules.midterm_structures.ClassIIWeightEstimation import *
import  input.GeneralConstants as const 

#Getting JSON files
with open('input/J1_constants.json', 'r') as f:
    J1 = json.load(f)

with open('input/L1_constants.json', 'r') as f:
    L1 = json.load(f)

with open('input/W1_constants.json', 'r') as f:
    W1 = json.load(f)

# Computing masses

#---------------------------------J1---------------------------------
weight_J1 = VtolWeightEstimation()
weight_J1.add_component(Wing(J1["mtom"], J1["S"], const.n_ult, J1["A"]))
weight_J1.add_component(Fuselage(J1["name"], J1["mtom"], np.pi*J1["w_fuse"], J1["l_fuse"], const.npax + 1 ))
weight_J1.add_component(LandingGear(J1["mtom"]))
weight_J1.add_component(Powertrain(J1["power_hover"], const.p_density))
weight_J1.add_component(HorizontalTail(J1["mtom"], J1["S_h"], J1["A_h"], J1["t_r_h"]))
weight_J1.add_component(Nacelle(J1["mtom"]))
weight_J1.add_component(H2System(J1["mission_energy"]/3.6e6, J1["power_cruise"]/1e3, J1["power_hover"]/1e3))
weight_J1.add_component(Miscallenous(J1["mtom"],J1["oem"], const.npax + 1))

mass_J1 = weight_J1.compute_mass()

J1_results = [i.mass for i in weight_J1.components]
J1["oem"] = mass_J1
J1["mtom"] = mass_J1 + const.m_pl
J1["wing_weight"] = J1_results[0]
J1["fuselage_weight"] = J1_results[1]
J1["lg_weight"] = J1_results[2]
J1["powertrain_weight"] = J1_results[3]
J1["hortail_weight"] = J1_results[4]
J1["nacelle_weight"] = J1_results[5]
J1["h2_weight"] = J1_results[6]
J1["misc_weight"] = J1_results[7]

#---------------------------------L1---------------------------------
weight_L1 = VtolWeightEstimation()
weight_L1.add_component(Wing(L1["mtom"], L1["S1"], const.n_ult, L1["A1"]))
weight_L1.add_component(Wing(L1["mtom"], L1["S2"], const.n_ult, L1["A2"]))
weight_L1.add_component(Fuselage(L1["name"], L1["mtom"], np.pi*L1["w_fuse"], L1["l_fuse"], const.npax + 1 ))
weight_L1.add_component(LandingGear(L1["mtom"]))
weight_L1.add_component(Powertrain(L1["power_hover"], const.p_density))
weight_L1.add_component(Nacelle(L1["mtom"]))
weight_L1.add_component(H2System(L1["mission_energy"]/3.6e6, L1["power_cruise"]/1e3, L1["power_hover"]/1e3))
weight_L1.add_component(Miscallenous(L1["mtom"],L1["oem"], const.npax + 1))

mass_L1 = weight_L1.compute_mass()

L1_results = [i.mass for i in weight_L1.components]
L1["oem"] = mass_L1
L1["mtom"] = mass_L1 + const.m_pl
L1["wing1_weight"] = L1_results[0]
L1["wing2_weight"] = L1_results[1]
L1["fuselage_weight"] = L1_results[2]
L1["lg_weight"] = L1_results[3]
L1["powertrain_weight"] = L1_results[4]
L1["nacelle_weight"] = L1_results[5]
L1["h2_weight"] = L1_results[6]
L1["misc_weight"] = L1_results[7]

#--------------------------------- W1---------------------------------
weight_W1 = VtolWeightEstimation()
weight_W1.add_component(Wing(W1["mtom"], W1["S1"], const.n_ult, W1["A1"]))
weight_W1.add_component(Wing(W1["mtom"], W1["S2"], const.n_ult, W1["A2"]))
weight_W1.add_component(Fuselage(W1["name"], W1["mtom"], np.pi*W1["w_fuse"], W1["l_fuse"], const.npax + 1 ))
weight_W1.add_component(LandingGear(W1["mtom"]))
weight_W1.add_component(Powertrain(W1["power_hover"], const.p_density))
weight_W1.add_component(Nacelle(W1["mtom"]))
weight_W1.add_component(H2System(W1["mission_energy"]/3.6e6, W1["power_cruise"]/1e3, W1["power_hover"]/1e3))
weight_W1.add_component(Miscallenous(W1["mtom"],W1["oem"], const.npax + 1))

mass_W1 = weight_W1.compute_mass()

W1_results = [i.mass for i in weight_W1.components]
W1["oem"] = mass_W1
W1["mtom"] = mass_W1 + const.m_pl
W1["wing1_weight"] = W1_results[0]
W1["wing2_weight"] = W1_results[1]
W1["fuselage_weight"] = W1_results[2]
W1["lg_weight"] = W1_results[3]
W1["powertrain_weight"] = W1_results[4]
W1["nacelle_weight"] = W1_results[5]
W1["h2_weight"] = W1_results[6]
W1["misc_weight"] = W1_results[7]

#--------------------------------- Printing results---------------------------------

print("\n", mass_J1, [[i.id for i in weight_J1.components], [i.mass for i in weight_J1.components]],"\n")
print(mass_L1,[[i.id for i in weight_L1.components], [i.mass for i in weight_L1.components]], "\n")
print(mass_W1,[[i.id for i in weight_W1.components], [i.mass for i in weight_W1.components]])

#--------------------------------- Write results to JSON---------------------------------
TEST = False # Set to true if you want to write to your downloads folders instead of rep0

download_dir = os.path.join(os.path.expanduser("~"), "Downloads")
os.chdir(str(list(pl.Path(__file__).parents)[2]))

if TEST:
    dir = [os.path.join(download_dir, "J1.json"),
     os.path.join(download_dir, "L1.json"),
     os.path.join(download_dir, "W1.json")]
else:
    dir = ['input/J1_constants.json',
    'input/L1_constants.json',
    'input/W1_constants.json']

# Writing to JSON files
with open(dir[0], 'w') as f:
    json.dump(J1, f, indent=6)

with open(dir[1], 'w') as f:
    json.dump(L1, f, indent=6)

with open(dir[2], 'w') as f:
    json.dump(W1, f, indent=6)