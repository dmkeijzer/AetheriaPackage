import sys
import pathlib as pl
sys.path.append(str(list(pl.Path(__file__).parents)[2]))
import json
import os
import numpy as np

# Import from modules and input folder
import input.data_structures.GeneralConstants  as const
from  modules.aero.prop_wing_interaction  import *
from input.data_structures.ISA_tool import ISA

os.chdir(str(list(pl.Path(__file__).parents)[2]))
# import CL_cruise from json files

# Define the directory and filenames of the JSON files
TEST = int(input("\n\nType 1 if you want to write the JSON data to your downlo1ad folder instead of the repo, type 0 otherwise:\n")) # Set to true if you want to write to your downloads folders instead of rep0
dict_directory = "input/data_structures"
file_name = "aetheria_constants.json"

with open(os.path.join(dict_directory, file_name)) as f:
    data = json.load(f)
download_dir = os.path.join(os.path.expanduser("~"), "Downloads")

atm = ISA(const.h_cruise)
t_cr = atm.temperature()
rho_cr = atm.density()
mhu = atm.viscosity_dyn()

diameter_propellers = 2*np.sqrt(data['diskarea']/(np.pi*6))
CL_wing_cruise = data['ld_cr']*data['cd']

slipstream_lift_var = prop_lift_slipstream(mach=data['mach_cruise'], sweep_half=-data['sweep_le'], T=1300, S_W=data['S'], n_e=4, D=diameter_propellers, b_W=data['b'], V_0=const.v_cr, angle_of_attack=0.03, CL_wing=CL_wing_cruise, i_cs=0, rho=rho_cr, delta_alpha_zero_f=0, alpha_0=data['alpha_zero_L'])[1]
prop_lift_var = prop_lift_thrust(T=1300, rho=rho_cr, V_0=const.v_cr, S_W=data['S'], angle_of_attack=0.03)
CL_total = slipstream_lift_var + prop_lift_var
print(CL_total)