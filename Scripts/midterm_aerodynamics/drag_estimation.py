import sys
import pathlib as pl
sys.path.append(str(list(pl.Path(__file__).parents)[2]))
import json
import os
import numpy as np

# Import from modules and input folder
import input.GeneralConstants  as const
from  modules.class2drag.clean_class2drag  import *

os.chdir(str(list(pl.Path(__file__).parents)[2]))
# import CL_cruise from json files

# Define the directory and filenames of the JSON files
dict_directory = "input"
dict_names = ["J1_constants.json", "L1_constants.json", "W1_constants.json"]

atm = ISA(const.h_cruise)
t_cr = atm.temperature()
rho_cr = atm.density()
mhu = atm.viscosity_dyn()

# Loop through the JSON files
for dict_name in dict_names:
    # Load data from JSON file
    with open(os.path.join(dict_directory, dict_name)) as jsonFile:
        data = json.load(jsonFile)
        if data["name"] == "J1":
            mac = data["mac"]
        else: 
            mac = (data["mac1"] + data["mac2"])/2

        #General flight variables
        re_var = Reynolds(rho_cr, const.v_cr, mac, mhu)
        M_var = Mach_cruise(const.v_cr, const.gamma, const.R,t_cr)
        if data["name"]== "J1":
            Oswald_eff_var = Oswald_eff(A)
        else: 
            Oswald_eff_var = Oswald_eff_tandem(b1,b2,h)

        #Form factor
        FF_fus_var = FF_fus(l, d)
        FF_wing_var = FF_wing(toc, xcm, M_var, sweep_m)

        #Wetted area
        S_wet_fus_var = S_wet_fus(d, const.l1, const.l2, const.l3)
        S_wet_wing_var = 2*S          # PLACEHOLDER

        #Miscoulanisous drag
        CD_upsweep_var = CD_upsweep(u, d, S_wet_fus_var)
        CD_base_var = CD_base(M_var, A_base, S_wet_fus_var)

        #Skin friction coefficienct
        C_fe_fus_var = C_fe_fus(const.frac_lam_fus, re_var)
        C_fe_wing_var = C_fe_wing(const.frac_lam_wing, re_var)

        #Total cd
        CD_fus_var = CD_fus(C_fe_fus_var, FF_fus_var, S_wet_fus_var)
        CD_wing_var = CD_wing(C_fe_wing_var, FF_wing_var, S_wet_wing_var)
        CD0_var = CD0(S, CD_fus_var, CD_wing_var, CD_upsweep_var, CD_base_var)
       
        #Summation and L/D calculation
        CDi_var = CDi(CL, A, e_var)
        CD_var = CD(CD0_var, CDi_var)
        lift_over_drag_var = lift_over_drag(CL, CD_var)
