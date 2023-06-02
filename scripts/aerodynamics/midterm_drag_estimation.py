import sys
import pathlib as pl
sys.path.append(str(list(pl.Path(__file__).parents)[2]))
import json
import os
import numpy as np

# Import from modules and input folder
import input.data_structures.GeneralConstants as const
from  modules.aero.clean_class2drag import *
from input.data_structures.ISA_tool import ISA

os.chdir(str(list(pl.Path(__file__).parents)[2]))
# import CL_cruise from json files

# Define the directory and filenames of the JSON files
TEST = int(input("\n\nType 1 if you want to write the JSON data to your download folder instead of the repo, type 0 otherwise:\n")) # Set to true if you want to write to your downloads folders instead of rep0
dict_directory = "input"
dict_names = ["J1_constants.json", "L1_constants.json", "W1_constants.json"]
download_dir = os.path.join(os.path.expanduser("~"), "Downloads")

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
        re_var = Reynolds(rho_cr, const.v_cr, mac, mhu, const.k)
        M_var = Mach_cruise(const.v_cr, const.gamma, const.R, t_cr)
        if data["name"]== "J1":
            Oswald_eff_var = Oswald_eff(data["A"])
        else: 
            Oswald_eff_var = Oswald_eff_tandem(data["b1"],data["b2"],data["h_wings"])

        # Writing to JSON file
        data["e"] = Oswald_eff_var
        data["depsda"] = 0.1
        
        #Form factor
        FF_fus_var = FF_fus(data["l_fuselage"], data["d_fuselage"])
        FF_wing_var = FF_wing(const.toc, const.xcm, M_var, data["sweep_m"])

        #Wetted area
        S_wet_fus_var = S_wet_fus(data["d_fuselage"], data["l_cockpit"], data["l_cabin"], data["l_tail"])
        S_wet_wing_var = 2*data["S"]        # from ADSEE slides

        #Miscoulanisous drag
        CD_upsweep_var = CD_upsweep(data["upsweep"], data["d_fuselage"], S_wet_fus_var)
        CD_base_var = CD_base(M_var, const.A_base, S_wet_fus_var)

        #Skin friction coefficienct
        C_fe_fus_var = C_fe_fus(const.frac_lam_fus, re_var, M_var)
        C_fe_wing_var = C_fe_wing(const.frac_lam_wing, re_var, M_var)

        #Total cd
        CD_fus_var = CD_fus(C_fe_fus_var, FF_fus_var, S_wet_fus_var)
        CD_wing_var = CD_wing(data["name"], C_fe_wing_var, FF_wing_var, S_wet_wing_var, data["S"])
        CD0_var = CD0(data["S"], CD_fus_var, CD_wing_var, CD_upsweep_var, CD_base_var)
       
        #Summation and L/D calculation
        if data["name"] == "J1":
            CDi_var = CDi(data["name"],data["cL_cruise"], data["A"], data["e"])
        else:
            A = (data["A1"]+data["A2"])/2
            CDi_var = CDi(data["name"],data["cL_cruise"], A, data["e"])

        CD_var = CD(CD0_var, CDi_var)
        lift_over_drag_var = lift_over_drag(data["cL_cruise"], CD_var)
        print("CD0_wing", CD_wing_var/data["S"])
        
        
        # Writing to JSON file
        data["ld_cr"] = lift_over_drag_var
        data["cd"] = CD_var
        data["cd0"] = CD0_var
        data["cd_upsweep"] = CD_upsweep_var
        data["cd_base"] = CD_base_var

        if TEST:
            with open(os.path.join(download_dir, dict_name), "w") as jsonFile:
                json.dump(data, jsonFile, indent= 6)
        else:
            with open(os.path.join(dict_directory, dict_name), "w") as jsonFile:
                json.dump(data, jsonFile, indent= 6)