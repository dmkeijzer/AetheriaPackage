# -*- coding: utf-8 -*-
import numpy as np
import sys
import os
import pathlib as pl
import json

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
download_dir = os.path.join(os.path.expanduser("~"), "Downloads")
import matplotlib.pyplot as plt

from modules.midterm_structures import *
import nvm_script

nvm_script



write_bool = int(input("\n\nType 1 if you want to write the JSON data to your download folder instead of the repo, type 0 otherwise:\n"))
def root_wingbox_stresses(dict_directory,dict_name,i):
    #First open the corresponding json file for the appropriate dictionary
    with open(dict_directory+"\\"+dict_name, "r") as jsonFile:
        data = json.loads(jsonFile.read())
    #If it's a tandem aircraft, take the rear wing since these are the most critical
    if data["tandem_bool"] ==1:
        c_root = data["c_root2"]
        wing_weight = data["wing2_weight"]
        b = data["b2"]
    else:
        c_root = data["c_root"]
        wing_weight = data["wing_weight"]
        b = data["b"]
    #Determine root chord height with the thickness to chord ratio
    h_root = c_root*data['thickness_to_chord']

    #Determine values
    thickness = 3E-3
    h_wingbox = 0.8*h_root
    w_wingbox = 0.6*c_root
    #h_wingbox = 0.8 * 1
    #w_wingbox = 0.6 * 2

    #Calculate geometric properties
    data["i_xx"] = i_xx_thinwalled(w_wingbox,h_wingbox,3E-3)
    data["i_zz"] = i_zz_thinwalled(w_wingbox,h_wingbox,3E-3)
    data["j_y"]  = j_y_thinwalled(w_wingbox,h_wingbox,3E-3)
    data["area_enclosed"] = enclosed_area_thinwalled(w_wingbox,h_wingbox,thickness)
    data["area"] = area_thinwalled(w_wingbox,h_wingbox,thickness)

    #Determine bending stress
    max_bending_stress_cr,min_bending_stress_cr = bending_stress(data["Mx_cr"],data["Mz_cr"],data["i_xx"],data["i_zz"],data["i_xz"],w_wingbox,h_wingbox)
    max_bending_stress_vf,min_bending_stress_vf = bending_stress(data["Mx_vf"],data["Mz_vf"],data["i_xx"],data["i_zz"],data["i_xz"],w_wingbox,h_wingbox)
    normal_stress_cr = normal_stress(data["Vy_cr"],data["area"])
    normal_stress_vf = normal_stress(data["Vy_vf"],data["area"])

    max_axial_cr = max(abs(max_bending_stress_cr+normal_stress_cr),abs(min_bending_stress_cr+normal_stress_cr))
    max_axial_vf = max(abs(max_bending_stress_vf+normal_stress_vf),abs(min_bending_stress_vf+normal_stress_vf))
    """CALC SHEAR FOR CRUISE"""
    #Determine shear stress due to shear loads
    tau_ab_shear,tau_bc_shear,tau_cd_shear,tau_da_shear = shear_thin_walled_rectangular_section(w_wingbox, h_wingbox, thickness, data["i_xx"],data["i_zz"],data['Vx_cr'],data['Vz_cr'])
    #Determine shear stress due to torsion
    tau_torsion = torsion_thinwalled_closed(data["T_cr"],thickness,data["area_enclosed"])
    #Combine shear stresses
    tau_ab = tau_ab_shear + tau_torsion
    tau_bc = tau_bc_shear + tau_torsion
    tau_cd = tau_cd_shear + tau_torsion
    tau_da = tau_da_shear + tau_torsion
    max_shear_cr = max(max(abs(tau_ab)),max(abs(tau_bc)),max(abs(tau_cd)),max(abs(tau_da)))

    """CALC SHEAR FOR VERTICAL FLIGHT"""
    #Determine shear stress due to shear loads
    tau_ab_shear,tau_bc_shear,tau_cd_shear,tau_da_shear = shear_thin_walled_rectangular_section(w_wingbox, h_wingbox, thickness, data["i_xx"],data["i_zz"],data['Vx_vf'],data['Vz_vf'])
    #Determine shear stress due to torsion
    tau_torsion = torsion_thinwalled_closed(data["T_vf"],thickness,data["area_enclosed"])
    #Combine shear stresses
    tau_ab = tau_ab_shear + tau_torsion
    tau_bc = tau_bc_shear + tau_torsion
    tau_cd = tau_cd_shear + tau_torsion
    tau_da = tau_da_shear + tau_torsion
    max_shear_vf = max(max(abs(tau_ab)),max(abs(tau_bc)),max(abs(tau_cd)),max(abs(tau_da)))

    """Fatigue equations"""
    N_wohler = wohlers_curve(3.15E14, 4.10,max_axial_cr/1000000)
    N_paris = paris_law(1E-12,1.225,max_axial_cr/1000000,3.0,thickness/2,0.45E-3)
    x = np.arange(0,w_wingbox,1E-5)
    y = np.arange(0,h_wingbox,1E-5)

    """CALC CRITICAL BENDING STRESS"""
    sigma_cr = critical_buckling_stress(4.00, thickness, w_wingbox)
    '''
    plt.plot(x,tau_ab,label="AB")
    plt.plot(y,tau_bc,label="BC")
    plt.plot(x,tau_cd,label="CD")
    plt.plot(y,tau_da,label="DA")
    plt.legend()
    plt.xlim(0,1.5)
    plt.show()

    print(f"AB goes from {tau_ab[0]} to {tau_ab[-1]}")
    print(f"BC goes from {tau_bc[0]} to {tau_bc[-1]}")
    print(f"CD goes from {tau_cd[0]} to {tau_cd[-1]}")
    print(f"DA goes from {tau_da[0]} to {tau_da[-1]}")
    '''
    print("CRUISE SITUATION")

    print(f"Max shear stress = {round(max_shear_cr/1000000,2)}[MPa]")
    print(f"Max bending stress = {round(max_axial_cr/1000000,2)}[MPa]\n")
    print(f"N_wohlers curve = {N_wohler/1000000}[*10^6]")
    print(f"N_paris curve = {N_paris / 1000000}[*10^6]\n")

    print("VERTICAL FLIGHT SITUATION")
    print(f"Max shear stress = {round(max_shear_vf/1000000,2)}[MPa]")
    print(f"Max bending stress = {round(max_axial_vf/1000000,2)}[MPa]\n")

    print("CRITICAL BUCKLING STRESS")
    print(f"Critical Buckling Stress = {round(sigma_cr/1000000,2)}[MPa]\n")

    print(f"")
    data["max_shear_cr"],data["max_axial_cr"] = max_shear_cr,max_bending_stress_cr
    data["max_shear_vf"],data["max_axial_vf"] = max_shear_vf,max_bending_stress_vf
    data["crit_buck_stress"] = sigma_cr
    if write_bool:
        with open(download_dir+"\\"+dict_name, "w") as jsonFile:
            json.dump(data, jsonFile,indent=2)
        print("Data written to downloads folder.")
    else:
        with open(dict_directory+"\\"+dict_name, "w") as jsonFile:
            json.dump(data, jsonFile,indent=2)
        print("Old files were overwritten.")
    return

dict_directory = str(list(pl.Path(__file__).parents)[2])+"\\input"          #determine file path
dict_name = ["J1_constants.json",  "L1_constants.json","W1_constants.json"] #define list with all the constants for each configuration
for i in range(len(dict_name)):                                             #iterate over each value
    print(dict_name[i])
    root_wingbox_stresses(dict_directory,dict_name[i],i)
