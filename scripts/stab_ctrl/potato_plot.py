import matplotlib.pyplot as plt
import json
import numpy as np
import sys
import os
import pathlib as pl

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

# design_name="J1"     # enter J1, W1, or L1
#
# if design_name=="J1":
#     with open('input/J1_constants.json') as file:
#         constants = json.load(file)
#     x1=3
#     x2=8.5
# elif design_name=="L1":
#     with open('input/L1_constants.json') as file:
#         constants = json.load(file)
#     x1=0.5
#     x2=8
# elif design_name=="W1":
#     with open('input/W1_constants.json') as file:
#         constants = json.load(file)
#     x1=0.5
#     x2=9
# else:
#     print("ERROR")
#
# #Assumption, nacelle position = powertrain position
# #IN THE LOADING ARRAY, ALWAYS PUT THE CARGO FIRST. EACH INPUT IS: [POSITION,MASS]
# loading_array=[[5.056,125],[1.723,77],[3.453,77],[3.453,77],[4.476,77],[4.476,77]]
# h2_position=6.3
#
# rotor_locations=constants['x_rotor_loc']
# nacelle_position=np.mean(rotor_locations)
# nacelle_weight=constants['nacelle_weight']
# h2_weight=constants['h2_weight']
# lg_weight=constants['lg_weight']
# fuselage_weight=constants['fuselage_weight']
# fuselage_pos=constants['l_fuse']*0.45               #BASED ON ADSEE 3
# powertrain_weight=constants['powertrain_weight']
# misc_weight=constants['misc_weight']
#
# if design_name=="J1":
#     wing_weight=constants['wing_weight']
#     hor_tail_weight=constants['hortail_weight']
# else:
#     wing_weight=constants['wing1_weight']
#     hor_tail_weight=constants['wing2_weight']
#
#
#
# lg_pos=fuselage_pos #For now, assume it coincides with the cg of the fuselage
# misc_position=fuselage_pos #For now, assume it coincides with the cg of the fuselage
#
# #####Calculation
# mass_array=[]
# mass_pos_array=[]
# oem_mass=wing_weight+hor_tail_weight+powertrain_weight+misc_weight+h2_weight+nacelle_weight+lg_weight+fuselage_weight
# oem_num=(x1*wing_weight+x2*hor_tail_weight+h2_position*h2_weight+nacelle_position*powertrain_weight+misc_position*misc_weight+nacelle_position*nacelle_weight+lg_weight*lg_pos+fuselage_weight*fuselage_pos)
# oem_pos=oem_num/oem_mass
#
# mass_array.append(oem_mass)
# mass_pos_array.append(oem_pos)
#
# mass=oem_mass
# mass_num=oem_num
# mass_pos=oem_pos
#
# for i in loading_array:
#     mass=mass+i[1]
#     mass_num=mass_num+i[1]*i[0]
#     mass_pos=mass_num/mass
#     mass_array.append(mass)
#     mass_pos_array.append(mass_pos)
#
#
# mass_array2=[]
# mass_pos_array2=[]
#
# mass=oem_mass+loading_array[0][1]
# mass_num=oem_num+loading_array[0][1]*loading_array[0][0]
# mass_pos=mass_num/mass
#
# mass_array2.append(mass)
# mass_pos_array2.append(mass_pos)
#
# loading_array_wo_cargo=loading_array[1:]
#
# for j in reversed(loading_array_wo_cargo):
#     mass=mass+j[1]
#     mass_num=mass_num+j[1]*j[0]
#     mass_pos=mass_num/mass
#     mass_array2.append(mass)
#     mass_pos_array2.append(mass_pos)
#
# plt.plot(mass_pos_array, mass_array, 'blue')
# plt.plot(mass_pos_array2, mass_array2, 'blue')
# plt.xlabel("cg location with respect to leading edge of fuselage [m]")
# plt.ylabel("Mass [kg]")
# #plt.savefig('L1_potato.png')
    


def J1loading(x1, x2,constants):
    loading_array = [[5.056, 125], [1.723, 77], [3.453, 77], [3.453, 77], [4.476, 77], [4.476, 77]]
    h2_position = 6.3
    rotor_locations = constants['x_rotor_loc']
    nacelle_position = np.mean(rotor_locations)
    nacelle_weight = constants['nacelle_weight']
    h2_weight = constants['h2_weight']
    lg_weight = constants['lg_weight']
    fuselage_weight = constants['fuselage_weight']
    fuselage_pos = constants['l_fuse'] * 0.45  # BASED ON ADSEE 3
    powertrain_weight = constants['powertrain_weight']
    misc_weight = constants['misc_weight']
    wing_weight=constants['wing_weight']
    hor_tail_weight=constants['vtail_weight']
    lg_pos = fuselage_pos  # For now, assume it coincides with the cg of the fuselage
    misc_position = fuselage_pos  # For now, assume it coincides with the cg of the fuselage
    mass_array = []
    mass_pos_array = []
    oem_mass = wing_weight + hor_tail_weight + powertrain_weight + misc_weight + h2_weight + nacelle_weight + lg_weight + fuselage_weight
    oem_num = (
                x1 * wing_weight + x2 * hor_tail_weight + h2_position * h2_weight + nacelle_position * powertrain_weight + misc_position * misc_weight + nacelle_position * nacelle_weight + lg_weight * lg_pos + fuselage_weight * fuselage_pos)
    oem_pos = oem_num / oem_mass

    mass_array.append(oem_mass)
    mass_pos_array.append(oem_pos)

    mass = oem_mass
    mass_num = oem_num
    mass_pos = oem_pos

    for i in loading_array:
        mass = mass + i[1]
        mass_num = mass_num + i[1] * i[0]
        mass_pos = mass_num / mass
        mass_array.append(mass)
        mass_pos_array.append(mass_pos)

    mass_array2 = []
    mass_pos_array2 = []

    mass = oem_mass + loading_array[0][1]
    mass_num = oem_num + loading_array[0][1] * loading_array[0][0]
    mass_pos = mass_num / mass

    mass_array2.append(mass)
    mass_pos_array2.append(mass_pos)

    loading_array_wo_cargo = loading_array[1:]

    for j in reversed(loading_array_wo_cargo):
        mass = mass + j[1]
        mass_num = mass_num + j[1] * j[0]
        mass_pos = mass_num / mass
        mass_array2.append(mass)
        mass_pos_array2.append(mass_pos)
    res = {"frontcg": min(mass_pos_array), "rearcg": max(mass_pos_array2)}
    res_margin = {"frontcg": min(mass_pos_array)-0.1*(max(mass_pos_array2)-min(mass_pos_array)), "rearcg": max(mass_pos_array2)+0.1*(max(mass_pos_array2)-min(mass_pos_array))}
    return res, res_margin

if __name__ == "__main__":
    from input.data_structures import *
    Fuse = Fuselage()
    Wing = Wing()
    Fuse.load()
    Wing.load()
    lf = Fuse.length_fuselage
    x1 = Wing.x_lewing + 0.24 * Wing.chord_mac + Wing.x_lemac
    print(J1loading(x1, lf))