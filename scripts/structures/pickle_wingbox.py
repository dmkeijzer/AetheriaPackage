import pickle
import numpy as np

with open(r"C:\Users\damie\OneDrive\Desktop\Damien\DSE\AetheriaPackage\output\structures\wingbox_output.pkl", "rb") as f:
    res = pickle.load(f)
 

str = [ "t spar", "t rib", "Rib pitch", 
        "Pitch stringer", "Height Stringer", "t stringer", "Stringer Flange Width", "thickness"]
vec_res = res.x


print(f"\nExited succesfully = {res.success} [kg]")
print(f"\nExecution time = {np.round(res.execution_time, 1)} [s] = {np.round(res.execution_time/60, 1)} [min]")
print(f"\nRequired iterations = {res.nit} ")
print(f"\nWing weight = {res.fun} [kg]\n")

i = 0

for str_ele, res_ele  in zip(str, vec_res):
    i += 1
    if i > 2:
        print(f"{str_ele} = {np.round(res_ele*1000, 4)} [mm]")
    else:     
        print(f"{str_ele} = {np.round(res_ele*1000, 4)} [mm]")