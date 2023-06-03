import sys
import os
import pathlib as pl
from scipy.optimize import minimize, LinearConstraint, Bounds
import numpy as np
import pickle

sys.path.append(str(list(pl.Path(__file__).parents)[2]))

import  modules.structures.wingbox_georgina as wb
import input.data_structures.GeneralConstants as const
from input.data_structures.material import Material
from input.data_structures.wing import Wing
from input.data_structures.engine import Engine
from input.data_structures.ISA_tool import ISA

#------------ Instantiate classes and load values from JSON -----------------------
WingClass = Wing()
MatClass = Material()
EngClass = Engine()
ISAClass = ISA(const.h_cruise)

WingClass.load()
MatClass.load()
EngClass.load()
#-----------------------------------------------------------------------------------------

#--------------------------- Assign correct values to global values in wingbox_georgina -----------


wb.taper = WingClass.taper
wb.rho = MatClass.rho
wb.W_eng = EngClass.mass_pertotalengine
wb.E = MatClass.E
wb.poisson = MatClass.poisson
wb.pb= MatClass.pb
wb.beta= MatClass.beta
wb.g= MatClass.g
wb.sigma_yield = MatClass.sigma_yield
wb.m_crip = MatClass.m_crip
wb.sigma_uts = MatClass.sigma_uts
wb.n_max= const.n_max_req


#------------------------------- Run script and save----------------------------------------------

x0=np.array([WingClass.span, WingClass.chord_root, 0.003, 0.002, 0.01, 0.01, 0.010,0.003,0.004,0.0022])    # :param x0: Initial estimate Design vector X = [b, cr, tsp, trib, L, bst, hst, tst, wst, t]
# bnds = wb.create_bounds(WingClass) # create bounds
# bnds = ((8, 12), (1, 4), (0.001, 0.009), (0.001, 0.003), (0.003, 0.015), (0.001, 0.01),(0.001, 0.01),(0.001, 0.003),(0.004, 0.005),(0.001, 0.003))

# This will not work:
#bounds = Bounds([-2.0,-0.5,-2.0],[2.0,0.8,0.7])

# This works
lb = [8, 1, 0.001, 0.001, 0.003, 0.001, 0.001, 0.001, 0.002, 0.001]
ub = [12, 4, 0.009, 0.003, 0.015, 0.02, 0.10, 0.004, 0.005, 0.003]
# A = [list(i) for i in np.eye(10)]
# lcons = LinearConstraint(A, lb=lb, ub=ub, keep_feasible=True)
bnds = Bounds(lb, ub, keep_feasible= True)
# Res = minimize(RosenbrockN, x0, method = 'trust-constr', constraints=lcons) 
res = wb.wingbox_optimization(x0, bnds, WingClass)

with open(r"output/structures/wingbox_output.pkl", "wb") as f:
    pickle.dump(res, f)
    print("Succesfully loaded data structure into wingbox_output.pkl")
 
#------------------------------- Print out results  ----------------------------------------------
str = ["span", "chord root", "t spar", "t rib", "Rib pitch", 
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
        print(f"{str_ele} = {np.round(res_ele, 4)} [m]")

#---------------------------------------------------------------------------------------------------


