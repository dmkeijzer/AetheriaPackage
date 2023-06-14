import sys
import os
import pathlib as pl
from scipy.optimize import minimize
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time

sys.path.append(str(list(pl.Path(__file__).parents)[2]))

import  modules.structures.wingbox_georgina as wb
import input.data_structures.GeneralConstants as const
from input.data_structures.material import Material
from input.data_structures.wing import Wing
from input.data_structures.engine import Engine
from input.data_structures.ISA_tool import ISA
from input.data_structures.aero import Aero
import modules.structures.wingbox_steven as ws
#------------ Instantiate classes and load values from JSON -----------------------
WingClass = Wing()
MatClass = Material()
EngClass = Engine()
AeroClass =  Aero()
ISAClass = ISA(const.h_cruise)

WingClass.load()
MatClass.load()
EngClass.load()
AeroClass.load()
#-----------------------------------------------------------------------------------------

#--------------------------- Assign correct values to global values in wingbox_georgina -----------

t_sp = 0.005
h_st = 0.1344444
t_st = 0.017333
t_sk = 5e-4
y = np.linspace(0,WingClass.span/2,10)


WingboxClass = ws.Wingbox(WingClass, EngClass, MatClass, AeroClass, HOVER=True)
#print(WingboxClass.N_xy(t_sp, h_st,t_st,t_sk,y))

print(f"Global buckling = {WingboxClass.global_local(h_st,t_st,t_sk)/1e6}[MPa]") #IS ANALYZED OVER THE ENTIRE SPAN NOT FOR EACH SECTION
print(f"Post buckling = {WingboxClass.post_buckling(t_sp, h_st,t_st,t_sk,y)/1e6}[MPa]")
print(f"Von mises = {WingboxClass.von_Mises(t_sp, h_st,t_st,t_sk,y)/1e6}[MPa]")
print(f"Buckling constraint = {WingboxClass.buckling_constr( t_sp, h_st,t_st,t_sk,y)}[-]")
print(f"Flange local buckling = {WingboxClass.flange_loc_loc(t_st,t_sk,h_st)/1e6}[MPa]")
print(f"Local column buckling = {WingboxClass.local_column(h_st,t_st,t_sk)/1e6}[MPa]")
print(f"Crippling constraint = {WingboxClass.crippling( h_st,t_st,t_sk)/1e6}[MPa]")
print(f"Web flange = {WingboxClass.web_flange( h_st,t_st,t_sk)/1e6}[MPa]")


t_sp_lst = np.linspace(5e-3,1e-1,10)
h_st_lst = np.linspace(1e-2,0.15,10)
t_st_lst = np.linspace(1e-3,5e-2,10)
t_sk_lst = np.linspace(5e-4,3e-2,10)

n_str_lst = np.linspace(1,10,10)
n_ribs_lst = np.linspace(2, 10, 9)


# # t_sp_lst = np.arange(1e-4,1e-1,0.5e-3)#200
# # h_st_lst = np.arange(1e-3,1e-1,1e-3)#100
# # t_st_lst = np.arange(1e-4,5e-2,2e-4)#250
# # t_sk_lst = np.arange(1e-4,5e-2,2e-4)


# start_time = time.time()
# open("modules/structures/results_steven.txt","w").close()
# print(f"Class setup took {time.time()-start_time}[sec]")
# for t_sp in t_sp_lst:
#     for h_st in h_st_lst:
#         for t_st in t_st_lst:
#             for t_sk in t_sk_lst:
#                 WingboxClass.checkconstraints(t_sp,h_st,t_st,t_sk,y)
# print(f"Running through each iteration took {time.time()-start_time}")
# # Optclass = ws.Wingbox(WingClass, EngClass, MatClass, AeroClass)
# #print(np.sum(Optclass.post_buckling(1e-3,1e-3,1e-3,1e-3,1e-3,1e-3,1e-3)))

# # optimizertest = ws.WingboxOptimizer(x0,WingClass, EngClass, MatClass, AeroClass)
