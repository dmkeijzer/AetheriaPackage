
import pytest
import os
import sys
import pathlib as pl
import numpy as np

sys.path.append(str(list(pl.Path(__file__).parents)[1]))
os.chdir(str(list(pl.Path(__file__).parents)[1]))


import modules.structures.wingbox_georgina as  wb
from input.data_structures.wing import Wing
from input.data_structures.engine import Engine
from input.data_structures.material import Material
from input.data_structures.aero import Aero

#------------- SETUP ----------------------------------------------------------
WingClass =  Wing()
EngineClass = Engine()
MatClass = Material()
AeroClass = Aero()

WingClass.taper = 0.45
WingClass.span = 7
WingClass.chord_root = 1.5
WingClass.chord_tip  = 0.7
WingClass.sweep_LE = 3
WingClass.surface = 10
WingClass.chord_mac = 1.6


MatClass.rho = 2710
MatClass.E =  70e9
MatClass.poisson = 0.3 
MatClass.pb =  2.5
MatClass.beta =  1.42
MatClass.g =  5
MatClass.sigma_yield =  430e6
MatClass.m_crip = 0.85
MatClass.sigma_uts = 640e6

AeroClass.cL_cruise = 0.43



# print(m_eng(7, 1.5,0.003,0.003, 0.12, 0.005, 0.005,0.0025,0.004,[0.002]))
# print(post_buckling(10,1.7, 1e-2, 5e-2,1e-2,1e-2,1e-2,1e-2,1e-2,[1e-2]))

t_spar = 0.003
t_rib = 5e-3
L = 10e-2
b_st = 5e-2
h_st = 2e-2
t_st = 1e-3
w_st = 2e-2
t = 2e-3

tmax =  t
tmin =  t

WingboxClass = wb.Wingbox(WingClass, EngineClass, MatClass, AeroClass)
WingboxClass.n_max = 2.5
WingboxClass.engine_weight = 41.8
WingboxClass.max_rib_pitch = L 
WingboxClass.str_pitch = b_st
WingboxClass.lift_func = lambda y:-151.7143*9.81*y+531*9.81


#------------------------------------------------------------------------------------------
# web flange = 486900483.7870751
# crippling = -7442667.720708497
# local - column= 7794407.777376909
# flange loc loc= 170567009.39318538


def test_web_flange():
    res_optimization = WingboxClass.web_flange(h_st , t_st, tmax, tmin)[0]
    comparsion_value_notebook = 486900483.7870751
    print(f" web flange optimizatino = {res_optimization:.2e}")
    print(f"  web flange notebook = {comparsion_value_notebook:.2e}")
    assert res_optimization*comparsion_value_notebook > 0

def test_crippling():
    res_optimization = WingboxClass.crippling(h_st , t_st, w_st,  tmax, tmin)[0]
    comparsion_value_notebook = -7442667.720708497
    print(f" crippling optimizatino = {res_optimization:.2e}")
    print(f"  crippling notebook = {comparsion_value_notebook:.2e}")
    assert res_optimization*comparsion_value_notebook > 0

def test_local_column():
    res_optimization = WingboxClass.local_column(h_st , t_st, w_st,  tmax, tmin)[0]
    comparsion_value_notebook = 7794407.777376909
    print(f" local column optimizatino = {res_optimization:.2e}")
    print(f" local column notebook = {comparsion_value_notebook:.2e}")
    assert res_optimization*comparsion_value_notebook > 0

def test_flange_loc_loc():
    res_optimization = WingboxClass.flange_loc_loc( t_st, w_st,  tmax, tmin)[0]
    comparsion_value_notebook = 170567009.39318538
    print(f"flagne loc loc optimizatino = {res_optimization:.2e}")
    print(f" flange loc loc  notebook = {comparsion_value_notebook:.2e}")
    assert res_optimization*comparsion_value_notebook > 0




test_web_flange()
test_crippling()
test_local_column()
test_flange_loc_loc()