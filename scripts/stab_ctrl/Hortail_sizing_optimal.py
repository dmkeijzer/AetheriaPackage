import json
import sys
import pathlib as pl
import os
import numpy as np
import pandas as pd

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from modules.stab_crtl.wing_loc_horzstab_sizing import wing_location_horizontalstab_size
from scripts.stab_ctrl.vee_tail_rudder_elevator_sizing import get_control_surface_to_tail_chord_ratio
from input.data_structures import *

WingClass = Wing().load()
FuseClass = Fuselage().load()
HorTailClass = HorTail().load()

ShS = wing_location_horizontalstab_size(WingClass, FuseClass, HorTailClass, CLh_approach = 1)[0,1]
Sh = ShS * WingClass.surface

get_control_surface_to_tail_chord_ratio(V_stall,Lambdah2,b,Fuselage_volume,S_hor,downwash_angle_landing,aoa_landing,CL_h,CL_a_h,V_tail_to_V_ratio,l_v,S,c,taper_h, AR_h,Cn_beta_req=-0.0571,beta_h=1,eta_h=0.95,total_deflection=20*np.pi/180,design_cross_wind_speed=5.14,step=0.1*np.pi/180)