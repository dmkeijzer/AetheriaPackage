import json
import sys
import pathlib as pl
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




def y_pos_inboard(Wing, Engine, ppclearance = 0.1):
    outboardy = y_pos_outboard(Wing, Engine)
    return outboardy - 2*Engine.prop_radius - ppclearance

def y_pos_outboard(Wing, Engine):
    return Wing.span / 2 + Engine.hub_radius

def y_pos_tail(VTail, Engine):
    return VTail.span * np.cos(VTail.dihedral) / 2 + Engine.hub_radius

def x_pos_inboard(Wing, Engine, ppclearance = 0.1):
    x_lewing_at_rotor_y = Wing.x_lewing + np.sin(Wing.sweep_LE) * y_pos_inboard(Wing, Engine, ppclearance = ppclearance)
    return x_lewing_at_rotor_y - Engine.pylon_length

def x_pos_outboard(Wing):
    return Wing.x_lewing + np.sin(Wing.sweep_LE) * Wing.span / 2 + 0.2 * Wing.chord_tip

def x_pos_tail(Fuse, VTail):
    return Fuse.length_fuselage + 0.25 * (VTail.surface / VTail.span)




if __name__ == "__main__":
    sys.path.append(str(list(pl.Path(__file__).parents)[2]))
    os.chdir(str(list(pl.Path(__file__).parents)[2]))
    from input.data_structures import *
    from modules.stab_ctrl.loading_diagram import loading_diagram

    with open("input/data_structures/aetheria_constants.json") as f_in:
        data = json.load(f_in)



    Wing = Wing()
    Engine = Engine()
    Fuse = Fuselage()
    VTail = VeeTail()

    Wing.load()
    Engine.load()
    Fuse.load()
    VTail.load()

    print(y_pos_inboard(Wing, Engine))
    print(y_pos_outboard(Wing, Engine))
    print(y_pos_tail(VTail, Engine))
    print()
    print(x_pos_inboard(Wing, Engine))
    print(x_pos_outboard(Wing))
    print(x_pos_tail(Fuse, VTail))
    print()
    print(loading_diagram(Wing.x_lewing + Wing.x_lemac + 0.24*Wing.chord_mac, Fuse.length_fuselage, data))
