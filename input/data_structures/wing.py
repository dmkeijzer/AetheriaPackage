from dataclasses import dataclass
import json
import sys
import os
import pathlib as pl

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from input.data_structures.GeneralConstants import *

@dataclass
class Wing():
    surface: float = None
    taper: float = None
    aspectratio: float = None
    span: float = None
    chord_root: float = None
    chord_tip: float = None
    chord_mac: float = None
    y_mac: float = None
    sweep_LE: float = None
    quarterchord_sweep: float = None
    x_lemac: float = None
    effective_aspectratio: float = None
    effective_span: float = None

    # Aero part of the wing
    cd: float = None
    cd0: float = None
    cL_alpha: float = None
    cL_cruise: float = None
    cm_ac: float = None
    e: float = None
    cm_alpha: float = None
    cL_approach: float = None
    cL_alpha0: float = None
    mach_cruise: float = None
    cL_alpha0_approach: float = None
    x_lewing: float = None
    v_stall: float = None
    v_approach: float = None
    alpha_approach: float = None

    def load(self):
        with open(r"input/data_structures/aetheria_constants.json") as jsonFile:
            data = json.load(jsonFile)

        self.surface = data["S"]
        self.taper = data["taper"]
        self.aspectratio = data["A"]
        self.span = data["b"]
        self.chord_root = data["c_root"]
        self.chord_tip = data["c_tip"]
        self.chord_mac = data["mac"]
        self.y_mac = data["y_mac"]
        self.sweep_LE = data["sweep_le"]
        self.quarterchord_sweep = data["quarterchord_sweep"]
        self.x_lemac = data["x_lemac"]
        self.cd = data["cd"]
        self.cd0 = data["cd0"]
        self.cL_alpha = data["clalpha"]
        self.cL_cruise = data["cL_cruise"]
        self.cm_ac = data["cm_ac"]
        self.e = data["e"]
        self.cm_alpha = data["cm_alpha"]
        self.cL_approach = data["cLmax"]
        self.cL_alpha0 = data["cL0"]
        self.mach_cruise = v_cr / a_cr
        self.cL_alpha0_approach = data["cL0"]
        self.x_lewing = data["x_lewing"]
        self.v_stall = data["v_stall"]
        self.alpha_approach = data["alpha_approach"]
        self.thickness_to_chord = data["thickness_to_chord"]

        # self.effective_aspectratio =  data[""] # Left out for now since it is not implemented yet
        # self.effective_span =  data[""] # Left out for now since it is not implemented yet

        return data

    def dump(self):

        with open(r"input/data_structures/aetheria_constants.json") as jsonFile:
            data = json.load(jsonFile)

        data["S"] = self.surface
        data["taper"] = self.taper
        data["A"] = self.aspectratio
        data["b"] = self.span
        data["c_root"] = self.chord_root
        data["c_tip"] = self.chord_tip
        data["mac"] = self.chord_mac
        data["y_mac"] = self.y_mac
        data["sweep_le"] = self.sweep_LE
        data["quarterchord_sweep"] = self.quarterchord_sweep
        data["x_lemac"] = self.x_lemac
        data["cd"] = self.cd
        data["x_lewing"] = self.x_lewing
        data["thickness_to_chord"] = self.thickness_to_chord

        with open(r"input/data_structures/aetheria_constants.json", "w") as jsonFile:
            json.dump(data, jsonFile, indent=4)


if __name__ == "__main__":
    WingClass = Wing()
    WingClass.load()

    print(WingClass.surface)
    print(WingClass.chord_tip)
