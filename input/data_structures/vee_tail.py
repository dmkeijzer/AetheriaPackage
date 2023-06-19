from dataclasses import dataclass
import json
import sys
import os
import pathlib as pl
import numpy as np

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

@dataclass
class VeeTail():
    CL_cruise: float = None
    Vh_V2: float = None
    length_wing2vtail: float = None
    rudder_max: float = None
    elevator_min: float = None
    dihedral: float = None
    surface: float = None
    taper: float = 1
    c_control_surface_to_c_vee_ratio: float = None
    #CD0: float = None WRONG
    ruddervator_efficiency: float = None
    span: float = None
    vtail_weight: float = None
    quarterchord_sweep: float = None
    thickness_to_chord: float = None


    @property
    def aspectratio(self):
        return self.span**2/self.surface
    
    @property
    def chord_root(self):
        return  2 *self.surface / ((1 + self.taper) * self.span)

    @property
    def chord_tip(self):
        return self.chord_root * self.taper

    @property
    def chord_mac(self):
        return  (2 / 3) * self.chord_root  * ((1 + self.taper + self.taper ** 2) / (1 + self.taper))




    def load(self):
        """ Initializes the class automatically from the JSON file
        """
        with open(r"input/data_structures/aetheria_constants.json") as jsonFile:
            data = json.load(jsonFile)
        self.CL_cruise = data["CL_cruise_h"]
        self.Vh_V2 = data["Vh_V2"]
        self.length_wing2vtail = data["length_wing2vtail"]
        self.rudder_max = data["rudder_max"]
        self.elevator_min = data["elevator_min"]
        self.dihedral = data["dihedral_vtail"]
        self.surface = data["surface_vtail"]
        self.c_control_surface_to_c_vee_ratio = data["c_control_surface_to_c_vee_ratio"]
        #self.CD0 = data["CD0_vtail"] WRONG
        self.ruddervator_efficiency = data["ruddervator_efficiency"]
        self.span = data["span_vtail"]
        self.vtail_weight= data["vtail_weight"]
        self.x_lewing = data["x_lewing"] + data["x_lemac"] + 0.24*data["mac"]+self.length_wing2vtail-0.25*self.chord_mac
        self.thickness_to_chord = data["thickness_to_chord_tail"]
        self.spar_thickness = data["spar_thickness_tail"]
        self.stringer_height = data["stringer_height_tail"]
        self.stringer_width = data["stringer_width_tail"]
        self.stringer_thickness = data["stringer_thickness_tail"]
        self.wingskin_thickness = data["wingskin_thickness_tail"]
        self.wing_weight = self.vtail_weight
        self.quarterchord_sweep = 1


    @property
    def sweep_LE(self):
        return 0.25 * (2 * self.chord_root /self.span) * (1 - self.taper) + np.tan(np.radians(self.quarterchord_sweep))

    def dump(self):
        with open(r"input/data_structures/aetheria_constants.json") as jsonFile:
            data = json.load(jsonFile)
        """Dumps values into json file"""
        data["CL_cruise_h"] = self.CL_cruise
        data["length_wing2vtail"] = self.length_wing2vtail
        data["rudder_max"] = self.rudder_max
        data["elevator_min"] = self.elevator_min
        data["dihedral_vtail"] = self.dihedral
        data["surface_vtail"] = self.surface
        data["c_control_surface_to_c_vee_ratio"] = self.c_control_surface_to_c_vee_ratio
        data["ruddervator_efficiency"] = self.ruddervator_efficiency
        data["span_vtail"] = self.span
        # data["vtail_weight"] = self.vtail_weight
        data["x_levtail"] = self.x_lewing
        data["vtail_weight"] = self.wing_weight
        data["spar_thickness_tail"] = self.spar_thickness
        data["stringer_height_tail"] = self.stringer_height
        data["stringer_width_tail"] = self.stringer_width
        data["stringer_thickness_tail"] = self.stringer_thickness
        data["wingskin_thickness_tail"] = self.wingskin_thickness


        with open(r"input/data_structures/aetheria_constants.json", "w") as jsonFile:
            json.dump(data, jsonFile, indent=6)