from dataclasses import dataclass
import json
import sys
import os
import pathlib as pl

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
    c_control_surface_to_c_vee_ratio: float = None
    #CD0: float = None WRONG
    ruddervator_efficiency: float = None
    span: float = None
    vtail_weight: float = None



    @property
    def aspectratio(self):
        return self.span**2/self.surface

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


        with open(r"input/data_structures/aetheria_constants.json", "w") as jsonFile:
            json.dump(data, jsonFile, indent=6)