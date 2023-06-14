from dataclasses import dataclass
import json
import sys
import os
import pathlib as pl

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

@dataclass
class Stab():

    Cm_de: float = None
    Cn_dr: float = None
    Cxa: float = None
    Cxq: float = None
    Cza: float = None
    Czq: float = None
    Cma: float = None
    Cmq: float = None
    Cz_adot: float = None
    Cm_adot: float = None
    muc: float = None
    Cxu: float = None
    Czu: float = None
    Cx0: float = None
    Cz0: float = None
    Cmu: float = None
    Cyb: float = None
    Cyp: float = None
    Cyr: float = None
    Clb: float = None
    Clp: float = None
    Clr: float = None
    Cnb: float = None
    Cnp: float = None
    Cnr: float = None
    Cy_dr: float = None
    Cy_beta_dot: float = None
    Cn_beta_dot: float = None
    mub: float = None
    sym_eigvals: float = None
    asym_eigvals: float = None
    cg_front: float = None
    cg_rear: float = None

    def load(self):
        """ Initializes the class automatically from the JSON file
        """
        with open(r"input/data_structures/aetheria_constants.json") as jsonFile:
            data = json.load(jsonFile)
        self.Cm_de = data["Cm_de"]
        self.Cn_dr = data["Cn_dr"]
        self.Cxa = data["Cxa"]
        self.Cxq = data["Cxq"]
        self.Cza = data["Cza"]
        self.Czq = data["Czq"]
        self.Cma = data["Cma"]
        self.Cmq = data["Cmq"]
        self.Cz_adot = data["Cz_adot"]
        self.Cm_adot = data["Cm_adot"]
        self.muc = data["muc"]
        self.Cxu = data["Cxu"]
        self.Czu = data["Czu"]
        self.Cx0 = data["Cx0"]
        self.Cz0 = data["Cz0"]
        self.Cmu = data["Cmu"]
        self.Cyb = data["Cyb"]
        self.Cyp = data["Cyp"]
        self.Cyr = data["Cyr"]
        self.Clb = data["Clb"]
        self.Clp = data["Clp"]
        self.Clr = data["Clr"]
        self.Cnb = data["Cnb"]
        self.Cnp = data["Cnp"]
        self.Cnr = data["Cnr"]
        self.Cy_dr = data["Cy_dr"]
        self.Cy_beta_dot = data["Cy_beta_dot"]
        self.Cn_beta_dot = data["Cn_beta_dot"]
        self.mub = data["mub"]
        self.sym_eigvals = data["sym_eigvals"]
        self.asym_eigvals = data["asym_eigvals"]
        self.cg_front = data["cg_front"]
        self.cg_rear = data["cg_rear"]

    def dump(self):
        with open(r"input/data_structures/aetheria_constants.json") as jsonFile:
            data = json.load(jsonFile)
        """Dumps values into json file"""

        data["Cn_dr"] = self.Cn_dr
        data["Cm_de"] = self.Cm_de
        data["Cxa"] = self.Cxa
        data["Cxq"] = self.Cxq
        data["Cza"] = self.Cza
        data["Czq"] = self.Czq
        data["Cma"] = self.Cma
        data["Cmq"] = self.Cmq
        data["Cz_adot"] = self.Cz_adot
        data["Cm_adot"] = self.Cm_adot
        data["muc"] = self.muc
        data["Cxu"] = self.Cxu
        data["Czu"] = self.Czu
        data["Cx0"] = self.Cx0
        data["Cz0"] = self.Cz0
        data["Cmu"] = self.Cmu
        data["Cyb"] = self.Cyb
        data["Cyp"] = self.Cyp
        data["Cyr"] = self.Cyr
        data["Clb"] = self.Clb
        data["Clp"] = self.Clp
        data["Clr"] = self.Clr
        data["Cnb"] = self.Cnb
        data["Cnp"] = self.Cnp
        data["Cnr"] = self.Cnr
        data["Cy_dr"] = self.Cy_dr
        data["Cy_beta_dot"] = self.Cy_beta_dot
        data["Cn_beta_dot"] = self.Cn_beta_dot
        data["mub"] = self.mub
        data["sym_eigvals"] = self.sym_eigvals
        data["asym_eigvals"] = self.asym_eigvals
        data["cg_front"] = self.cg_front
        data['cg_rear'] = self.cg_rear


        with open(r"input/data_structures/aetheria_constants.json", "w") as jsonFile:
            json.dump(data, jsonFile, indent=6)