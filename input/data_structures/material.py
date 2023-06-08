from dataclasses import dataclass
import json
import sys
import pathlib as pl
import os

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

@dataclass
class Material():
    E: float = None # Youngs modules
    poisson: float = None # Poisson ratio
    beta: float = None # Length of the fuseglage
    sigma_yield: float = None # Length of the cabin
    sigma_uts: float = None # Ultimate tensile strength
    m_crip: float = None #  Upsweep of the fuselage
    rho: float = None #  Density of material
    pb: float = None #  post buckling ratio
    g: float = None #  Density of material
    shear_modulus: float = None #Shear modulus

    def load(self):
        """ Initializes the class automatically from the JSON file
        """
        with open(r"input/data_structures/aetheria_constants.json") as jsonFile:
            data = json.load(jsonFile)["material"]
        self.E = data['youngsmodulus']
        self.poisson = data["poisson"]
        self.beta = data["beta"]
        self.sigma_yield = data["sigma_yield"]
        self.sigma_uts = data["ultimate_tensile_stress"]
        self.m_crip = data["m_crip"]
        self.rho = data["rho"]
        self.pb = data["pb"]
        self.g = data["g"]
        self.shear_modulus = data["shear_modulus"]