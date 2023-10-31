# -*- coding: utf-8 -*-import json
import sys
import pathlib as pl
import os
import json
from pydantic import BaseModel, FilePath

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

import input.GeneralConstants as const


class HydrogenTank(BaseModel):
    """
    This class is to estimate the parameters of a Hydrogen tank.

    :param EnergyDensity  [kWh/kg]
    :param VolumeDensity [kWh/l]
    :param cost [US$/kWh] (not loaded)
    """
    label: str = "Tank"
    energyDensity: float = 1.8
    volumeDensity: float = 0.6*3.6e6*1000 # J/m^3
    volumeDensity_h2: float = 2e6*1000 # J/m^3
    cost: float =  16
    fuel_cell_eff: float =  0.55


    def mass(self,energy) -> float:
        """
        :return: Mass of the battery
        """
        return energy / self.energyDensity

    def volume(self,energy) -> float:
        """
        :return: Volume of the tank [m^3]
        """
        return energy / self.volumeDensity / self.fuel_cell_eff 

    def price(self,energy) -> float:
        """
        :return: Approx cost of the battery [US$]
        """
        return energy * self.cost

    def dump(self, file_path: FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)

        data["Tank"] = self.model_dump()

        with open(file_path, "w") as jsonFile:
            json.dump(data, jsonFile, indent=4)


if __name__ == "__main__":
    test = HydrogenTank()
    test.load()
    print(test.volume(786963069.9958103/3.6e6))
