from dataclasses import dataclass
import json

@dataclass
class FuellCell:
    """This class is to estimate the parameters of a Fuell Cell"""

    PowerDensity_kWkg: float
    powerRequired_kW:float 
    VolumeDensity_kWL: float 
    Cost : float = None
    Efficiency: float = None

    def load(self):
        jsonfilename = "aetheria_constants.json"
        with open(jsonfilename) as jsonfile:
            data = json.open(jsonfile)
            raise NotImplementedError

    def mass(self):
        """
        :param power: Power requirement for the fuell cell[kW]
        :param sp_P_den: Power density of the fuell cell[kW/kg]
        :return: Mass of the battery
        """
        return self.powerRequired_kW / self.PowerDensity

    def volume(self):
        """
        :param Powery: Power required from the fuell cell [kW]
        :param vol_en_den: Volumetric energy density of the fuell cell [kW/l]
        :return: Volume of the fuell cell [m^3]
        """
        return self.powerRequired_kW /self.VolumeDensity  * 0.001

    def price(self):
        """
        :param power: Required power for the fuell cell [Wh]
        :param cost: Cost per Wh of the battery [US$/kW]
        :return: Approx cost of the battery [US$]
        """
        return self.powerRequired_kW * self.Cost

